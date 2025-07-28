"""
Response Parser for KBLI Classification Results

This module provides functionality to parse, validate, and save LLM responses
for KBLI classification tasks. It handles JSON validation, error recovery,
and JSONL file output.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


class ResponseParser:
    """
    Parser for LLM responses in KBLI classification tasks.
    
    This class validates JSON responses from LLMs, handles malformed data,
    and saves results to JSONL files for further processing.
    """
    
    # Expected response schema
    REQUIRED_FIELDS = {
        'is_correct': bool,
        'confidence_score': (int, float),
        'reasoning': str,
        'alternative_codes': list,
        'alternative_reasoning': str
    }
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize response parser.
        
        Parameters
        ----------
        output_dir : str or Path, optional
            Directory for output files. If None, uses default.
        """
        
        if output_dir is None:
            output_dir = Path.cwd() / "result" / "pilot" / "extract_llm"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Response parser initialized with output directory: {self.output_dir}")
    
    def extract_json_from_response(self, response_text: str) -> Optional[Dict]:
        """
        Extract JSON object from LLM response text.
        
        Parameters
        ----------
        response_text : str
            Raw response text from LLM.
        
        Returns
        -------
        Dict or None
            Parsed JSON object, or None if extraction fails.
        """
        
        if not response_text or not response_text.strip():
            logger.warning("Empty response text")
            return None
        
        # Try direct JSON parsing first
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from code blocks
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',  # ```json { ... } ```
            r'```\s*(\{.*?\})\s*```',      # ``` { ... } ```
            r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',  # Any { ... } block
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    json_obj = json.loads(match.strip())
                    logger.debug("Successfully extracted JSON from response")
                    return json_obj
                except json.JSONDecodeError:
                    continue
        
        # Try to find JSON-like content and fix common issues
        try:
            # Look for content between first { and last }
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_candidate = response_text[start_idx:end_idx + 1]
                
                # Try some common fixes
                fixes = [
                    lambda x: x,  # No fix
                    lambda x: x.replace("true", "true").replace("false", "false"),  # Ensure lowercase booleans
                    lambda x: re.sub(r'(\w+):', r'"\1":', x),  # Quote unquoted keys
                    lambda x: re.sub(r',\s*}', '}', x),  # Remove trailing commas
                    lambda x: re.sub(r',\s*]', ']', x),  # Remove trailing commas in arrays
                ]
                
                for fix in fixes:
                    try:
                        fixed_json = fix(json_candidate)
                        json_obj = json.loads(fixed_json)
                        logger.debug("Successfully extracted and fixed JSON from response")
                        return json_obj
                    except (json.JSONDecodeError, Exception):
                        continue
        
        except Exception as e:
            logger.warning(f"Error during JSON extraction: {str(e)}")
        
        logger.warning("Failed to extract valid JSON from response")
        return None
    
    def validate_response(self, json_obj: Dict) -> Dict[str, Any]:
        """
        Validate parsed JSON response against expected schema.
        
        Parameters
        ----------
        json_obj : Dict
            Parsed JSON object to validate.
        
        Returns
        -------
        Dict
            Validation result with 'is_valid', 'errors', 'cleaned_data' fields.
        """
        
        validation_result = {
            'is_valid': True,
            'errors': [],
            'cleaned_data': {}
        }
        
        if not isinstance(json_obj, dict):
            validation_result['is_valid'] = False
            validation_result['errors'].append("Response is not a JSON object")
            return validation_result
        
        # Check required fields
        for field, expected_type in self.REQUIRED_FIELDS.items():
            if field not in json_obj:
                validation_result['errors'].append(f"Missing required field: {field}")
                validation_result['is_valid'] = False
                continue
            
            value = json_obj[field]
            
            # Type validation
            if not isinstance(value, expected_type):
                # Try to convert some types
                if field == 'confidence_score' and isinstance(value, str):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        validation_result['errors'].append(f"Field '{field}' has invalid type. Expected {expected_type}, got {type(value)}")
                        validation_result['is_valid'] = False
                        continue
                elif field == 'is_correct' and isinstance(value, str):
                    if value.lower() in ['true', '1', 'yes']:
                        value = True
                    elif value.lower() in ['false', '0', 'no']:
                        value = False
                    else:
                        validation_result['errors'].append(f"Field '{field}' has invalid boolean value: {value}")
                        validation_result['is_valid'] = False
                        continue
                else:
                    validation_result['errors'].append(f"Field '{field}' has invalid type. Expected {expected_type}, got {type(value)}")
                    validation_result['is_valid'] = False
                    continue
            
            # Additional validations
            if field == 'confidence_score':
                if not (0.0 <= value <= 1.0):
                    validation_result['errors'].append(f"confidence_score must be between 0.0 and 1.0, got {value}")
                    validation_result['is_valid'] = False
            
            elif field == 'alternative_codes':
                if not isinstance(value, list):
                    validation_result['errors'].append(f"alternative_codes must be a list, got {type(value)}")
                    validation_result['is_valid'] = False
                else:
                    # Validate each code in the list
                    clean_codes = []
                    for code in value:
                        if isinstance(code, (str, int)):
                            clean_codes.append(str(code).strip())
                        else:
                            validation_result['errors'].append(f"Invalid alternative code type: {type(code)}")
                    value = clean_codes
            
            validation_result['cleaned_data'][field] = value
        
        # Add any additional fields that might be present
        for field, value in json_obj.items():
            if field not in self.REQUIRED_FIELDS:
                validation_result['cleaned_data'][field] = value
        
        if validation_result['errors']:
            logger.warning(f"Validation errors: {validation_result['errors']}")
        
        return validation_result
    
    def parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse and validate LLM response.
        
        Parameters
        ----------
        response_text : str
            Raw response text from LLM.
        
        Returns
        -------
        Dict
            Parsing result with 'success', 'data', 'errors', 'raw_response' fields.
        """
        
        result = {
            'success': False,
            'data': None,
            'errors': [],
            'raw_response': response_text
        }
        
        # Extract JSON
        json_obj = self.extract_json_from_response(response_text)
        if json_obj is None:
            result['errors'].append("Failed to extract JSON from response")
            return result
        
        # Validate JSON
        validation = self.validate_response(json_obj)
        if not validation['is_valid']:
            result['errors'].extend(validation['errors'])
            return result
        
        # Success
        result['success'] = True
        result['data'] = validation['cleaned_data']
        
        logger.debug("Response parsed and validated successfully")
        return result
    
    def create_result_record(
        self,
        uuid: str,
        text: str,
        kbli_code: str,
        run_id: int,
        model: str,
        response_data: Dict,
        api_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Create a complete result record for JSONL output.
        
        Parameters
        ----------
        uuid : str
            Unique identifier for the dataset row.
        text : str
            Original job description text.
        kbli_code : str
            KBLI code being evaluated.
        run_id : int
            Run number (1, 2, or 3 for self-consistency).
        model : str
            Name of the model used.
        response_data : Dict
            Parsed and validated response data.
        api_metadata : Dict, optional
            API call metadata (timing, tokens, etc.).
        
        Returns
        -------
        Dict
            Complete result record.
        """
        
        record = {
            # Dataset information
            'uuid': uuid,
            'text': text,
            'kbli_code': kbli_code,
            
            # Processing information
            'run_id': run_id,
            'model': model,
            'timestamp': datetime.now().isoformat(),
            
            # LLM response fields
            'is_correct': response_data.get('is_correct'),
            'confidence_score': response_data.get('confidence_score'),
            'reasoning': response_data.get('reasoning'),
            'alternative_codes': response_data.get('alternative_codes', []),
            'alternative_reasoning': response_data.get('alternative_reasoning', ''),
        }
        
        # Add API metadata if available
        if api_metadata:
            record['api_metadata'] = api_metadata
        
        return record
    
    def append_to_jsonl(self, output_file: Union[str, Path], record: Dict) -> bool:
        """
        Append a record to JSONL file.
        
        Parameters
        ----------
        output_file : str or Path
            Path to JSONL output file.
        record : Dict
            Record to append.
        
        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'a', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
            
            logger.debug(f"Record appended to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to append record to {output_file}: {str(e)}")
            return False
    
    def read_existing_results(self, output_file: Union[str, Path]) -> Dict[str, List[int]]:
        """
        Read existing results from JSONL file to determine resume point.
        
        Parameters
        ----------
        output_file : str or Path
            Path to JSONL output file.
        
        Returns
        -------
        Dict[str, List[int]]
            Dictionary mapping UUIDs to list of completed run_ids.
        """
        
        existing_results = {}
        output_path = Path(output_file)
        
        if not output_path.exists():
            logger.info(f"No existing results file found: {output_path}")
            return existing_results
        
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        uuid = record.get('uuid')
                        run_id = record.get('run_id')
                        
                        if uuid and run_id is not None:
                            if uuid not in existing_results:
                                existing_results[uuid] = []
                            existing_results[uuid].append(run_id)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {str(e)}")
                        continue
            
            total_records = sum(len(runs) for runs in existing_results.values())
            logger.info(f"Found {total_records} existing results for {len(existing_results)} UUIDs")
            
        except Exception as e:
            logger.error(f"Error reading existing results: {str(e)}")
        
        return existing_results
    
    def get_output_filename(self, dataset_name: str, model_name: str) -> str:
        """
        Generate output filename based on dataset and model names.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset (extracted from filename).
        model_name : str
            Name of the model used.
        
        Returns
        -------
        str
            Generated filename.
        """
        
        # Clean names for filename
        clean_dataset = re.sub(r'[^\w\-_]', '_', dataset_name)
        clean_model = re.sub(r'[^\w\-_]', '_', model_name)
        
        return f"{clean_dataset}_{clean_model}.jsonl"
    
    def get_processing_stats(self, existing_results: Dict[str, List[int]], total_uuids: int) -> Dict:
        """
        Calculate processing statistics.
        
        Parameters
        ----------
        existing_results : Dict[str, List[int]]
            Existing results mapping.
        total_uuids : int
            Total number of UUIDs in dataset.
        
        Returns
        -------
        Dict
            Processing statistics.
        """
        
        completed_uuids = len([uuid for uuid, runs in existing_results.items() if len(runs) >= 3])
        partial_uuids = len([uuid for uuid, runs in existing_results.items() if 0 < len(runs) < 3])
        remaining_uuids = total_uuids - len(existing_results)
        
        total_api_calls_needed = total_uuids * 3
        completed_api_calls = sum(len(runs) for runs in existing_results.values())
        remaining_api_calls = total_api_calls_needed - completed_api_calls
        
        return {
            'total_uuids': total_uuids,
            'completed_uuids': completed_uuids,
            'partial_uuids': partial_uuids,
            'remaining_uuids': remaining_uuids,
            'total_api_calls_needed': total_api_calls_needed,
            'completed_api_calls': completed_api_calls,
            'remaining_api_calls': remaining_api_calls,
            'progress_percentage': (completed_api_calls / total_api_calls_needed) * 100 if total_api_calls_needed > 0 else 0
        }


# Convenience functions
def create_response_parser(output_dir: Optional[Union[str, Path]] = None) -> ResponseParser:
    """
    Create a ResponseParser instance with default settings.
    
    Parameters
    ----------
    output_dir : str or Path, optional
        Directory for output files.
    
    Returns
    -------
    ResponseParser
        Initialized response parser.
    """
    
    return ResponseParser(output_dir=output_dir)


def parse_and_save_response(
    response_text: str,
    uuid: str,
    text: str,
    kbli_code: str,
    run_id: int,
    model: str,
    output_file: Union[str, Path],
    api_metadata: Optional[Dict] = None
) -> Dict:
    """
    Parse response and save to JSONL file in one step.
    
    Parameters
    ----------
    response_text : str
        Raw response text from LLM.
    uuid : str
        Unique identifier for the dataset row.
    text : str
        Original job description text.
    kbli_code : str
        KBLI code being evaluated.
    run_id : int
        Run number (1, 2, or 3).
    model : str
        Name of the model used.
    output_file : str or Path
        Path to JSONL output file.
    api_metadata : Dict, optional
        API call metadata.
    
    Returns
    -------
    Dict
        Processing result with success status and details.
    """
    
    parser = ResponseParser()
    
    # Parse response
    parse_result = parser.parse_response(response_text)
    
    if not parse_result['success']:
        return {
            'success': False,
            'errors': parse_result['errors'],
            'uuid': uuid,
            'run_id': run_id
        }
    
    # Create record
    record = parser.create_result_record(
        uuid=uuid,
        text=text,
        kbli_code=kbli_code,
        run_id=run_id,
        model=model,
        response_data=parse_result['data'],
        api_metadata=api_metadata
    )
    
    # Save to file
    success = parser.append_to_jsonl(output_file, record)
    
    return {
        'success': success,
        'record': record if success else None,
        'errors': [] if success else ['Failed to save to JSONL file'],
        'uuid': uuid,
        'run_id': run_id
    }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        print("Testing Response Parser...")
        
        # Create parser
        parser = ResponseParser()
        
        # Test JSON extraction and validation
        test_responses = [
            # Valid response
            '''```json
            {
                "is_correct": true,
                "confidence_score": 0.85,
                "reasoning": "Job description clearly matches the KBLI code for meat processing activities.",
                "alternative_codes": [],
                "alternative_reasoning": ""
            }
            ```''',
            
            # Response with issues that need fixing
            '''{
                is_correct: false,
                confidence_score: "0.65",
                reasoning: "Job description seems to be about retail, not manufacturing.",
                alternative_codes: ["47111", "47112"],
                alternative_reasoning: "Retail codes would be more appropriate."
            }''',
            
            # Invalid response
            '''This is not a valid JSON response at all.'''
        ]
        
        for i, response in enumerate(test_responses, 1):
            print(f"\n--- Test Response {i} ---")
            result = parser.parse_response(response)
            print(f"Success: {result['success']}")
            if result['success']:
                print(f"Data: {result['data']}")
            else:
                print(f"Errors: {result['errors']}")
        
        # Test JSONL file operations
        print(f"\n--- Testing JSONL Operations ---")
        
        test_output = Path.cwd() / "test_output.jsonl"
        
        # Create test record
        test_record = parser.create_result_record(
            uuid="test-uuid-123",
            text="test job description",
            kbli_code="10110",
            run_id=1,
            model="gemini-2.0-flash-exp",
            response_data={
                'is_correct': True,
                'confidence_score': 0.9,
                'reasoning': 'Test reasoning',
                'alternative_codes': [],
                'alternative_reasoning': ''
            }
        )
        
        # Save record
        success = parser.append_to_jsonl(test_output, test_record)
        print(f"JSONL save success: {success}")
        
        # Read back
        existing = parser.read_existing_results(test_output)
        print(f"Existing results: {existing}")
        
        # Clean up
        if test_output.exists():
            test_output.unlink()
        
        print("\n✅ Response parser testing completed!")
        
    except Exception as e:
        logger.error(f"Error in testing: {str(e)}")
        print(f"❌ Testing failed: {str(e)}")
