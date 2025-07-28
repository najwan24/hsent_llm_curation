#!/usr/bin/env python3
"""
KBLI Classification - LLM Label Extraction Script

This script processes KBLI classification datasets using LLM models with
self-consistency (N_RUNS=3) and saves results to JSONL format.

Features:
- Resume capability based on deterministic UUIDs
- Multi-model support with different rate limits
- Self-consistency with 3 API calls per row
- Immediate JSONL saving for data safety
- Comprehensive logging and progress tracking
- Graceful quota exhaustion handling
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.gemini_client import GeminiClient
from src.llm.response_parser import ResponseParser
from src.llm.prompt_builder import KBLIPromptBuilder
from src.data.load_main_data import load_main_dataset

# Configure logging
logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path, dataset_name: str, model_name: str) -> None:
    """
    Set up comprehensive logging for the processing session.
    
    Parameters
    ----------
    log_dir : Path
        Directory for log files.
    dataset_name : str
        Name of the dataset being processed.
    model_name : str
        Name of the model being used.
    """
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{dataset_name}_{model_name}_{timestamp}.log"
    log_path = log_dir / log_filename
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger.info(f"=== KBLI LLM Label Extraction Started ===")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"Timestamp: {timestamp}")


def load_dataset_with_validation(input_path: Path) -> pd.DataFrame:
    """
    Load and validate the input dataset.
    
    Parameters
    ----------
    input_path : Path
        Path to the input CSV file.
    
    Returns
    -------
    pd.DataFrame
        Loaded and validated dataset.
        
    Raises
    ------
    ValueError
        If dataset validation fails.
    """
    
    logger.info(f"Loading dataset from: {input_path}")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    
    # Load dataset
    try:
        df = pd.read_csv(input_path, dtype=str)
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {str(e)}")
    
    # Validate required columns - handle both 'uuid' and 'sample_id' for UUID column
    uuid_column = None
    if 'uuid' in df.columns:
        uuid_column = 'uuid'
    elif 'sample_id' in df.columns:
        uuid_column = 'sample_id'
        # Rename for consistency
        df = df.rename(columns={'sample_id': 'uuid'})
        logger.info("Renamed 'sample_id' column to 'uuid' for processing")
    else:
        raise ValueError("Missing UUID column: expected 'uuid' or 'sample_id'")
    
    required_columns = ['uuid', 'text', 'kbli_code']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for empty values
    for col in required_columns:
        empty_count = df[col].isna().sum() + (df[col] == '').sum()
        if empty_count > 0:
            logger.warning(f"Found {empty_count} empty values in column '{col}'")
    
    # Remove rows with missing required data
    original_len = len(df)
    df = df.dropna(subset=required_columns)
    df = df[df['uuid'].str.strip() != '']
    df = df[df['text'].str.strip() != '']
    df = df[df['kbli_code'].str.strip() != '']
    
    if len(df) < original_len:
        logger.warning(f"Removed {original_len - len(df)} rows with missing data")
    
    # Validate UUID uniqueness
    duplicate_uuids = df['uuid'].duplicated().sum()
    if duplicate_uuids > 0:
        logger.warning(f"Found {duplicate_uuids} duplicate UUIDs - keeping first occurrence")
        df = df.drop_duplicates(subset=['uuid'], keep='first')
    
    logger.info(f"Dataset validation completed: {len(df)} valid rows")
    return df


def determine_processing_plan(
    df: pd.DataFrame,
    existing_results: Dict[str, List[int]],
) -> List[Tuple[str, str, str, int]]:
    """
    Determine which API calls need to be made based on existing results.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to process.
    existing_results : Dict[str, List[int]]
        Existing results mapping UUIDs to completed run_ids.
    
    Returns
    -------
    List[Tuple[str, str, str, int]]
        List of (uuid, text, kbli_code, run_id) tuples to process.
    """
    
    processing_plan = []
    
    for _, row in df.iterrows():
        uuid = row['uuid']
        text = row['text']
        kbli_code = row['kbli_code']
        
        # Determine which runs are needed for this UUID
        completed_runs = existing_results.get(uuid, [])
        needed_runs = [run_id for run_id in [1, 2, 3] if run_id not in completed_runs]
        
        # Add to processing plan
        for run_id in needed_runs:
            processing_plan.append((uuid, text, kbli_code, run_id))
    
    logger.info(f"Processing plan created: {len(processing_plan)} API calls needed")
    return processing_plan


def process_single_request(
    uuid: str,
    text: str,
    kbli_code: str,
    run_id: int,
    prompt_builder: KBLIPromptBuilder,
    gemini_client: GeminiClient,
    response_parser: ResponseParser,
    output_file: Path
) -> Dict:
    """
    Process a single API request and save the result.
    
    Parameters
    ----------
    uuid : str
        UUID of the dataset row.
    text : str
        Job description text.
    kbli_code : str
        KBLI code to evaluate.
    run_id : int
        Run number (1, 2, or 3).
    prompt_builder : KBLIPromptBuilder
        Prompt builder instance.
    gemini_client : GeminiClient
        Gemini API client.
    response_parser : ResponseParser
        Response parser instance.
    output_file : Path
        Output JSONL file path.
    
    Returns
    -------
    Dict
        Processing result with success status and metadata.
    """
    
    try:
        # Build prompt
        prompt = prompt_builder.build_prompt(text, kbli_code)
        logger.debug(f"Built prompt for UUID {uuid} run {run_id} ({len(prompt)} chars)")
        
        # Make API call
        api_response = gemini_client.generate_response(prompt)
        
        if not api_response['success']:
            logger.error(f"API call failed for UUID {uuid} run {run_id}: {api_response['error']}")
            
            # Check for quota exhaustion
            if api_response['metadata'].get('quota_exhausted'):
                return {
                    'success': False,
                    'quota_exhausted': True,
                    'error': api_response['error'],
                    'uuid': uuid,
                    'run_id': run_id
                }
            
            return {
                'success': False,
                'quota_exhausted': False,
                'error': api_response['error'],
                'uuid': uuid,
                'run_id': run_id
            }
        
        # Parse and save response
        parse_result = response_parser.parse_response(api_response['content'])
        
        if not parse_result['success']:
            logger.warning(f"Failed to parse response for UUID {uuid} run {run_id}: {parse_result['errors']}")
            return {
                'success': False,
                'quota_exhausted': False,
                'error': f"Parse error: {parse_result['errors']}",
                'uuid': uuid,
                'run_id': run_id
            }
        
        # Create result record
        record = response_parser.create_result_record(
            uuid=uuid,
            text=text,
            kbli_code=kbli_code,
            run_id=run_id,
            model=gemini_client.model_name,
            response_data=parse_result['data'],
            api_metadata=api_response['metadata']
        )
        
        # Save to JSONL file
        save_success = response_parser.append_to_jsonl(output_file, record)
        
        if not save_success:
            logger.error(f"Failed to save result for UUID {uuid} run {run_id}")
            return {
                'success': False,
                'quota_exhausted': False,
                'error': "Failed to save result",
                'uuid': uuid,
                'run_id': run_id
            }
        
        logger.debug(f"Successfully processed UUID {uuid} run {run_id}")
        return {
            'success': True,
            'quota_exhausted': False,
            'record': record,
            'uuid': uuid,
            'run_id': run_id
        }
        
    except Exception as e:
        logger.error(f"Unexpected error processing UUID {uuid} run {run_id}: {str(e)}")
        return {
            'success': False,
            'quota_exhausted': False,
            'error': f"Unexpected error: {str(e)}",
            'uuid': uuid,
            'run_id': run_id
        }


def main():
    """Main processing function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="KBLI Classification - LLM Label Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process mini_test_with_ids.csv with gemini-2.0-flash-exp
    python 01_extract_llm_labels.py --input data/processed/mini_test_with_ids.csv --model gemini-2.0-flash-exp
    
    # Resume processing
    python 01_extract_llm_labels.py --input data/processed/mini_test_with_ids.csv --model gemini-2.0-flash-exp --resume
    
    # Use different model
    python 01_extract_llm_labels.py --input data/processed/mini_test_with_ids.csv --model gemini-1.5-pro
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input CSV file with UUID column')
    parser.add_argument('--model', '-m', type=str, default='gemini-2.0-flash-exp',
                       help='Gemini model to use (default: gemini-2.0-flash-exp)')
    parser.add_argument('--temperature', '-t', type=float, default=0.7,
                       help='Temperature for text generation (default: 0.7)')
    parser.add_argument('--output-dir', '-o', type=str, 
                       default='results/pilot/extract_llm',
                       help='Output directory for JSONL files')
    parser.add_argument('--log-dir', '-l', type=str,
                       default='logs/llm_processing',
                       help='Directory for log files')
    parser.add_argument('--resume', '-r', action='store_true',
                       help='Resume processing from existing results')
    parser.add_argument('--max-requests', type=int, default=None,
                       help='Maximum number of API requests to make (for testing)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    
    # Extract dataset name
    dataset_name = input_path.stem
    
    # Setup logging
    setup_logging(log_dir, dataset_name, args.model)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Log configuration
    logger.info(f"Configuration:")
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Resume: {args.resume}")
    logger.info(f"  Max requests: {args.max_requests or 'unlimited'}")
    
    try:
        # Load and validate dataset
        df = load_dataset_with_validation(input_path)
        
        # Initialize components
        logger.info("Initializing components...")
        
        # Create Gemini client
        gemini_client = GeminiClient(
            model_name=args.model,
            temperature=args.temperature
        )
        
        # Create prompt builder
        prompt_builder = KBLIPromptBuilder()
        
        # Create response parser
        response_parser = ResponseParser(output_dir=output_dir)
        
        # Setup output file
        output_filename = response_parser.get_output_filename(dataset_name, args.model)
        output_file = output_dir / output_filename
        
        logger.info(f"Output file: {output_file}")
        
        # Read existing results for resuming
        existing_results = {}
        if args.resume or output_file.exists():
            existing_results = response_parser.read_existing_results(output_file)
        
        # Calculate processing statistics
        stats = response_parser.get_processing_stats(existing_results, len(df))
        logger.info(f"Processing Statistics:")
        logger.info(f"  Total UUIDs: {stats['total_uuids']}")
        logger.info(f"  Completed UUIDs: {stats['completed_uuids']}")
        logger.info(f"  Partial UUIDs: {stats['partial_uuids']}")
        logger.info(f"  Remaining UUIDs: {stats['remaining_uuids']}")
        logger.info(f"  Progress: {stats['progress_percentage']:.1f}%")
        logger.info(f"  Remaining API calls: {stats['remaining_api_calls']}")
        
        # Check if processing is complete
        if stats['remaining_api_calls'] == 0:
            logger.info("✅ All processing is already complete!")
            return
        
        # Create processing plan
        processing_plan = determine_processing_plan(df, existing_results)
        
        # Apply max_requests limit if specified
        if args.max_requests and args.max_requests < len(processing_plan):
            logger.info(f"Limiting to {args.max_requests} requests (testing mode)")
            processing_plan = processing_plan[:args.max_requests]
        
        # Display model information
        model_info = gemini_client.get_model_info()
        logger.info(f"Model Information:")
        logger.info(f"  Name: {model_info['display_name']}")
        logger.info(f"  Rate limit: {model_info['requests_per_minute']} requests/minute")
        logger.info(f"  Temperature: {model_info['temperature']}")
        
        # Estimate processing time
        estimated_minutes = len(processing_plan) / model_info['requests_per_minute']
        logger.info(f"Estimated processing time: {estimated_minutes:.1f} minutes")
        
        # Start processing
        logger.info(f"Starting processing of {len(processing_plan)} API calls...")
        
        successful_requests = 0
        failed_requests = 0
        quota_exhausted = False
        
        # Progress bar
        with tqdm(total=len(processing_plan), desc="Processing", unit="req") as pbar:
            
            for uuid, text, kbli_code, run_id in processing_plan:
                
                # Process single request
                result = process_single_request(
                    uuid=uuid,
                    text=text,
                    kbli_code=kbli_code,
                    run_id=run_id,
                    prompt_builder=prompt_builder,
                    gemini_client=gemini_client,
                    response_parser=response_parser,
                    output_file=output_file
                )
                
                # Update counters
                if result['success']:
                    successful_requests += 1
                    pbar.set_postfix({
                        'Success': successful_requests,
                        'Failed': failed_requests,
                        'UUID': uuid[:8],
                        'Run': run_id
                    })
                else:
                    failed_requests += 1
                    
                    # Check for quota exhaustion
                    if result.get('quota_exhausted'):
                        quota_exhausted = True
                        logger.error("🚫 API quota exhausted - stopping processing")
                        break
                
                pbar.update(1)
        
        # Final statistics
        logger.info(f"Processing completed:")
        logger.info(f"  Successful requests: {successful_requests}")
        logger.info(f"  Failed requests: {failed_requests}")
        logger.info(f"  Total processed: {successful_requests + failed_requests}")
        
        if quota_exhausted:
            logger.info(f"⚠️  Processing stopped due to quota exhaustion")
            logger.info(f"   Run the script again tomorrow to continue processing")
            sys.exit(1)
        elif failed_requests > 0:
            logger.warning(f"⚠️  {failed_requests} requests failed - check logs for details")
            sys.exit(1)
        else:
            logger.info(f"✅ All requests completed successfully!")
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.exception("Exception details:")
        sys.exit(1)


if __name__ == "__main__":
    main()
