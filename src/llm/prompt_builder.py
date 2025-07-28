"""
Module for building KBLI classification prompts using hierarchical codebook data.

This module provides functions to construct prompts for KBLI classification tasks
by combining job descriptions from datasets with hierarchical context from the
KBLI codebook. It uses a master prompt template to ensure consistent formatting.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)


class KBLIPromptBuilder:
    """
    Class for building KBLI classification prompts with hierarchical context.
    
    This class loads the master prompt template and hierarchical codebook,
    then provides methods to build prompts for individual job descriptions.
    """
    
    def __init__(
        self,
        master_prompt_path: Optional[Union[str, Path]] = None,
        hierarchical_codebook_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the prompt builder.
        
        Parameters
        ----------
        master_prompt_path : str or Path, optional
            Path to the master prompt template file. If None, uses default path.
        hierarchical_codebook_path : str or Path, optional
            Path to the hierarchical codebook CSV file. If None, uses default path.
        
        Raises
        ------
        FileNotFoundError
            If required files are not found.
        """
        
        # Set default paths if not provided
        if master_prompt_path is None:
            project_root = Path.cwd()
            master_prompt_path = project_root / "data" / "external" / "master_prompt.txt"
        
        if hierarchical_codebook_path is None:
            project_root = Path.cwd()
            hierarchical_codebook_path = project_root / "data" / "external" / "kbli_codebook_hierarchical.csv"
        
        self.master_prompt_path = Path(master_prompt_path)
        self.hierarchical_codebook_path = Path(hierarchical_codebook_path)
        
        # Load template and codebook
        self.master_template = self._load_master_template()
        self.hierarchical_codebook = self._load_hierarchical_codebook()
        
        logger.info(f"Prompt builder initialized with {len(self.hierarchical_codebook)} hierarchical codes")
    
    def _load_master_template(self) -> str:
        """
        Load the master prompt template from file.
        
        Returns
        -------
        str
            Master prompt template content.
            
        Raises
        ------
        FileNotFoundError
            If the master prompt template file is not found.
        """
        
        if not self.master_prompt_path.exists():
            raise FileNotFoundError(f"Master prompt template not found: {self.master_prompt_path}")
        
        logger.debug(f"Loading master template from: {self.master_prompt_path}")
        
        try:
            with open(self.master_prompt_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            logger.debug(f"Master template loaded successfully ({len(template)} characters)")
            return template
            
        except Exception as e:
            logger.error(f"Failed to load master template: {str(e)}")
            raise
    
    def _load_hierarchical_codebook(self) -> pd.DataFrame:
        """
        Load the hierarchical codebook CSV file.
        
        Returns
        -------
        pd.DataFrame
            Hierarchical codebook DataFrame.
            
        Raises
        ------
        FileNotFoundError
            If the hierarchical codebook file is not found.
        """
        
        if not self.hierarchical_codebook_path.exists():
            raise FileNotFoundError(f"Hierarchical codebook not found: {self.hierarchical_codebook_path}")
        
        logger.debug(f"Loading hierarchical codebook from: {self.hierarchical_codebook_path}")
        
        try:
            # Load with all columns as string to preserve code formatting
            codebook = pd.read_csv(self.hierarchical_codebook_path, dtype=str)
            
            # Validate expected columns
            expected_columns = [
                'code_5', 'title_5', 'desc_5',
                'code_4', 'title_4', 
                'code_3', 'title_3',
                'code_2', 'title_2',
                'code_1', 'title_1'
            ]
            
            missing_columns = [col for col in expected_columns if col not in codebook.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in hierarchical codebook: {missing_columns}")
            
            logger.debug(f"Hierarchical codebook loaded successfully ({len(codebook)} entries)")
            return codebook
            
        except Exception as e:
            logger.error(f"Failed to load hierarchical codebook: {str(e)}")
            raise
    
    def get_hierarchy_context(self, kbli_code: str) -> str:
        """
        Generate hierarchical context string for a given KBLI code.
        
        Parameters
        ----------
        kbli_code : str
            5-digit KBLI code (e.g., "10211")
        
        Returns
        -------
        str
            Formatted hierarchical context string.
            
        Raises
        ------
        ValueError
            If the KBLI code is not found in the hierarchical codebook.
        
        Examples
        --------
        >>> builder = KBLIPromptBuilder()
        >>> context = builder.get_hierarchy_context("10211")
        >>> print(context)
        Section A: Pertanian, Kehutanan dan Perikanan
        Division 01: Pertanian Tanaman, Peternakan, Perburuan dan Kegiatan YBDI
        ...
        """
        
        # Find the code in the hierarchical codebook
        code_row = self.hierarchical_codebook[self.hierarchical_codebook['code_5'] == str(kbli_code)]
        
        if code_row.empty:
            raise ValueError(f"KBLI code '{kbli_code}' not found in hierarchical codebook")
        
        # Get the first (and should be only) matching row
        row = code_row.iloc[0]
        
        # Build hierarchical context string
        context_parts = []
        
        # Section (Level 1)
        if pd.notna(row['code_1']) and pd.notna(row['title_1']):
            context_parts.append(f"Section {row['code_1']}: {row['title_1']}")
        
        # Division (Level 2)
        if pd.notna(row['code_2']) and pd.notna(row['title_2']):
            context_parts.append(f"Division {row['code_2']}: {row['title_2']}")
        
        # Group (Level 3)
        if pd.notna(row['code_3']) and pd.notna(row['title_3']):
            context_parts.append(f"Group {row['code_3']}: {row['title_3']}")
        
        # Class (Level 4)
        if pd.notna(row['code_4']) and pd.notna(row['title_4']):
            context_parts.append(f"Class {row['code_4']}: {row['title_4']}")
        
        # Sub-Class (Level 5)
        if pd.notna(row['code_5']) and pd.notna(row['title_5']):
            context_parts.append(f"Sub-Class {row['code_5']}: {row['title_5']}")
        
        # Description
        if pd.notna(row['desc_5']) and row['desc_5'].strip():
            context_parts.append(f"Description: {row['desc_5']}")
        
        hierarchy_context = "\n".join(context_parts)
        
        logger.debug(f"Generated hierarchy context for code {kbli_code} ({len(context_parts)} levels)")
        
        return hierarchy_context
    
    def build_prompt(self, job_description: str, kbli_code: str) -> str:
        """
        Build a complete prompt for KBLI classification.
        
        Parameters
        ----------
        job_description : str
            Job description text from the dataset.
        kbli_code : str
            5-digit KBLI code to evaluate.
        
        Returns
        -------
        str
            Complete prompt ready for LLM classification.
            
        Raises
        ------
        ValueError
            If the KBLI code is not found or inputs are invalid.
        
        Examples
        --------
        >>> builder = KBLIPromptBuilder()
        >>> prompt = builder.build_prompt(
        ...     "jagal sapi menghasilkan potong sapi di bidang jagal sapi",
        ...     "10110"
        ... )
        >>> print(prompt[:100])
        You are an expert classifier for the Indonesian Standard Industrial...
        """
        
        # Validate inputs
        if not job_description or not job_description.strip():
            raise ValueError("Job description cannot be empty")
        
        if not kbli_code or not str(kbli_code).strip():
            raise ValueError("KBLI code cannot be empty")
        
        # Clean inputs
        job_description = job_description.strip()
        kbli_code = str(kbli_code).strip()
        
        # Get hierarchical context
        try:
            hierarchy_context = self.get_hierarchy_context(kbli_code)
        except ValueError as e:
            logger.error(f"Failed to get hierarchy context: {str(e)}")
            raise
        
        # Build the prompt using the template
        try:
            prompt = self.master_template.format(
                code_to_check=kbli_code,
                hierarchy_context=hierarchy_context,
                job_description=job_description
            )
            
            logger.debug(f"Built prompt for code {kbli_code} ({len(prompt)} characters)")
            
            return prompt
            
        except Exception as e:
            logger.error(f"Failed to build prompt: {str(e)}")
            raise
    
    def build_prompt_from_dataset_row(self, row: pd.Series) -> str:
        """
        Build a prompt from a dataset row containing 'text' and 'kbli_code' columns.
        
        Parameters
        ----------
        row : pd.Series
            Dataset row with 'text' and 'kbli_code' columns.
        
        Returns
        -------
        str
            Complete prompt ready for LLM classification.
            
        Raises
        ------
        KeyError
            If required columns are missing from the row.
        ValueError
            If the data is invalid.
        
        Examples
        --------
        >>> import pandas as pd
        >>> builder = KBLIPromptBuilder()
        >>> row = pd.Series({
        ...     'text': 'jagal sapi menghasilkan potong sapi',
        ...     'kbli_code': '10110'
        ... })
        >>> prompt = builder.build_prompt_from_dataset_row(row)
        """
        
        # Validate required columns
        required_columns = ['text', 'kbli_code']
        missing_columns = [col for col in required_columns if col not in row.index]
        if missing_columns:
            raise KeyError(f"Missing required columns in dataset row: {missing_columns}")
        
        # Extract data
        job_description = row['text']
        kbli_code = row['kbli_code']
        
        # Build and return prompt
        return self.build_prompt(job_description, kbli_code)
    
    def batch_build_prompts(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Build prompts for multiple dataset rows.
        
        Parameters
        ----------
        dataset : pd.DataFrame
            Dataset with 'text' and 'kbli_code' columns.
        
        Returns
        -------
        pd.DataFrame
            Dataset with additional 'prompt' column containing built prompts.
            
        Raises
        ------
        KeyError
            If required columns are missing from the dataset.
        
        Examples
        --------
        >>> import pandas as pd
        >>> builder = KBLIPromptBuilder()
        >>> df = pd.DataFrame({
        ...     'text': ['job desc 1', 'job desc 2'],
        ...     'kbli_code': ['10110', '10120']
        ... })
        >>> df_with_prompts = builder.batch_build_prompts(df)
        >>> 'prompt' in df_with_prompts.columns
        True
        """
        
        # Validate required columns
        required_columns = ['text', 'kbli_code']
        missing_columns = [col for col in required_columns if col not in dataset.columns]
        if missing_columns:
            raise KeyError(f"Missing required columns in dataset: {missing_columns}")
        
        logger.info(f"Building prompts for {len(dataset)} dataset rows")
        
        # Create a copy to avoid modifying the original
        result_df = dataset.copy()
        
        # Build prompts for each row
        prompts = []
        failed_count = 0
        
        for idx, row in dataset.iterrows():
            try:
                prompt = self.build_prompt_from_dataset_row(row)
                prompts.append(prompt)
            except Exception as e:
                logger.warning(f"Failed to build prompt for row {idx}: {str(e)}")
                prompts.append(None)  # or empty string
                failed_count += 1
        
        # Add prompts column
        result_df['prompt'] = prompts
        
        success_count = len(dataset) - failed_count
        logger.info(f"Built {success_count}/{len(dataset)} prompts successfully")
        
        if failed_count > 0:
            logger.warning(f"{failed_count} prompts failed to build")
        
        return result_df
    
    def get_available_codes(self) -> list:
        """
        Get list of all available KBLI codes in the hierarchical codebook.
        
        Returns
        -------
        list
            List of available 5-digit KBLI codes.
        """
        
        return self.hierarchical_codebook['code_5'].tolist()
    
    def validate_code(self, kbli_code: str) -> bool:
        """
        Check if a KBLI code exists in the hierarchical codebook.
        
        Parameters
        ----------
        kbli_code : str
            5-digit KBLI code to validate.
        
        Returns
        -------
        bool
            True if the code exists, False otherwise.
        """
        
        return str(kbli_code) in self.hierarchical_codebook['code_5'].values
    
    def get_codebook_info(self) -> Dict[str, int]:
        """
        Get information about the loaded hierarchical codebook.
        
        Returns
        -------
        Dict[str, int]
            Dictionary with codebook statistics.
        """
        
        return {
            'total_codes': len(self.hierarchical_codebook),
            'unique_sections': self.hierarchical_codebook['code_1'].nunique(),
            'unique_divisions': self.hierarchical_codebook['code_2'].nunique(),
            'unique_groups': self.hierarchical_codebook['code_3'].nunique(),
            'unique_classes': self.hierarchical_codebook['code_4'].nunique(),
        }


# Convenience functions for standalone usage
def create_prompt_builder(
    master_prompt_path: Optional[Union[str, Path]] = None,
    hierarchical_codebook_path: Optional[Union[str, Path]] = None
) -> KBLIPromptBuilder:
    """
    Create a KBLIPromptBuilder instance with default or custom paths.
    
    Parameters
    ----------
    master_prompt_path : str or Path, optional
        Path to the master prompt template file.
    hierarchical_codebook_path : str or Path, optional
        Path to the hierarchical codebook CSV file.
    
    Returns
    -------
    KBLIPromptBuilder
        Initialized prompt builder instance.
    """
    
    return KBLIPromptBuilder(master_prompt_path, hierarchical_codebook_path)


def build_single_prompt(
    job_description: str,
    kbli_code: str,
    prompt_builder: Optional[KBLIPromptBuilder] = None
) -> str:
    """
    Build a single prompt using default or provided prompt builder.
    
    Parameters
    ----------
    job_description : str
        Job description text.
    kbli_code : str
        5-digit KBLI code.
    prompt_builder : KBLIPromptBuilder, optional
        Prompt builder instance. If None, creates a new one.
    
    Returns
    -------
    str
        Complete prompt ready for LLM classification.
    """
    
    if prompt_builder is None:
        prompt_builder = create_prompt_builder()
    
    return prompt_builder.build_prompt(job_description, kbli_code)


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        print("Testing KBLI Prompt Builder...")
        
        # Create prompt builder
        builder = KBLIPromptBuilder()
        
        # Get codebook info
        info = builder.get_codebook_info()
        print(f"Codebook info: {info}")
        
        # Test single prompt building
        job_desc = "jagal sapi menghasilkan potong sapi di bidang jagal sapi"
        kbli_code = "10110"
        
        print(f"\nTesting single prompt for code {kbli_code}...")
        
        # Get hierarchy context
        hierarchy = builder.get_hierarchy_context(kbli_code)
        print(f"Hierarchy context:\n{hierarchy}")
        
        # Build complete prompt
        prompt = builder.build_prompt(job_desc, kbli_code)
        print(f"\nPrompt built successfully ({len(prompt)} characters)")
        print("First 200 characters of prompt:")
        print(prompt[:200] + "...")
        
        # Test with invalid code
        try:
            invalid_prompt = builder.build_prompt(job_desc, "99999")
        except ValueError as e:
            print(f"\nExpected error for invalid code: {str(e)}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        logger.error(f"Error in testing: {str(e)}")
        raise
