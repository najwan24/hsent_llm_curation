"""
Module for creating hierarchical KBLI codebook structure.

This module transforms the flat KBLI codebook into a hierarchical format
suitable for LLM prompt creation, with all hierarchy levels in one row.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import logging

# Configure logging
logger = logging.getLogger(__name__)


def create_lookup_from_dataframe(df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    """
    Create a fast lookup dictionary for all KBLI codes.
    
    Parameters
    ----------
    df : pd.DataFrame
        KBLI codebook DataFrame with columns: kategori, digit, kode, judul, deskripsi
    
    Returns
    -------
    Dict[str, Tuple[str, str]]
        Dictionary mapping code -> (title, description)
        Key: code string (e.g., "A", "01", "011", "0111", "01111")
        Value: tuple of (title, description)
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'kode': ['A', '01'], 
    ...     'judul': ['Agriculture', 'Crop Production'],
    ...     'deskripsi': ['Desc A', 'Desc 01']
    ... })
    >>> lookup = create_lookup_from_dataframe(df)
    >>> lookup['A']
    ('Agriculture', 'Desc A')
    """
    
    logger.debug(f"Creating lookup dictionary from {len(df)} codebook entries")
    
    lookup = {}
    for _, row in df.iterrows():
        code = str(row['kode'])
        title = str(row['judul']) if pd.notna(row['judul']) else ""
        description = str(row['deskripsi']) if pd.notna(row['deskripsi']) else ""
        lookup[code] = (title, description)
    
    logger.debug(f"Created lookup dictionary with {len(lookup)} entries")
    return lookup


def build_hierarchy_for_code(code_5: str, lookup_dict: Dict[str, Tuple[str, str]], kategori: str) -> Dict[str, str]:
    """
    Build hierarchical structure for a single 5-digit KBLI code.
    
    Parameters
    ----------
    code_5 : str
        5-digit KBLI code (e.g., "01111")
    lookup_dict : Dict[str, Tuple[str, str]]
        Lookup dictionary for all codes
    kategori : str
        Category letter (e.g., "A", "B", "C")
    
    Returns
    -------
    Dict[str, str]
        Dictionary with hierarchical structure:
        - code_5, title_5, desc_5 (5-digit level)
        - code_4, title_4 (4-digit level)
        - code_3, title_3 (3-digit level)
        - code_2, title_2 (2-digit level)
        - code_1, title_1 (1-digit/category level)
    
    Examples
    --------
    >>> lookup = {'A': ('Agriculture', 'Desc'), '01': ('Crops', ''), '01111': ('Corn', '')}
    >>> hierarchy = build_hierarchy_for_code('01111', lookup, 'A')
    >>> hierarchy['code_5']
    '01111'
    >>> hierarchy['title_1']
    'Agriculture'
    """
    
    # Extract hierarchical codes
    code_4 = code_5[:4]  # First 4 characters
    code_3 = code_5[:3]  # First 3 characters  
    code_2 = code_5[:2]  # First 2 characters
    code_1 = kategori    # Category letter
    
    # Build hierarchical entry
    entry = {
        'code_5': code_5,
        'title_5': lookup_dict.get(code_5, ("", ""))[0],
        'desc_5': lookup_dict.get(code_5, ("", ""))[1],
        'code_4': code_4,
        'title_4': lookup_dict.get(code_4, ("", ""))[0],
        'code_3': code_3,
        'title_3': lookup_dict.get(code_3, ("", ""))[0],
        'code_2': code_2,
        'title_2': lookup_dict.get(code_2, ("", ""))[0],
        'code_1': code_1,
        'title_1': lookup_dict.get(code_1, ("", ""))[0]
    }
    
    return entry


def create_hierarchical_codebook(
    input_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    validate_output: bool = True
) -> Path:
    """
    Transform KBLI codebook into hierarchical format.
    
    Creates a flat DataFrame where each row contains the full hierarchy
    for a 5-digit KBLI code, making it easy to use in LLM prompts.
    
    Parameters
    ----------
    input_path : str or Path, optional
        Path to the original KBLI codebook CSV. If None, uses default path.
    output_path : str or Path, optional
        Path where hierarchical codebook will be saved. If None, uses default.
    validate_output : bool, default True
        Whether to validate the output after creation.
    
    Returns
    -------
    Path
        Path where the hierarchical codebook was saved.
    
    Raises
    ------
    FileNotFoundError
        If input file doesn't exist.
    ValueError
        If codebook structure is invalid or output validation fails.
    
    Examples
    --------
    >>> output_path = create_hierarchical_codebook()
    >>> print(f"Hierarchical codebook saved to: {output_path}")
    """
    
    # Set default paths if not provided
    if input_path is None:
        project_root = Path.cwd()
        input_path = project_root / "data" / "external" / "kbli_codebook.csv"
    
    if output_path is None:
        project_root = Path.cwd()
        output_path = project_root / "data" / "external" / "kbli_codebook_hierarchical.csv"
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Validate input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"KBLI codebook file not found: {input_path}")
    
    logger.info(f"Loading KBLI codebook from: {input_path}")
    
    # Load the codebook using our existing function
    try:
        from ..data.load_codebook import load_kbli_codebook
        df = load_kbli_codebook(str(input_path))
        logger.info(f"Loaded {len(df)} entries from codebook")
    except Exception as e:
        logger.error(f"Failed to load codebook: {str(e)}")
        raise
    
    # Validate required columns
    required_columns = ['kategori', 'digit', 'kode', 'judul', 'deskripsi']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in codebook: {missing_columns}")
    
    # Create lookup dictionary for all codes
    lookup_dict = create_lookup_from_dataframe(df)
    logger.info(f"Created lookup dictionary with {len(lookup_dict)} entries")
    
    # Get only the 5-digit codes to iterate over
    df_5_digit = df[df['digit'] == 5].copy()
    logger.info(f"Found {len(df_5_digit)} 5-digit codes to process")
    
    if len(df_5_digit) == 0:
        raise ValueError("No 5-digit codes found in the codebook")
    
    # Build hierarchical data
    logger.info("Building hierarchical structure...")
    hierarchical_data = []
    
    for _, row in df_5_digit.iterrows():
        code_5 = str(row['kode'])
        kategori = str(row['kategori'])
        
        try:
            entry = build_hierarchy_for_code(code_5, lookup_dict, kategori)
            hierarchical_data.append(entry)
        except Exception as e:
            logger.warning(f"Failed to build hierarchy for code {code_5}: {str(e)}")
            continue
    
    if not hierarchical_data:
        raise ValueError("No hierarchical entries could be created")
    
    logger.info(f"Built hierarchical structure for {len(hierarchical_data)} codes")
    
    # Convert to DataFrame
    hierarchical_df = pd.DataFrame(hierarchical_data)
    
    # Ensure column order
    expected_columns = [
        'code_5', 'title_5', 'desc_5',
        'code_4', 'title_4', 
        'code_3', 'title_3',
        'code_2', 'title_2',
        'code_1', 'title_1'
    ]
    
    # Reorder columns to match expected format
    hierarchical_df = hierarchical_df[expected_columns]
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    try:
        hierarchical_df.to_csv(output_path, index=False)
        logger.info(f"Hierarchical codebook saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save hierarchical codebook: {str(e)}")
        raise
    
    # Validate output if requested
    if validate_output:
        validation_result = validate_hierarchical_codebook(hierarchical_df)
        if not validation_result['is_valid']:
            logger.warning(f"Output validation failed: {validation_result}")
        else:
            logger.info("Output validation passed successfully")
    
    # Log summary
    logger.info("Hierarchical codebook creation completed:")
    logger.info(f"  Input: {input_path} ({len(df)} total entries)")
    logger.info(f"  Output: {output_path} ({len(hierarchical_df)} hierarchical entries)")
    logger.info(f"  Columns: {list(hierarchical_df.columns)}")
    
    return output_path


def validate_hierarchical_codebook(df: pd.DataFrame) -> dict:
    """
    Validate the structure and content of hierarchical codebook.
    
    Parameters
    ----------
    df : pd.DataFrame
        Hierarchical codebook DataFrame to validate.
    
    Returns
    -------
    dict
        Dictionary containing validation results:
        - total_records: Total number of records
        - expected_columns: List of expected columns
        - missing_columns: List of missing columns
        - empty_code_5: Number of records with empty 5-digit codes
        - empty_titles: Number of records with empty titles
        - is_valid: Boolean indicating if all checks passed
    
    Examples
    --------
    >>> df = pd.DataFrame({'code_5': ['01111'], 'title_5': ['Corn']})
    >>> result = validate_hierarchical_codebook(df)
    >>> result['is_valid']
    False  # Missing required columns
    """
    
    expected_columns = [
        'code_5', 'title_5', 'desc_5',
        'code_4', 'title_4', 
        'code_3', 'title_3',
        'code_2', 'title_2',
        'code_1', 'title_1'
    ]
    
    # Check columns
    missing_columns = [col for col in expected_columns if col not in df.columns]
    
    # Check data quality
    total_records = len(df)
    empty_code_5 = df['code_5'].isnull().sum() if 'code_5' in df.columns else total_records
    
    # Count empty titles across all levels
    empty_titles = 0
    title_columns = ['title_5', 'title_4', 'title_3', 'title_2', 'title_1']
    for col in title_columns:
        if col in df.columns:
            empty_titles += (df[col].isnull() | (df[col] == "")).sum()
    
    # Overall validation
    is_valid = (len(missing_columns) == 0 and empty_code_5 == 0 and total_records > 0)
    
    result = {
        'total_records': total_records,
        'expected_columns': expected_columns,
        'missing_columns': missing_columns,
        'empty_code_5': empty_code_5,
        'empty_titles': empty_titles,
        'is_valid': is_valid
    }
    
    logger.debug(f"Hierarchical codebook validation: {result}")
    
    return result


def get_hierarchical_sample(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Get a sample of hierarchical codebook for inspection.
    
    Parameters
    ----------
    df : pd.DataFrame
        Hierarchical codebook DataFrame.
    n : int, default 3
        Number of sample records to return.
    
    Returns
    -------
    pd.DataFrame
        Sample DataFrame with n records.
    """
    
    if len(df) < n:
        logger.warning(f"Dataset has only {len(df)} records, returning all")
        return df.copy()
    
    sample_df = df.head(n).copy()
    logger.info(f"Generated sample of {len(sample_df)} records")
    
    return sample_df


def load_hierarchical_codebook(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load hierarchical codebook from CSV file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the hierarchical codebook CSV file.
    
    Returns
    -------
    pd.DataFrame
        Hierarchical codebook DataFrame.
    
    Raises
    ------
    FileNotFoundError
        If the file doesn't exist.
    ValueError
        If the file structure is invalid.
    """
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Hierarchical codebook file not found: {file_path}")
    
    logger.info(f"Loading hierarchical codebook from: {file_path}")
    
    try:
        df = pd.read_csv(file_path, dtype=str)  # Keep all as string to preserve codes
        logger.info(f"Loaded hierarchical codebook with {len(df)} records")
        
        # Validate structure
        validation = validate_hierarchical_codebook(df)
        if not validation['is_valid']:
            logger.warning(f"Loaded codebook has validation issues: {validation['missing_columns']}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load hierarchical codebook: {str(e)}")
        raise


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        print("Testing hierarchical codebook creation...")
        
        # Create hierarchical codebook
        output_path = create_hierarchical_codebook()
        print(f"Hierarchical codebook created at: {output_path}")
        
        # Load and display sample
        df = load_hierarchical_codebook(output_path)
        print(f"\nLoaded hierarchical codebook with {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        
        # Show sample
        sample = get_hierarchical_sample(df, 3)
        print("\nSample hierarchical structure:")
        print(sample.to_string())
        
        # Validate
        validation = validate_hierarchical_codebook(df)
        print(f"\nValidation result: {validation}")
        
    except Exception as e:
        logger.error(f"Error in testing: {str(e)}")
        raise
