"""
Module for loading KBLI (Klasifikasi Baku Lapangan Usaha Indonesia) codebook data.

This module provides functions to load and process the KBLI codebook dataset
with proper data types and validation.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_kbli_codebook(
    file_path: Optional[str] = None,
    encoding: str = "utf-8",
    validate_data: bool = True
) -> pd.DataFrame:
    """
    Load KBLI codebook dataset with proper data types.
    
    This function loads the KBLI codebook CSV file and ensures the 'kode' column
    is treated as string type to preserve leading zeros and maintain consistency.
    
    Parameters
    ----------
    file_path : str, optional
        Path to the KBLI codebook CSV file. If None, uses the default path
        relative to the project structure.
    encoding : str, default 'utf-8'
        Character encoding of the CSV file.
    validate_data : bool, default True
        Whether to perform basic data validation after loading.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the KBLI codebook with proper data types:
        - kategori: string
        - digit: int64
        - kode: string (preserves leading zeros)
        - judul: string
        - deskripsi: string
    
    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    pd.errors.EmptyDataError
        If the CSV file is empty.
    ValueError
        If required columns are missing or data validation fails.
    
    Examples
    --------
    >>> # Load with default path
    >>> codebook_df = load_kbli_codebook()
    
    >>> # Load with custom path
    >>> codebook_df = load_kbli_codebook("path/to/custom/kbli_codebook.csv")
    
    >>> # Load without validation
    >>> codebook_df = load_kbli_codebook(validate_data=False)
    """
    
    # Set default file path if not provided
    if file_path is None:
        # Assuming the standard project structure
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        file_path = project_root / "data" / "external" / "kbli_codebook.csv"
    
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"KBLI codebook file not found: {file_path}")
    
    logger.info(f"Loading KBLI codebook from: {file_path}")
    
    # Define data types for columns
    dtype_mapping = {
        'kategori': 'string',
        'digit': 'int64',
        'kode': 'string',  # Keep as string to preserve leading zeros
        'judul': 'string',
        'deskripsi': 'string'
    }
    
    try:
        # Load the CSV file with specified data types
        df = pd.read_csv(
            file_path,
            dtype=dtype_mapping,
            encoding=encoding,
            na_values=['', 'NA', 'N/A', 'null', 'NULL']
        )
        
        logger.info(f"Successfully loaded {len(df)} records from KBLI codebook")
        
        # Perform data validation if requested
        if validate_data:
            _validate_kbli_data(df)
        
        return df
    
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"The file {file_path} is empty")
    except Exception as e:
        logger.error(f"Error loading KBLI codebook: {str(e)}")
        raise


def _validate_kbli_data(df: pd.DataFrame) -> None:
    """
    Validate the loaded KBLI codebook data.
    
    Parameters
    ----------
    df : pd.DataFrame
        The loaded KBLI codebook DataFrame.
    
    Raises
    ------
    ValueError
        If validation fails.
    """
    
    required_columns = ['kategori', 'digit', 'kode', 'judul', 'deskripsi']
    
    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for empty DataFrame
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Check for null values in critical columns
    critical_columns = ['kode', 'judul']
    for col in critical_columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            logger.warning(f"Found {null_count} null values in column '{col}'")
    
    # Validate digit column values
    invalid_digits = df[~df['digit'].isin([1, 2, 3, 4, 5])]['digit'].unique()
    if len(invalid_digits) > 0:
        logger.warning(f"Found unexpected digit values: {invalid_digits}")
    
    # Validate kode column (should be strings)
    if not df['kode'].dtype == 'string':
        logger.warning("'kode' column is not of string type")
    
    logger.info("Data validation completed successfully")


def get_kbli_by_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Filter KBLI codebook by category.
    
    Parameters
    ----------
    df : pd.DataFrame
        The KBLI codebook DataFrame.
    category : str
        The category to filter by (e.g., 'A', 'B', 'C', etc.).
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only records for the specified category.
    """
    
    if 'kategori' not in df.columns:
        raise ValueError("DataFrame must contain 'kategori' column")
    
    filtered_df = df[df['kategori'] == category.upper()].copy()
    logger.info(f"Found {len(filtered_df)} records for category '{category}'")
    
    return filtered_df


def get_kbli_by_digit_level(df: pd.DataFrame, digit_level: int) -> pd.DataFrame:
    """
    Filter KBLI codebook by digit level.
    
    Parameters
    ----------
    df : pd.DataFrame
        The KBLI codebook DataFrame.
    digit_level : int
        The digit level to filter by (1-5).
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only records for the specified digit level.
    """
    
    if 'digit' not in df.columns:
        raise ValueError("DataFrame must contain 'digit' column")
    
    if digit_level not in [1, 2, 3, 4, 5]:
        raise ValueError("digit_level must be between 1 and 5")
    
    filtered_df = df[df['digit'] == digit_level].copy()
    logger.info(f"Found {len(filtered_df)} records for digit level {digit_level}")
    
    return filtered_df


def get_codebook_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a summary of the KBLI codebook dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The KBLI codebook DataFrame.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing summary statistics of the codebook.
    """
    
    summary = {
        'total_records': len(df),
        'categories': sorted(df['kategori'].unique().tolist()) if 'kategori' in df.columns else [],
        'digit_levels': sorted(df['digit'].unique().tolist()) if 'digit' in df.columns else [],
        'records_by_category': df['kategori'].value_counts().to_dict() if 'kategori' in df.columns else {},
        'records_by_digit': df['digit'].value_counts().to_dict() if 'digit' in df.columns else {},
        'null_values': df.isnull().sum().to_dict()
    }
    
    return summary


# Example usage and testing
if __name__ == "__main__":
    try:
        # Load the KBLI codebook
        kbli_df = load_kbli_codebook()
        
        # Display basic information
        print(f"Loaded KBLI codebook with {len(kbli_df)} records")
        print(f"Columns: {list(kbli_df.columns)}")
        print(f"Data types:\n{kbli_df.dtypes}")
        
        # Show sample data
        print("\nSample data:")
        print(kbli_df.head())
        
        # Get summary
        summary = get_codebook_summary(kbli_df)
        print(f"\nSummary: {summary}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
