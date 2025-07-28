"""
Module for loading main dataset with standardized column structure.

This module provides functions to load and process datasets that follow the
standard format with columns: text, kbli_code, category, kbli_count.
The kbli_code column is handled as string type to preserve leading zeros.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_main_dataset(
    file_path: Union[str, Path],
    encoding: str = "utf-8",
    validate_data: bool = True,
    expected_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load main dataset with standardized column structure.
    
    This function loads CSV files with the standard format:
    - text: string (job descriptions/activities)
    - kbli_code: string (KBLI classification codes, preserves leading zeros)
    - category: string (category letters like A, B, C, etc.)
    - kbli_count: int64 (count values)
    
    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file to load.
    encoding : str, default 'utf-8'
        Character encoding of the CSV file.
    validate_data : bool, default True
        Whether to perform basic data validation after loading.
    expected_columns : List[str], optional
        List of expected column names. If None, uses the standard format:
        ['text', 'kbli_code', 'category', 'kbli_count']
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the dataset with proper data types:
        - text: string
        - kbli_code: string (preserves leading zeros)
        - category: string
        - kbli_count: int64
    
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
    >>> # Load mini_test.csv
    >>> df = load_main_dataset("data/processed/mini_test.csv")
    
    >>> # Load with custom path
    >>> df = load_main_dataset("/path/to/custom/dataset.csv")
    
    >>> # Load without validation
    >>> df = load_main_dataset("dataset.csv", validate_data=False)
    
    >>> # Load with custom expected columns
    >>> df = load_main_dataset("dataset.csv", 
    ...                       expected_columns=['text', 'code', 'cat', 'count'])
    """
    
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    logger.info(f"Loading main dataset from: {file_path}")
    
    # Set default expected columns if not provided
    if expected_columns is None:
        expected_columns = ['text', 'kbli_code', 'category', 'kbli_count']
    
    # Define data types for standard columns
    dtype_mapping = {
        'text': 'string',
        'kbli_code': 'string',  # Keep as string to preserve leading zeros
        'category': 'string',
        'kbli_count': 'int64'
    }
    
    try:
        # Load the CSV file with specified data types
        df = pd.read_csv(
            file_path,
            dtype=dtype_mapping,
            encoding=encoding,
            na_values=['', 'NA', 'N/A', 'null', 'NULL']
        )
        
        logger.info(f"Successfully loaded {len(df)} records from dataset")
        
        # Perform data validation if requested
        if validate_data:
            _validate_main_dataset(df, expected_columns)
        
        return df
    
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"The file {file_path} is empty")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def _validate_main_dataset(df: pd.DataFrame, expected_columns: List[str]) -> None:
    """
    Validate the loaded main dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset DataFrame.
    expected_columns : List[str]
        List of expected column names.
    
    Raises
    ------
    ValueError
        If validation fails.
    """
    
    # Check if all expected columns are present
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for empty DataFrame
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    # Check for null values in critical columns
    critical_columns = ['text', 'kbli_code'] if 'kbli_code' in df.columns else ['text']
    for col in critical_columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                logger.warning(f"Found {null_count} null values in column '{col}'")
    
    # Validate kbli_code column if it exists
    if 'kbli_code' in df.columns:
        if not df['kbli_code'].dtype == 'string':
            logger.warning("'kbli_code' column is not of string type")
        
        # Check for invalid kbli_code formats (should be numeric strings)
        invalid_codes = df[~df['kbli_code'].str.match(r'^\d+$', na=False)]['kbli_code'].dropna()
        if len(invalid_codes) > 0:
            logger.warning(f"Found {len(invalid_codes)} non-numeric kbli_code values")
    
    # Validate category column if it exists
    if 'category' in df.columns:
        unique_categories = df['category'].unique()
        expected_categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                             'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U']
        invalid_categories = [cat for cat in unique_categories if cat not in expected_categories]
        if invalid_categories:
            logger.warning(f"Found unexpected category values: {invalid_categories}")
    
    # Validate kbli_count column if it exists
    if 'kbli_count' in df.columns:
        if df['kbli_count'].dtype != 'int64':
            logger.warning("'kbli_count' column is not of int64 type")
        
        negative_counts = df[df['kbli_count'] < 0]['kbli_count']
        if len(negative_counts) > 0:
            logger.warning(f"Found {len(negative_counts)} negative kbli_count values")
    
    logger.info("Data validation completed successfully")


def load_mini_test_dataset(
    file_path: Optional[str] = None,
    encoding: str = "utf-8",
    validate_data: bool = True
) -> pd.DataFrame:
    """
    Convenience function to load the mini_test.csv dataset.
    
    Parameters
    ----------
    file_path : str, optional
        Path to the mini_test.csv file. If None, uses the default path
        relative to the project structure.
    encoding : str, default 'utf-8'
        Character encoding of the CSV file.
    validate_data : bool, default True
        Whether to perform basic data validation after loading.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the mini_test dataset.
    
    Examples
    --------
    >>> # Load with default path
    >>> df = load_mini_test_dataset()
    
    >>> # Load with custom path
    >>> df = load_mini_test_dataset("path/to/mini_test.csv")
    """
    
    # Set default file path if not provided
    if file_path is None:
        # Assuming the standard project structure
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        file_path = project_root / "data" / "processed" / "mini_test.csv"
    
    return load_main_dataset(file_path, encoding=encoding, validate_data=validate_data)


def filter_by_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Filter dataset by category.
    
    Parameters
    ----------
    df : pd.DataFrame
        The main dataset DataFrame.
    category : str
        The category to filter by (e.g., 'A', 'B', 'C', etc.).
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only records for the specified category.
    """
    
    if 'category' not in df.columns:
        raise ValueError("DataFrame must contain 'category' column")
    
    filtered_df = df[df['category'] == category.upper()].copy()
    logger.info(f"Found {len(filtered_df)} records for category '{category}'")
    
    return filtered_df


def filter_by_kbli_code(df: pd.DataFrame, kbli_code: str) -> pd.DataFrame:
    """
    Filter dataset by specific KBLI code.
    
    Parameters
    ----------
    df : pd.DataFrame
        The main dataset DataFrame.
    kbli_code : str
        The KBLI code to filter by (e.g., '10110', '01111').
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only records for the specified KBLI code.
    """
    
    if 'kbli_code' not in df.columns:
        raise ValueError("DataFrame must contain 'kbli_code' column")
    
    filtered_df = df[df['kbli_code'] == str(kbli_code)].copy()
    logger.info(f"Found {len(filtered_df)} records for KBLI code '{kbli_code}'")
    
    return filtered_df


def filter_by_kbli_pattern(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """
    Filter dataset by KBLI code pattern (e.g., codes starting with specific digits).
    
    Parameters
    ----------
    df : pd.DataFrame
        The main dataset DataFrame.
    pattern : str
        The pattern to match (e.g., '101' for codes starting with 101).
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing records matching the pattern.
    """
    
    if 'kbli_code' not in df.columns:
        raise ValueError("DataFrame must contain 'kbli_code' column")
    
    filtered_df = df[df['kbli_code'].str.startswith(pattern, na=False)].copy()
    logger.info(f"Found {len(filtered_df)} records matching pattern '{pattern}'")
    
    return filtered_df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a summary of the main dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The main dataset DataFrame.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing summary statistics of the dataset.
    """
    
    summary = {
        'total_records': len(df),
        'columns': list(df.columns),
        'data_types': df.dtypes.to_dict()
    }
    
    # Add category-specific summary if category column exists
    if 'category' in df.columns:
        summary['unique_categories'] = sorted(df['category'].unique().tolist())
        summary['records_by_category'] = df['category'].value_counts().to_dict()
    
    # Add KBLI code-specific summary if kbli_code column exists
    if 'kbli_code' in df.columns:
        summary['unique_kbli_codes'] = len(df['kbli_code'].unique())
        summary['top_kbli_codes'] = df['kbli_code'].value_counts().head(10).to_dict()
    
    # Add kbli_count statistics if column exists
    if 'kbli_count' in df.columns:
        summary['kbli_count_stats'] = {
            'min': int(df['kbli_count'].min()),
            'max': int(df['kbli_count'].max()),
            'mean': float(df['kbli_count'].mean()),
            'median': float(df['kbli_count'].median())
        }
    
    # Add null value counts
    summary['null_values'] = df.isnull().sum().to_dict()
    
    return summary


def sample_dataset(df: pd.DataFrame, n: int = 5, random_state: Optional[int] = 42) -> pd.DataFrame:
    """
    Get a random sample from the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The main dataset DataFrame.
    n : int, default 5
        Number of samples to return.
    random_state : int, optional
        Random seed for reproducible sampling.
    
    Returns
    -------
    pd.DataFrame
        Sample DataFrame with n records.
    """
    
    if len(df) < n:
        logger.warning(f"Dataset has only {len(df)} records, returning all records")
        return df.copy()
    
    sample_df = df.sample(n=n, random_state=random_state).copy()
    logger.info(f"Generated random sample of {len(sample_df)} records")
    
    return sample_df


# Example usage and testing
if __name__ == "__main__":
    try:
        # Load the mini_test dataset
        df = load_mini_test_dataset()
        
        # Display basic information
        print(f"Loaded dataset with {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        
        # Show sample data
        print("\nSample data:")
        print(df.head())
        
        # Test filtering by category
        category_c = filter_by_category(df, 'C')
        print(f"\nCategory C records: {len(category_c)}")
        
        # Get summary
        summary = get_dataset_summary(df)
        print(f"\nDataset summary:")
        for key, value in summary.items():
            if key not in ['top_kbli_codes']:  # Skip detailed dictionary output
                print(f"{key}: {value}")
        
        # Show sample of different KBLI codes
        print(f"\nSample KBLI codes: {df['kbli_code'].unique()[:10].tolist()}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
