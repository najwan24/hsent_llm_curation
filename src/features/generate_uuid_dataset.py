"""
Module for adding deterministic UUID4 identifiers to datasets.

This module provides functions to add unique sample IDs to datasets while
maintaining deterministic behavior (same input produces same UUID).
"""

import pandas as pd
import uuid
import hashlib
from pathlib import Path
from typing import Optional, Union
import logging

# Configure logging
logger = logging.getLogger(__name__)


def generate_deterministic_uuid(seed_string: str) -> str:
    """
    Generate a deterministic UUID4 based on a seed string.
    
    Same seed string will always produce the same UUID, ensuring reproducibility
    while maintaining the UUID4 format.
    
    Parameters
    ----------
    seed_string : str
        String to use as seed for UUID generation.
    
    Returns
    -------
    str
        36-character UUID4 string (e.g., '550e8400-e29b-41d4-a716-446655440000')
    
    Examples
    --------
    >>> uuid_str = generate_deterministic_uuid("sample_text_123")
    >>> len(uuid_str)
    36
    >>> uuid_str == generate_deterministic_uuid("sample_text_123")
    True
    """
    
    # Create a hash from the seed string
    hash_object = hashlib.md5(seed_string.encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    
    # Convert hash to UUID format
    # UUID4 format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    # where x is any hexadecimal digit and y is one of 8, 9, A, or B
    uuid_str = f"{hash_hex[:8]}-{hash_hex[8:12]}-4{hash_hex[13:16]}-{hash_hex[16:17]}{hash_hex[17:20]}-{hash_hex[20:32]}"
    
    # Ensure the UUID is valid by creating a UUID object and converting back
    try:
        uuid_obj = uuid.UUID(uuid_str)
        return str(uuid_obj)
    except ValueError:
        # Fallback: use uuid5 for guaranteed valid UUID
        namespace = uuid.NAMESPACE_DNS
        return str(uuid.uuid5(namespace, seed_string))


def add_uuid_to_dataset(
    df: pd.DataFrame, 
    column_name: str = 'sample_id',
    seed_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Add deterministic UUID column to a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to add UUID column to.
    column_name : str, default 'sample_id'
        Name of the UUID column to add.
    seed_column : str, optional
        Column to use as seed for UUID generation. If None, uses row index
        combined with first text column.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with UUID column added as the first column.
    
    Raises
    ------
    ValueError
        If the DataFrame is empty or seed_column doesn't exist.
    
    Examples
    --------
    >>> df = pd.DataFrame({'text': ['hello', 'world'], 'label': [1, 2]})
    >>> df_with_uuid = add_uuid_to_dataset(df)
    >>> 'sample_id' in df_with_uuid.columns
    True
    >>> df_with_uuid.columns[0]
    'sample_id'
    """
    
    if df.empty:
        raise ValueError("Cannot add UUID to empty DataFrame")
    
    logger.info(f"Adding UUID column '{column_name}' to dataset with {len(df)} records")
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Determine seed for UUID generation
    if seed_column and seed_column in df.columns:
        logger.debug(f"Using column '{seed_column}' as seed for UUID generation")
        seed_values = df_copy[seed_column].astype(str)
    else:
        # Use row index combined with first text-like column
        text_columns = df_copy.select_dtypes(include=['object', 'string']).columns
        if len(text_columns) > 0:
            first_text_col = text_columns[0]
            logger.debug(f"Using row index + '{first_text_col}' as seed for UUID generation")
            seed_values = df_copy.index.astype(str) + "_" + df_copy[first_text_col].astype(str)
        else:
            logger.debug("Using row index as seed for UUID generation")
            seed_values = df_copy.index.astype(str)
    
    # Generate UUIDs
    logger.debug("Generating deterministic UUIDs...")
    uuids = [generate_deterministic_uuid(seed) for seed in seed_values]
    
    # Verify UUID uniqueness
    unique_uuids = set(uuids)
    if len(unique_uuids) != len(uuids):
        duplicates = len(uuids) - len(unique_uuids)
        logger.warning(f"Found {duplicates} duplicate UUIDs. This may indicate duplicate seed values.")
    
    # Add UUID column as first column
    df_copy.insert(0, column_name, uuids)
    
    logger.info(f"Successfully added {len(uuids)} UUIDs with {len(unique_uuids)} unique values")
    
    return df_copy


def get_output_path(input_path: Union[str, Path], dataset_name: Optional[str] = None) -> Path:
    """
    Generate output path for dataset with UUIDs.
    
    Parameters
    ----------
    input_path : str or Path
        Path to the input dataset file.
    dataset_name : str, optional
        Custom name for the dataset. If None, derives from input filename.
    
    Returns
    -------
    Path
        Output path in format: data/processed/{dataset_name}_with_ids.csv
    
    Examples
    --------
    >>> path = get_output_path("data/raw/mini_test.csv")
    >>> str(path)
    'data/processed/mini_test_with_ids.csv'
    """
    
    input_path = Path(input_path)
    
    if dataset_name is None:
        # Extract dataset name from input filename
        dataset_name = input_path.stem
    
    # Construct output path
    project_root = Path.cwd()
    output_dir = project_root / "data" / "processed"
    output_path = output_dir / f"{dataset_name}_with_ids.csv"
    
    return output_path


def process_dataset_with_uuid(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    dataset_name: Optional[str] = None,
    column_name: str = 'sample_id',
    seed_column: Optional[str] = None,
    validate_columns: Optional[list] = None
) -> Path:
    """
    Process a dataset by adding UUID column and saving to output path.
    
    Parameters
    ----------
    input_path : str or Path
        Path to the input dataset CSV file.
    output_path : str or Path, optional
        Path where the output will be saved. If None, auto-generates path.
    dataset_name : str, optional
        Name for the dataset (used in auto-generated output path).
    column_name : str, default 'sample_id'
        Name of the UUID column to add.
    seed_column : str, optional
        Column to use as seed for UUID generation.
    validate_columns : list, optional
        List of column names that must be present in the dataset.
    
    Returns
    -------
    Path
        Path where the processed dataset was saved.
    
    Raises
    ------
    FileNotFoundError
        If input file doesn't exist.
    ValueError
        If required columns are missing or validation fails.
    
    Examples
    --------
    >>> output = process_dataset_with_uuid("data/processed/mini_test.csv")
    >>> print(f"Processed dataset saved to: {output}")
    """
    
    input_path = Path(input_path)
    
    # Validate input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Processing dataset: {input_path}")
    
    # Load the dataset
    try:
        # Import the load_main_data module
        from ..data.load_main_data import load_main_dataset
        df = load_main_dataset(input_path)
        logger.info(f"Loaded dataset with {len(df)} records and {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise
    
    # Validate required columns if specified
    if validate_columns:
        missing_columns = [col for col in validate_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        logger.debug(f"Column validation passed for: {validate_columns}")
    
    # Add UUID column
    try:
        df_with_uuid = add_uuid_to_dataset(df, column_name, seed_column)
        logger.info("UUID column added successfully")
    except Exception as e:
        logger.error(f"Failed to add UUID column: {str(e)}")
        raise
    
    # Determine output path
    if output_path is None:
        output_path = get_output_path(input_path, dataset_name)
    else:
        output_path = Path(output_path)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the processed dataset
    try:
        df_with_uuid.to_csv(output_path, index=False)
        logger.info(f"Dataset with UUIDs saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save dataset: {str(e)}")
        raise
    
    # Log summary
    logger.info(f"Processing completed successfully:")
    logger.info(f"  Input: {input_path} ({len(df)} records)")
    logger.info(f"  Output: {output_path} ({len(df_with_uuid)} records)")
    logger.info(f"  UUID column: '{column_name}' added as first column")
    
    return output_path


def validate_uuid_integrity(df: pd.DataFrame, uuid_column: str = 'sample_id') -> dict:
    """
    Validate the integrity of UUID column in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing UUID column to validate.
    uuid_column : str, default 'sample_id'
        Name of the UUID column to validate.
    
    Returns
    -------
    dict
        Dictionary containing validation results:
        - total_records: Total number of records
        - unique_uuids: Number of unique UUIDs
        - duplicates: Number of duplicate UUIDs
        - null_uuids: Number of null UUIDs
        - valid_format: Number of UUIDs with valid format
        - is_valid: Boolean indicating if all checks passed
    
    Examples
    --------
    >>> df = pd.DataFrame({'sample_id': ['550e8400-e29b-41d4-a716-446655440000'], 'data': [1]})
    >>> result = validate_uuid_integrity(df)
    >>> result['is_valid']
    True
    """
    
    if uuid_column not in df.columns:
        logger.error(f"UUID column '{uuid_column}' not found in DataFrame")
        return {'is_valid': False, 'error': f"Column '{uuid_column}' not found"}
    
    total_records = len(df)
    uuid_series = df[uuid_column]
    
    # Count unique UUIDs
    unique_uuids = uuid_series.nunique()
    duplicates = total_records - unique_uuids
    
    # Count null UUIDs
    null_uuids = uuid_series.isnull().sum()
    
    # Validate UUID format
    valid_format_count = 0
    for uuid_str in uuid_series.dropna():
        try:
            uuid.UUID(str(uuid_str))
            valid_format_count += 1
        except (ValueError, AttributeError):
            pass
    
    # Overall validation
    is_valid = (duplicates == 0 and null_uuids == 0 and 
                valid_format_count == (total_records - null_uuids))
    
    result = {
        'total_records': total_records,
        'unique_uuids': unique_uuids,
        'duplicates': duplicates,
        'null_uuids': null_uuids,
        'valid_format': valid_format_count,
        'is_valid': is_valid
    }
    
    logger.info(f"UUID integrity check: {result}")
    
    return result


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.DEBUG)
    
    # Test with sample data
    sample_df = pd.DataFrame({
        'text': ['sample text 1', 'sample text 2', 'sample text 3'],
        'kbli_code': ['10110', '10120', '10130'],
        'category': ['C', 'C', 'C'],
        'kbli_count': [71, 108, 225]
    })
    
    print("Testing UUID generation...")
    df_with_uuid = add_uuid_to_dataset(sample_df)
    print(f"Result shape: {df_with_uuid.shape}")
    print(f"Columns: {list(df_with_uuid.columns)}")
    print("\nSample data:")
    print(df_with_uuid.head())
    
    # Test UUID integrity
    integrity = validate_uuid_integrity(df_with_uuid)
    print(f"\nUUID Integrity: {integrity}")
