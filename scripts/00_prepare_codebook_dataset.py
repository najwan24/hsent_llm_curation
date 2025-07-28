#!/usr/bin/env python3
"""
ACSES Pilot Study - Codebook and Dataset Preparation Script

This script orchestrates the preparation of:
1. Hierarchical KBLI codebook for LLM prompts
2. Datasets with deterministic UUID identifiers

Usage:
    python 00_prepare_codebook_dataset.py [options]
    
    Options:
        --codebook-input PATH       Path to original KBLI codebook CSV
        --codebook-output PATH      Path for hierarchical codebook output  
        --dataset-input PATH        Path to dataset CSV to add UUIDs
        --dataset-output PATH       Path for dataset with UUIDs output
        --dataset-name NAME         Name for the dataset (used in auto path)
        --validate-columns COLS     Comma-separated list of required columns
        --log-level LEVEL           Logging level (DEBUG, INFO, WARNING, ERROR)
        --skip-codebook            Skip codebook processing
        --skip-dataset             Skip dataset processing

Examples:
    # Process codebook only
    python 00_prepare_codebook_dataset.py --skip-dataset
    
    # Process dataset with validation
    python 00_prepare_codebook_dataset.py --dataset-input data/processed/mini_test.csv --validate-columns text,kbli_code,category
    
    # Full processing with custom paths
    python 00_prepare_codebook_dataset.py --codebook-input data/external/kbli_codebook.csv --dataset-input data/processed/mini_test.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our modules
from src.features.create_hierarchical_codebook import create_hierarchical_codebook, validate_hierarchical_codebook
from src.features.generate_uuid_dataset import process_dataset_with_uuid, validate_uuid_integrity


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration with file and console handlers.
    
    Creates logs in logs/codebook_preparation/ directory with timestamp.
    
    Parameters
    ----------
    log_level : str, default "INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    
    # Create logs directory
    log_dir = Path("logs") / "codebook_preparation"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"codebook_prep_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")
    
    return logger


def validate_paths_and_columns(args: argparse.Namespace, logger: logging.Logger) -> None:
    """
    Validate input paths and column specifications.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments
    logger : logging.Logger
        Logger instance for messages
    
    Raises
    ------
    FileNotFoundError
        If required input files don't exist
    ValueError
        If path or column validation fails
    """
    
    logger.info("Validating input paths and arguments...")
    
    # Validate codebook input if processing codebook
    if not args.skip_codebook:
        if args.codebook_input and not Path(args.codebook_input).exists():
            raise FileNotFoundError(f"Codebook input file not found: {args.codebook_input}")
        logger.debug(f"Codebook input path validated: {args.codebook_input}")
    
    # Validate dataset input if processing dataset
    if not args.skip_dataset:
        if args.dataset_input and not Path(args.dataset_input).exists():
            raise FileNotFoundError(f"Dataset input file not found: {args.dataset_input}")
        logger.debug(f"Dataset input path validated: {args.dataset_input}")
        
        # Validate column specification
        if args.validate_columns:
            columns = [col.strip() for col in args.validate_columns.split(',')]
            if len(columns) == 0:
                raise ValueError("No columns specified for validation")
            logger.debug(f"Column validation requested for: {columns}")
    
    logger.info("Path and argument validation completed successfully")


def process_hierarchical_codebook(args: argparse.Namespace, logger: logging.Logger) -> Optional[Path]:
    """
    Process the KBLI codebook into hierarchical format.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments
    logger : logging.Logger
        Logger instance for messages
    
    Returns
    -------
    Path or None
        Path to created hierarchical codebook, or None if skipped
    
    Raises
    ------
    Exception
        If codebook processing fails
    """
    
    if args.skip_codebook:
        logger.info("Skipping codebook processing as requested")
        return None
    
    logger.info("=" * 60)
    logger.info("STARTING HIERARCHICAL CODEBOOK PROCESSING")
    logger.info("=" * 60)
    
    try:
        # Create hierarchical codebook
        output_path = create_hierarchical_codebook(
            input_path=args.codebook_input,
            output_path=args.codebook_output,
            validate_output=True
        )
        
        logger.info(f"✅ Hierarchical codebook created successfully: {output_path}")
        
        # Additional validation and reporting
        from src.features.create_hierarchical_codebook import load_hierarchical_codebook
        df = load_hierarchical_codebook(output_path)
        
        validation = validate_hierarchical_codebook(df)
        if validation['is_valid']:
            logger.info("✅ Hierarchical codebook validation passed")
        else:
            logger.warning(f"⚠️ Hierarchical codebook validation issues: {validation}")
        
        # Log sample for verification
        logger.info("\nSample of hierarchical codebook structure:")
        sample = df.head(2)
        for idx, row in sample.iterrows():
            logger.info(f"Code {row['code_5']}: {row['title_5']}")
            logger.info(f"  Level 4: {row['code_4']} - {row['title_4']}")
            logger.info(f"  Level 3: {row['code_3']} - {row['title_3']}")
            logger.info(f"  Level 2: {row['code_2']} - {row['title_2']}")
            logger.info(f"  Level 1: {row['code_1']} - {row['title_1']}")
            logger.info("")
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Failed to process hierarchical codebook: {str(e)}")
        raise


def process_dataset_with_uuids(args: argparse.Namespace, logger: logging.Logger) -> Optional[Path]:
    """
    Process dataset by adding deterministic UUID identifiers.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments
    logger : logging.Logger
        Logger instance for messages
    
    Returns
    -------
    Path or None
        Path to processed dataset, or None if skipped
    
    Raises
    ------
    Exception
        If dataset processing fails
    """
    
    if args.skip_dataset:
        logger.info("Skipping dataset processing as requested")
        return None
    
    if not args.dataset_input:
        logger.info("No dataset input specified, skipping dataset processing")
        return None
    
    logger.info("=" * 60)
    logger.info("STARTING DATASET UUID PROCESSING")
    logger.info("=" * 60)
    
    try:
        # Parse validation columns if specified
        validate_columns = None
        if args.validate_columns:
            validate_columns = [col.strip() for col in args.validate_columns.split(',')]
            logger.info(f"Will validate presence of columns: {validate_columns}")
        
        # Process dataset with UUIDs
        output_path = process_dataset_with_uuid(
            input_path=args.dataset_input,
            output_path=args.dataset_output,
            dataset_name=args.dataset_name,
            column_name='sample_id',
            validate_columns=validate_columns
        )
        
        logger.info(f"✅ Dataset with UUIDs created successfully: {output_path}")
        
        # Additional validation and reporting
        import pandas as pd
        df = pd.read_csv(output_path)
        
        uuid_validation = validate_uuid_integrity(df, 'sample_id')
        if uuid_validation['is_valid']:
            logger.info("✅ UUID integrity validation passed")
        else:
            logger.warning(f"⚠️ UUID integrity validation issues: {uuid_validation}")
        
        # Log sample for verification
        logger.info(f"\nDataset summary:")
        logger.info(f"  Total records: {len(df)}")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  UUID column position: {df.columns.get_loc('sample_id') + 1}")
        
        logger.info("\nSample records with UUIDs:")
        sample = df.head(3)
        for idx, row in sample.iterrows():
            logger.info(f"  UUID: {row['sample_id']}")
            if 'text' in df.columns:
                text_preview = str(row['text'])[:50] + "..." if len(str(row['text'])) > 50 else str(row['text'])
                logger.info(f"  Text: {text_preview}")
            logger.info("")
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Failed to process dataset with UUIDs: {str(e)}")
        raise


def main():
    """
    Main function to orchestrate codebook and dataset preparation.
    """
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="ACSES Pilot Study - Codebook and Dataset Preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Codebook options
    parser.add_argument('--codebook-input', type=str,
                       help='Path to original KBLI codebook CSV (default: data/external/kbli_codebook.csv)')
    parser.add_argument('--codebook-output', type=str,
                       help='Path for hierarchical codebook output (default: data/external/kbli_codebook_hierarchical.csv)')
    
    # Dataset options
    parser.add_argument('--dataset-input', type=str,
                       help='Path to dataset CSV to add UUIDs')
    parser.add_argument('--dataset-output', type=str,
                       help='Path for dataset with UUIDs output (auto-generated if not specified)')
    parser.add_argument('--dataset-name', type=str,
                       help='Name for the dataset (used in auto-generated output path)')
    parser.add_argument('--validate-columns', type=str,
                       help='Comma-separated list of required columns to validate')
    
    # Processing options
    parser.add_argument('--skip-codebook', action='store_true',
                       help='Skip codebook processing')
    parser.add_argument('--skip-dataset', action='store_true',
                       help='Skip dataset processing')
    
    # Logging options
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        logger.info("🚀 Starting ACSES Pilot Study - Codebook and Dataset Preparation")
        logger.info("=" * 80)
        logger.info(f"Arguments: {vars(args)}")
        logger.info("=" * 80)
        
        # Validate inputs
        validate_paths_and_columns(args, logger)
        
        # Process hierarchical codebook
        codebook_output = process_hierarchical_codebook(args, logger)
        
        # Process dataset with UUIDs
        dataset_output = process_dataset_with_uuids(args, logger)
        
        # Final summary
        logger.info("=" * 80)
        logger.info("🎉 PROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        if codebook_output:
            logger.info(f"📄 Hierarchical codebook: {codebook_output}")
        
        if dataset_output:
            logger.info(f"📊 Dataset with UUIDs: {dataset_output}")
        
        if not codebook_output and not dataset_output:
            logger.info("ℹ️  No processing performed (both codebook and dataset were skipped)")
        
        logger.info("\n✅ All tasks completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Critical error during processing: {str(e)}")
        logger.error("Processing failed. Check the logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
