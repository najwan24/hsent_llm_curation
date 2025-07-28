#!/usr/bin/env python3
"""
Example usage of KBLI Prompt Builder

This script demonstrates how to use the KBLIPromptBuilder class to generate
prompts for KBLI classification tasks using job descriptions and codes.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.prompt_builder import KBLIPromptBuilder, create_prompt_builder, build_single_prompt
from src.data.load_main_data import load_mini_test_dataset
import pandas as pd

def example_single_prompt():
    """Example of building a single prompt"""
    
    print("=" * 60)
    print("Example 1: Building a Single Prompt")
    print("=" * 60)
    
    # Create prompt builder
    builder = KBLIPromptBuilder()
    
    # Example job description and KBLI code
    job_description = "jagal sapi menghasilkan potong sapi di bidang jagal sapi"
    kbli_code = "10110"
    
    print(f"Job Description: {job_description}")
    print(f"KBLI Code: {kbli_code}")
    
    # Build prompt
    prompt = builder.build_prompt(job_description, kbli_code)
    
    print(f"\nGenerated prompt ({len(prompt)} characters):")
    print("-" * 40)
    print(prompt)
    print("-" * 40)


def example_batch_processing():
    """Example of batch processing multiple prompts"""
    
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing Dataset")
    print("=" * 60)
    
    # Load dataset
    df = load_mini_test_dataset()
    print(f"Loaded dataset with {len(df)} records")
    
    # Create prompt builder
    builder = KBLIPromptBuilder()
    
    # Process first 5 rows
    sample_df = df.head(5)
    print(f"\nProcessing {len(sample_df)} sample records...")
    
    # Build prompts for all rows
    df_with_prompts = builder.batch_build_prompts(sample_df)
    
    print(f"Results:")
    print(f"  Original columns: {list(sample_df.columns)}")
    print(f"  New columns: {list(df_with_prompts.columns)}")
    print(f"  Prompts generated: {df_with_prompts['prompt'].notna().sum()}")
    
    # Show summary of each generated prompt
    print("\nSample prompts summary:")
    for idx, row in df_with_prompts.iterrows():
        text_preview = row['text'][:50] + "..." if len(row['text']) > 50 else row['text']
        prompt_length = len(row['prompt']) if pd.notna(row['prompt']) else 0
        print(f"  Row {idx}: Code {row['kbli_code']}, Text: '{text_preview}', Prompt: {prompt_length} chars")


def example_hierarchy_context():
    """Example of getting hierarchy context"""
    
    print("\n" + "=" * 60)
    print("Example 3: Getting Hierarchy Context")
    print("=" * 60)
    
    builder = KBLIPromptBuilder()
    
    # Example codes with different hierarchies
    example_codes = ["10110", "10211", "01111", "47111"]
    
    for code in example_codes:
        try:
            hierarchy = builder.get_hierarchy_context(code)
            print(f"\nHierarchy for code {code}:")
            print(hierarchy)
        except ValueError as e:
            print(f"\nError for code {code}: {str(e)}")


def example_validation_and_info():
    """Example of validation and getting codebook info"""
    
    print("\n" + "=" * 60)
    print("Example 4: Validation and Codebook Info")
    print("=" * 60)
    
    builder = KBLIPromptBuilder()
    
    # Get codebook info
    info = builder.get_codebook_info()
    print("Codebook Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test code validation
    test_codes = ["10110", "99999", "01111", "invalid"]
    print(f"\nCode Validation Tests:")
    
    for code in test_codes:
        is_valid = builder.validate_code(code)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"  {code}: {status}")
    
    # Show some available codes
    available_codes = builder.get_available_codes()
    print(f"\nSample of available codes (first 10):")
    for code in available_codes[:10]:
        print(f"  {code}")


def example_convenience_functions():
    """Example of using convenience functions"""
    
    print("\n" + "=" * 60)
    print("Example 5: Using Convenience Functions")
    print("=" * 60)
    
    # Using create_prompt_builder function
    builder = create_prompt_builder()
    print("Created prompt builder using convenience function")
    
    # Using build_single_prompt function
    job_desc = "membuat tempe dari kacang kedelai"
    code = "10391"
    
    prompt = build_single_prompt(job_desc, code)
    print(f"\nBuilt single prompt using convenience function:")
    print(f"Job: {job_desc}")
    print(f"Code: {code}")
    print(f"Prompt length: {len(prompt)} characters")
    
    # Show first part of the prompt
    print(f"\nFirst 200 characters of prompt:")
    print(prompt[:200] + "...")


def main():
    """Run all examples"""
    
    print("KBLI Prompt Builder - Usage Examples")
    print("====================================")
    
    try:
        # Run all examples
        example_single_prompt()
        example_batch_processing()
        example_hierarchy_context()
        example_validation_and_info()
        example_convenience_functions()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        raise


if __name__ == "__main__":
    main()
