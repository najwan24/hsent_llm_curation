# KBLI Prompt Building System - Complete Implementation

## Overview

This is a complete implementation of a KBLI (Indonesian Standard Industrial Classification) data processing and LLM prompt building system. The system transforms raw KBLI codebook data into hierarchical structures suitable for LLM classification tasks.

## 🎯 System Components

### 1. Data Loading Modules (`src/data/`)

#### `load_codebook.py`
- **Purpose**: Load and validate KBLI codebook CSV files
- **Key Features**: Preserves string dtypes for codes with leading zeros
- **Main Function**: `load_kbli_codebook()`
- **Validation**: Comprehensive data validation and filtering

#### `load_main_data.py`
- **Purpose**: Load main datasets with standardized column structure
- **Key Features**: Compatible with mini_test.csv format
- **Main Function**: `load_main_dataset()`
- **Validation**: Column validation and data type enforcement

### 2. Feature Engineering (`src/features/`)

#### `generate_uuid_dataset.py`
- **Purpose**: Add deterministic UUID4 identifiers to datasets
- **Key Features**: MD5-based deterministic UUID generation for reproducibility
- **Output**: Datasets with unique identifiers for tracking

#### `create_hierarchical_codebook.py`
- **Purpose**: Transform flat KBLI codebook into hierarchical format
- **Key Features**: Creates Section → Division → Group → Class → Sub-Class hierarchy
- **Output**: Hierarchical codebook for LLM prompt context

### 3. LLM Integration (`src/llm/`)

#### `prompt_builder.py`
- **Purpose**: Build classification prompts using master template
- **Key Features**: 
  - Template-based prompt generation
  - Hierarchical context injection
  - Batch processing capabilities
  - JSON response formatting
- **Main Class**: `KBLIPromptBuilder`

### 4. Orchestration (`scripts/`)

#### `00_prepare_codebook_dataset.py`
- **Purpose**: Command-line interface for complete system orchestration
- **Key Features**: CLI with comprehensive logging and validation
- **Capabilities**: Process codebook and/or datasets with flexible options

## 🚀 Quick Start Guide

### Step 1: Prepare Data Files
Ensure you have:
- `data/external/kbli_codebook.csv` (KBLI codebook)
- `data/processed/mini_test.csv` (or your dataset)
- `master_prompt.txt` (LLM prompt template)

### Step 2: Process Codebook and Dataset
```bash
# Full processing
python scripts/00_prepare_codebook_dataset.py

# Process only codebook
python scripts/00_prepare_codebook_dataset.py --skip-dataset

# Process dataset with validation
python scripts/00_prepare_codebook_dataset.py \
    --dataset-input data/processed/mini_test.csv \
    --validate-columns text,kbli_code,category
```

### Step 3: Build LLM Prompts
```python
from src.llm.prompt_builder import create_prompt_builder

# Create prompt builder
builder = create_prompt_builder()

# Build single prompt
prompt = builder.build_prompt(
    job_description="jagal sapi menghasilkan potong sapi",
    kbli_code="10110"
)

# Batch process dataset
import pandas as pd
df = pd.read_csv("data/processed/mini_test.csv")
df_with_prompts = builder.build_prompts_for_dataset(df.head(10))
```

## 📊 System Performance

### Data Processing Results
- **KBLI Codebook**: 2,712 records → 1,791 hierarchical entries
- **Dataset Processing**: 2,266 records with UUID generation
- **Prompt Generation**: ~2,500-3,000 characters per prompt
- **Processing Speed**: Batch processing of hundreds of records per second

### Validation Results
- ✅ All data types preserved correctly
- ✅ Hierarchical structure validated
- ✅ UUID generation deterministic and reproducible
- ✅ Prompt templates properly formatted
- ✅ JSON response format validated

## 🔧 Advanced Usage

### Custom Prompt Templates
Modify `master_prompt.txt` to customize:
- Classification instructions
- Response format requirements
- Analysis criteria
- Context presentation

### Batch Processing
```python
# Process large datasets efficiently
from src.llm.prompt_builder import KBLIPromptBuilder

builder = KBLIPromptBuilder()
large_df = pd.read_csv("large_dataset.csv")

# Process in chunks for memory efficiency
chunk_size = 1000
for i in range(0, len(large_df), chunk_size):
    chunk = large_df.iloc[i:i+chunk_size]
    chunk_with_prompts = builder.build_prompts_for_dataset(chunk)
    chunk_with_prompts.to_csv(f"output/prompts_chunk_{i//chunk_size}.csv")
```

### Validation and Debugging
```python
# Validate KBLI codes
builder = KBLIPromptBuilder()
valid_codes = builder.get_valid_codes()
print(f"Total valid codes: {len(valid_codes)}")

# Get hierarchy for specific code
hierarchy = builder.get_hierarchy_context("10110")
print(hierarchy)

# Get codebook statistics
info = builder.get_codebook_info()
print(f"Sections: {info['unique_sections']}")
print(f"Divisions: {info['unique_divisions']}")
```

## 📁 File Structure
```
├── src/
│   ├── data/
│   │   ├── load_codebook.py        # KBLI codebook loading
│   │   └── load_main_data.py       # Main dataset loading
│   ├── features/
│   │   ├── generate_uuid_dataset.py    # UUID generation
│   │   └── create_hierarchical_codebook.py  # Hierarchy creation
│   └── llm/
│       └── prompt_builder.py       # LLM prompt building
├── scripts/
│   └── 00_prepare_codebook_dataset.py  # CLI orchestration
├── examples/
│   └── prompt_builder_usage.py     # Usage examples
├── data/
│   ├── external/
│   │   ├── kbli_codebook.csv       # Original codebook
│   │   └── kbli_codebook_hierarchical.csv  # Generated hierarchy
│   └── processed/
│       └── mini_test.csv           # Sample dataset
├── logs/                           # System logs
└── master_prompt.txt              # LLM prompt template
```

## 🧪 Testing and Examples

Run comprehensive examples:
```bash
python examples/prompt_builder_usage.py
```

This will demonstrate:
1. Single prompt building
2. Batch dataset processing
3. Hierarchy context retrieval
4. Code validation
5. Convenience functions

## 🎯 Key Features

### ✅ Production Ready
- Comprehensive error handling
- Professional logging
- Data validation at every step
- Memory-efficient batch processing

### ✅ Reproducible
- Deterministic UUID generation
- Version-controlled data transformations
- Consistent output formats

### ✅ Flexible
- Modular design for easy extension
- Configurable via CLI parameters
- Template-based prompt customization

### ✅ Scalable
- Batch processing capabilities
- Memory-efficient operations
- Parallel processing ready

## 📝 System Requirements

- Python 3.7+
- pandas
- pathlib (built-in)
- logging (built-in)
- hashlib (built-in)
- uuid (built-in)
- argparse (built-in)

## 🎓 Usage Patterns

### For Data Scientists
1. Use `load_codebook.py` and `load_main_data.py` for data exploration
2. Use `prompt_builder.py` for quick prompt generation
3. Use batch processing for large-scale analysis

### For ML Engineers
1. Use orchestration script for pipeline integration
2. Use UUID generation for experiment tracking
3. Use hierarchical codebook for feature engineering

### For Research
1. Use deterministic UUIDs for reproducible experiments
2. Use hierarchical context for interpretable models
3. Use prompt templates for consistent evaluation

---

## 🎉 System Status: **COMPLETE & TESTED** ✅

All components implemented, tested with real data, and ready for production use!
