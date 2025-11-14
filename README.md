# Singapore Foursquare Synthetic Dataset Generation

Transaction-only synthetic data generation pipeline for Singapore Foursquare check-in data.

## Overview

This project converts Foursquare Singapore check-in data (2013) into synthetic transaction datasets. All check-ins are converted to transactions - views and reviews are empty placeholders.

## Project Structure

```
singapore_foursquare_dataset/
├── src/                    # Source code (modular by function)
│   ├── preprocessing/      # Data preprocessing scripts
│   ├── generation/         # Transaction generation
│   ├── validation/         # Quality validation
│   └── utils/              # Utility functions
├── config/                 # Configuration files
│   ├── parameters.json
│   └── paths.py
├── data/                   # All data (organized by stage)
│   ├── raw/                # Original datasets (never modify)
│   ├── interim/            # Temporary transformations
│   ├── processed/          # Clean inputs
│   └── synthetic/          # Generated outputs
├── scripts/                # Executable entry points
├── tests/                  # Unit tests
└── .github/                # Documentation & CI/CD
```

## Quick Start

### Python Environment Setup

**1. Create a virtual environment:**

```powershell
# Using venv (Python 3.9+)
python -m venv .env

# Or using conda
conda create -n fsq-env python=3.9
```

**2. Activate the environment:**

```powershell
# On Windows (PowerShell)
.\.env\Scripts\Activate.ps1

# On Windows (Command Prompt)
.env\Scripts\activate.bat

# On Linux/Mac
source .env/bin/activate

# Using conda
conda activate fsq-env
```

### Installation

```powershell
# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Environment Configuration (Optional)

If you need to configure environment-specific variables (e.g., API keys, custom paths), create a `.env` file in the project root:

```powershell
# Copy the example template
Copy-Item .env.example .env

# Edit with your configuration
notepad .env
```

Example `.env` file contents:
```bash
# Python environment (if needed)
PYTHON_ENV=production

# Custom data paths (optional - overrides default config)
# DATA_RAW_PATH=./data/raw
# DATA_PROCESSED_PATH=./data/processed
# DATA_SYNTHETIC_PATH=./data/synthetic

# API keys (if needed for future features)
# GEOPY_API_KEY=your_api_key_here
# DATABRICKS_TOKEN=your_token_here

# Processing parameters (optional)
# MIN_INTERACTIONS=5
# BATCH_SIZE=100
```

**Note:** The `.env` file is already included in `.gitignore` and will not be committed to the repository.

### Running the Pipeline

```powershell
# 1. Preprocess data
python scripts/preprocess_fsq_data.py

# 2. Generate synthetic datasets
python scripts/generate_synthetic_data.py

# 3. Postprocess synthetic data (5-core filtering)
python scripts/postprocess_synthetic_data.py

# 4. Validate outputs
python scripts/validate_output.py
```

### Using Makefile

```bash
make install      # Install dependencies
make preprocess   # Preprocess raw FSQ data
make generate     # Generate synthetic datasets
make postprocess  # Apply 5-core filtering to synthetic data
make validate     # Validate outputs
make clean        # Remove generated files
make test         # Run tests
```

## Data Pipeline

1. **Raw Data** (`data/raw/`) → Original FSQ datasets (never modify)
2. **Interim Data** (`data/interim/`) → Temporary transformations (planning area enrichment)
3. **Processed Data** (`data/processed/`) → Clean inputs for generation
4. **Synthetic Data** (`data/synthetic/`) → Generated outputs
5. **Postprocessed Data** (`data/synthetic/postprocessed/`) → 5-core filtered outputs

## Architecture

**Core Principle:** ALL Foursquare check-ins → transactions only

- ✅ Transactions: Real check-in data with original timestamps
- ⚠️ Views: Empty placeholder (0 entries)
- ⚠️ Reviews: Empty placeholder (0 entries)

No ML temporal models needed - uses original FSQ timestamps directly.

## Key Files

- `src/preprocessing/add_planning_area.py` - Add geographic region information to checkins
- `src/preprocessing/fsq_to_input_json.py` - FSQ CSV → JSON conversion
- `src/generation/transaction_generator.py` - Main synthetic data generation
- `scripts/postprocess_synthetic_data.py` - Apply 5-core filtering to synthetic data
- `src/validation/validate_transactions.py` - Transaction quality checks
- `config/paths.py` - Centralized path definitions

## Requirements

- Python >= 3.9
- pandas >= 1.5.0
- geopandas >= 0.12.0
- numpy >= 1.21.0
- scikit-learn >= 1.2.0
- tqdm >= 4.64.0

See `requirements.txt` for complete list.

## Documentation

- `.github/copilot-instructions.md` - AI agent coding guidelines
- `.github/REFACTOR_GUIDE.md` - Project migration guide
- `.github/REFACTORING_SUMMARY.md` - Recent refactoring changes
- `data/raw/README.md` - Raw data documentation

## Development

### Running Tests

The project includes comprehensive tests to verify data consistency between raw and synthetic datasets.

**Install test dependencies:**
```powershell
pip install pytest pytest-cov
```

**Run all tests:**
```powershell
# Using pytest
pytest tests/ -v

# Using Makefile
make test
```

**Run specific test file:**
```powershell
pytest tests/test_user_consistency.py -v
```

**Run with detailed output:**
```powershell
pytest tests/ -v -s
```

**Run specific test class:**
```powershell
# Test only user consistency
pytest tests/test_user_consistency.py::TestUserConsistency -v

# Test only POI consistency
pytest tests/test_user_consistency.py::TestPOIConsistency -v
```

**Run with coverage report:**
```powershell
pytest tests/ --cov=src --cov-report=html
```

**Available Test Suites:**
- **TestUserConsistency** - Verifies all users in synthetic data exist in raw FSQ dataset
- **TestPOIConsistency** - Verifies all POIs in synthetic transactions exist in raw data
- **TestTransactionStructure** - Validates transaction-only structure (empty views/reviews)
- **TestDataIntegrity** - Checks overall data quality and completeness

### Code Style

- Follow PEP 8 standards
- Use type hints (PEP 484)
- Write PEP 257 docstrings
- Maximum line length: 79 characters

## Rollback

If you need to rollback to the pre-refactor state:

```powershell
git reset --hard 28fa49d4
```

This will restore the project to the old structure (json_gen/, Input_data/, configuration/).

## Branch

Current development branch: `databricks_integration_datapipeline`

## License

Research use only

## Contributors

Your Team

---

**Last Updated:** November 14, 2025  
**Version:** 1.0.0 (Post-refactor)
