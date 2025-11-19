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
```
# Using venv (Python 3.9+)
python -m venv .env
```
**2. Activate the environment:**
```
# On Windows (Command Prompt)
.env\Scripts\activate.bat

# On Linux/Mac
source .env/bin/activate
```
### Installation
```powershell
# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Running the Pipeline

```powershell
# 1. Preprocess data
python scripts/preprocess_fsq_data.py

# 2. Generate synthetic datasets
python scripts/generate_synthetic_data.py

# 3. Postprocess synthetic data (5-core filtering)
python scripts/postprocess_synthetic_data.py

# 4. Extract unique POIs from filtered data
python src/utils/poi_extraction.py

# 5. Validate outputs
python scripts/validate_output.py
```

## Data Pipeline

1. **Raw Data** (`data/raw/`) → Original FSQ datasets (never modify)
2. **Interim Data** (`data/interim/`) → Temporary transformations (planning area enrichment)
3. **Processed Data** (`data/processed/`) → Clean inputs for generation
4. **Synthetic Data** (`data/synthetic/`) → Generated outputs
5. **Postprocessed Data** (`data/synthetic/postprocessed/`) → 5-core filtered outputs
6. **POI Extraction** → Unique POIs extracted from filtered data for validation/analysis

### Pipeline Steps Explained

**Step 1: Preprocess** - Enriches raw data with geographic information and converts to JSON
- Adds planning area/subzone information to checkins via GeoJSON lookup
- Converts FSQ CSV format to structured JSON for the generator
- Output: `data/processed/input_filtered.json` and `input_all_categories.json`

**Step 2: Generate** - Creates synthetic user interaction data
- Generates synthetic transactions based on real FSQ checkin patterns
- Preserves temporal patterns and user behavior
- Output: `data/synthetic/fsq_to_synthetic_*.json`

**Step 3: Postprocess** - Applies 5-core filtering for data quality
- Ensures users have ≥5 interactions
- Ensures POIs have ≥5 visits
- Removes sparse data that could affect model training
- Output: `data/synthetic/postprocessed/*_5core_filtered.json`

**Step 4: Extract POIs** - Creates POI reference dataset
- Extracts all unique POIs from filtered synthetic data
- Includes POI metadata (name, categories, location, planning area)
- Useful for validation, analysis, and downstream applications
- Output: `src/utils/all_pois_just5_core_synthetic.json`

**Step 5: Validate** - Checks data quality and consistency
- Validates transaction structure
- Verifies user/POI consistency with original data
- Generates quality reports

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
- `src/utils/poi_extraction.py` - Extract unique POIs from filtered synthetic data
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

**Run specific test file:**
```powershell
pytest tests/test_user_consistency.py -v
```

**Run specific test class:**
```powershell
# Test only user consistency
pytest tests/test_user_consistency.py::TestUserConsistency -v

# Test only POI consistency
pytest tests/test_user_consistency.py::TestPOIConsistency -v
```

## Branch
Current development branch: `transaction_json_generator`

