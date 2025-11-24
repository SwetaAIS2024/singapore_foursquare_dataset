# Singapore Foursquare Synthetic Dataset Generation

Transaction-only synthetic data generation pipeline for Singapore Foursquare check-in data.

## Overview

This project converts Foursquare Singapore check-in data (2013) into synthetic transaction datasets. All check-ins are converted to transactions - views and reviews are empty placeholders.

---

## Project Structure

```
singapore_foursquare_dataset/
├── src/                    # Source code (modular by function)
│   ├── preprocessing/      # Data preprocessing scripts
│   ├── generation/         # Transaction generation
│   └── utils/              # Utility functions
├── config/                 # Configuration files
│   └── paths.py
├── data/                   # All data (organized by stage)
│   ├── raw/                # Original datasets (never modify)
│   ├── processed/          # Clean inputs
│   ├── synthetic/          # Generated synthetic output
│   └── synthetic_postprocess/  # Generated post processed synthetic output 
├── scripts/                # Executable entry points
├── tests/                  # Unit tests
│   ├── test_user_consistency.py
└── .github/                # Documentation & CI/CD
```

### Python Environment Setup

- python -m venv .env
- source .env/bin/activate
- pip install -r requirements.txt


#### Running the scripts
```bash
# 1. Preprocess data, this is for the fsq to json conversion
python scripts/preprocess_fsq_data.py

# 2. Generate synthetic datasets
python scripts/generate_synthetic_data.py

# 3. Postprocess synthetic data (5-core filtering)
python scripts/postprocess_synthetic_data.py

# Default: Replace categories only (with backup)
python scripts/postprocess_synthetic_data.py

# Create both versions (with backup) ->>> USE THIS TO HAVE BOTH THE VERSIONS
python scripts/postprocess_synthetic_data.py --preserve-original

# Custom mapping file
python scripts/postprocess_synthetic_data.py --mapping-file path/to/mapping.csv

# Disable category grouping
python scripts/postprocess_synthetic_data.py --no-grouping

# Skip backup (fast mode, use with caution)
python scripts/postprocess_synthetic_data.py --no-backup

# 4. Extract unique POIs from filtered data (synthetic postprocess)
python src/utils/poi_extraction.py

# 5. Extrapolate views and reviews from transactions
# This generates the final dataset with complete user interaction funnel
python scripts/extrapolate_views_reviews.py

# Default paths (reads filtered_5core_filtered.json, outputs to data/final_dataset/)
python scripts/extrapolate_views_reviews.py

# Custom paths
python scripts/extrapolate_views_reviews.py \
  --transaction-file data/synthetic_postprocess/all_categories_5core_filtered.json \
  --output-dir data/final_dataset \
  --log-level INFO

# 6. Extract unique POIs from final dataset (with views/transactions/reviews)
python src/utils/poi_extraction_final.py

# 7. Validate outputs by running the tests
python tests/test_user_consistency.py
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
- Output: `data/synthetic_postprocess/*_5core_filtered.json`

**Step 4: Extract POIs (Postprocess)** - Creates POI reference from filtered synthetic data
- Extracts all unique POIs from 5-core filtered synthetic data
- Includes POI metadata (name, categories, location, planning area)
- Useful for validation, analysis, and downstream applications
- Output: `src/utils/all_pois_filtered_synthetic.json` and `src/utils/all_pois_all_categories_synthetic.json`

**Step 5: Extrapolate Views & Reviews** - Completes user interaction funnel
- Generates views (20-90 min before each transaction)
- Generates reviews (1-7 days after 40% of transactions)
- Applies realistic spatial-temporal patterns based on user behavior research
- Maintains strict funnel order: View → Transaction → Review
- Output: `data/final_dataset/final_dataset_with_views_reviews.json`
- Statistics: `data/final_dataset/extrapolation_stats.json`
- Documentation: `docs/VIEWS_REVIEWS_GENERATION_LOGIC.md`

**Step 6: Extract POIs (Final)** - Creates comprehensive POI reference with interaction statistics
- Extracts all unique POIs from final dataset (with views, transactions, reviews)
- Includes POI metadata with full interaction statistics
- 100% POI name coverage from reference data
- Output: `src/utils/all_pois_final_dataset.json`
- Statistics: `data/final_dataset/poi_extraction_stats.json`

**Step 7: Validate** - Comprehensive automated testing
- Validates both filtered and all-categories datasets
- Verifies user/POI consistency with raw FSQ data
- Tests transaction structure and required fields
- Checks data integrity (non-empty datasets, all users have transactions)
- 12 automated tests covering all critical validations


## Architecture

### Intermediate Dataset (Postprocess Output)
**Core Principle:** ALL Foursquare check-ins → transactions only

- ✅ Transactions: Real check-in data with original timestamps
- ⚠️ Views: Empty placeholder (0 entries)
- ⚠️ Reviews: Empty placeholder (0 entries)

### Final Dataset (After Views/Reviews Extrapolation)
**Complete User Interaction Funnel:**

- ✅ **Views**: Generated from transactions (20-90 min before, from home/work locations)
- ✅ **Transactions**: Original check-in data at POI locations
- ✅ **Reviews**: Generated from transactions (1-7 days after, 40% rate, from home locations)

**Funnel Guarantees:**
- Every transaction has a preceding view (1:1 ratio)
- 40% of transactions have reviews
- Strict chronological order maintained
- Realistic spatial-temporal patterns

See `docs/VIEWS_REVIEWS_GENERATION_LOGIC.md` for detailed generation logic.

## Key Files

### Core Pipeline
- `src/preprocessing/add_planning_area.py` - Add geographic region information to checkins
- `src/preprocessing/fsq_to_input_json.py` - FSQ CSV → JSON conversion
- `src/generation/transaction_generator.py` - Main synthetic data generation
- `scripts/postprocess_synthetic_data.py` - Apply 5-core filtering to synthetic data
- `src/utils/poi_extraction.py` - Extract unique POIs from filtered synthetic data

### Final Dataset Generation
- `scripts/extrapolate_views_reviews.py` - Generate views and reviews from transactions
- `src/utils/poi_extraction_final.py` - Extract POIs from final dataset with statistics

### Testing & Configuration
- `tests/test_user_consistency.py` - User-transaction JSON quality check
- `config/paths.py` - Centralized path definitions

### Documentation
- `docs/VIEWS_REVIEWS_GENERATION_LOGIC.md` - Detailed explanation of views/reviews generation
- `docs/CATEGORY_GROUPING_IMPLEMENTATION.md` - Category mapping documentation
- `data/final_dataset/README.md` - Final dataset structure and usage

## Requirements

- **Python >= 3.9** ([Download](https://www.python.org/downloads/))
- pandas >= 1.5.0
- geopandas >= 0.12.0
- numpy >= 1.21.0
- scikit-learn >= 1.2.0
- tqdm >= 4.64.0

See `requirements.txt` for complete list. All dependencies are installed automatically when you install the package.

### Running Tests

The project includes comprehensive automated tests (12 tests) to verify data consistency between raw and synthetic datasets.

**Test Coverage:**
- User consistency (exist in raw data, no duplicates)
- POI consistency (valid counts and structure)
- Transaction structure (required fields, correct format)
- Data integrity (non-empty datasets, all users have transactions)

**Run tests:**
```bash
# Run all tests with verbose output
pytest tests/test_user_consistency.py -v

# Or run directly
python tests/test_user_consistency.py
```


