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
# 1. Preprocess data - this step can be skipped if the planning area is not needed in the 
# JSON dataset
python scripts/preprocess_fsq_data.py

# 2. Generate synthetic datasets
python scripts/generate_synthetic_data.py

# 3. Postprocess synthetic data (5-core filtering)
python scripts/postprocess_synthetic_data.py

# 4. Extract unique POIs from filtered data
python src/utils/poi_extraction.py

# 5. Validate outputs by running the tests
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
- Output: `data/synthetic/postprocessed/*_5core_filtered.json`

**Step 4: Extract POIs** - Creates POI reference dataset
- Extracts all unique POIs from filtered synthetic data
- Includes POI metadata (name, categories, location, planning area)
- Useful for validation, analysis, and downstream applications
- Output: `src/utils/all_pois_filtered_synthetic.json` and `src/utils/all_pois_all_categories_synthetic.json`

**Step 5: Validate** - Comprehensive automated testing
- Validates both filtered and all-categories datasets
- Verifies user/POI consistency with raw FSQ data
- Tests transaction structure and required fields
- Checks data integrity (non-empty datasets, all users have transactions)
- 12 automated tests covering all critical validations


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
- `tests/test_user_consistency.py` - user - transaction JSON quality check
- `config/paths.py` - Centralized path definitions

## Requirements

- **Python >= 3.9** ([Download](https://www.python.org/downloads/))
- pandas >= 1.5.0
- geopandas >= 0.12.0
- numpy >= 1.21.0
- scikit-learn >= 1.2.0
- tqdm >= 4.64.0

See `requirements.txt` for complete list. All dependencies are installed automatically when you install the package.

### CLI Options Reference

```bash
fsq-pipeline [OPTIONS]

Required Arguments:
  --input PATH              Path to input CSV file (raw FSQ check-ins) or preprocessed JSON
  --output PATH             Path to output JSON file (synthetic data)

Optional Arguments:
  --min-interactions INT    Minimum interactions for 5-core filtering (default: 5)
  --dataset-name TEXT       Name for this dataset (default: custom_dataset)
  --geojson-path PATH       Path to planning_area.geojson (uses package default if not provided)
  
Skip Options:
  --skip-preprocessing      Skip preprocessing step (use if input is already preprocessed JSON)
  --skip-postprocessing     Skip postprocessing step (no 5-core filtering)
  
Output Options:
  --json-output             Output progress as JSON (for programmatic parsing)
  --verbose                 Show detailed error messages and stack traces
  --version                 Show version and exit
  --help                    Show help message and exit
```

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

## Common Use Cases

### Use Case 1: Quick Generation with Defaults
```bash
fsq-pipeline --input my_checkins.csv --output synthetic.json
```

### Use Case 2: Higher Quality Filtering
```bash
# Require at least 10 interactions per user/POI
fsq-pipeline \
  --input my_checkins.csv \
  --output synthetic.json \
  --min-interactions 10
```

### Use Case 3: Already Preprocessed Data
```bash
# Skip preprocessing if you already have JSON format
fsq-pipeline \
  --input preprocessed.json \
  --output synthetic.json \
  --skip-preprocessing
```

### Use Case 4: No Filtering (Keep All Data)
```bash
# Skip 5-core filtering to keep all generated transactions
fsq-pipeline \
  --input my_checkins.csv \
  --output synthetic.json \
  --skip-postprocessing
```

### Use Case 5: Programmatic Integration (.NET Example)
```csharp
public async Task<string> GenerateSyntheticData(string inputPath)
{
    var outputPath = Path.GetTempFileName() + ".json";
    
    var process = await Process.Start(new ProcessStartInfo
    {
        FileName = "fsq-pipeline",
        Arguments = $"--input \"{inputPath}\" --output \"{outputPath}\" --json-output",
        RedirectStandardOutput = true,
        UseShellExecute = false
    });
    
    await process.WaitForExitAsync();
    
    if (process.ExitCode == 0)
    {
        return outputPath;
    }
    throw new Exception("Pipeline failed");
}
```

---
