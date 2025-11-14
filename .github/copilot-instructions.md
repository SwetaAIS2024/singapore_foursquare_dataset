# Singapore Foursquare Synthetic Dataset Generation - AI Coding Guide

## Critical Architecture Decision

**This project uses TRANSACTION-ONLY conversion**: All Foursquare checkins become transactions. Views and reviews are empty placeholders, NOT generated data. There is NO funnel logic, NO Markov model, NO 60:30:10 ratios.

```python
# Core function in direct_conversion_fsq_to_json.py (lines 136-151)
def assign_transactions_only(visits):
    """ALL checkins → transactions. Views/reviews = empty."""
    transaction_indices = set(range(len(visits)))  # 100% transactions
    return {
        "transactions": transaction_indices,
        "views": set(),      # Always empty
        "reviews": set()     # Always empty
    }
```

## Essential File Locations

```
json_gen/                                   # Current active pipeline
├── func/
│   └── direct_conversion_fsq_to_json.py    # MAIN: Transaction-only generation
├── input_fsq_json/
│   ├── fsq_to_input_json.py                # Preprocesses FSQ → input JSON
│   ├── input_filtered.json                 # Filtered categories input
│   └── input_all_categories.json           # All categories input
├── output_syn_json/
│   ├── fsq_to_synthetic_filtered.json      # Output: filtered transactions
│   ├── fsq_to_synthetic_all_categories.json # Output: all transactions
│   └── preprocessing/                      # Post-processing utilities
│       └── process_synthetic_data.py       # Data cleaning & filtering
├── validation_graph/
│   └── validation.py                       # Validates transaction quality
└── utils/
    ├── comparison_syn_input/comp.py        # Compares input vs output stats
    ├── fsq_add_planning_area/              # Geo-enrichment utilities
    └── extract_all_poi_id_interactions/    # POI extraction tools

c0_Configuration/
├── config_params.json                      # Pipeline parameters (eps, clusters, etc.)
└── config_paths.py                         # Central path definitions

Input_data/input_data/                      # Raw Foursquare datasets
├── FSQ_SG_2013_Checkins.csv               # Original check-in data
└── FSQ_SG_2013_POI.csv                    # POI metadata (name, category, coords)
```

### Planned: Databricks Integration (Future)

The following structure is planned for cloud-scale processing on Databricks Community Edition:

```
databricks/                                 # TODO: Add Databricks integration
├── notebooks/
│   ├── 01_setup_and_upload.py             # Environment setup
│   ├── 02_data_processing.py              # Spark-based processing
│   ├── 03_clustering.py                   # MLlib clustering
│   ├── 04_synthetic_generation.py         # Distributed generation
│   └── 05_validation.py                   # Quality validation
├── utils/
│   ├── databricks_config.py               # Cloud configuration
│   └── spark_helpers.py                   # Spark optimization utilities
└── workflows/
    └── pipeline_workflow.json             # Workflow orchestration
```

**Note**: When adding Databricks integration, use DBFS paths (`/FileStore/shared_uploads/`) and implement Spark DataFrame processing for large datasets.

## Data Flow & Key Patterns

### 1. Timestamp Handling - EXACT Preservation

**CRITICAL**: Use exact FSQ timestamps. NO offsets, NO jitter, NO modifications.

```python
# ✅ CORRECT - Use exact timestamp from input
timestamp = entry['_parsed_timestamp'].strftime("%Y-%m-%dT%H:%M:%SZ")
lat, lon = float(entry["lat"]), float(entry["lon"])

# ❌ WRONG - Never add random offsets
timestamp = original_time + timedelta(minutes=random.randint(-30, 30))  # NEVER DO THIS
```

Input format: `"2012-04-03T18:19:55Z"` → Output: Same ISO 8601 string

### 2. Input JSON Schema (from `fsq_to_input_json.py`)

```json
{
  "user_id": "21418",
  "user_metadata": [
    {
      "poi_id": "POI-ID-114d7492",
      "poi_category": "Chinese Restaurant",
      "planning_area": "HOUGANG",
      "timestamp": "2012-04-03T18:19:55Z",  // ISO 8601 format
      "lat": "1.359951",
      "lon": "103.884701"
    }
  ]
}
```

### 3. Output JSON Schema (Transactions Only)

```json
{
  "user": {
    "userId": "21418",
    "age": 35,
    "gender": "male",
    "location": {"city": "Singapore", "country": "SG"}
  },
  "interaction": {
    "views": [],           // ALWAYS EMPTY
    "transactions": [      // ALL FSQ checkins here
      {
        "timestamp": "2012-04-03T18:19:55Z",  // Exact from input
        "poiId": "POI-ID-114d7492",
        "poiCategories": ["Chinese Restaurant"],
        "transactionId": "738291",  // Random generated
        "amount": 45.50,            // Random based on category
        "currency": "SGD",
        "paymentMethod": "Credit Card",
        "userLocation": {"latitude": 1.359951, "longitude": 103.884701}
      }
    ],
    "reviews": []          // ALWAYS EMPTY
  }
}
```

## Development Workflow

### Run Complete Pipeline

```powershell
# 1. Activate environment
cd c:\Users\admin\Desktop\sweta\MPS_syn_data_gen\singapore_foursquare_dataset
.\code_venv\Scripts\Activate.ps1  # Or: python -m venv code_venv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Preprocess FSQ data to input JSON (if needed)
cd json_gen\input_fsq_json
python fsq_to_input_json.py

# 4. Generate synthetic transactions
cd ..\func
python direct_conversion_fsq_to_json.py

# 5. Validate output
cd ..\validation_graph
python validation.py

# 6. Compare input vs output statistics
cd ..\utils\comparison_syn_input
python comp.py
```

### Quick Validation Commands

```powershell
# Check output files exist
Get-ChildItem json_gen\output_syn_json\*.json

# Count users in output
python -c "import json; data=json.load(open('json_gen/output_syn_json/fsq_to_synthetic_filtered.json')); print(f'Users: {len(data)}')"

# View validation summary
Get-Content json_gen\validation_graph\user_poi_time_graphs\summary_statistics.txt
```

## Validation Rules (from `validation.py`)

### Transaction Quality Checks

1. **Timestamp Uniqueness**: No duplicate timestamps per user
   ```python
   all_timestamps = [txn["timestamp"] for txn in profile["interaction"]["transactions"]]
   assert len(all_timestamps) == len(set(all_timestamps))
   ```

2. **POI Overlap**: Multiple users should visit popular POIs (realistic)
   ```python
   # Target: 70-80% of POIs visited by 2+ users
   overlap_rate = pois_visited_by_multiple / total_unique_pois
   ```

3. **Temporal Distribution**: Events spread across realistic time span
   ```python
   # Check: First event to last event span ~6-8 months for 2012 FSQ data
   span_days = (last_event - first_event).days
   ```

4. **Category Distribution**: Match input FSQ data proportions
   ```python
   # Top categories should align with input distribution
   category_counts = Counter([txn["poiCategories"][0] for txn in all_transactions])
   ```

## Common Pitfalls & Solutions

### ❌ DO NOT

```python
# 1. Never generate views or reviews
assigns = {"views": some_indices, ...}  # WRONG - views must be empty

# 2. Never modify timestamps
time_offset = timedelta(hours=random.randint(-2, 2))  # WRONG - no offsets

# 3. Never add funnel logic
if is_popular_poi:
    assign_as_view_then_transaction()  # WRONG - no funnel assignment

# 4. Never jitter coordinates
lat = original_lat + random.uniform(-0.001, 0.001)  # WRONG - exact coords only
```

### ✅ DO

```python
# 1. Convert ALL checkins to transactions
transaction_indices = set(range(len(visits)))  # 100% conversion

# 2. Use exact timestamps
timestamp = entry['_parsed_timestamp'].strftime("%Y-%m-%dT%H:%M:%SZ")

# 3. Validate empty placeholders
assert len(profile["interaction"]["views"]) == 0
assert len(profile["interaction"]["reviews"]) == 0

# 4. Use exact coordinates
lat, lon = float(entry["lat"]), float(entry["lon"])
```

## Configuration Management

**Central config** in `c0_Configuration/config_params.json`:
```json
{
  "eps_km": 0.5,              // DBSCAN spatial clustering
  "min_samples": 10,
  "n_clusters": 20,           // K-means clusters
  "n_users": 1000             // Sampling target
}
```

**Path imports** from `config_paths.py`:
```python
from c0_Configuration.config_paths import JSON_INPUT, JSON_OUTPUT
# Always use these paths - never hardcode
```

## Error Handling Pattern

```python
def safe_string(val, default="Unknown"):
    """Standard NaN/None handler used across codebase"""
    import math
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    if isinstance(val, str) and val.lower() in ['nan', 'null', 'none', '']:
        return default
    return str(val).strip() or default
```

Use `safe_string()` for all POI names/categories to handle missing data.

## Debugging Tips

```python
# 1. Print processing summary in direct_conversion_fsq_to_json.py
print(f"User {user_id}: {len(visits)} checkins → {len(assigns['transactions'])} transactions")

# 2. Check for timestamp duplicates
from collections import Counter
timestamp_counts = Counter([txn["timestamp"] for txn in transactions])
duplicates = {ts: count for ts, count in timestamp_counts.items() if count > 1}

# 3. Verify empty arrays
assert all(len(p["interaction"]["views"]) == 0 for p in app_profiles)
assert all(len(p["interaction"]["reviews"]) == 0 for p in app_profiles)
```

## Project-Specific Terms

- **FSQ**: Foursquare (original check-in dataset from 2013)
- **Transaction-only**: Core architecture - no views/reviews generated
- **Planning Area**: Singapore administrative regions (from `planning_area.geojson`)
- **Direct translation**: 1:1 FSQ checkin → synthetic transaction conversion
- **Filtered vs All Categories**: Two output versions - one with relevant categories only, one with all POI types

## When Adding Features

1. **Maintain transaction-only architecture**: Don't add view/review generation
2. **Preserve exact timestamps**: No temporal models or offsets
3. **Update validation**: Add checks to `validation.py` for new fields
4. **Follow path conventions**: Use `config_paths.py` for all file paths
5. **Use safe_string()**: Handle NaN/None values consistently
6. **Plan for Databricks**: When adding cloud integration, use Spark DataFrames and DBFS paths

## Future Development Roadmap

### Planned Features
- **Databricks Integration**: Cloud-scale processing with Apache Spark and MLflow
- **API Backend**: FastAPI endpoints for on-demand dataset generation
- **DVC Pipeline**: Data version control for reproducible experiments
- **CI/CD**: GitHub Actions for automated testing and validation

When implementing these, reference existing patterns in `direct_conversion_fsq_to_json.py` for core logic and `validation.py` for quality checks.

---

**Last Updated**: 2025-11-14  
**Current Branch**: `databricks_integration_datapipeline`  
**Key Files**: `direct_conversion_fsq_to_json.py`, `validation.py`, `config_paths.py`
