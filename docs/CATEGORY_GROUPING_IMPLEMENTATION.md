# Category Grouping Implementation Summary

## ✅ Changes Implemented

I've successfully updated your `postprocess_synthetic_data.py` script to support **two modes** of category mapping:

### 1. **Preserve Original Mode** (New)
- Adds a new `groupCategory` field to each transaction
- Keeps the original `poiCategories` intact
- Useful for analysis that needs both detail and grouped views

### 2. **Replace Mode** (Existing, Enhanced)
- Replaces `poiCategories` with the group category
- Creates a cleaner dataset with only group categories
- Useful for production models

### 3. **Both Versions Simultaneously** (New)
- Use `--preserve-original` flag to create BOTH versions
- Gives you maximum flexibility

## 🎯 Key Features

### Command-Line Interface
```bash
# Default: Replace categories only (with backup)
python scripts/postprocess_synthetic_data.py

# Create both versions (with backup)
python scripts/postprocess_synthetic_data.py --preserve-original

# Custom mapping file
python scripts/postprocess_synthetic_data.py --mapping-file path/to/mapping.csv

# Disable category grouping
python scripts/postprocess_synthetic_data.py --no-grouping

# Skip backup (fast mode, use with caution)
python scripts/postprocess_synthetic_data.py --no-backup
```

### Function Parameters
```python
apply_category_grouping(data, category_mapping, preserve_original=False)
```
- `preserve_original=True`: Adds `groupCategory` field
- `preserve_original=False`: Replaces `poiCategories`

### Process Dataset Parameters
```python
process_dataset(
    input_file, 
    output_dir, 
    dataset_name,
    min_interactions=5,
    category_mapping_file=None,
    apply_grouping=True,
    preserve_original=False,  # NEW
    skip_backup=False         # NEW
)
```

### Main Function Parameters
```python
main(
    category_mapping_file=None,
    apply_grouping=True,
    preserve_original=False,  # NEW
    skip_backup=False         # NEW
)
```

## 📁 Output Files

### Without `--preserve-original` (Default)
```
data/synthetic_postprocess/
├── filtered_grouped.json                  # Categories replaced
├── filtered_5core_filtered.json           # Final 5-core output
├── all_categories_grouped.json
└── all_categories_5core_filtered.json
```

### With `--preserve-original`
```
data/synthetic_postprocess/
├── filtered_with_groupCategory.json       # Original + groupCategory field ✨
├── filtered_grouped_replaced.json         # Categories replaced ✨
├── filtered_5core_filtered.json           # Final 5-core output
├── all_categories_with_groupCategory.json ✨
├── all_categories_grouped_replaced.json   ✨
├── all_categories_5core_filtered.json
└── backups/                               # Timestamped backups 🔐
    ├── filtered_5core_filtered_backup_20251124_172445.json
    ├── filtered_poi_data_backup_20251124_172445.txt
    ├── backup_manifest_20251124_172445.json
    └── ...
```

## 📊 Data Structure Examples

### Version 1: With groupCategory Field
```json
{
  "user": {
    "userId": "user_123",
    "age": 25,
    "gender": "F"
  },
  "interaction": {
    "transactions": [
      {
        "poiId": "poi_456",
        "poiName": "Taj Indian Restaurant",
        "poiCategories": ["Indian Restaurant"],      ← Original preserved
        "groupCategory": "Restaurant",                ← Group added
        "timestamp": "2013-05-15T12:30:00Z",
        "userLocation": {...}
      }
    ]
  }
}
```

### Version 2: Replaced Categories
```json
{
  "user": {
    "userId": "user_123",
    "age": 25,
    "gender": "F"
  },
  "interaction": {
    "transactions": [
      {
        "poiId": "poi_456",
        "poiName": "Taj Indian Restaurant",
        "poiCategories": ["Restaurant"],              ← Replaced with group
        "timestamp": "2013-05-15T12:30:00Z",
        "userLocation": {...}
      }
    ]
  }
}
```

## 🔄 Processing Pipeline

```
1. Backup Existing Files (if enabled, default: True)
   ├─→ Smart backup: Only backs up files for current dataset
   ├─→ Creates timestamped backups in backups/ folder
   └─→ Saves backup manifest JSON
   ↓
2. Load Synthetic Data
   ↓
3. Apply Category Grouping (if enabled)
   ├─→ If preserve_original=True:
   │   ├─→ Create Version 1: with groupCategory field
   │   └─→ Create Version 2: with replaced categories
   └─→ If preserve_original=False:
       └─→ Create Version 1: with replaced categories
   ↓
4. Apply 5-Core Filtering (on replaced version)
   ↓
5. Save Outputs:
   ├─→ Grouped JSON files (before 5-core)
   ├─→ 5-core filtered JSON
   ├─→ POI metadata
   ├─→ User metadata
   └─→ Summary report
```

## 📝 Usage Examples

### Example 1: Default (Replace Only)
```bash
source grp_venv/bin/activate
python scripts/postprocess_synthetic_data.py
```

**What happens:**
- Loads synthetic data
- Replaces `poiCategories` with group categories
- Applies 5-core filtering
- Saves `filtered_grouped.json` and `filtered_5core_filtered.json`

### Example 2: Create Both Versions
```bash
source grp_venv/bin/activate
python scripts/postprocess_synthetic_data.py --preserve-original
```

**What happens:**
- Loads synthetic data
- Creates `filtered_with_groupCategory.json` (original + groupCategory)
- Creates `filtered_grouped_replaced.json` (replaced)
- Applies 5-core filtering on replaced version
- Saves final `filtered_5core_filtered.json`

### Example 3: Custom Mapping
```bash
source grp_venv/bin/activate
python scripts/postprocess_synthetic_data.py \
  --mapping-file config/category_mapping.csv \
  --preserve-original
```

### Example 4: Skip Category Grouping
```bash
source grp_venv/bin/activate
python scripts/postprocess_synthetic_data.py --no-grouping
```

## 📋 Category Mapping File

You already have a mapping file at:
- `config/category_mapping.csv`

**Format:**
```csv
original_category,group_category
Indian Restaurant,Restaurant
Chinese Restaurant,Asian Restaurant
Italian Restaurant,Restaurant
Coffee Shop,Cafe, Coffee, and Tea House
Bar,Bar
...
```

## ✅ What's Been Updated

1. **`backup_postprocessed_files()` function** 🆕
   - Smart backup system: Only backs up files that will be modified
   - Dataset-specific: Backs up only relevant files for current dataset
   - Creates timestamped backups with manifest
   - Configurable via `skip_backup` parameter

2. **`apply_category_grouping()` function**
   - Added `preserve_original` parameter
   - Supports two modes: preserve or replace

3. **`process_dataset()` function**
   - Added `preserve_original` parameter
   - Added `skip_backup` parameter
   - Creates both versions when flag is True
   - Uses `copy.deepcopy()` to avoid data corruption
   - Backs up files before processing

4. **`main()` function**
   - Added `preserve_original` parameter
   - Added `skip_backup` parameter
   - Passes flags through to all datasets

5. **Command-line interface**
   - Added `--preserve-original` flag
   - Added `--no-grouping` flag
   - Added `--mapping-file` option
   - Added `--no-backup` flag 🆕
   - Includes helpful examples in `--help`

6. **Documentation**
   - Created `docs/CATEGORY_GROUPING_USAGE.md`
   - Updated `docs/CATEGORY_GROUPING_IMPLEMENTATION.md`
   - Comprehensive usage guide with backup information
   - Examples and troubleshooting

## 🎯 Benefits

1. **Flexibility**: Choose between preserve or replace modes
2. **Both Versions**: Get both with a single flag
3. **Backward Compatible**: Default behavior unchanged
4. **Easy to Use**: Simple command-line interface
5. **Well Documented**: Clear examples and usage guide
6. **Safe by Default**: Automatic backups protect your data 🔐
7. **Smart Backups**: Only backs up files that will be modified
8. **Fast Mode Available**: Use `--no-backup` for rapid iteration

## 🚀 Next Steps

1. **Run the script** with your desired mode:
   ```bash
   source grp_venv/bin/activate
   python scripts/postprocess_synthetic_data.py --preserve-original
   ```

2. **Check the outputs** in `data/synthetic_postprocess/`

3. **Verify category mapping** - Check console output for unmapped categories

4. **Use the preserved version** for analysis that needs both detail and grouping

5. **Use the replaced version** for production models

## 📚 Documentation

- **Usage Guide**: `docs/CATEGORY_GROUPING_USAGE.md`
- **Category Mapping**: `config/README_CATEGORY_MAPPING.md`
- **This Summary**: `docs/CATEGORY_GROUPING_IMPLEMENTATION.md`

## 🔧 Technical Details

- Uses `copy.deepcopy()` to create independent copies
- Processes all transactions in place
- Maintains statistics for both versions
- Preserves original data structure
- Compatible with 5-core filtering
- Works with both CSV and JSON mapping files
- Smart backup system:
  - Only backs up files that will be modified
  - Creates timestamped backups with YYYYMMDD_HHMMSS format
  - Saves backup manifest in JSON format
  - Uses `shutil.copy2()` to preserve metadata
  - Path-safe: Converts Path objects to strings for JSON serialization

## ❓ Questions?

If you have questions or need modifications:
1. Check the console output for detailed logs
2. Review the usage guide in `docs/CATEGORY_GROUPING_USAGE.md`
3. Look at the `--help` output for quick reference
4. Examine the generated reports in `data/synthetic_postprocess/`
