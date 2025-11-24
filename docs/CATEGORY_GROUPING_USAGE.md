# Category Grouping Usage Guide

## Overview

The postprocessing script now supports **two modes** for category mapping:

1. **Preserve Original** - Adds a `groupCategory` field while keeping the original `poiCategories`
2. **Replace** - Replaces `poiCategories` with the group category

You can create **both versions** simultaneously with the `--preserve-original` flag.

## Quick Start

### Option 1: Replace Categories Only (Default)
```bash
python scripts/postprocess_synthetic_data.py
```

**Output:**
- `filtered_grouped.json` - Categories replaced with group categories
- `filtered_5core_filtered.json` - Final 5-core filtered data

### Option 2: Create Both Versions
```bash
python scripts/postprocess_synthetic_data.py --preserve-original
```

**Output:**
- `filtered_with_groupCategory.json` - Original + groupCategory field added
- `filtered_grouped_replaced.json` - Categories replaced
- `filtered_5core_filtered.json` - Final 5-core filtered data (with replaced categories)

### Option 3: Custom Mapping File
```bash
python scripts/postprocess_synthetic_data.py --mapping-file config/category_mapping.csv --preserve-original
```

### Option 4: Disable Category Grouping
```bash
python scripts/postprocess_synthetic_data.py --no-grouping
```

### Option 5: Skip Backup (Fast Mode)
```bash
python scripts/postprocess_synthetic_data.py --preserve-original --no-backup
```

**Warning:** This skips backing up existing files. Use only when you're confident or have manual backups.

## Output File Structure

### Version 1: With groupCategory Field (Preserved)
```json
{
  "user": {...},
  "interaction": {
    "transactions": [
      {
        "poiId": "poi_123",
        "poiName": "Example Restaurant",
        "poiCategories": ["Indian Restaurant"],
        "groupCategory": "Restaurant",
        "timestamp": "2013-05-15T12:30:00Z",
        ...
      }
    ]
  }
}
```

### Version 2: Replaced Categories
```json
{
  "user": {...},
  "interaction": {
    "transactions": [
      {
        "poiId": "poi_123",
        "poiName": "Example Restaurant",
        "poiCategories": ["Restaurant"],
        "timestamp": "2013-05-15T12:30:00Z",
        ...
      }
    ]
  }
}
```

## Category Mapping File

Ensure you have a category mapping file at:
- `config/category_mapping.json` (preferred), or
- `config/category_mapping.csv`

### CSV Format
```csv
original_category,group_category
Indian Restaurant,Restaurant
Chinese Restaurant,Restaurant
Italian Restaurant,Restaurant
Coffee Shop,Cafe
Bar,Bar
```

### JSON Format
```json
{
  "Indian Restaurant": "Restaurant",
  "Chinese Restaurant": "Restaurant",
  "Italian Restaurant": "Restaurant",
  "Coffee Shop": "Cafe",
  "Bar": "Bar"
}
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--preserve-original` | Create both versions (preserved + replaced) | `False` |
| `--no-grouping` | Disable category grouping entirely | `False` (grouping enabled) |
| `--mapping-file PATH` | Use custom mapping file | `config/category_mapping.{json,csv}` |
| `--no-backup` | Skip backing up existing files (use with caution) | `False` (backup enabled) |

## Examples

### Example 1: Default Processing
```bash
python scripts/postprocess_synthetic_data.py
```
- Applies category grouping (replace mode)
- Uses default mapping file
- Applies 5-core filtering

### Example 2: Create Both Versions
```bash
python scripts/postprocess_synthetic_data.py --preserve-original
```
- Creates TWO versions before 5-core filtering:
  1. With `groupCategory` field (original preserved)
  2. With replaced `poiCategories`
- Final 5-core output uses replaced categories

### Example 3: Custom Mapping
```bash
python scripts/postprocess_synthetic_data.py \
  --mapping-file /path/to/my_mapping.csv \
  --preserve-original
```

### Example 4: Skip Category Grouping
```bash
python scripts/postprocess_synthetic_data.py --no-grouping
```
- Only performs 5-core filtering
- No category transformation

## Programmatic Usage

You can also use the functions programmatically:

```python
from scripts.postprocess_synthetic_data import main

# Default: replace only
main()

# Create both versions
main(preserve_original=True)

# Custom mapping file
main(
    category_mapping_file='path/to/mapping.csv',
    preserve_original=True
)

# Disable grouping
main(apply_grouping=False)
```

## Processing Pipeline

The complete pipeline is:

1. **Backup existing files** (optional, enabled by default)
   - Smart backup: Only backs up files that will be modified for the current dataset
   - Creates timestamped backups in `data/synthetic_postprocess/backups/`
   - Saves a manifest JSON file with backup details
2. **Load synthetic data** from JSON files
3. **Apply category grouping** (optional)
   - If `preserve_original=True`: Create 2 versions
   - If `preserve_original=False`: Create 1 version (replace)
4. **Apply 5-core filtering** on the replaced version
5. **Save outputs**:
   - Grouped data (before 5-core)
   - 5-core filtered JSON
   - POI metadata
   - User metadata
   - Summary report

## Output Files

When running with `--preserve-original`, you'll get:

```
data/synthetic_postprocess/
├── filtered_with_groupCategory.json       # Original + groupCategory field
├── filtered_grouped_replaced.json         # Replaced categories
├── filtered_5core_filtered.json           # Final 5-core filtered (replaced)
├── filtered_poi_data.txt
├── filtered_user_data.txt
├── filtered_5core_report.txt
├── all_categories_with_groupCategory.json
├── all_categories_grouped_replaced.json
├── all_categories_5core_filtered.json
├── all_categories_poi_data.txt
├── all_categories_user_data.txt
├── all_categories_5core_report.txt
└── backups/                               # Timestamped backups (if enabled)
    ├── filtered_5core_filtered_backup_20251124_172445.json
    ├── filtered_poi_data_backup_20251124_172445.txt
    ├── backup_manifest_20251124_172445.json
    └── ...
```

## Backup System

The script automatically backs up files before making changes. This ensures you can always recover previous results.

### Smart Backup Behavior

- **Dataset-specific**: Only backs up files for the dataset being processed
- **Minimal backups**: Only backs up files that will actually be modified
- **Timestamped**: Each backup has a timestamp for easy identification
- **Manifest file**: A JSON manifest tracks all backups

### Backup Examples

**Processing "filtered" dataset (default mode):**
```
Backs up only:
✅ filtered_5core_filtered.json
✅ filtered_poi_data.txt
✅ filtered_user_data.txt
✅ filtered_5core_report.txt
✅ filtered_grouped.json (if exists)
```

**Processing "filtered" dataset with --preserve-original:**
```
Backs up only:
✅ filtered_5core_filtered.json
✅ filtered_poi_data.txt
✅ filtered_user_data.txt
✅ filtered_5core_report.txt
✅ filtered_with_groupCategory.json (if exists)
✅ filtered_grouped_replaced.json (if exists)
```

**Skipping backup (fast mode):**
```bash
python scripts/postprocess_synthetic_data.py --no-backup
```

### Backup Manifest

Each backup session creates a manifest file:

```json
{
  "backup_dir": "/path/to/backups",
  "timestamp": "20251124_172445",
  "files_backed_up": [
    {
      "original": "/path/to/filtered_5core_filtered.json",
      "backup": "/path/to/backups/filtered_5core_filtered_backup_20251124_172445.json",
      "size": 12345678
    }
  ],
  "files_not_found": []
}
```

## Validation

After running, check the console output for:

1. **Mapping Statistics:**
   - Categories mapped
   - Categories unmapped
   - Group category distribution

2. **5-core Statistics:**
   - Users retained
   - POIs retained
   - Interactions retained

3. **Unmapped Categories Warning:**
   - Lists any categories not found in mapping file
   - Add these to your mapping file for complete coverage

## Troubleshooting

### Problem: "Category mapping file not found"
**Solution:** Create a mapping file at `config/category_mapping.csv` or specify with `--mapping-file`

### Problem: Many unmapped categories
**Solution:** Check that category names match exactly (case-sensitive). Extract actual categories first:

```python
import json

with open('data/synthetic/fsq_to_synthetic_all_categories.json', 'r') as f:
    data = json.load(f)

categories = set()
for user in data:
    for txn in user['interaction']['transactions']:
        if txn['poiCategories']:
            categories.add(txn['poiCategories'][0])

print(f"Found {len(categories)} unique categories")
for cat in sorted(categories):
    print(f"  - {cat}")
```

### Problem: Want to use groupCategory in analysis
**Solution:** Use the `filtered_with_groupCategory.json` file, which has both original and group categories.

## Best Practices

1. **Always check unmapped categories** - Add them to your mapping file
2. **Use preserve_original** during development to validate mappings
3. **Use replace mode** for final production datasets
4. **Validate category distribution** in the console output
5. **Keep mapping file in version control** for reproducibility
6. **Keep backups enabled** unless you're certain (default behavior)
7. **Review backup manifest** to verify what was backed up
8. **Use --no-backup** only when iterating rapidly during development

## Support

For issues or questions:
1. Check console output for detailed error messages
2. Verify mapping file format and location
3. Ensure category names match exactly (case-sensitive)
4. Review the generated `*_5core_report.txt` files
