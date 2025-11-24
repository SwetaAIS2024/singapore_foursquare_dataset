# Backup System Documentation

## Overview

The postprocessing script now includes an **intelligent backup system** that automatically protects your data before making any changes. The system is smart, dataset-specific, and creates timestamped backups with detailed manifests.

## Key Features

### 🎯 Smart Backup
- **Dataset-specific**: Only backs up files for the dataset being processed
- **Minimal overhead**: Only backs up files that will actually be modified
- **Timestamped**: Each backup has a unique timestamp (YYYYMMDD_HHMMSS)
- **Manifest tracking**: JSON manifest records all backup operations

### 🔐 Safety First
- **Enabled by default**: Automatic protection unless explicitly disabled
- **Non-destructive**: Original files are preserved in `backups/` folder
- **Recoverable**: Easy to restore from timestamped backups

### ⚡ Flexible
- **Skip when needed**: Use `--no-backup` for rapid iteration
- **Configurable**: Control via command-line flag or programmatic API

## How It Works

### Backup Location
```
data/synthetic_postprocess/
└── backups/
    ├── filtered_5core_filtered_backup_20251124_172445.json
    ├── filtered_poi_data_backup_20251124_172445.txt
    ├── filtered_user_data_backup_20251124_172445.txt
    ├── filtered_5core_report_backup_20251124_172445.txt
    ├── backup_manifest_20251124_172445.json
    └── ...
```

### Backup Manifest
Each backup session creates a manifest file:

```json
{
  "backup_dir": "/path/to/synthetic_postprocess/backups",
  "timestamp": "20251124_172445",
  "files_backed_up": [
    {
      "original": "/path/to/filtered_5core_filtered.json",
      "backup": "/path/to/backups/filtered_5core_filtered_backup_20251124_172445.json",
      "size": 12345678
    },
    {
      "original": "/path/to/filtered_poi_data.txt",
      "backup": "/path/to/backups/filtered_poi_data_backup_20251124_172445.txt",
      "size": 123456
    }
  ],
  "files_not_found": []
}
```

## Usage

### Command-Line

#### Default: With Backup (Recommended)
```bash
python scripts/postprocess_synthetic_data.py --preserve-original
```

**Console Output:**
```
================================================================================
BACKING UP EXISTING POSTPROCESSED FILES
================================================================================

✅ Backed up: filtered_5core_filtered.json
   → filtered_5core_filtered_backup_20251124_172445.json
✅ Backed up: filtered_poi_data.txt
   → filtered_poi_data_backup_20251124_172445.txt
✅ Backed up: filtered_user_data.txt
   → filtered_user_data_backup_20251124_172445.txt
✅ Backed up: filtered_5core_report.txt
   → filtered_5core_report_backup_20251124_172445.txt

📋 Backup Summary:
   Files backed up: 4
   Backup location: /path/to/backups
   Manifest: backup_manifest_20251124_172445.json

✅ Backup complete!
```

#### Skip Backup (Fast Mode)
```bash
python scripts/postprocess_synthetic_data.py --preserve-original --no-backup
```

**Console Output:**
```
⚠️  Backup skipped (--no-backup flag enabled)
```

### Programmatic API

```python
from scripts.postprocess_synthetic_data import main

# With backup (default)
main(preserve_original=True)

# Without backup
main(preserve_original=True, skip_backup=True)
```

## What Gets Backed Up?

The backup system is **smart** and only backs up files that will be modified:

### Processing "filtered" Dataset (Default Mode)
```
✅ filtered_5core_filtered.json    (will be regenerated)
✅ filtered_poi_data.txt            (will be regenerated)
✅ filtered_user_data.txt           (will be regenerated)
✅ filtered_5core_report.txt        (will be regenerated)
✅ filtered_grouped.json            (if exists, will be overwritten)
```

### Processing "filtered" with --preserve-original
```
✅ filtered_5core_filtered.json         (will be regenerated)
✅ filtered_poi_data.txt                (will be regenerated)
✅ filtered_user_data.txt               (will be regenerated)
✅ filtered_5core_report.txt            (will be regenerated)
✅ filtered_with_groupCategory.json     (if exists, will be overwritten)
✅ filtered_grouped_replaced.json       (if exists, will be overwritten)
```

### Processing "all_categories" Dataset
```
✅ all_categories_5core_filtered.json   (will be regenerated)
✅ all_categories_poi_data.txt          (will be regenerated)
✅ all_categories_user_data.txt         (will be regenerated)
✅ all_categories_5core_report.txt      (will be regenerated)
✅ all_categories_grouped.json          (if exists, will be overwritten)
```

**Note:** Files that don't exist are skipped (not an error).

## When to Use --no-backup

### ✅ Use --no-backup when:
- Rapid iteration during development
- You have manual backups
- Disk space is limited
- You're testing on temporary data
- Processing speed is critical

### ❌ Don't use --no-backup when:
- Processing production data
- You don't have other backups
- Results are important
- First time running with new parameters
- Unsure about the outcome

## Restoring from Backup

### Manual Restoration

1. **Find the backup files:**
   ```bash
   cd data/synthetic_postprocess/backups
   ls -lt  # List by most recent
   ```

2. **Identify the timestamp you want to restore:**
   ```bash
   cat backup_manifest_20251124_172445.json
   ```

3. **Copy files back:**
   ```bash
   cp filtered_5core_filtered_backup_20251124_172445.json \
      ../filtered_5core_filtered.json
   
   cp filtered_poi_data_backup_20251124_172445.txt \
      ../filtered_poi_data.txt
   ```

### Script-based Restoration

You can create a restoration script:

```python
import json
import shutil
from pathlib import Path

def restore_from_manifest(manifest_file):
    """Restore files from a backup manifest."""
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    for backup_info in manifest['files_backed_up']:
        backup_path = backup_info['backup']
        original_path = backup_info['original']
        
        print(f"Restoring {Path(original_path).name}...")
        shutil.copy2(backup_path, original_path)
    
    print(f"✅ Restored {len(manifest['files_backed_up'])} files")

# Usage
restore_from_manifest(
    'data/synthetic_postprocess/backups/backup_manifest_20251124_172445.json'
)
```

## Managing Backups

### Disk Space Considerations

Backups can accumulate over time. To manage disk space:

#### Check backup folder size:
```bash
du -sh data/synthetic_postprocess/backups
```

#### List backups by size:
```bash
ls -lhS data/synthetic_postprocess/backups/*.json
```

#### Remove old backups (older than 7 days):
```bash
find data/synthetic_postprocess/backups -name "*backup_*" -mtime +7 -delete
```

#### Keep only last 5 backups:
```bash
cd data/synthetic_postprocess/backups
ls -t backup_manifest_*.json | tail -n +6 | xargs rm -f
# Then remove corresponding backup files based on timestamps
```

### Automated Cleanup Script

Create a cleanup script:

```python
#!/usr/bin/env python
# cleanup_old_backups.py

from pathlib import Path
from datetime import datetime, timedelta

def cleanup_old_backups(backup_dir, days_to_keep=7):
    """Remove backups older than specified days."""
    backup_dir = Path(backup_dir)
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    
    removed_count = 0
    for backup_file in backup_dir.glob("*backup_*"):
        # Extract timestamp from filename
        parts = backup_file.stem.split('_backup_')
        if len(parts) == 2:
            timestamp_str = parts[1]
            try:
                file_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                if file_date < cutoff_date:
                    backup_file.unlink()
                    removed_count += 1
                    print(f"Removed: {backup_file.name}")
            except ValueError:
                continue
    
    print(f"✅ Removed {removed_count} old backup files")

# Usage
cleanup_old_backups('data/synthetic_postprocess/backups', days_to_keep=7)
```

## Best Practices

1. **Keep backups enabled by default** - Safety first!
2. **Review backup manifests** - Verify what was backed up
3. **Periodic cleanup** - Remove old backups to save disk space
4. **Test restoration** - Verify you can restore from backups
5. **Use --no-backup sparingly** - Only when you're confident
6. **Check console output** - Verify backup succeeded before proceeding
7. **Keep manifests** - They help track what was backed up when

## Troubleshooting

### Problem: "Backup failed - permission denied"
**Solution:** Check write permissions on the output directory:
```bash
chmod -R u+w data/synthetic_postprocess/
```

### Problem: "Disk space full"
**Solution:** Clean up old backups:
```bash
# Remove backups older than 7 days
find data/synthetic_postprocess/backups -mtime +7 -delete

# Or use --no-backup flag
python scripts/postprocess_synthetic_data.py --no-backup
```

### Problem: "TypeError: Object of type PosixPath is not JSON serializable"
**Solution:** This has been fixed in the latest version. Update your script.

### Problem: "Too many backup files"
**Solution:** Implement automated cleanup (see Managing Backups section above).

## Technical Implementation

### Functions

#### `create_backup(file_path, backup_dir=None)`
Creates a single timestamped backup file.

```python
backup_path = create_backup(
    file_path='data/synthetic_postprocess/filtered_5core_filtered.json',
    backup_dir='data/synthetic_postprocess/backups'
)
```

#### `backup_postprocessed_files(output_dir, dataset_name=None, preserve_original=False)`
Backs up all relevant files for a dataset.

```python
backup_info = backup_postprocessed_files(
    output_dir=Path('data/synthetic_postprocess'),
    dataset_name='filtered',
    preserve_original=True
)
```

### Internal Logic

1. **Determine files to backup** based on:
   - Dataset name (filtered, all_categories)
   - Processing mode (preserve_original flag)
   - Existing files in output directory

2. **Create backup directory** if it doesn't exist

3. **For each file:**
   - Check if file exists
   - Create timestamped copy using `shutil.copy2()`
   - Record backup information

4. **Save manifest** with all backup details

5. **Print summary** to console

## See Also

- [Category Grouping Usage Guide](CATEGORY_GROUPING_USAGE.md)
- [Implementation Summary](CATEGORY_GROUPING_IMPLEMENTATION.md)
- [Main README](../README.md)

## Support

For questions or issues with the backup system:
1. Check console output for detailed error messages
2. Verify backup directory permissions
3. Check available disk space
4. Review backup manifest files
5. Test with `--no-backup` to isolate backup-related issues
