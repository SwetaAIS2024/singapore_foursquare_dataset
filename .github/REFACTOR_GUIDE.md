# Project Refactoring Guide - Light Refactor

**Status:** 📋 Planning  
**Estimated Time:** 2-3 hours  
**Risk Level:** Low (no code logic changes)  
**Date:** 2025-11-14

---

## 🎯 Objective

Reorganize the project structure to follow Python best practices while maintaining all existing functionality.

**Before:**
```
singapore_foursquare_dataset/
├── json_gen/                    # Generic name
├── c0_Configuration/            # Non-standard naming
└── Input_data/                  # Mixed case
```

**After:**
```
singapore_foursquare_dataset/
├── src/                         # Standard Python source
├── config/                      # Standard config
├── data/                        # Organized data hierarchy
└── scripts/                     # Entry points
```

---

## 📋 Pre-Migration Checklist

- [ ] Commit all current changes: `git add . && git commit -m "Pre-refactor checkpoint"`
- [ ] Create refactor branch: `git checkout -b refactor/light-restructure`
- [ ] Backup data files (optional but recommended)
- [ ] Note any custom paths in your environment

---

## 🔄 Step-by-Step Migration

### **Phase 1: Create New Directory Structure** (10 min)

Run these commands from the repository root:

```powershell
# Create main directories
New-Item -ItemType Directory -Force -Path "src"
New-Item -ItemType Directory -Force -Path "config"
New-Item -ItemType Directory -Force -Path "data"
New-Item -ItemType Directory -Force -Path "scripts"
New-Item -ItemType Directory -Force -Path "logs"
New-Item -ItemType Directory -Force -Path "tests"

# Create src subdirectories
New-Item -ItemType Directory -Force -Path "src\preprocessing"
New-Item -ItemType Directory -Force -Path "src\generation"
New-Item -ItemType Directory -Force -Path "src\validation"
New-Item -ItemType Directory -Force -Path "src\utils"

# Create data subdirectories
New-Item -ItemType Directory -Force -Path "data\raw"
New-Item -ItemType Directory -Force -Path "data\interim"
New-Item -ItemType Directory -Force -Path "data\processed"
New-Item -ItemType Directory -Force -Path "data\synthetic"

# Create Python package markers
New-Item -ItemType File -Force -Path "src\__init__.py"
New-Item -ItemType File -Force -Path "src\preprocessing\__init__.py"
New-Item -ItemType File -Force -Path "src\generation\__init__.py"
New-Item -ItemType File -Force -Path "src\validation\__init__.py"
New-Item -ItemType File -Force -Path "src\utils\__init__.py"
New-Item -ItemType File -Force -Path "tests\__init__.py"
```

---

### **Phase 2: Move Configuration Files** (5 min)

```powershell
# Move configuration
Move-Item -Path "c0_Configuration\config_params.json" -Destination "config\parameters.json"
Move-Item -Path "c0_Configuration\config_paths.py" -Destination "config\paths.py"

# Remove old directory (after confirming move)
Remove-Item -Path "c0_Configuration" -Recurse
```

**Update `config/paths.py`:**
```python
# Old path references - FIND AND REPLACE:
# BASE_DIR = Path(__file__).parent.parent
# INPUT_DATA = BASE_DIR / "c1_Data_Collection_and_Processing" / "input_data"

# New path references:
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_INTERIM = BASE_DIR / "data" / "interim"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_SYNTHETIC = BASE_DIR / "data" / "synthetic"

# Input files
FSQ_CHECKINS_CSV = DATA_RAW / "FSQ_SG_2013_Checkins.csv"
FSQ_POI_CSV = DATA_RAW / "FSQ_SG_2013_POI.csv"
RELEVANT_CATEGORIES_XLSX = DATA_RAW / "Relevant_POI_category.xlsx"

# Processed files
INPUT_FILTERED_JSON = DATA_PROCESSED / "input_filtered.json"
INPUT_ALL_CATEGORIES_JSON = DATA_PROCESSED / "input_all_categories.json"

# Output files
SYNTHETIC_FILTERED_JSON = DATA_SYNTHETIC / "fsq_to_synthetic_filtered.json"
SYNTHETIC_ALL_CATEGORIES_JSON = DATA_SYNTHETIC / "fsq_to_synthetic_all_categories.json"

# GeoJSON files
PLANNING_AREA_GEOJSON = BASE_DIR / "planning_area.geojson"
SUBZONE_GEOJSON = BASE_DIR / "subzone.geojson"
```

---

### **Phase 3: Move Data Files** (10 min)

```powershell
# Move raw data
Move-Item -Path "Input_data\input_data\FSQ_SG_2013_Checkins.csv" -Destination "data\raw\"
Move-Item -Path "Input_data\input_data\FSQ_SG_2013_POI.csv" -Destination "data\raw\"

# If you have the category file, move it too
# Move-Item -Path "Input_data\Relevant_POI_category.xlsx" -Destination "data\raw\" -ErrorAction SilentlyContinue

# Move processed inputs
Move-Item -Path "json_gen\input_fsq_json\input_filtered.json" -Destination "data\processed\" -ErrorAction SilentlyContinue
Move-Item -Path "json_gen\input_fsq_json\input_all_categories.json" -Destination "data\processed\" -ErrorAction SilentlyContinue

# Move synthetic outputs
Move-Item -Path "json_gen\output_syn_json\fsq_to_synthetic_filtered.json" -Destination "data\synthetic\" -ErrorAction SilentlyContinue
Move-Item -Path "json_gen\output_syn_json\fsq_to_synthetic_all_categories.json" -Destination "data\synthetic\" -ErrorAction SilentlyContinue

# Remove old directories (after confirming moves)
Remove-Item -Path "Input_data" -Recurse -ErrorAction SilentlyContinue
```

---

### **Phase 4: Move Source Code** (15 min)

#### **4.1 Preprocessing**
```powershell
# Move preprocessing scripts
Move-Item -Path "json_gen\input_fsq_json\fsq_to_input_json.py" -Destination "src\preprocessing\"
Move-Item -Path "json_gen\utils\fsq_add_planning_area\append_planning_area_2_fsq.py" -Destination "src\preprocessing\add_planning_area.py"
```

#### **4.2 Generation**
```powershell
# Move generation script
Move-Item -Path "json_gen\func\direct_conversion_fsq_to_json.py" -Destination "src\generation\transaction_generator.py"
```

#### **4.3 Validation**
```powershell
# Move validation scripts
Move-Item -Path "json_gen\validation_graph\validation.py" -Destination "src\validation\validate_transactions.py"
Move-Item -Path "json_gen\utils\comparison_syn_input\comp.py" -Destination "src\validation\compare_distributions.py"
```

#### **4.4 Utils**
```powershell
# Move utility scripts
Move-Item -Path "json_gen\utils\extract_all_poi_id_interactions\poi_extraction_all.py" -Destination "src\utils\poi_extraction.py"

# Move postprocessing
Move-Item -Path "json_gen\output_syn_json\postprocessing\process_synthetic_data.py" -Destination "src\validation\process_synthetic_data.py"
```

#### **4.5 Clean up old directories**
```powershell
# After confirming all files moved
Remove-Item -Path "json_gen" -Recurse -Force
```

---

### **Phase 5: Update Import Statements** (20 min)

You'll need to update imports in all moved files. Here's what to change:

#### **In `src/generation/transaction_generator.py`:**

```python
# OLD:
from c0_Configuration.config_paths import JSON_INPUT_FILTERED, JSON_OUTPUT_FILTERED

# NEW:
from config.paths import INPUT_FILTERED_JSON, SYNTHETIC_FILTERED_JSON
```

**Find and replace in file:**
```python
# Update path variables
JSON_INPUT_FILTERED → INPUT_FILTERED_JSON
JSON_INPUT_ALL → INPUT_ALL_CATEGORIES_JSON
JSON_OUTPUT_FILTERED → SYNTHETIC_FILTERED_JSON
JSON_OUTPUT_ALL → SYNTHETIC_ALL_CATEGORIES_JSON
```

#### **In `src/preprocessing/fsq_to_input_json.py`:**

```python
# OLD:
from c0_Configuration.config_paths import CHECKINS_PATH_CSV, POI_COORDINATES_CSV

# NEW:
from config.paths import FSQ_CHECKINS_CSV, FSQ_POI_CSV
```

#### **In `src/preprocessing/add_planning_area.py`:**

```python
# OLD:
from c0_Configuration.config_paths import CHECKINS_PATH_CSV, PLANNING_CACHE

# NEW:
from config.paths import FSQ_CHECKINS_CSV, PLANNING_AREA_GEOJSON
```

---

### **Phase 6: Create Entry Point Scripts** (15 min)

#### **Create `scripts/generate_synthetic_data.py`:**

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry point for synthetic data generation.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.transaction_generator import process_input_to_synthetic
from config.paths import (
    INPUT_FILTERED_JSON,
    INPUT_ALL_CATEGORIES_JSON,
    SYNTHETIC_FILTERED_JSON,
    SYNTHETIC_ALL_CATEGORIES_JSON
)

def main():
    """Run the complete synthetic data generation pipeline."""
    print("="*60)
    print("SYNTHETIC DATA GENERATION - DUAL OUTPUT VERSION")
    print("="*60)
    print("Generating synthetic datasets from both input files:")
    print(f"1. Filtered categories: {INPUT_FILTERED_JSON}")
    print(f"2. All categories: {INPUT_ALL_CATEGORIES_JSON}")
    
    try:
        # Process filtered categories dataset
        print("\n" + "🎯" * 20)
        filtered_profiles = process_input_to_synthetic(
            INPUT_FILTERED_JSON, 
            SYNTHETIC_FILTERED_JSON, 
            "Filtered Categories Dataset"
        )
        
        # Process all categories dataset
        print("\n" + "📊" * 20)
        all_profiles = process_input_to_synthetic(
            INPUT_ALL_CATEGORIES_JSON, 
            SYNTHETIC_ALL_CATEGORIES_JSON, 
            "All Categories Dataset"
        )
        
        # Final summary
        print("\n" + "="*60)
        print("GENERATION COMPLETE - SUMMARY")
        print("="*60)
        print(f"📂 Output Files Generated:")
        print(f"   1. Filtered dataset: {SYNTHETIC_FILTERED_JSON}")
        print(f"      └── Users: {len(filtered_profiles)}")
        print(f"      └── Transactions: {sum(len(p['interaction']['transactions']) for p in filtered_profiles)}")
        print(f"   2. All categories dataset: {SYNTHETIC_ALL_CATEGORIES_JSON}")
        print(f"      └── Users: {len(all_profiles)}")
        print(f"      └── Transactions: {sum(len(p['interaction']['transactions']) for p in all_profiles)}")
        
        print(f"\n🎉 Successfully generated both synthetic datasets!")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
```

#### **Create `scripts/validate_output.py`:**

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate synthetic dataset quality.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.validate_transactions import main as validate_main

if __name__ == "__main__":
    validate_main()
```

#### **Create `scripts/preprocess_fsq_data.py`:**

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess raw FSQ data to input JSON format.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.fsq_to_input_json import main as preprocess_main

if __name__ == "__main__":
    preprocess_main()
```

---

### **Phase 7: Create Supporting Files** (10 min)

#### **Create `setup.py`:**

```python
from setuptools import setup, find_packages

setup(
    name="singapore-fsq-synthetic",
    version="1.0.0",
    description="Singapore Foursquare Synthetic Dataset Generation",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "geopandas>=0.12.0",
        "shapely>=2.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.64.0",
    ],
)
```

#### **Create `Makefile`:**

```makefile
.PHONY: install clean generate validate preprocess test help

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make clean       - Remove generated files"
	@echo "  make preprocess  - Preprocess raw FSQ data"
	@echo "  make generate    - Generate synthetic datasets"
	@echo "  make validate    - Validate synthetic outputs"
	@echo "  make test        - Run tests"

install:
	pip install -r requirements.txt
	pip install -e .

clean:
	rm -rf data/synthetic/*
	rm -rf data/processed/*
	rm -rf logs/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

preprocess:
	python scripts/preprocess_fsq_data.py

generate:
	python scripts/generate_synthetic_data.py

validate:
	python scripts/validate_output.py

test:
	pytest tests/ -v
```

#### **Create `data/raw/README.md`:**

```markdown
# Raw Data Directory

## Contents

- `FSQ_SG_2013_Checkins.csv` - Original Foursquare Singapore check-in data (2013)
- `FSQ_SG_2013_POI.csv` - POI metadata (names, categories, coordinates)

## Important

⚠️ **DO NOT MODIFY FILES IN THIS DIRECTORY**

These are the original raw datasets. All transformations should be done in the processing pipeline.

## Data Provenance

- **Source**: Foursquare Singapore Dataset
- **Year**: 2013
- **Format**: CSV (Tab-separated)
- **License**: [Add license information]

## Processing Pipeline

1. **Raw Data** (this directory) → Never modified
2. **Interim Data** (`data/interim/`) → Temporary transformations
3. **Processed Data** (`data/processed/`) → Clean input for generation
4. **Synthetic Data** (`data/synthetic/`) → Final outputs
```

#### **Create `.gitkeep` for logs:**

```powershell
New-Item -ItemType File -Force -Path "logs\.gitkeep"
```

---

### **Phase 8: Update .gitignore** (5 min)

Add to `.gitignore`:

```gitignore
# Data directories (keep structure, ignore large files)
data/raw/*.csv
data/interim/*.txt
data/interim/*.json
data/processed/*.json
data/synthetic/*.json

# But keep directory structure
!data/raw/README.md
!data/**/.gitkeep

# Logs
logs/*.log
logs/*.txt
!logs/.gitkeep

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.eggs/

# Virtual environments
venv/
code_venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Jupyter
.ipynb_checkpoints/
```

---

### **Phase 9: Testing** (20 min)

Test the refactored structure:

```powershell
# 1. Test imports
python -c "from config.paths import BASE_DIR; print(f'✅ Config import works: {BASE_DIR}')"

# 2. Test preprocessing (if data exists)
python scripts/preprocess_fsq_data.py

# 3. Test generation
python scripts/generate_synthetic_data.py

# 4. Test validation
python scripts/validate_output.py

# 5. Verify outputs exist
Get-ChildItem data\synthetic\*.json
```

---

### **Phase 10: Update Documentation** (10 min)

#### **Update `.github/copilot-instructions.md`:**

Replace the "Essential File Locations" section with:

```markdown
## Essential File Locations

```
src/                                        # All source code
├── preprocessing/
│   ├── fsq_to_input_json.py               # FSQ CSV → JSON conversion
│   └── add_planning_area.py               # Add geographic regions
├── generation/
│   └── transaction_generator.py           # MAIN: Transaction-only generation
├── validation/
│   ├── validate_transactions.py           # Transaction quality checks
│   ├── compare_distributions.py           # Input vs output comparison
│   └── process_synthetic_data.py          # Post-processing & filtering
└── utils/
    └── poi_extraction.py                  # POI extraction utilities

config/
├── parameters.json                        # Pipeline configuration
└── paths.py                               # Centralized path definitions

data/
├── raw/                                   # Original FSQ datasets (never modify)
│   ├── FSQ_SG_2013_Checkins.csv
│   └── FSQ_SG_2013_POI.csv
├── interim/                               # Temporary transformations
├── processed/                             # Clean inputs for generation
│   ├── input_filtered.json
│   └── input_all_categories.json
└── synthetic/                             # Generated outputs
    ├── fsq_to_synthetic_filtered.json
    └── fsq_to_synthetic_all_categories.json

scripts/                                   # Entry point scripts
├── preprocess_fsq_data.py                # Step 1: Preprocess raw data
├── generate_synthetic_data.py            # Step 2: Generate synthetic data
└── validate_output.py                    # Step 3: Validate outputs
```
```

#### **Update README.md:**

Add a "Project Structure" section:

```markdown
## Project Structure

```
singapore_foursquare_dataset/
├── src/                    # Source code (modular by function)
├── config/                 # Configuration files
├── data/                   # All data (organized by stage)
├── scripts/                # Executable entry points
├── tests/                  # Unit tests
└── .github/                # CI/CD and documentation
```

## Quick Start

```powershell
# 1. Install dependencies
pip install -r requirements.txt
pip install -e .

# 2. Preprocess data
python scripts/preprocess_fsq_data.py

# 3. Generate synthetic datasets
python scripts/generate_synthetic_data.py

# 4. Validate outputs
python scripts/validate_output.py
```
```

---

## ✅ Post-Migration Checklist

- [ ] All files moved to new locations
- [ ] All imports updated and working
- [ ] Configuration paths updated
- [ ] Scripts run successfully
- [ ] Output files generated in correct locations
- [ ] Git tracked files updated
- [ ] Documentation updated
- [ ] `.github/copilot-instructions.md` updated
- [ ] Commit changes: `git commit -m "Refactor: reorganize project structure"`
- [ ] Test on clean environment

---

## 🔄 Rollback Plan

If something goes wrong:

```powershell
# Discard all changes and return to pre-refactor state
git reset --hard HEAD
git clean -fd

# Or restore from specific commit
git reflog
git reset --hard <commit-hash>
```

---

## 📊 Before/After Comparison

### **Running the Pipeline**

**Before:**
```powershell
cd json_gen/func
python direct_conversion_fsq_to_json.py
```

**After:**
```powershell
python scripts/generate_synthetic_data.py
# OR
make generate
```

### **Imports**

**Before:**
```python
from c0_Configuration.config_paths import JSON_OUTPUT
```

**After:**
```python
from config.paths import SYNTHETIC_FILTERED_JSON
```

---

## 🆘 Troubleshooting

### **Issue: Module not found errors**

```powershell
# Ensure you're running from repository root
cd c:\Users\admin\Desktop\sweta\MPS_syn_data_gen\singapore_foursquare_dataset

# Install in editable mode
pip install -e .
```

### **Issue: Path errors**

```powershell
# Verify paths.py is correctly updated
python -c "from config.paths import BASE_DIR, DATA_RAW; print(BASE_DIR); print(DATA_RAW)"
```

### **Issue: Files not found**

```powershell
# Check file actually moved
Get-ChildItem -Recurse -Filter "direct_conversion_fsq_to_json.py"

# Should show: src\generation\transaction_generator.py
```

---

## 📞 Support

If you encounter issues during refactoring:

1. Check the specific phase where error occurred
2. Verify files exist in new locations
3. Check git status: `git status`
4. Review import statements in affected files
5. Use rollback plan if needed

---

**Last Updated:** 2025-11-14  
**Estimated Total Time:** 2-3 hours  
**Difficulty:** Medium  
**Breaking Changes:** None (pure restructure)
