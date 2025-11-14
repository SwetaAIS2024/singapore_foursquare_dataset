#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess raw FSQ data to input JSON format.

This script performs two preprocessing steps:
1. Add planning area information to raw checkin data
2. Convert FSQ CSV files to JSON format for synthetic data generation
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.add_planning_area import main as add_planning_area_main
from src.preprocessing.fsq_to_input_json import main as fsq_to_json_main


def main():
    """
    Run the complete preprocessing pipeline.
    """
    print("\n" + "="*80)
    print("SINGAPORE FSQ DATA PREPROCESSING PIPELINE")
    print("="*80 + "\n")
    
    # Step 1: Add planning areas to checkin data
    print("\n" + "="*80)
    print("STEP 1: Adding Planning Area Information")
    print("="*80 + "\n")
    try:
        add_planning_area_main()
        print("\n✅ Step 1 Complete: Planning areas added successfully")
    except Exception as e:
        print(f"\n❌ Step 1 Failed: {e}")
        print("Aborting preprocessing pipeline.")
        return 1
    
    # Step 2: Convert to JSON format
    print("\n" + "="*80)
    print("STEP 2: Converting to JSON Format")
    print("="*80 + "\n")
    try:
        fsq_to_json_main()
        print("\n✅ Step 2 Complete: JSON conversion successful")
    except Exception as e:
        print(f"\n❌ Step 2 Failed: {e}")
        return 1
    
    print("\n" + "="*80)
    print("✅ PREPROCESSING PIPELINE COMPLETE")
    print("="*80 + "\n")
    return 0


if __name__ == "__main__":
    exit(main())
