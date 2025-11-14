#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry point for synthetic data generation.

This script runs the complete synthetic data generation pipeline,
converting Foursquare check-in data to synthetic transaction datasets.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.generation.transaction_generator import process_input_to_synthetic
from config.paths import (
    INPUT_TO_GENERATOR_FILTERED_JSON,
    INPUT_TO_GENERATOR_ALL_CATEGORIES_JSON,
    SYNTHETIC_FILTERED_JSON,
    SYNTHETIC_ALL_CATEGORIES_JSON
)


def main():
    """Run the complete synthetic data generation pipeline."""
    print("=" * 60)
    print("SYNTHETIC DATA GENERATION - DUAL OUTPUT VERSION")
    print("=" * 60)
    print("Generating synthetic datasets from both input files:")
    print(f"1. Filtered categories: {INPUT_TO_GENERATOR_FILTERED_JSON}")
    print(f"2. All categories: {INPUT_TO_GENERATOR_ALL_CATEGORIES_JSON}")
    
    try:
        # Process filtered categories dataset
        print("\n" + "🎯" * 20)
        filtered_profiles = process_input_to_synthetic(
            str(INPUT_TO_GENERATOR_FILTERED_JSON), 
            str(SYNTHETIC_FILTERED_JSON), 
            "Filtered Categories Dataset"
        )
        
        # Process all categories dataset
        print("\n" + "📊" * 20)
        all_profiles = process_input_to_synthetic(
            str(INPUT_TO_GENERATOR_ALL_CATEGORIES_JSON), 
            str(SYNTHETIC_ALL_CATEGORIES_JSON), 
            "All Categories Dataset"
        )
        
        # Final summary
        print("\n" + "=" * 60)
        print("GENERATION COMPLETE - SUMMARY")
        print("=" * 60)
        print(f"📂 Output Files Generated:")
        print(f"   1. Filtered dataset: {SYNTHETIC_FILTERED_JSON}")
        if filtered_profiles:
            print(f"      └── Users: {len(filtered_profiles)}")
            total_txns = sum(
                len(p['interaction']['transactions'])
                for p in filtered_profiles
            )
            print(f"      └── Transactions: {total_txns}")
        
        print(f"   2. All categories dataset: {SYNTHETIC_ALL_CATEGORIES_JSON}")
        if all_profiles:
            print(f"      └── Users: {len(all_profiles)}")
            total_txns = sum(
                len(p['interaction']['transactions'])
                for p in all_profiles
            )
            print(f"      └── Transactions: {total_txns}")
        
        print(f"\n🎉 Successfully generated both synthetic datasets!")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
