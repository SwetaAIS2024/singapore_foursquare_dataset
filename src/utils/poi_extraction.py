#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
POI Extraction Utility

Extracts unique POIs from 5-core filtered synthetic datasets.
Creates a reference POI dataset with metadata for validation and analysis.
"""
import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.paths import (
    DATA_SYNTHETIC_POST,
    INPUT_TO_GENERATOR_ALL_CATEGORIES_JSON,
    INPUT_TO_GENERATOR_FILTERED_JSON
)


# POI reference file for names and metadata
poi_reference_file = INPUT_TO_GENERATOR_ALL_CATEGORIES_JSON

# Input: 5-core filtered synthetic datasets
input_files = [
    {
        "input": DATA_SYNTHETIC_POST / "all_categories_5core_filtered.json",
        "output": Path(__file__).parent / "all_pois_all_categories_synthetic.json"
    },
    {
        "input": DATA_SYNTHETIC_POST / "filtered_5core_filtered.json",
        "output": Path(__file__).parent / "all_pois_filtered_synthetic.json"
    }
]

def safe_float(val):
    """Convert value to float safely, returning None on failure."""
    try:
        return float(val)
    except Exception:
        return None


def safe_string(val, default="Unknown"):
    """
    Safely convert value to string, handling NaN, None, and empty values.
    
    Parameters:
    val: Value to convert
    default: Default value to return if conversion fails
    
    Returns:
    str: Converted string or default value
    """
    import math
    
    if val is None:
        return default
    
    # Handle NaN values (both float('nan') and string 'NaN')
    if isinstance(val, float) and math.isnan(val):
        return default
    
    # Handle string representations of NaN
    if isinstance(val, str) and val.lower() in ['nan', 'null', 'none', '']:
        return default
    
    # Return the string representation of the value
    return str(val).strip() or default


def load_poi_reference(reference_file):
    """
    Load POI names and metadata from the input JSON file.
    
    Returns:
    dict: Dictionary mapping poi_id to POI metadata
    """
    poi_lookup = {}
    
    if not reference_file.exists():
        print(f"⚠️  POI reference file not found: {reference_file}")
        return poi_lookup
    
    print(f"📖 Loading POI reference from: {reference_file}")
    
    with open(reference_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for user_data in data:
        user_metadata = user_data.get('user_metadata', [])
        for visit in user_metadata:
            poi_id = visit.get('poi_id')
            if not poi_id or poi_id in poi_lookup:
                continue
            
            poi_lookup[poi_id] = {
                'name': safe_string(visit.get('poi_name'), 'Unknown'),
                'category': safe_string(visit.get('poi_category'), 'Unknown'),
                'planning_area': visit.get('planning_area'),
                'lat': safe_float(visit.get('lat')),
                'lon': safe_float(visit.get('lon'))
            }
    
    print(f"✅ Loaded {len(poi_lookup):,} POI references\n")
    return poi_lookup


def main():
    """
    Main execution function for POI extraction.
    """
    print("\n" + "="*80)
    print("POI EXTRACTION FROM 5-CORE FILTERED SYNTHETIC DATA")
    print("="*80 + "\n")
    
    # Load POI reference data for names and metadata
    poi_lookup = load_poi_reference(poi_reference_file)
    
    # Process each input file separately
    for file_config in input_files:
        input_file = file_config["input"]
        output_file = file_config["output"]
        
        if not input_file.exists():
            print(f"⚠️  Skipping {input_file.name} - file not found\n")
            continue
        
        print(f"{'='*80}")
        print(f"Processing: {input_file.name}")
        print(f"{'='*80}")
        print(f"📂 Reading from: {input_file}")
        
        unique_pois = {}
        
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"👥 Processing {len(data):,} users from {input_file.name}...")

        poi_count = 0
        
        for i, user_data in enumerate(data):
            # Extract transactions from the new structure
            interaction = user_data.get("interaction", {})
            transactions = interaction.get("transactions", [])
            
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1:,} users...")
            
            for transaction in transactions:
                poi_count += 1
                poi_id = transaction.get("poiId")
                
                # Skip if no POI ID or already seen
                if not poi_id or poi_id in unique_pois:
                    continue
                
                # Get POI info from reference first, fallback to transaction data
                poi_ref = poi_lookup.get(poi_id, {})
                
                poi_name = poi_ref.get('name') or safe_string(transaction.get("poiName"), "Unknown")
                poi_categories = transaction.get("poiCategories", [])
                if not poi_categories and poi_ref.get('category'):
                    poi_categories = [poi_ref.get('category')]
                
                planning_area = transaction.get("planning_area") or poi_ref.get('planning_area')
                user_location = transaction.get("userLocation", {})
                lat = safe_float(user_location.get("latitude")) or poi_ref.get('lat')
                lon = safe_float(user_location.get("longitude")) or poi_ref.get('lon')
                
                # Only require POI ID - everything else is optional
                unique_pois[poi_id] = {
                    "poiId": poi_id,
                    "poiName": poi_name,
                    "poiCategories": poi_categories if poi_categories else [],
                    "planningArea": planning_area if planning_area else None,
                    "userLocation": {
                        "latitude": lat,
                        "longitude": lon
                    }
                }
                
                if len(unique_pois) <= 5:  # Show first few POIs
                    print(f"   ✓ Added POI: {poi_id} - {poi_name}")
        
        print(f"📊 Processed {poi_count:,} total transactions")
        print(f"✅ Found {len(unique_pois):,} unique POIs from {input_file.name}")

        unique_pois_list = list(unique_pois.values())

        # Save to output file
        print(f"💾 Saving POI data to: {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(unique_pois_list, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved {len(unique_pois_list):,} POIs to {output_file.name}")
        
        if unique_pois_list:
            print(f"\n📋 Sample POI format:")
            sample_poi = unique_pois_list[0]
            print(json.dumps(sample_poi, indent=2, ensure_ascii=False))
        
        print()  # Empty line between files
    
    print(f"{'='*80}")
    print(f"✅ EXTRACTION COMPLETE")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())