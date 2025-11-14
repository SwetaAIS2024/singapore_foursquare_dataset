#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Postprocessing script for synthetic JSON datasets.

Applies 5-core filtering to synthetic data:
- Users and POIs must have at least 5 interactions
- Outputs filtered JSON in the same format as input
- Generates summary reports and metadata files

Based on the original FSQ preprocessing notebook but adapted for synthetic JSON data structure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import os
from collections import defaultdict

from config.paths import (
    DATA_SYNTHETIC,
    SYNTHETIC_FILTERED_JSON,
    SYNTHETIC_ALL_CATEGORIES_JSON
)

def load_synthetic_data(json_file):
    """Load synthetic data from JSON file."""
    print(f"Loading synthetic data from: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data):,} user profiles")
    return data

def extract_interactions(data):
    """Extract user interactions from synthetic data structure."""
    interactions = []
    
    for user_profile in data:
        user_id = user_profile['user']['userId']
        user_info = user_profile['user']
        
        # Extract transactions (main interaction type)
        transactions = user_profile['interaction']['transactions']
        
        for txn in transactions:
            interactions.append({
                'user_id': user_id,
                'poi_id': txn['poiId'],
                'poi_name': txn.get('poiName', 'Unknown'),
                'lat': txn['userLocation']['latitude'],
                'lon': txn['userLocation']['longitude'],
                'timestamp': txn['timestamp'],
                'category': txn['poiCategories'][0] if txn['poiCategories'] else 'Unknown',
                'planning_area': txn.get('planning_area', 'Unknown'),
                'user_age': user_info['age'],
                'user_gender': user_info['gender'],
                'user_device': user_info['device']['platform']
            })
    
    print(f"Extracted {len(interactions):,} total interactions")
    return interactions

def five_core_filter(df, user_col='user_id', item_col='poi_id', min_interactions=5):
    """Apply 5-core filtering: users and POIs must have at least 5 interactions."""
    print(f"Applying 5-core filtering (min_interactions={min_interactions})")
    print(f"Initial: {len(df):,} interactions, {df[user_col].nunique():,} users, {df[item_col].nunique():,} POIs")
    
    iteration = 0
    while True:
        iteration += 1
        user_counts = df[user_col].value_counts()
        item_counts = df[item_col].value_counts()

        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index

        df_filtered = df[df[user_col].isin(valid_users) & df[item_col].isin(valid_items)]

        print(f"  Iteration {iteration}: {len(df_filtered):,} interactions, "
              f"{df_filtered[user_col].nunique():,} users, {df_filtered[item_col].nunique():,} POIs")

        if len(df_filtered) == len(df):
            break
        df = df_filtered

    print(f"Final after 5-core: {len(df_filtered):,} interactions, "
          f"{df_filtered[user_col].nunique():,} users, {df_filtered[item_col].nunique():,} POIs")
    return df_filtered

def create_poi_dictionary(df):
    """Create POI dictionary with metadata."""
    poi_dict = {}
    
    for _, row in df.drop_duplicates('poi_id').iterrows():
        poi_dict[row['poi_id']] = {
            'name': row['poi_name'],
            'lat': row['lat'],
            'lon': row['lon'],
            'cat': row['category'],
            'planning_area': row['planning_area']
        }
    
    print(f"Created POI dictionary with {len(poi_dict):,} unique POIs")
    return poi_dict


def create_user_dictionary(df):
    """Create user dictionary with metadata."""
    user_dict = {}
    
    for _, row in df.drop_duplicates('user_id').iterrows():
        user_dict[row['user_id']] = {
            'age': row['user_age'],
            'gender': row['user_gender'],
            'device': row['user_device']
        }
    
    print(f"Created user dictionary with {len(user_dict):,} unique users")
    return user_dict

def save_poi_data(poi_dict, output_dir, dataset_name):
    """Save POI dictionary in format compatible with validation models."""
    poi_file = output_dir / f"{dataset_name}_poi_data.txt"
    
    with open(poi_file, 'w', encoding='utf-8') as f:
        for poi_id, data in poi_dict.items():
            line = f"{poi_id}\t{data['name']}\t{data['lat']}\t{data['lon']}\t{data['cat']}\t{data['planning_area']}\n"
            f.write(line)
    
    print(f"Saved POI data to: {poi_file}")


def save_user_data(user_dict, output_dir, dataset_name):
    """Save user dictionary."""
    user_file = output_dir / f"{dataset_name}_user_data.txt"
    
    with open(user_file, 'w', encoding='utf-8') as f:
        for user_id, data in user_dict.items():
            line = f"{user_id}\t{data['age']}\t{data['gender']}\t{data['device']}\n"
            f.write(line)
    
    print(f"Saved user data to: {user_file}")

def save_filtered_json(original_data, df_filtered, output_dir, dataset_name):
    """Save filtered data back in original JSON format."""
    
    # Get the set of valid users and POIs after 5-core filtering
    valid_users = set(df_filtered['user_id'].unique())
    valid_pois = set(df_filtered['poi_id'].unique())
    
    print(f"Filtering JSON data: keeping {len(valid_users):,} users and {len(valid_pois):,} POIs")
    
    # Filter the original JSON data
    filtered_data = []
    
    for user_profile in original_data:
        user_id = user_profile['user']['userId']
        
        if user_id not in valid_users:
            continue
        
        # Keep the user structure but filter transactions
        filtered_user = {
            'user': user_profile['user'].copy(),
            'interaction': {
                'views': [],  # Keep empty as in original
                'transactions': [],
                'reviews': []  # Keep empty as in original
            }
        }
        
        # Filter transactions to only include valid POIs
        for txn in user_profile['interaction']['transactions']:
            if txn['poiId'] in valid_pois:
                filtered_user['interaction']['transactions'].append(txn)
        
        # Only keep users who still have transactions after filtering
        if len(filtered_user['interaction']['transactions']) > 0:
            filtered_data.append(filtered_user)
    
    # Save the filtered JSON
    output_file = output_dir / f"{dataset_name}_5core_filtered.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved filtered JSON to: {output_file}")
    print(f"  Original users: {len(original_data):,}")
    print(f"  Filtered users: {len(filtered_data):,}")
    print(f"  Retention rate: {len(filtered_data)/len(original_data)*100:.2f}%")

def generate_basic_summary_report(df_original, df_filtered, poi_dict, user_dict, output_dir, dataset_name):
    """Generate basic summary report for 5-core filtering only."""
    report_file = output_dir / f"{dataset_name}_5core_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"5-CORE FILTERING REPORT FOR {dataset_name.upper()}\n")
        f.write("=" * 60 + "\n\n")
        
        # Original data statistics
        f.write("ORIGINAL DATA:\n")
        f.write(f"  Total interactions: {len(df_original):,}\n")
        f.write(f"  Unique users: {df_original['user_id'].nunique():,}\n")
        f.write(f"  Unique POIs: {df_original['poi_id'].nunique():,}\n")
        f.write(f"  Unique categories: {df_original['category'].nunique():,}\n\n")
        
        # After 5-core filtering
        f.write("AFTER 5-CORE FILTERING:\n")
        f.write(f"  Total interactions: {len(df_filtered):,}\n")
        f.write(f"  Unique users: {df_filtered['user_id'].nunique():,}\n")
        f.write(f"  Unique POIs: {df_filtered['poi_id'].nunique():,}\n")
        f.write(f"  Data retention: {len(df_filtered)/len(df_original)*100:.2f}%\n\n")
        
        # User retention
        f.write("USER RETENTION:\n")
        f.write(f"  Users retained: {df_filtered['user_id'].nunique():,} / {df_original['user_id'].nunique():,} ")
        f.write(f"({df_filtered['user_id'].nunique()/df_original['user_id'].nunique()*100:.2f}%)\n\n")
        
        # POI retention  
        f.write("POI RETENTION:\n")
        f.write(f"  POIs retained: {df_filtered['poi_id'].nunique():,} / {df_original['poi_id'].nunique():,} ")
        f.write(f"({df_filtered['poi_id'].nunique()/df_original['poi_id'].nunique()*100:.2f}%)\n\n")
        
        # Category distribution
        f.write("POI CATEGORY DISTRIBUTION (AFTER 5-CORE):\n")
        cat_counts = df_filtered['category'].value_counts()
        for cat, count in cat_counts.head(15).items():
            f.write(f"  {cat:<30}: {count:5d} ({count/len(df_filtered)*100:5.2f}%)\n")
        
        f.write(f"\n  Total categories: {len(cat_counts)}\n\n")
        
        # User interaction distribution
        f.write("USER INTERACTION DISTRIBUTION (AFTER 5-CORE):\n")
        user_interactions = df_filtered['user_id'].value_counts()
        f.write(f"  Min interactions per user: {user_interactions.min()}\n")
        f.write(f"  Max interactions per user: {user_interactions.max()}\n")
        f.write(f"  Avg interactions per user: {user_interactions.mean():.2f}\n")
        f.write(f"  Median interactions per user: {user_interactions.median():.2f}\n\n")
        
        # POI interaction distribution
        f.write("POI INTERACTION DISTRIBUTION (AFTER 5-CORE):\n")
        poi_interactions = df_filtered['poi_id'].value_counts()
        f.write(f"  Min interactions per POI: {poi_interactions.min()}\n")
        f.write(f"  Max interactions per POI: {poi_interactions.max()}\n")
        f.write(f"  Avg interactions per POI: {poi_interactions.mean():.2f}\n")
        f.write(f"  Median interactions per POI: {poi_interactions.median():.2f}\n")
    
    print(f"Generated 5-core filtering report: {report_file}")

def process_dataset(input_file, output_dir, dataset_name, min_interactions=5):
    """
    Main processing pipeline - 5-core filtering only.
    
    Parameters:
    - input_file: Path to input JSON file
    - output_dir: Path to output directory
    - dataset_name: Name of the dataset (used in output filenames)
    - min_interactions: 5-core filtering threshold (users and POIs must have ≥5 interactions)
    """
    
    print(f"Processing {dataset_name.upper()}")
    print(f"Applying 5-core filtering with minimum {min_interactions} interactions per user/POI\n")
    
    # Load data
    synthetic_data = load_synthetic_data(input_file)
    
    # Extract interactions
    interactions = extract_interactions(synthetic_data)
    df_original = pd.DataFrame(interactions)
    
    # Apply 5-core filtering
    df_filtered = five_core_filter(df_original, min_interactions=min_interactions)
    
    # Create dictionaries
    poi_dict = create_poi_dictionary(df_filtered)
    user_dict = create_user_dictionary(df_filtered)
    
    # Save filtered data in original JSON format (main output)
    save_filtered_json(synthetic_data, df_filtered, output_dir, dataset_name)
    
    # Save basic metadata
    save_poi_data(poi_dict, output_dir, dataset_name)
    save_user_data(user_dict, output_dir, dataset_name)
    
    # Generate basic summary report
    generate_basic_summary_report(df_original, df_filtered, poi_dict, user_dict, output_dir, dataset_name)
    
    print(f"✅ 5-core filtering complete for {dataset_name}\n")
    return {
        'df_filtered': df_filtered,
        'poi_dict': poi_dict,
        'user_dict': user_dict
    }

def main():
    """Main execution function for postprocessing synthetic data."""
    
    # Define output directory for postprocessed files
    output_dir = DATA_SYNTHETIC / "postprocessed"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Processing parameters
    min_interactions = 5  # 5-core filtering threshold
    
    print("\n" + "="*80)
    print("5-CORE FILTERING PIPELINE FOR SYNTHETIC DATA")
    print("="*80)
    print(f"Input files:")
    print(f"  1. Filtered: {SYNTHETIC_FILTERED_JSON}")
    print(f"  2. All categories: {SYNTHETIC_ALL_CATEGORIES_JSON}")
    print(f"Output directory: {output_dir}")
    print(f"Parameters: min_interactions={min_interactions}")
    print(f"Processing: 5-core filtering only\n")
    
    # Process filtered dataset
    if SYNTHETIC_FILTERED_JSON.exists():
        print(f"\n{'='*80}")
        print("PROCESSING FILTERED DATASET")
        print(f"{'='*80}\n")
        filtered_results = process_dataset(
            SYNTHETIC_FILTERED_JSON, 
            output_dir, 
            "filtered",
            min_interactions=min_interactions
        )
    else:
        print(f"❌ Filtered file not found: {SYNTHETIC_FILTERED_JSON}")
        filtered_results = None
    
    # Process all categories dataset
    if SYNTHETIC_ALL_CATEGORIES_JSON.exists():
        print(f"\n{'='*80}")
        print("PROCESSING ALL CATEGORIES DATASET")
        print(f"{'='*80}\n")
        all_results = process_dataset(
            SYNTHETIC_ALL_CATEGORIES_JSON,
            output_dir,
            "all_categories", 
            min_interactions=min_interactions
        )
    else:
        print(f"❌ All categories file not found: {SYNTHETIC_ALL_CATEGORIES_JSON}")
        all_results = None
    
    # Final summary
    print(f"\n{'='*80}")
    print("✅ 5-CORE FILTERING PIPELINE COMPLETE")
    print(f"{'='*80}\n")
    
    if filtered_results:
        print(f"✅ Filtered dataset: {len(filtered_results['df_filtered']):,} interactions, "
              f"{filtered_results['df_filtered']['user_id'].nunique():,} users, "
              f"{filtered_results['df_filtered']['poi_id'].nunique():,} POIs")
        print(f"   📄 Output: {output_dir / 'filtered_5core_filtered.json'}")
    
    if all_results:
        print(f"✅ All categories dataset: {len(all_results['df_filtered']):,} interactions, "
              f"{all_results['df_filtered']['user_id'].nunique():,} users, "
              f"{all_results['df_filtered']['poi_id'].nunique():,} POIs")
        print(f"   📄 Output: {output_dir / 'all_categories_5core_filtered.json'}")
    
    print(f"\n📂 All output files saved to: {output_dir}")
    print("🎯 5-core filtered JSON datasets ready - same format as input!\n")
    
    return 0 if (filtered_results or all_results) else 1


if __name__ == "__main__":
    exit(main())