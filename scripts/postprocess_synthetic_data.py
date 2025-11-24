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
import shutil
from datetime import datetime
from collections import defaultdict

from config.paths import (
    DATA_SYNTHETIC_POST,
    SYNTHETIC_FILTERED_JSON,
    SYNTHETIC_ALL_CATEGORIES_JSON,
    BASE_DIR
)

def load_category_mapping(mapping_file=None):
    """
    Load category mapping from CSV/JSON file.
    Maps 180 detailed POI categories to 50 group categories.
    
    Parameters:
    - mapping_file: Path to mapping file (CSV or JSON format)
                   Expected format: original_category -> group_category
                   CSV: original_category,group_category
                   JSON: {"original_category": "group_category", ...}
    
    Returns:
    - dict: Mapping dictionary {original_category: group_category}
    """
    if mapping_file is None:
        # Default path - look for mapping file in config or data directory
        mapping_file = BASE_DIR / "config" / "category_mapping.json"
        if not mapping_file.exists():
            mapping_file = BASE_DIR / "config" / "category_mapping.csv"
    
    mapping_file = Path(mapping_file)
    
    if not mapping_file.exists():
        print(f"⚠️  Warning: Category mapping file not found: {mapping_file}")
        print(f"   Skipping category grouping. To enable, provide mapping file.")
        return None
    
    print(f"Loading category mapping from: {mapping_file}")
    
    category_mapping = {}
    
    try:
        if mapping_file.suffix == '.json':
            # Load JSON format
            with open(mapping_file, 'r', encoding='utf-8') as f:
                category_mapping = json.load(f)
        
        elif mapping_file.suffix == '.csv':
            # Load CSV format
            df_mapping = pd.read_csv(mapping_file)
            if 'original_category' in df_mapping.columns and 'group_category' in df_mapping.columns:
                category_mapping = dict(zip(
                    df_mapping['original_category'],
                    df_mapping['group_category']
                ))
            else:
                print(f"❌ Error: CSV must have columns 'original_category' and 'group_category'")
                return None
        
        else:
            print(f"❌ Error: Unsupported file format. Use .json or .csv")
            return None
        
        print(f"✅ Loaded mapping for {len(category_mapping)} categories")
        
        # Show some examples
        sample_mappings = list(category_mapping.items())[:5]
        print(f"   Sample mappings:")
        for orig, group in sample_mappings:
            print(f"     '{orig}' → '{group}'")
        
        return category_mapping
    
    except Exception as e:
        print(f"❌ Error loading category mapping: {e}")
        return None


def create_backup(file_path, backup_dir=None):
    """
    Create a timestamped backup of a file.
    
    Parameters:
    - file_path: Path to the file to backup
    - backup_dir: Directory to store backups (default: {output_dir}/backups)
    
    Returns:
    - Path to the backup file, or None if file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return None
    
    # Create backup directory
    if backup_dir is None:
        backup_dir = file_path.parent / "backups"
    else:
        backup_dir = Path(backup_dir)
    
    backup_dir.mkdir(exist_ok=True, parents=True)
    
    # Create timestamped backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
    backup_path = backup_dir / backup_name
    
    # Copy file to backup
    shutil.copy2(file_path, backup_path)
    
    return backup_path


def backup_postprocessed_files(output_dir, dataset_name=None, preserve_original=False):
    """
    Backup only the files that will be modified by the current run.
    
    Parameters:
    - output_dir: Directory containing postprocessed files
    - dataset_name: Name of dataset being processed (e.g., 'filtered', 'all_categories')
                   If None, backs up all files
    - preserve_original: Whether preserve_original mode is enabled
    
    Returns:
    - Dictionary with backup information
    """
    output_dir = Path(output_dir)
    backup_dir = output_dir / "backups"
    
    print(f"\n{'='*80}")
    print("BACKING UP EXISTING POSTPROCESSED FILES")
    print(f"{'='*80}\n")
    
    backup_info = {
        'backup_dir': str(backup_dir),
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'files_backed_up': [],
        'files_not_found': []
    }
    
    # Determine which files to backup based on dataset_name
    if dataset_name:
        # Only backup files for this specific dataset
        files_to_backup = [
            f"{dataset_name}_5core_filtered.json",
            f"{dataset_name}_poi_data.txt",
            f"{dataset_name}_user_data.txt",
            f"{dataset_name}_5core_report.txt"
        ]
        # Add grouped files if they exist (might be created by this run)
        if preserve_original:
            files_to_backup.extend([
                f"{dataset_name}_with_groupCategory.json",
                f"{dataset_name}_grouped_replaced.json"
            ])
        else:
            files_to_backup.append(f"{dataset_name}_grouped.json")
    else:
        # Backup all files (fallback for safety)
        files_to_backup = [
            "filtered_5core_filtered.json",
            "filtered_grouped.json",
            "all_categories_5core_filtered.json",
            "all_categories_grouped.json",
            "filtered_poi_data.txt",
            "filtered_user_data.txt",
            "all_categories_poi_data.txt",
            "all_categories_user_data.txt",
            "filtered_5core_report.txt",
            "all_categories_5core_report.txt"
        ]
    
    for filename in files_to_backup:
        file_path = output_dir / filename
        
        if file_path.exists():
            backup_path = create_backup(file_path, backup_dir)
            if backup_path:
                backup_info['files_backed_up'].append({
                    'original': str(file_path),
                    'backup': str(backup_path),
                    'size': file_path.stat().st_size
                })
                print(f"✅ Backed up: {filename}")
                print(f"   → {backup_path.name}")
        else:
            backup_info['files_not_found'].append(str(file_path))
    
    # Save backup manifest
    manifest_path = backup_dir / f"backup_manifest_{backup_info['timestamp']}.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(backup_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 Backup Summary:")
    print(f"   Files backed up: {len(backup_info['files_backed_up'])}")
    print(f"   Backup location: {backup_dir}")
    print(f"   Manifest: {manifest_path.name}")
    
    if backup_info['files_not_found']:
        print(f"\n⚠️  Files not found (skipped):")
        for file_path in backup_info['files_not_found']:
            print(f"   - {Path(file_path).name}")
    
    print(f"\n✅ Backup complete!\n")
    
    return backup_info


def apply_category_grouping(data, category_mapping, preserve_original=False):
    """
    Apply category grouping to synthetic data.
    Can either replace categories or add group_category as a separate field.
    
    Parameters:
    - data: List of user profiles (synthetic data JSON structure)
    - category_mapping: Dictionary mapping original_category -> group_category
    - preserve_original: If True, adds 'groupCategory' field and keeps original.
                        If False, replaces poiCategories with group category.
    
    Returns:
    - List of user profiles with updated categories
    - Statistics dictionary with mapping results
    """
    if category_mapping is None:
        print("⚠️  No category mapping provided, skipping category grouping")
        return data, None
    
    print(f"\n{'='*80}")
    print("APPLYING CATEGORY GROUPING")
    print(f"{'='*80}")
    print(f"Mapping {len(category_mapping)} categories to group categories")
    print(f"Mode: {'PRESERVE ORIGINAL (add groupCategory field)' if preserve_original else 'REPLACE (overwrite poiCategories)'}\n")
    
    # Statistics tracking
    stats = {
        'total_transactions': 0,
        'categories_mapped': 0,
        'categories_unmapped': 0,
        'unmapped_categories': set(),
        'group_category_distribution': defaultdict(int),
        'original_category_distribution': defaultdict(int)
    }
    
    # Process each user profile
    for user_profile in data:
        transactions = user_profile['interaction']['transactions']
        
        for txn in transactions:
            stats['total_transactions'] += 1
            
            # Get original category (first in list)
            if txn['poiCategories'] and len(txn['poiCategories']) > 0:
                original_category = txn['poiCategories'][0]
                stats['original_category_distribution'][original_category] += 1
                
                # Map to group category
                if original_category in category_mapping:
                    group_category = category_mapping[original_category]
                    
                    if preserve_original:
                        # Add as separate field, keep original
                        txn['groupCategory'] = group_category
                    else:
                        # Replace original with group category
                        txn['poiCategories'] = [group_category]
                    
                    stats['categories_mapped'] += 1
                    stats['group_category_distribution'][group_category] += 1
                else:
                    # Category not in mapping
                    if preserve_original:
                        txn['groupCategory'] = original_category  # Use original as fallback
                    
                    stats['categories_unmapped'] += 1
                    stats['unmapped_categories'].add(original_category)
                    stats['group_category_distribution'][original_category] += 1
    
    # Print statistics
    print(f"Category Grouping Results:")
    print(f"  Total transactions processed: {stats['total_transactions']:,}")
    print(f"  Categories mapped: {stats['categories_mapped']:,} ({stats['categories_mapped']/stats['total_transactions']*100:.2f}%)")
    print(f"  Categories unmapped: {stats['categories_unmapped']:,} ({stats['categories_unmapped']/stats['total_transactions']*100:.2f}%)")
    
    if stats['unmapped_categories']:
        print(f"\n⚠️  Warning: {len(stats['unmapped_categories'])} categories not found in mapping:")
        for cat in sorted(list(stats['unmapped_categories']))[:10]:
            count = stats['original_category_distribution'][cat]
            print(f"     - '{cat}' ({count} transactions)")
        if len(stats['unmapped_categories']) > 10:
            print(f"     ... and {len(stats['unmapped_categories']) - 10} more")
    
    print(f"\nGroup Category Distribution (Top 20):")
    sorted_groups = sorted(stats['group_category_distribution'].items(), 
                          key=lambda x: x[1], reverse=True)
    for cat, count in sorted_groups[:20]:
        pct = count / stats['total_transactions'] * 100
        print(f"  {cat:<30}: {count:6,} ({pct:5.2f}%)")
    
    print(f"\n✅ Category grouping complete!")
    print(f"   Original categories: {len(stats['original_category_distribution'])}")
    print(f"   Group categories: {len(stats['group_category_distribution'])}")
    
    return data, stats


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

def process_dataset(input_file, output_dir, dataset_name, min_interactions=5, 
                   category_mapping_file=None, apply_grouping=True, preserve_original=False, skip_backup=False):
    """
    Main processing pipeline - 5-core filtering + optional category grouping.
    
    Parameters:
    - input_file: Path to input JSON file
    - output_dir: Path to output directory
    - dataset_name: Name of the dataset (used in output filenames)
    - min_interactions: 5-core filtering threshold (users and POIs must have ≥5 interactions)
    - category_mapping_file: Path to category mapping file (CSV or JSON)
    - apply_grouping: Whether to apply category grouping (default: True)
    - preserve_original: If True, creates TWO versions:
                        1. Original categories with groupCategory field added
                        2. Replaced categories (poiCategories overwritten)
    - skip_backup: Skip backing up existing files (default: False)
    """
    
    print(f"Processing {dataset_name.upper()}")
    print(f"Applying 5-core filtering with minimum {min_interactions} interactions per user/POI")
    if apply_grouping:
        print(f"Category grouping: ENABLED")
        if preserve_original:
            print(f"  Mode: CREATE TWO VERSIONS (preserved + replaced)")
        else:
            print(f"  Mode: REPLACE ONLY")
    else:
        print(f"Category grouping: DISABLED")
    print()
    
    # Backup existing postprocessed files for this dataset only (unless skip_backup=True)
    if not skip_backup:
        backup_postprocessed_files(output_dir, dataset_name=dataset_name, preserve_original=preserve_original)
    else:
        print("⚠️  Backup skipped (--no-backup flag enabled)\n")
    
    # Load data
    synthetic_data = load_synthetic_data(input_file)
    
    # ==================================================
    # STEP 1: APPLY CATEGORY GROUPING (OPTIONAL)
    # ==================================================
    grouping_stats = None
    if apply_grouping:
        category_mapping = load_category_mapping(category_mapping_file)
        if category_mapping:
            
            if preserve_original:
                # Create TWO versions
                print(f"\n{'='*80}")
                print("CREATING VERSION 1: WITH GROUP CATEGORY FIELD (PRESERVED)")
                print(f"{'='*80}")
                
                import copy
                synthetic_data_preserved = copy.deepcopy(synthetic_data)
                synthetic_data_preserved, stats_preserved = apply_category_grouping(
                    synthetic_data_preserved, category_mapping, preserve_original=True
                )
                
                # Save preserved version before 5-core
                grouped_preserved = output_dir / f"{dataset_name}_with_groupCategory.json"
                with open(grouped_preserved, 'w', encoding='utf-8') as f:
                    json.dump(synthetic_data_preserved, f, ensure_ascii=False, indent=2)
                print(f"\n📄 Saved preserved version to: {grouped_preserved}")
                
                print(f"\n{'='*80}")
                print("CREATING VERSION 2: REPLACED CATEGORIES")
                print(f"{'='*80}")
                
                synthetic_data, grouping_stats = apply_category_grouping(
                    synthetic_data, category_mapping, preserve_original=False
                )
                
                # Save replaced version before 5-core
                grouped_replaced = output_dir / f"{dataset_name}_grouped_replaced.json"
                with open(grouped_replaced, 'w', encoding='utf-8') as f:
                    json.dump(synthetic_data, f, ensure_ascii=False, indent=2)
                print(f"\n📄 Saved replaced version to: {grouped_replaced}")
                
            else:
                # Single version - replace only
                synthetic_data, grouping_stats = apply_category_grouping(
                    synthetic_data, category_mapping, preserve_original=False
                )
                
                # Save grouped data before 5-core filtering
                grouped_output = output_dir / f"{dataset_name}_grouped.json"
                with open(grouped_output, 'w', encoding='utf-8') as f:
                    json.dump(synthetic_data, f, ensure_ascii=False, indent=2)
                print(f"\n📄 Saved grouped data to: {grouped_output}")
        else:
            print("⏭️  Skipping category grouping (no mapping file found)")
    else:
        print("⏭️  Category grouping disabled")
    
    print(f"\n{'='*80}")
    print("APPLYING 5-CORE FILTERING")
    print(f"{'='*80}\n")
    
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

def main(category_mapping_file=None, apply_grouping=True, preserve_original=False, skip_backup=False):
    """
    Main execution function for postprocessing synthetic data.
    
    Parameters:
    - category_mapping_file: Path to category mapping file (optional)
    - apply_grouping: Whether to apply category grouping (default: True)
    - preserve_original: Create two versions - with groupCategory field AND replaced (default: False)
    - skip_backup: Skip backing up existing files (default: False)
    """
    
    # Define output directory for postprocessed files
    output_dir = DATA_SYNTHETIC_POST
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Processing parameters
    min_interactions = 5  # 5-core filtering threshold
    
    print("\n" + "="*80)
    print("POSTPROCESSING PIPELINE FOR SYNTHETIC DATA")
    print("="*80)
    print(f"Input files:")
    print(f"  1. Filtered: {SYNTHETIC_FILTERED_JSON}")
    print(f"  2. All categories: {SYNTHETIC_ALL_CATEGORIES_JSON}")
    print(f"Output directory: {output_dir}")
    print(f"Parameters:")
    print(f"  - min_interactions: {min_interactions}")
    print(f"  - category_grouping: {'ENABLED' if apply_grouping else 'DISABLED'}")
    if apply_grouping:
        print(f"  - preserve_original: {'YES (2 versions)' if preserve_original else 'NO (replace only)'}")
    if category_mapping_file:
        print(f"  - mapping_file: {category_mapping_file}")
    print()
    
    # Process filtered dataset
    if SYNTHETIC_FILTERED_JSON.exists():
        print(f"\n{'='*80}")
        print("PROCESSING FILTERED DATASET")
        print(f"{'='*80}\n")
        filtered_results = process_dataset(
            SYNTHETIC_FILTERED_JSON, 
            output_dir, 
            "filtered",
            min_interactions=min_interactions,
            category_mapping_file=category_mapping_file,
            apply_grouping=apply_grouping,
            preserve_original=preserve_original,
            skip_backup=skip_backup
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
            min_interactions=min_interactions,
            category_mapping_file=category_mapping_file,
            apply_grouping=apply_grouping,
            preserve_original=preserve_original,
            skip_backup=skip_backup
        )
    else:
        print(f"❌ All categories file not found: {SYNTHETIC_ALL_CATEGORIES_JSON}")
        all_results = None
    
    # Final summary
    print(f"\n{'='*80}")
    print("✅ POSTPROCESSING PIPELINE COMPLETE")
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
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Postprocess synthetic data with 5-core filtering and category grouping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Replace categories only
  python postprocess_synthetic_data.py
  
  # Create both versions (preserved + replaced)
  python postprocess_synthetic_data.py --preserve-original
  
  # Use custom mapping file
  python postprocess_synthetic_data.py --mapping-file path/to/mapping.csv --preserve-original
  
  # Disable category grouping
  python postprocess_synthetic_data.py --no-grouping
        """
    )
    
    parser.add_argument(
        '--no-grouping', 
        action='store_true', 
        help='Disable category grouping'
    )
    parser.add_argument(
        '--preserve-original', 
        action='store_true', 
        help='Create two versions: with groupCategory field AND replaced categories'
    )
    parser.add_argument(
        '--mapping-file', 
        type=str, 
        help='Path to category mapping file (CSV or JSON)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip backing up existing files (use with caution)'
    )
    
    args = parser.parse_args()
    
    exit(main(
        category_mapping_file=args.mapping_file,
        apply_grouping=not args.no_grouping,
        preserve_original=args.preserve_original,
        skip_backup=args.no_backup
    ))