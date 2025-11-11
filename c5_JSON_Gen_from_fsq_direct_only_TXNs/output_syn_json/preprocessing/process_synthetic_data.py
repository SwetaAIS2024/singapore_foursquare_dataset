#!/usr/bin/env python3
"""
Preprocessing script for synthetic JSON datasets
Converts synthetic user interaction data to trajectory format compatible with validation models

Based on the original FSQ preprocessing notebook but adapted for synthetic JSON data structure
"""

import json
import pandas as pd
import io
import os
from datetime import datetime
from collections import defaultdict, Counter
import argparse

def load_synthetic_data(json_file):
    """Load synthetic data from JSON file"""
    print(f"Loading synthetic data from: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} user profiles")
    return data

def extract_interactions(data):
    """Extract user interactions from synthetic data structure"""
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
    
    print(f"Extracted {len(interactions)} total interactions")
    return interactions

def five_core_filter(df, user_col='user_id', item_col='poi_id', min_interactions=5):
    """Apply 5-core filtering: users and POIs must have at least 5 interactions"""
    print(f"Applying 5-core filtering (min_interactions={min_interactions})")
    print(f"Initial: {len(df)} interactions, {df[user_col].nunique()} users, {df[item_col].nunique()} POIs")
    
    iteration = 0
    while True:
        iteration += 1
        user_counts = df[user_col].value_counts()
        item_counts = df[item_col].value_counts()

        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index

        df_filtered = df[df[user_col].isin(valid_users) & df[item_col].isin(valid_items)]

        print(f"  Iteration {iteration}: {len(df_filtered)} interactions, "
              f"{df_filtered[user_col].nunique()} users, {df_filtered[item_col].nunique()} POIs")

        if len(df_filtered) == len(df):
            break
        df = df_filtered

    print(f"Final after 5-core: {len(df_filtered)} interactions, "
          f"{df_filtered[user_col].nunique()} users, {df_filtered[item_col].nunique()} POIs")
    return df_filtered

def create_poi_dictionary(df):
    """Create POI dictionary with metadata"""
    poi_dict = {}
    
    for _, row in df.drop_duplicates('poi_id').iterrows():
        poi_dict[row['poi_id']] = {
            'name': row['poi_name'],
            'lat': row['lat'],
            'lon': row['lon'],
            'cat': row['category'],
            'planning_area': row['planning_area']
        }
    
    print(f"Created POI dictionary with {len(poi_dict)} unique POIs")
    return poi_dict

def create_user_dictionary(df):
    """Create user dictionary with metadata"""
    user_dict = {}
    
    for _, row in df.drop_duplicates('user_id').iterrows():
        user_dict[row['user_id']] = {
            'age': row['user_age'],
            'gender': row['user_gender'],
            'device': row['user_device']
        }
    
    print(f"Created user dictionary with {len(user_dict)} unique users")
    return user_dict

# COMMENTED OUT: Trajectory segmentation functions (not needed for 5-core filtering only)
# def format_checkins_for_user(group):
#     """Format checkins for a single user - sort by timestamp"""
#     # Sort by timestamp
#     sorted_group = group.sort_values('timestamp')
#     return '|'.join(f"{row.poi_id},{row.lat},{row.lon},{row.timestamp}" 
#                    for row in sorted_group.itertuples())

# def segment_trajectories(user_checkins, min_len=5, max_len=200):
#     """Segment long user trajectories into chunks using notebook method (backward overlapping)"""
#     user_trajs = {}
#     
#     print(f"Segmenting trajectories (min_len={min_len}, max_len={max_len})")
#     
#     for _, row in user_checkins.iterrows():
#         traj_set = []
#         user_id = row['user_id']
#         checkin_list = row['checkins'].split('|')
#         
#         if len(checkin_list) < min_len:
#             continue
#         elif len(checkin_list) > max_len:
#             # Use notebook method: backward overlapping chunks
#             trajs = [checkin_list[max(i - max_len, 0):i] for i in range(len(checkin_list), 0, -max_len)]
#             trajs = trajs[::-1]  # Reverse to get chronological order
#             trajs = [t for t in trajs if len(t) >= min_len]
#             traj_set.extend(trajs)
#         else:
#             traj_set.append(checkin_list)
#         
#         if len(traj_set) > 0:
#             user_trajs[user_id] = traj_set
# 
#     # Statistics
#     num_users = len(user_trajs)
#     num_trajs = sum(len(trajs) for trajs in user_trajs.values())
#     
#     print(f"Segmentation results: {num_users} users, {num_trajs} trajectories")
#     return user_trajs

# def save_trajectory_data(user_trajs, output_dir, dataset_name):
#     """Save trajectory data in format compatible with validation models (notebook method)"""
#     
#     # Main trajectory file - Multiple lines per user (like notebook)
#     traj_file = os.path.join(output_dir, f"{dataset_name}_trajectories.txt")
#     with io.open(traj_file, 'w', encoding='utf8') as f:
#         for user_id in user_trajs:
#             traj_seq = ['|'.join([c for c in t]) for t in user_trajs[user_id]]
#             if len(traj_seq) > 0:
#                 for traj in traj_seq:
#                     f.write(str(user_id) + '\t' + traj + '\n')
#     
#     print(f"Saved trajectories to: {traj_file}")
#     
#     # Statistics file  
#     stats_file = os.path.join(output_dir, f"{dataset_name}_statistics.txt")
#     with io.open(stats_file, 'w', encoding='utf8') as f:
#         length_dist = defaultdict(int)
#         for user_id in user_trajs:
#             for traj in user_trajs[user_id]:
#                 length = len(traj)
#                 length_dist[length] += 1
#         
#         # Write length distribution
#         for length in sorted(length_dist.keys()):
#             freq = length_dist[length]
#             f.write(f"{length}\t{freq}\n")
#     
#     print(f"Saved statistics to: {stats_file}")

def save_poi_data(poi_dict, output_dir, dataset_name):
    """Save POI dictionary in format compatible with validation models"""
    poi_file = os.path.join(output_dir, f"{dataset_name}_poi_data.txt")
    
    with open(poi_file, 'w', encoding='utf-8') as f:
        for poi_id, data in poi_dict.items():
            line = f"{poi_id}\t{data['name']}\t{data['lat']}\t{data['lon']}\t{data['cat']}\t{data['planning_area']}\n"
            f.write(line)
    
    print(f"Saved POI data to: {poi_file}")

def save_user_data(user_dict, output_dir, dataset_name):
    """Save user dictionary"""
    user_file = os.path.join(output_dir, f"{dataset_name}_user_data.txt")
    
    with open(user_file, 'w', encoding='utf-8') as f:
        for user_id, data in user_dict.items():
            line = f"{user_id}\t{data['age']}\t{data['gender']}\t{data['device']}\n"
            f.write(line)
    
    print(f"Saved user data to: {user_file}")

def save_filtered_json(original_data, df_filtered, output_dir, dataset_name):
    """Save filtered data back in original JSON format"""
    
    # Get the set of valid users and POIs after 5-core filtering
    valid_users = set(df_filtered['user_id'].unique())
    valid_pois = set(df_filtered['poi_id'].unique())
    
    print(f"Filtering JSON data: keeping {len(valid_users)} users and {len(valid_pois)} POIs")
    
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
    output_file = os.path.join(output_dir, f"{dataset_name}_5core_filtered.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved filtered JSON to: {output_file}")
    print(f"  Original users: {len(original_data)}")
    print(f"  Filtered users: {len(filtered_data)}")
    print(f"  Retention rate: {len(filtered_data)/len(original_data)*100:.2f}%")

# COMMENTED OUT: Comprehensive summary report with trajectory data  
# def generate_summary_report(df_original, df_filtered, user_trajs, poi_dict, user_dict, output_dir, dataset_name):
#     """Generate comprehensive summary report"""
#     report_file = os.path.join(output_dir, f"{dataset_name}_processing_report.txt")
#     
#     with open(report_file, 'w', encoding='utf-8') as f:
#         f.write(f"PREPROCESSING REPORT FOR {dataset_name.upper()}\n")
#         f.write("=" * 60 + "\n\n")
#         
#         # Original data statistics
#         f.write("ORIGINAL DATA:\n")
#         f.write(f"  Total interactions: {len(df_original):,}\n")
#         f.write(f"  Unique users: {df_original['user_id'].nunique():,}\n")
#         f.write(f"  Unique POIs: {df_original['poi_id'].nunique():,}\n")
#         f.write(f"  Unique categories: {df_original['category'].nunique():,}\n\n")
#         
#         # After 5-core filtering
#         f.write("AFTER 5-CORE FILTERING:\n")
#         f.write(f"  Total interactions: {len(df_filtered):,}\n")
#         f.write(f"  Unique users: {df_filtered['user_id'].nunique():,}\n")
#         f.write(f"  Unique POIs: {df_filtered['poi_id'].nunique():,}\n")
#         f.write(f"  Data retention: {len(df_filtered)/len(df_original)*100:.2f}%\n\n")
#         
#         # Trajectory segmentation
#         total_trajs = sum(len(trajs) for trajs in user_trajs.values())
#         f.write("TRAJECTORY SEGMENTATION:\n")
#         f.write(f"  Users with valid trajectories: {len(user_trajs):,}\n")
#         f.write(f"  Total trajectory segments: {total_trajs:,}\n")
#         f.write(f"  Avg trajectories per user: {total_trajs/len(user_trajs):.2f}\n\n")
#         
#         # Length distribution
#         length_counts = defaultdict(int)
#         for trajs in user_trajs.values():
#             for traj in trajs:
#                 length_counts[len(traj)] += 1
#         
#         f.write("TRAJECTORY LENGTH DISTRIBUTION:\n")
#         for length in sorted(length_counts.keys()):
#             count = length_counts[length]
#             f.write(f"  Length {length:3d}: {count:4d} trajectories\n")
#         
#         f.write(f"\nAverge trajectory length: {sum(l*c for l,c in length_counts.items())/sum(length_counts.values()):.2f}\n\n")
#         
#         # Category distribution
#         f.write("POI CATEGORY DISTRIBUTION:\n")
#         cat_counts = df_filtered['category'].value_counts()
#         for cat, count in cat_counts.head(15).items():
#             f.write(f"  {cat:<30}: {count:5d} ({count/len(df_filtered)*100:5.2f}%)\n")
#         
#         f.write(f"\n  Total categories: {len(cat_counts)}\n")
#     
#     print(f"Generated processing report: {report_file}")

def generate_basic_summary_report(df_original, df_filtered, poi_dict, user_dict, output_dir, dataset_name):
    """Generate basic summary report for 5-core filtering only"""
    report_file = os.path.join(output_dir, f"{dataset_name}_5core_report.txt")
    
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

def process_dataset(input_file, output_dir, dataset_name, min_interactions=5, min_len=5, max_len=200):
    """
    Main processing pipeline - 5-core filtering only
    
    Parameters:
    - min_interactions: 5-core filtering threshold (users and POIs must have ≥5 interactions)
    - min_len: NOT USED (trajectory processing commented out)
    - max_len: NOT USED (trajectory processing commented out)
    """
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {dataset_name.upper()} - 5-CORE FILTERING ONLY")
    print(f"{'='*60}")
    print(f"Applying 5-core filtering with minimum {min_interactions} interactions per user/POI")
    
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
    
    # COMMENTED OUT: Additional preprocessing steps - only keeping 5-core filtering
    # # Format checkins by user
    # print("Formatting user checkins...")
    # user_checkins = df_filtered.groupby('user_id').apply(format_checkins_for_user).reset_index()
    # user_checkins.columns = ['user_id', 'checkins']
    # user_checkins['num_checkins'] = user_checkins['checkins'].apply(lambda x: len(str(x).split('|')))
    # user_checkins = user_checkins.sort_values(by='num_checkins', ascending=False).reset_index(drop=True)
    # 
    # # Segment trajectories
    # user_trajs = segment_trajectories(user_checkins, min_len=min_len, max_len=max_len)
    # 
    # # Save all outputs
    # save_trajectory_data(user_trajs, output_dir, dataset_name)
    # save_poi_data(poi_dict, output_dir, dataset_name)
    # save_user_data(user_dict, output_dir, dataset_name)
    # generate_summary_report(df_original, df_filtered, user_trajs, poi_dict, user_dict, output_dir, dataset_name)
    
    # Save filtered data in original JSON format (main output)
    save_filtered_json(synthetic_data, df_filtered, output_dir, dataset_name)
    
    # Save basic metadata
    save_poi_data(poi_dict, output_dir, dataset_name)
    save_user_data(user_dict, output_dir, dataset_name)
    
    # Generate basic summary report
    generate_basic_summary_report(df_original, df_filtered, poi_dict, user_dict, output_dir, dataset_name)
    
    print(f"\n✅ 5-core filtering complete for {dataset_name}")
    return {
        'df_filtered': df_filtered,
        'poi_dict': poi_dict,
        'user_dict': user_dict
    }

def main():
    """Main execution function"""
    
    # File paths
    base_dir = r"C:\Users\admin\Desktop\sweta\MPS_syn_data_gen\singapore_foursquare_dataset\synthetic_data_v2\c5_JSON_Gen_from_fsq_direct_only_TXNs\output_syn_json"
    
    filtered_file = os.path.join(base_dir, "fsq_to_synthetic_filtered_with_names.json")
    all_categories_file = os.path.join(base_dir, "fsq_to_synthetic_all_categories_with_names.json")
    output_dir = os.path.join(base_dir, "preprocessing")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Processing parameters
    min_interactions = 5  # 5-core filtering
    min_len = 5          # Minimum trajectory length
    max_len = 200        # Maximum trajectory length
    
    print("5-CORE FILTERING PIPELINE FOR SYNTHETIC DATA")
    print("=" * 60)
    print(f"Input files:")
    print(f"  1. Filtered: {filtered_file}")
    print(f"  2. All categories: {all_categories_file}")
    print(f"Output directory: {output_dir}")
    print(f"Parameters: min_interactions={min_interactions}")
    print(f"Processing: 5-core filtering only (trajectory segmentation DISABLED)")
    
    # Process filtered dataset
    if os.path.exists(filtered_file):
        filtered_results = process_dataset(
            filtered_file, 
            output_dir, 
            "filtered",
            min_interactions=min_interactions,
            min_len=min_len,
            max_len=max_len
        )
    else:
        print(f"❌ Filtered file not found: {filtered_file}")
        filtered_results = None
    
    # Process all categories dataset
    if os.path.exists(all_categories_file):
        all_results = process_dataset(
            all_categories_file,
            output_dir,
            "all_categories", 
            min_interactions=min_interactions,
            min_len=min_len,
            max_len=max_len
        )
    else:
        print(f"❌ All categories file not found: {all_categories_file}")
        all_results = None
    
    # Final summary
    print(f"\n{'='*60}")
    print("5-CORE FILTERING PIPELINE COMPLETE")
    print(f"{'='*60}")
    
    if filtered_results:
        print(f"✅ Filtered dataset: {len(filtered_results['df_filtered'])} interactions, "
              f"{filtered_results['df_filtered']['user_id'].nunique()} users, "
              f"{filtered_results['df_filtered']['poi_id'].nunique()} POIs")
        print(f"   📄 Output: filtered_5core_filtered.json")
    
    if all_results:
        print(f"✅ All categories dataset: {len(all_results['df_filtered'])} interactions, "
              f"{all_results['df_filtered']['user_id'].nunique()} users, "
              f"{all_results['df_filtered']['poi_id'].nunique()} POIs")
        print(f"   📄 Output: all_categories_5core_filtered.json")
    
    print(f"\n📂 All output files saved to: {output_dir}")
    print("🎯 5-core filtered JSON datasets ready - same format as input!")

if __name__ == "__main__":
    main()