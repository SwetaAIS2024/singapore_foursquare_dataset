"""
Dataset Comparison Script: Input Foursquare vs Synthetic Output
Compares statistics per user and per POI-ID between the two datasets
"""

import json
import os
from collections import defaultdict
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# File paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up 3 levels: u5_comparison_syn_input -> utils -> c5_JSON_Dataset_Gen_from_fsq_direct -> synthetic_data_v2
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

INPUT_JSON = os.path.join(BASE_DIR, "json_gen", "input_fsq_json", "input.json")
OUTPUT_JSON = os.path.join(BASE_DIR, "json_gen", "output_syn_json", "fsq_to_synthetic.json")
COMPARISON_DIR = SCRIPT_DIR  # Already in u5_comparison_syn_input folder

os.makedirs(COMPARISON_DIR, exist_ok=True)

def load_json(file_path):
    """Load JSON file"""
    print(f"Loading: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"  ✓ Loaded {len(data)} records")
    return data

def extract_input_stats(input_data):
    """Extract statistics from input Foursquare dataset"""
    print("\nAnalyzing INPUT dataset...")
    
    user_stats = {}
    poi_stats = defaultdict(lambda: {'total_checkins': 0, 'unique_users': set(), 'categories': set()})
    
    for user_record in input_data:
        user_id = user_record['user_id']
        checkins = user_record['user_metadata']
        
        # Per-user stats
        user_stats[user_id] = {
            'total_checkins': len(checkins),
            'unique_pois': len(set(c['poi_id'] for c in checkins)),
            'poi_visits': defaultdict(int)
        }
        
        for checkin in checkins:
            poi_id = checkin['poi_id']
            user_stats[user_id]['poi_visits'][poi_id] += 1
            
            # Per-POI stats
            poi_stats[poi_id]['total_checkins'] += 1
            poi_stats[poi_id]['unique_users'].add(user_id)
            poi_stats[poi_id]['categories'].add(checkin.get('poi_category', 'Unknown'))
    
    # Convert sets to counts
    for poi_id in poi_stats:
        poi_stats[poi_id]['unique_users'] = len(poi_stats[poi_id]['unique_users'])
        poi_stats[poi_id]['categories'] = list(poi_stats[poi_id]['categories'])
    
    print(f"  ✓ Analyzed {len(user_stats)} users")
    print(f"  ✓ Analyzed {len(poi_stats)} POIs")
    
    return user_stats, dict(poi_stats)

def extract_output_stats(output_data):
    """Extract statistics from synthetic output dataset"""
    print("\nAnalyzing OUTPUT dataset...")
    
    user_stats = {}
    poi_stats = defaultdict(lambda: {
        'total_interactions': 0, 
        'views': 0, 
        'transactions': 0, 
        'reviews': 0,
        'unique_users': set(), 
        'categories': set()
    })
    
    for user_record in output_data:
        user_id = user_record['user']['userId']
        interactions = user_record['interaction']
        
        # Count interactions by type
        n_views = len(interactions.get('views', []))
        n_transactions = len(interactions.get('transactions', []))
        n_reviews = len(interactions.get('reviews', []))
        total_interactions = n_views + n_transactions + n_reviews
        
        # Collect all POIs
        all_pois = set()
        poi_visits = defaultdict(lambda: {'views': 0, 'transactions': 0, 'reviews': 0})
        
        for view in interactions.get('views', []):
            poi_id = view['poiId']
            all_pois.add(poi_id)
            poi_visits[poi_id]['views'] += 1
            poi_stats[poi_id]['views'] += 1
            poi_stats[poi_id]['total_interactions'] += 1
            poi_stats[poi_id]['unique_users'].add(user_id)
            if view.get('poiCategories'):
                poi_stats[poi_id]['categories'].add(view['poiCategories'][0])
        
        for txn in interactions.get('transactions', []):
            poi_id = txn['poiId']
            all_pois.add(poi_id)
            poi_visits[poi_id]['transactions'] += 1
            poi_stats[poi_id]['transactions'] += 1
            poi_stats[poi_id]['total_interactions'] += 1
            poi_stats[poi_id]['unique_users'].add(user_id)
            if txn.get('poiCategories'):
                poi_stats[poi_id]['categories'].add(txn['poiCategories'][0])
        
        for review in interactions.get('reviews', []):
            poi_id = review['poiId']
            all_pois.add(poi_id)
            poi_visits[poi_id]['reviews'] += 1
            poi_stats[poi_id]['reviews'] += 1
            poi_stats[poi_id]['total_interactions'] += 1
            poi_stats[poi_id]['unique_users'].add(user_id)
            if review.get('poiCategories'):
                poi_stats[poi_id]['categories'].add(review['poiCategories'][0])
        
        # Per-user stats
        user_stats[user_id] = {
            'total_interactions': total_interactions,
            'views': n_views,
            'transactions': n_transactions,
            'reviews': n_reviews,
            'unique_pois': len(all_pois),
            'poi_visits': dict(poi_visits)
        }
    
    # Convert sets to counts
    for poi_id in poi_stats:
        poi_stats[poi_id]['unique_users'] = len(poi_stats[poi_id]['unique_users'])
        poi_stats[poi_id]['categories'] = list(poi_stats[poi_id]['categories'])
    
    print(f"  ✓ Analyzed {len(user_stats)} users")
    print(f"  ✓ Analyzed {len(poi_stats)} POIs")
    
    return user_stats, dict(poi_stats)

def compare_user_stats(input_stats, output_stats):
    """Compare per-user statistics"""
    print("\n" + "="*70)
    print("USER-LEVEL COMPARISON")
    print("="*70)
    
    comparison_data = []
    
    for user_id in input_stats:
        if user_id not in output_stats:
            print(f"WARNING: User {user_id} in input but not in output")
            continue
        
        inp = input_stats[user_id]
        out = output_stats[user_id]
        
        comparison_data.append({
            'user_id': user_id,
            'input_checkins': inp['total_checkins'],
            'output_interactions': out['total_interactions'],
            'output_views': out['views'],
            'output_transactions': out['transactions'],
            'output_reviews': out['reviews'],
            'input_unique_pois': inp['unique_pois'],
            'output_unique_pois': out['unique_pois'],
            'interaction_diff': out['total_interactions'] - inp['total_checkins'],
            'poi_diff': out['unique_pois'] - inp['unique_pois']
        })
    
    df = pd.DataFrame(comparison_data)
    
    print(f"\nTotal Users Compared: {len(df)}")
    print(f"\nInteraction Count Comparison:")
    print(f"  Input Avg Checkins:         {df['input_checkins'].mean():.2f} ± {df['input_checkins'].std():.2f}")
    print(f"  Output Avg Interactions:    {df['output_interactions'].mean():.2f} ± {df['output_interactions'].std():.2f}")
    print(f"  Difference:                 {df['interaction_diff'].mean():.2f} ± {df['interaction_diff'].std():.2f}")
    
    print(f"\nPOI Count Comparison:")
    print(f"  Input Avg Unique POIs:      {df['input_unique_pois'].mean():.2f} ± {df['input_unique_pois'].std():.2f}")
    print(f"  Output Avg Unique POIs:     {df['output_unique_pois'].mean():.2f} ± {df['output_unique_pois'].std():.2f}")
    print(f"  Difference:                 {df['poi_diff'].mean():.2f} ± {df['poi_diff'].std():.2f}")
    
    print(f"\nInteraction Type Distribution:")
    print(f"  Views:        {df['output_views'].sum()} ({df['output_views'].sum()/df['output_interactions'].sum()*100:.1f}%)")
    print(f"  Transactions: {df['output_transactions'].sum()} ({df['output_transactions'].sum()/df['output_interactions'].sum()*100:.1f}%)")
    print(f"  Reviews:      {df['output_reviews'].sum()} ({df['output_reviews'].sum()/df['output_interactions'].sum()*100:.1f}%)")
    
    # Check preservation ratio
    exact_match = (df['interaction_diff'] == 0).sum()
    print(f"\nExact Match (same # interactions): {exact_match}/{len(df)} ({exact_match/len(df)*100:.1f}%)")
    
    return df

def compare_poi_stats(input_stats, output_stats):
    """Compare per-POI statistics"""
    print("\n" + "="*70)
    print("POI-LEVEL COMPARISON")
    print("="*70)
    
    comparison_data = []
    
    for poi_id in input_stats:
        if poi_id not in output_stats:
            print(f"WARNING: POI {poi_id} in input but not in output")
            continue
        
        inp = input_stats[poi_id]
        out = output_stats[poi_id]
        
        comparison_data.append({
            'poi_id': poi_id,
            'input_checkins': inp['total_checkins'],
            'output_interactions': out['total_interactions'],
            'output_views': out['views'],
            'output_transactions': out['transactions'],
            'output_reviews': out['reviews'],
            'input_unique_users': inp['unique_users'],
            'output_unique_users': out['unique_users'],
            'interaction_diff': out['total_interactions'] - inp['total_checkins'],
            'user_diff': out['unique_users'] - inp['unique_users']
        })
    
    df = pd.DataFrame(comparison_data)
    
    print(f"\nTotal POIs Compared: {len(df)}")
    print(f"\nInteraction Count Comparison:")
    print(f"  Input Avg Checkins:         {df['input_checkins'].mean():.2f} ± {df['input_checkins'].std():.2f}")
    print(f"  Output Avg Interactions:    {df['output_interactions'].mean():.2f} ± {df['output_interactions'].std():.2f}")
    print(f"  Difference:                 {df['interaction_diff'].mean():.2f} ± {df['interaction_diff'].std():.2f}")
    
    print(f"\nUser Count Comparison:")
    print(f"  Input Avg Unique Users:     {df['input_unique_users'].mean():.2f} ± {df['input_unique_users'].std():.2f}")
    print(f"  Output Avg Unique Users:    {df['output_unique_users'].mean():.2f} ± {df['output_unique_users'].std():.2f}")
    print(f"  Difference:                 {df['user_diff'].mean():.2f} ± {df['user_diff'].std():.2f}")
    
    # Check preservation ratio
    exact_match = (df['interaction_diff'] == 0).sum()
    print(f"\nExact Match (same # interactions): {exact_match}/{len(df)} ({exact_match/len(df)*100:.1f}%)")
    
    return df

def create_visualizations(user_df, poi_df):
    """Create comparison visualizations"""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Input vs Synthetic Dataset Comparison', fontsize=16, fontweight='bold')
    
    # 1. User interaction count scatter
    ax = axes[0, 0]
    ax.scatter(user_df['input_checkins'], user_df['output_interactions'], alpha=0.5, s=20)
    ax.plot([0, user_df['input_checkins'].max()], [0, user_df['input_checkins'].max()], 
            'r--', label='Perfect match', linewidth=2)
    ax.set_xlabel('Input Checkins', fontweight='bold')
    ax.set_ylabel('Output Interactions', fontweight='bold')
    ax.set_title('User: Checkin Count Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. User interaction difference distribution
    ax = axes[0, 1]
    ax.hist(user_df['interaction_diff'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero diff')
    ax.set_xlabel('Difference (Output - Input)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('User: Interaction Count Difference')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. User POI count comparison
    ax = axes[0, 2]
    ax.scatter(user_df['input_unique_pois'], user_df['output_unique_pois'], alpha=0.5, s=20, color='green')
    ax.plot([0, user_df['input_unique_pois'].max()], [0, user_df['input_unique_pois'].max()], 
            'r--', label='Perfect match', linewidth=2)
    ax.set_xlabel('Input Unique POIs', fontweight='bold')
    ax.set_ylabel('Output Unique POIs', fontweight='bold')
    ax.set_title('User: Unique POI Count Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. POI interaction count scatter
    ax = axes[1, 0]
    ax.scatter(poi_df['input_checkins'], poi_df['output_interactions'], alpha=0.5, s=20, color='orange')
    ax.plot([0, poi_df['input_checkins'].max()], [0, poi_df['input_checkins'].max()], 
            'r--', label='Perfect match', linewidth=2)
    ax.set_xlabel('Input Checkins', fontweight='bold')
    ax.set_ylabel('Output Interactions', fontweight='bold')
    ax.set_title('POI: Checkin Count Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 5. POI interaction difference distribution
    ax = axes[1, 1]
    ax.hist(poi_df['interaction_diff'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero diff')
    ax.set_xlabel('Difference (Output - Input)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('POI: Interaction Count Difference')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 6. Interaction type breakdown (pie chart)
    ax = axes[1, 2]
    interaction_types = ['Views', 'Transactions', 'Reviews']
    interaction_counts = [
        user_df['output_views'].sum(),
        user_df['output_transactions'].sum(),
        user_df['output_reviews'].sum()
    ]
    colors = ['#3498db', '#2ecc71', '#f39c12']
    ax.pie(interaction_counts, labels=interaction_types, autopct='%1.1f%%', 
           colors=colors, startangle=90)
    ax.set_title('Synthetic Dataset: Interaction Type Distribution')
    
    plt.tight_layout()
    output_path = os.path.join(COMPARISON_DIR, 'comparison_plots.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved plots: {output_path}")
    plt.close()

def save_comparison_reports(user_df, poi_df):
    """Save detailed comparison reports"""
    print("\n" + "="*70)
    print("SAVING COMPARISON REPORTS")
    print("="*70)
    
    # User-level report
    user_report_path = os.path.join(COMPARISON_DIR, 'user_comparison.csv')
    user_df.to_csv(user_report_path, index=False)
    print(f"  ✓ Saved user comparison: {user_report_path}")
    
    # POI-level report
    poi_report_path = os.path.join(COMPARISON_DIR, 'poi_comparison.csv')
    poi_df.to_csv(poi_report_path, index=False)
    print(f"  ✓ Saved POI comparison: {poi_report_path}")
    
    # Summary report
    summary_path = os.path.join(COMPARISON_DIR, 'comparison_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("INPUT vs SYNTHETIC DATASET COMPARISON SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("USER-LEVEL STATISTICS:\n")
        f.write("-"*70 + "\n")
        f.write(f"Total Users: {len(user_df)}\n\n")
        f.write(f"Interaction Counts:\n")
        f.write(f"  Input  - Mean: {user_df['input_checkins'].mean():.2f}, Std: {user_df['input_checkins'].std():.2f}\n")
        f.write(f"  Output - Mean: {user_df['output_interactions'].mean():.2f}, Std: {user_df['output_interactions'].std():.2f}\n")
        f.write(f"  Correlation: {user_df['input_checkins'].corr(user_df['output_interactions']):.4f}\n\n")
        
        f.write(f"Unique POI Counts:\n")
        f.write(f"  Input  - Mean: {user_df['input_unique_pois'].mean():.2f}, Std: {user_df['input_unique_pois'].std():.2f}\n")
        f.write(f"  Output - Mean: {user_df['output_unique_pois'].mean():.2f}, Std: {user_df['output_unique_pois'].std():.2f}\n")
        f.write(f"  Correlation: {user_df['input_unique_pois'].corr(user_df['output_unique_pois']):.4f}\n\n")
        
        exact_match = (user_df['interaction_diff'] == 0).sum()
        f.write(f"Exact Preservation: {exact_match}/{len(user_df)} ({exact_match/len(user_df)*100:.1f}%)\n\n")
        
        f.write("\nPOI-LEVEL STATISTICS:\n")
        f.write("-"*70 + "\n")
        f.write(f"Total POIs: {len(poi_df)}\n\n")
        f.write(f"Interaction Counts:\n")
        f.write(f"  Input  - Mean: {poi_df['input_checkins'].mean():.2f}, Std: {poi_df['input_checkins'].std():.2f}\n")
        f.write(f"  Output - Mean: {poi_df['output_interactions'].mean():.2f}, Std: {poi_df['output_interactions'].std():.2f}\n")
        f.write(f"  Correlation: {poi_df['input_checkins'].corr(poi_df['output_interactions']):.4f}\n\n")
        
        f.write(f"Unique User Counts:\n")
        f.write(f"  Input  - Mean: {poi_df['input_unique_users'].mean():.2f}, Std: {poi_df['input_unique_users'].std():.2f}\n")
        f.write(f"  Output - Mean: {poi_df['output_unique_users'].mean():.2f}, Std: {poi_df['output_unique_users'].std():.2f}\n")
        f.write(f"  Correlation: {poi_df['input_unique_users'].corr(poi_df['output_unique_users']):.4f}\n\n")
        
        exact_match = (poi_df['interaction_diff'] == 0).sum()
        f.write(f"Exact Preservation: {exact_match}/{len(poi_df)} ({exact_match/len(poi_df)*100:.1f}%)\n\n")
        
        f.write("\nINTERACTION TYPE DISTRIBUTION:\n")
        f.write("-"*70 + "\n")
        total = user_df['output_interactions'].sum()
        f.write(f"Views:        {user_df['output_views'].sum()} ({user_df['output_views'].sum()/total*100:.1f}%)\n")
        f.write(f"Transactions: {user_df['output_transactions'].sum()} ({user_df['output_transactions'].sum()/total*100:.1f}%)\n")
        f.write(f"Reviews:      {user_df['output_reviews'].sum()} ({user_df['output_reviews'].sum()/total*100:.1f}%)\n")
        f.write(f"Total:        {total}\n")
    
    print(f"  ✓ Saved summary: {summary_path}")

def main():
    """Main comparison function"""
    print("\n" + "="*70)
    print("DATASET SIMILARITY COMPARISON")
    print("Input FSQ vs Synthetic Output")
    print("="*70 + "\n")
    
    # Load datasets
    input_data = load_json(INPUT_JSON)
    output_data = load_json(OUTPUT_JSON)
    
    # Extract statistics
    input_user_stats, input_poi_stats = extract_input_stats(input_data)
    output_user_stats, output_poi_stats = extract_output_stats(output_data)
    
    # Compare statistics
    user_comparison_df = compare_user_stats(input_user_stats, output_user_stats)
    poi_comparison_df = compare_poi_stats(input_poi_stats, output_poi_stats)
    
    # Create visualizations
    create_visualizations(user_comparison_df, poi_comparison_df)
    
    # Save reports
    save_comparison_reports(user_comparison_df, poi_comparison_df)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print(f"Results saved to: {COMPARISON_DIR}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()