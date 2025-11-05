"""
Validation Script: POI-Time Graph Generation for Synthetic Dataset
Creates visualizations showing user interactions (views, transactions, reviews) over time
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from c0_Configuration.config_paths import JSON_OUTPUT, JSON_VAL

OUTPUT_DIR = os.path.join(JSON_VAL, "user_poi_time_graphs")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_synthetic_data(file_path):
    """Load synthetic dataset from JSON file"""
    print(f"Loading synthetic data from: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} users")
    return data

def parse_timestamp(timestamp_str):
    """Parse ISO 8601 timestamp string to datetime object"""
    return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

def filter_interactions_by_date_range(interactions, start_date, end_date):
    """Filter interactions to only include those within the specified date range"""
    filtered = []
    for interaction in interactions:
        if start_date <= interaction['timestamp'] <= end_date:
            filtered.append(interaction)
    return filtered

def get_dataset_date_range(data):
    """Get the earliest and latest dates from the entire dataset"""
    all_timestamps = []
    
    for user_data in data:
        # Extract all timestamps from views
        if 'views' in user_data['interaction']:
            for view in user_data['interaction']['views']:
                all_timestamps.append(parse_timestamp(view['timestamp']))
        
        # Extract all timestamps from transactions
        if 'transactions' in user_data['interaction']:
            for txn in user_data['interaction']['transactions']:
                all_timestamps.append(parse_timestamp(txn['timestamp']))
        
        # Extract all timestamps from reviews
        if 'reviews' in user_data['interaction']:
            for review in user_data['interaction']['reviews']:
                all_timestamps.append(parse_timestamp(review['timestamp']))
    
    if all_timestamps:
        return min(all_timestamps), max(all_timestamps)
    return None, None

def extract_user_interactions(user_data):
    """Extract all interactions (views, transactions, reviews) for a user"""
    interactions = []
    
    user_id = user_data['user']['userId']
    
    # Extract views
    if 'views' in user_data['interaction']:
        for view in user_data['interaction']['views']:
            interactions.append({
                'type': 'view',
                'timestamp': parse_timestamp(view['timestamp']),
                'poiId': view['poiId'],
                'poiCategories': view['poiCategories'][0] if view['poiCategories'] else 'Unknown',
                'userLocation': view['userLocation'],
                'duration': view.get('duration', 0)
            })
    
    # Extract transactions
    if 'transactions' in user_data['interaction']:
        for txn in user_data['interaction']['transactions']:
            interactions.append({
                'type': 'transaction',
                'timestamp': parse_timestamp(txn['timestamp']),
                'poiId': txn['poiId'],
                'poiCategories': txn['poiCategories'][0] if txn['poiCategories'] else 'Unknown',
                'userLocation': txn['userLocation'],
                'amount': txn.get('amount', 0)
            })
    
    # Extract reviews
    if 'reviews' in user_data['interaction']:
        for review in user_data['interaction']['reviews']:
            interactions.append({
                'type': 'review',
                'timestamp': parse_timestamp(review['timestamp']),
                'poiId': review['poiId'],
                'poiCategories': review['poiCategories'][0] if review['poiCategories'] else 'Unknown',
                'userLocation': review['userLocation'],
                'rating': review.get('rating', 0)
            })
    
    # Sort by timestamp
    interactions.sort(key=lambda x: x['timestamp'])
    
    return user_id, interactions

def create_funnel_sequences(interactions):
    """Identify complete View → Transaction → Review funnels"""
    funnels = []
    poi_interactions = defaultdict(list)
    
    # Group by POI
    for interaction in interactions:
        poi_interactions[interaction['poiId']].append(interaction)
    
    # Find complete funnels for each POI
    for poi_id, poi_events in poi_interactions.items():
        views = [e for e in poi_events if e['type'] == 'view']
        txns = [e for e in poi_events if e['type'] == 'transaction']
        reviews = [e for e in poi_events if e['type'] == 'review']
        
        # Match each transaction with closest preceding view and following review
        for txn in txns:
            # Find closest view before transaction
            preceding_views = [v for v in views if v['timestamp'] < txn['timestamp']]
            closest_view = max(preceding_views, key=lambda x: x['timestamp']) if preceding_views else None
            
            # Find closest review after transaction
            following_reviews = [r for r in reviews if r['timestamp'] > txn['timestamp']]
            closest_review = min(following_reviews, key=lambda x: x['timestamp']) if following_reviews else None
            
            if closest_view and closest_review:
                view_to_txn_mins = (txn['timestamp'] - closest_view['timestamp']).total_seconds() / 60
                txn_to_review_days = (closest_review['timestamp'] - txn['timestamp']).total_seconds() / 86400
                
                funnels.append({
                    'poi_id': poi_id,
                    'category': txn['poiCategories'],
                    'view': closest_view,
                    'transaction': txn,
                    'review': closest_review,
                    'view_to_txn_mins': view_to_txn_mins,
                    'txn_to_review_days': txn_to_review_days,
                    'complete': True
                })
    
    return funnels

def create_poi_time_graph(user_id, interactions, output_dir):
    """Create enhanced POI-time visualizations for a single user"""
    if not interactions:
        print(f"No interactions for user {user_id}, skipping...")
        return
    
    # Organize interactions by POI
    poi_interactions = defaultdict(lambda: {'view': [], 'transaction': [], 'review': []})
    
    for interaction in interactions:
        poi_id = interaction['poiId']
        interaction_type = interaction['type']
        poi_interactions[poi_id][interaction_type].append(interaction)
    
    # Get unique POIs and assign colors
    unique_pois = list(poi_interactions.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_pois)))
    poi_color_map = {poi: colors[i] for i, poi in enumerate(unique_pois)}
    
    # Identify complete funnels
    funnels = create_funnel_sequences(interactions)
    
    # Create figure with four subplots
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])  # Full width timeline
    ax2 = fig.add_subplot(gs[1, 0])  # Funnel flow diagram
    ax3 = fig.add_subplot(gs[1, 1])  # Temporal analysis
    ax4 = fig.add_subplot(gs[2, :])  # Sankey-style sequence
    
    fig.suptitle(f'User {user_id}: POI Interaction Analysis', fontsize=18, fontweight='bold', y=0.995)
    
    # ========== SUBPLOT 1: Enhanced Timeline with Funnel Arrows ==========
    y_position = 0
    poi_y_positions = {}
    
    for poi_id in unique_pois:
        poi_y_positions[poi_id] = y_position
        poi_data = poi_interactions[poi_id]
        color = poi_color_map[poi_id]
        
        # Get category
        if poi_data['view']:
            category = poi_data['view'][0]['poiCategories']
        elif poi_data['transaction']:
            category = poi_data['transaction'][0]['poiCategories']
        elif poi_data['review']:
            category = poi_data['review'][0]['poiCategories']
        else:
            category = 'Unknown'
        
        # Plot views
        if poi_data['view']:
            timestamps = [v['timestamp'] for v in poi_data['view']]
            ax1.scatter(timestamps, [y_position] * len(timestamps), 
                       marker='o', s=120, c=[color], alpha=0.7, 
                       edgecolors='darkblue', linewidths=1.5, zorder=3)
        
        # Plot transactions
        if poi_data['transaction']:
            timestamps = [t['timestamp'] for t in poi_data['transaction']]
            ax1.scatter(timestamps, [y_position] * len(timestamps), 
                       marker='s', s=180, c=[color], alpha=0.9, 
                       edgecolors='darkgreen', linewidths=2, zorder=4)
        
        # Plot reviews
        if poi_data['review']:
            timestamps = [r['timestamp'] for r in poi_data['review']]
            ax1.scatter(timestamps, [y_position] * len(timestamps), 
                       marker='*', s=250, c=[color], alpha=1.0, 
                       edgecolors='darkorange', linewidths=2, zorder=5)
        
        y_position += 1
    
    # Draw funnel arrows
    for funnel in funnels:
        poi_y = poi_y_positions[funnel['poi_id']]
        view_time = funnel['view']['timestamp']
        txn_time = funnel['transaction']['timestamp']
        review_time = funnel['review']['timestamp']
        
        # View → Transaction arrow
        ax1.annotate('', xy=(txn_time, poi_y), xytext=(view_time, poi_y),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='blue', alpha=0.4))
        
        # Transaction → Review arrow
        ax1.annotate('', xy=(review_time, poi_y), xytext=(txn_time, poi_y),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='orange', alpha=0.4))
    
    # Format subplot 1
    ax1.set_yticks(range(len(unique_pois)))
    ax1.set_yticklabels([f'{poi[:18]}...' for poi in unique_pois], fontsize=9)
    ax1.set_xlabel('Timestamp', fontsize=12, fontweight='bold')
    ax1.set_ylabel('POI ID', fontsize=12, fontweight='bold')
    ax1.set_title('Timeline with Funnel Flows (○ View → ■ Transaction → ★ Review)', 
                 fontsize=13, fontweight='bold', pad=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('#f8f9fa')
    
    # ========== SUBPLOT 2: Funnel Conversion Diagram ==========
    view_times = [i['timestamp'] for i in interactions if i['type'] == 'view']
    txn_times = [i['timestamp'] for i in interactions if i['type'] == 'transaction']
    review_times = [i['timestamp'] for i in interactions if i['type'] == 'review']
    
    funnel_stages = ['Views', 'Transactions', 'Reviews']
    funnel_counts = [len(view_times), len(txn_times), len(review_times)]
    funnel_colors_map = ['#3498db', '#2ecc71', '#f39c12']
    
    # Draw funnel as horizontal bars
    for i, (stage, count, color) in enumerate(zip(funnel_stages, funnel_counts, funnel_colors_map)):
        bar_width = count / max(funnel_counts) if max(funnel_counts) > 0 else 0
        ax2.barh(i, bar_width, height=0.6, color=color, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Add count labels
        if count > 0:
            ax2.text(bar_width/2, i, f'{count}', ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='white')
        
        # Add conversion rate
        if i > 0 and funnel_counts[i-1] > 0:
            conversion = (count / funnel_counts[i-1]) * 100
            ax2.text(bar_width + 0.05, i, f'{conversion:.1f}%', 
                    ha='left', va='center', fontsize=11, fontweight='bold')
    
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(funnel_stages, fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1.2)
    ax2.set_xlabel('Relative Volume', fontsize=12, fontweight='bold')
    ax2.set_title('Funnel Conversion Rates', fontsize=13, fontweight='bold', pad=10)
    ax2.invert_yaxis()
    ax2.set_facecolor('#f8f9fa')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # ========== SUBPLOT 3: Temporal Analysis ==========
    if funnels:
        view_to_txn = [f['view_to_txn_mins'] for f in funnels]
        txn_to_review = [f['txn_to_review_days'] for f in funnels]
        
        # Create dual-axis plot
        ax3_twin = ax3.twinx()
        
        # View to Transaction (minutes)
        ax3.hist(view_to_txn, bins=20, color='blue', alpha=0.6, edgecolor='black', label='View→Txn')
        ax3.set_xlabel('Time Difference', fontsize=12, fontweight='bold')
        ax3.set_ylabel('View → Transaction (minutes)', fontsize=11, fontweight='bold', color='blue')
        ax3.tick_params(axis='y', labelcolor='blue')
        
        # Transaction to Review (days)
        ax3_twin.hist(txn_to_review, bins=20, color='orange', alpha=0.6, edgecolor='black', label='Txn→Review')
        ax3_twin.set_ylabel('Transaction → Review (days)', fontsize=11, fontweight='bold', color='orange')
        ax3_twin.tick_params(axis='y', labelcolor='orange')
        
        ax3.set_title(f'Funnel Timing Distribution (n={len(funnels)} complete funnels)', 
                     fontsize=13, fontweight='bold', pad=10)
        ax3.set_facecolor('#f8f9fa')
        ax3.grid(alpha=0.3, linestyle='--')
        
        # Add median lines and annotations
        if view_to_txn:
            median_v2t = np.median(view_to_txn)
            ax3.axvline(median_v2t, color='darkblue', linestyle='--', linewidth=2)
            ax3.text(median_v2t, ax3.get_ylim()[1]*0.9, f'Median: {median_v2t:.1f}min', 
                    ha='center', fontsize=10, fontweight='bold', 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    else:
        ax3.text(0.5, 0.5, 'No complete funnels found', ha='center', va='center',
                fontsize=14, transform=ax3.transAxes)
        ax3.set_facecolor('#f8f9fa')
    
    # ========== SUBPLOT 4: Sequential Flow Visualization ==========
    # Sort all interactions chronologically
    sorted_interactions = sorted(interactions, key=lambda x: x['timestamp'])
    
    # Assign y-positions based on interaction type
    type_y_map = {'view': 3, 'transaction': 2, 'review': 1}
    
    for i, interaction in enumerate(sorted_interactions):
        y_pos = type_y_map[interaction['type']]
        timestamp = interaction['timestamp']
        poi_id = interaction['poiId']
        color = poi_color_map[poi_id]
        
        if interaction['type'] == 'view':
            ax4.scatter(timestamp, y_pos, marker='o', s=150, c=[color], 
                       alpha=0.7, edgecolors='darkblue', linewidths=1.5, zorder=3)
        elif interaction['type'] == 'transaction':
            ax4.scatter(timestamp, y_pos, marker='s', s=200, c=[color], 
                       alpha=0.9, edgecolors='darkgreen', linewidths=2, zorder=4)
        else:  # review
            ax4.scatter(timestamp, y_pos, marker='*', s=280, c=[color], 
                       alpha=1.0, edgecolors='darkorange', linewidths=2, zorder=5)
        
        # Draw connecting line to next interaction for same POI
        if i < len(sorted_interactions) - 1:
            next_interaction = sorted_interactions[i + 1]
            if next_interaction['poiId'] == poi_id:
                next_y = type_y_map[next_interaction['type']]
                ax4.plot([timestamp, next_interaction['timestamp']], [y_pos, next_y],
                        color=color, alpha=0.3, linewidth=2, zorder=1)
    
    ax4.set_yticks([1, 2, 3])
    ax4.set_yticklabels(['Reviews', 'Transactions', 'Views'], fontsize=12, fontweight='bold')
    ax4.set_xlabel('Timestamp', fontsize=12, fontweight='bold')
    ax4.set_title('Sequential Flow Pattern (Chronological Order)', fontsize=13, fontweight='bold', pad=10)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_ylim(0.5, 3.5)
    ax4.set_facecolor('#f8f9fa')
    
    # Add statistics text box
    stats_text = f"""
    Total Interactions: {len(interactions)}
    • Views: {len(view_times)} ({len(view_times)/len(interactions)*100:.1f}%)
    • Transactions: {len(txn_times)} ({len(txn_times)/len(interactions)*100:.1f}%)
    • Reviews: {len(review_times)} ({len(review_times)/len(interactions)*100:.1f}%)
    
    Complete Funnels: {len(funnels)}
    Unique POIs: {len(unique_pois)}
    """
    fig.text(0.02, 0.02, stats_text.strip(), fontsize=10, 
            verticalalignment='bottom', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.99])
    
    # Save figure
    output_path = os.path.join(output_dir, f'user_{user_id}_enhanced_analysis.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved enhanced graph for user {user_id}")
    
    return {
        'user_id': user_id,
        'total_interactions': len(interactions),
        'num_pois': len(unique_pois),
        'views': len(view_times),
        'transactions': len(txn_times),
        'reviews': len(review_times),
        'complete_funnels': len(funnels)
    }

def generate_summary_statistics(all_stats):
    """Generate and save summary statistics"""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    total_users = len(all_stats)
    total_interactions = sum(s['total_interactions'] for s in all_stats)
    total_views = sum(s['views'] for s in all_stats)
    total_txns = sum(s['transactions'] for s in all_stats)
    total_reviews = sum(s['reviews'] for s in all_stats)
    total_funnels = sum(s['complete_funnels'] for s in all_stats)
    avg_pois_per_user = np.mean([s['num_pois'] for s in all_stats])
    avg_interactions_per_user = np.mean([s['total_interactions'] for s in all_stats])
    avg_funnels_per_user = np.mean([s['complete_funnels'] for s in all_stats])
    
    # Calculate conversion rates
    view_to_txn_rate = (total_txns / total_views * 100) if total_views > 0 else 0
    txn_to_review_rate = (total_reviews / total_txns * 100) if total_txns > 0 else 0
    view_to_review_rate = (total_reviews / total_views * 100) if total_views > 0 else 0
    
    print(f"Total Users: {total_users}")
    print(f"Total Interactions: {total_interactions}")
    print(f"  - Views: {total_views} ({total_views/total_interactions*100:.1f}%)")
    print(f"  - Transactions: {total_txns} ({total_txns/total_interactions*100:.1f}%)")
    print(f"  - Reviews: {total_reviews} ({total_reviews/total_interactions*100:.1f}%)")
    print(f"\nConversion Rates:")
    print(f"  - View → Transaction: {view_to_txn_rate:.2f}%")
    print(f"  - Transaction → Review: {txn_to_review_rate:.2f}%")
    print(f"  - View → Review (complete): {view_to_review_rate:.2f}%")
    print(f"\nComplete Funnels: {total_funnels}")
    print(f"Average POIs per User: {avg_pois_per_user:.2f}")
    print(f"Average Interactions per User: {avg_interactions_per_user:.2f}")
    print(f"Average Complete Funnels per User: {avg_funnels_per_user:.2f}")
    print("="*70)
    
    # Save summary to file
    summary_path = os.path.join(OUTPUT_DIR, 'summary_statistics.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("SYNTHETIC DATASET VALIDATION SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total Users: {total_users}\n")
        f.write(f"Total Interactions: {total_interactions}\n")
        f.write(f"  - Views: {total_views} ({total_views/total_interactions*100:.1f}%)\n")
        f.write(f"  - Transactions: {total_txns} ({total_txns/total_interactions*100:.1f}%)\n")
        f.write(f"  - Reviews: {total_reviews} ({total_reviews/total_interactions*100:.1f}%)\n\n")
        f.write(f"Conversion Rates:\n")
        f.write(f"  - View → Transaction: {view_to_txn_rate:.2f}%\n")
        f.write(f"  - Transaction → Review: {txn_to_review_rate:.2f}%\n")
        f.write(f"  - View → Review (complete): {view_to_review_rate:.2f}%\n\n")
        f.write(f"Complete Funnels: {total_funnels}\n")
        f.write(f"Average POIs per User: {avg_pois_per_user:.2f}\n")
        f.write(f"Average Interactions per User: {avg_interactions_per_user:.2f}\n")
        f.write(f"Average Complete Funnels per User: {avg_funnels_per_user:.2f}\n\n")
        f.write("="*70 + "\n\n")
        f.write("PER-USER DETAILS:\n")
        f.write("-"*70 + "\n")
        for stat in all_stats:
            f.write(f"User {stat['user_id']}: {stat['total_interactions']} interactions ")
            f.write(f"({stat['views']}V, {stat['transactions']}T, {stat['reviews']}R) ")
            f.write(f"across {stat['num_pois']} POIs, {stat['complete_funnels']} complete funnels\n")
    
    print(f"\n✓ Summary saved to: {summary_path}")

def main():
    """Main validation function"""
    print("\n" + "="*70)
    print("POI-TIME GRAPH GENERATION FOR SYNTHETIC DATASET (1 WEEK)")
    print("="*70 + "\n")
    
    # Load data
    if not os.path.exists(JSON_OUTPUT):
        print(f"ERROR: Synthetic data file not found: {JSON_OUTPUT}")
        return
    
    data = load_synthetic_data(JSON_OUTPUT)
    
    # Get dataset date range
    print("Analyzing dataset date range...")
    earliest_date, latest_date = get_dataset_date_range(data)
    
    if not earliest_date or not latest_date:
        print("ERROR: No timestamps found in dataset")
        return
    
    print(f"Dataset spans: {earliest_date.date()} to {latest_date.date()}")
    print(f"Total duration: {(latest_date - earliest_date).days} days")
    
    # Define 1-week window (use the first week of the dataset)
    start_date = earliest_date
    end_date = earliest_date + timedelta(days=7)
    
    print(f"\n🔍 Filtering to 1-week window:")
    print(f"   Start: {start_date.date()} {start_date.time()}")
    print(f"   End:   {end_date.date()} {end_date.time()}")
    
    # Extract all users with their interaction counts in the 1-week window
    print(f"\nAnalyzing user activity in 1-week window...")
    user_activity = []
    
    for user_data in data:
        user_id, interactions = extract_user_interactions(user_data)
        filtered_interactions = filter_interactions_by_date_range(interactions, start_date, end_date)
        
        if filtered_interactions:
            user_activity.append({
                'user_data': user_data,
                'user_id': user_id,
                'interactions': filtered_interactions,
                'total_interactions': len(filtered_interactions)
            })
    
    # Sort by activity and take top 100
    user_activity.sort(key=lambda x: x['total_interactions'], reverse=True)
    top_users = user_activity[:100]
    
    print(f"Found {len(user_activity)} users with data in 1-week window")
    print(f"Selecting top 100 most active users for visualization")
    
    if top_users:
        print(f"\nTop 100 users activity range:")
        print(f"  - Most active: {top_users[0]['total_interactions']} interactions (User {top_users[0]['user_id']})")
        print(f"  - Least active (of top 100): {top_users[-1]['total_interactions']} interactions (User {top_users[-1]['user_id']})")
    
    # Process top 100 users
    print(f"\nGenerating graphs for top 100 users...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    all_stats = []
    
    for idx, user_info in enumerate(top_users, 1):
        user_id = user_info['user_id']
        filtered_interactions = user_info['interactions']
        
        print(f"[{idx}/100] Processing user {user_id} ({len(filtered_interactions)} interactions)...")
        stats = create_poi_time_graph(user_id, filtered_interactions, OUTPUT_DIR)
        all_stats.append(stats)
    
    # Generate summary
    if all_stats:
        generate_summary_statistics(all_stats)
    
    print(f"\n{'='*70}")
    print(f"FILTERING SUMMARY:")
    print(f"  - Total users in dataset: {len(data)}")
    print(f"  - Users with data in 1-week window: {len(user_activity)}")
    print(f"  - Top users selected for visualization: {len(all_stats)}")
    print(f"  - Total graphs generated: {len(all_stats)}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
