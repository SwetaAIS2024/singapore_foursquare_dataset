# -*- coding: utf-8 -*-
"""
Synthetic Foursquare Dataset Generation with Enhanced Funnel Logic
Fixes applied:
- Index-based view-transaction-review pairing (no POI-based matching)
- Proper handling of POI revisits in validation
- Realistic temporal delays (20-90 min V->T, 1-7 days T->R)
"""
import json
import random
import pandas as pd 
from collections import defaultdict
from datetime import datetime, timedelta
import math
import calendar
from tqdm import tqdm
import sys
import os

# Add parent directory to path for imports
from c0_Configuration.config_paths import (
    JSON_INPUT, JSON_OUTPUT, POST_SAMPLING_ANALYSIS_OUTPUT_DIR, JSON_FUNC
)

# Import ML-based temporal generator (ENHANCED VERSION)
try:
    from temporal_pattern_learner_enhanced import (
        EnhancedTemporalPatternLearner, 
        EnhancedTimeSeriesGenerator, 
        train_enhanced_model
    )
    ML_TEMPORAL_AVAILABLE = True
    print("✅ Using Enhanced Temporal Pattern Learner")
except ImportError:
    # Fallback to original version
    try:
        from temporal_pattern_learner import (
            TemporalPatternLearner as EnhancedTemporalPatternLearner,
            RealisticTimeSeriesGenerator as EnhancedTimeSeriesGenerator,
            train_temporal_model as train_enhanced_model
        )
        ML_TEMPORAL_AVAILABLE = True
        print("⚠️  Using original temporal learner (enhanced not found)")
    except ImportError:
        print("WARNING: ML temporal generator not available, will train on first run")
        ML_TEMPORAL_AVAILABLE = False
        EnhancedTemporalPatternLearner = None
        EnhancedTimeSeriesGenerator = None
        train_enhanced_model = None

# -----------------------------
# Helper Functions
# -----------------------------

def jitter_location(lat, lon, distance_km=2):
    max_deg = distance_km / 111.0
    lat_jitter = random.uniform(-max_deg, max_deg)
    lon_jitter = random.uniform(-max_deg, max_deg) / math.cos(math.radians(lat))
    return round(lat + lat_jitter, 6), round(lon + lon_jitter, 6)

def generate_user_home_location(planning_area_lat, planning_area_lon):
    """
    Generate a consistent 'home' location for a user within their planning area.
    This represents where they live and where views/reviews typically happen.
    """
    # Small jitter within 1-3km of planning area center (residential area)
    return jitter_location(planning_area_lat, planning_area_lon, distance_km=random.uniform(1, 3))

def get_view_location(user_home_lat, user_home_lon, poi_lat, poi_lon):
    """
    View happens from home, work, or nearby location (not at the POI).
    80% from home, 20% from other location within 5km of home.
    """
    if random.random() < 0.8:
        # View from home (with small jitter ~100m for GPS noise)
        return jitter_location(user_home_lat, user_home_lon, distance_km=0.1)
    else:
        # View from somewhere else nearby (work, friend's place, etc.)
        return jitter_location(user_home_lat, user_home_lon, distance_km=random.uniform(2, 5))

def get_review_location(user_home_lat, user_home_lon):
    """
    Review happens from home (90% of the time) with small GPS noise.
    """
    if random.random() < 0.9:
        return jitter_location(user_home_lat, user_home_lon, distance_km=0.1)
    else:
        # Occasional review from nearby location
        return jitter_location(user_home_lat, user_home_lon, distance_km=random.uniform(1, 3))

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two lat/lon points in km."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def random_payment():
    return random.choice(["credit_card", "mobile_wallet", "cash"])

def random_rating():
    return round(random.uniform(3.5, 5.0), 1)

def random_amount(category):
    base = {
        "restaurant": 20,
        "nightclub": 35,
        "pub": 25,
        "coffee shop": 6,
        "dim sum restaurant": 15
    }
    return round(random.uniform(0.8, 1.2) * base.get(category.lower(), 12), 2)

def generate_review_text(user_id, poi_id, category):
    return "Review text placeholder."

# ✅ FIX 5: Enforce category transition logic for realistic sequences
def reorder_visits_by_category_transitions(visits, temporal_generator):
    """
    Reorder visits to follow learned category transition patterns.
    This creates realistic visit sequences (e.g., dinner → bar → late-night food).
    
    Args:
        visits: List of visit dictionaries
        temporal_generator: Has category_transitions pattern
    
    Returns:
        Reordered visits list
    """
    if not visits or len(visits) <= 1:
        return visits
    
    # Check if we have category transitions
    if not temporal_generator or 'category_transitions' not in temporal_generator.patterns:
        return visits
    
    transitions = temporal_generator.patterns.get('category_transitions', {})
    if not transitions:
        return visits
    
    # Build sequence using Markov chain
    reordered = []
    remaining = visits.copy()
    
    # Start with first visit
    current_visit = remaining.pop(0)
    reordered.append(current_visit)
    current_category = current_visit.get('poi_category', '')
    
    # Build chain
    while remaining:
        # Get possible next categories from transitions
        next_category_probs = transitions.get(current_category, {})
        
        if next_category_probs:
            # Find visits matching preferred next categories
            candidates = []
            for i, visit in enumerate(remaining):
                visit_cat = visit.get('poi_category', '')
                if visit_cat in next_category_probs:
                    prob = next_category_probs[visit_cat]
                    candidates.append((i, visit, prob))
            
            if candidates:
                # Weighted random selection
                total_prob = sum(c[2] for c in candidates)
                rand_val = random.random() * total_prob
                cumsum = 0
                selected_idx = 0
                for i, visit, prob in candidates:
                    cumsum += prob
                    if cumsum >= rand_val:
                        selected_idx = i
                        break
                
                next_visit = remaining.pop(selected_idx)
                reordered.append(next_visit)
                current_category = next_visit.get('poi_category', '')
            else:
                # No matching category, take next one
                next_visit = remaining.pop(0)
                reordered.append(next_visit)
                current_category = next_visit.get('poi_category', '')
        else:
            # No transitions for current category, take next
            next_visit = remaining.pop(0)
            reordered.append(next_visit)
            current_category = next_visit.get('poi_category', '')
    
    return reordered

# ✅ FIX: Funnel Order Validation
def validate_funnel_order(users_data, verbose=True):
    """
    Validate that funnel order is maintained: View → Transaction → Review
    
    Returns:
    - dict with validation metrics
    """
    stats = {
        'total_users': len(users_data),
        'funnel_violations': [],
        'temporal_stats': {
            'view_to_txn_minutes': [],
            'txn_to_review_days': []
        },
        'spatial_stats': {
            'view_distance_from_poi_km': [],
            'review_distance_from_poi_km': []
        }
    }
    
    for user_data in users_data:
        interaction = user_data.get('interaction', {})
        views = interaction.get('views', [])
        transactions = interaction.get('transactions', [])
        reviews = interaction.get('reviews', [])
        
        # Build POI → timestamps map
        poi_to_view = {}
        poi_to_txn = {}
        poi_to_review = {}
        
        for view in views:
            poi_id = view['poiId']
            timestamp = datetime.strptime(view['timestamp'], "%Y-%m-%dT%H:%M:%SZ")
            if poi_id not in poi_to_view:
                poi_to_view[poi_id] = []
            poi_to_view[poi_id].append({'timestamp': timestamp, 'location': view['userLocation']})
        
        for txn in transactions:
            poi_id = txn['poiId']
            timestamp = datetime.strptime(txn['timestamp'], "%Y-%m-%dT%H:%M:%SZ")
            if poi_id not in poi_to_txn:
                poi_to_txn[poi_id] = []
            poi_to_txn[poi_id].append({'timestamp': timestamp, 'location': txn['userLocation']})
        
        for review in reviews:
            poi_id = review['poiId']
            timestamp = datetime.strptime(review['timestamp'], "%Y-%m-%dT%H:%M:%SZ")
            if poi_id not in poi_to_review:
                poi_to_review[poi_id] = []
            poi_to_review[poi_id].append({'timestamp': timestamp, 'location': review['userLocation']})
        
        # Validate order for each transaction
        for poi_id in poi_to_txn:
            # Handle multiple transactions per POI (revisits)
            for txn_data in poi_to_txn[poi_id]:
                txn_time = txn_data['timestamp']
                txn_loc = txn_data['location']
                
                # Check if transaction had a view
                if poi_id not in poi_to_view:
                    stats['funnel_violations'].append({
                        'user': user_data['user']['userId'],
                        'poi': poi_id,
                        'violation': 'Transaction without View'
                    })
                    continue
                
                # Find most recent view before transaction
                valid_views = [v for v in poi_to_view[poi_id] if v['timestamp'] < txn_time]
                if not valid_views:
                    stats['funnel_violations'].append({
                        'user': user_data['user']['userId'],
                        'poi': poi_id,
                        'violation': 'View after Transaction'
                    })
                    continue
                
                most_recent_view = max(valid_views, key=lambda x: x['timestamp'])
                view_time = most_recent_view['timestamp']
                view_loc = most_recent_view['location']
                
                # Calculate temporal delta: view → transaction
                delta_minutes = (txn_time - view_time).total_seconds() / 60
                stats['temporal_stats']['view_to_txn_minutes'].append(delta_minutes)
                
                # Calculate spatial distance: view location vs POI location
                view_distance = haversine_distance(
                    view_loc['latitude'], view_loc['longitude'],
                    txn_loc['latitude'], txn_loc['longitude']
                )
                stats['spatial_stats']['view_distance_from_poi_km'].append(view_distance)
                
                # Check review order: find review that matches THIS transaction
                if poi_id in poi_to_review:
                    # Find review that comes after this transaction (closest one)
                    valid_reviews = [r for r in poi_to_review[poi_id] if r['timestamp'] > txn_time]
                    
                    if valid_reviews:
                        # Use the closest review after this transaction
                        closest_review = min(valid_reviews, key=lambda x: x['timestamp'])
                        review_time = closest_review['timestamp']
                        review_loc = closest_review['location']
                        
                        # Calculate temporal delta: transaction → review
                        delta_days = (review_time - txn_time).total_seconds() / 86400
                        stats['temporal_stats']['txn_to_review_days'].append(delta_days)
                        
                        # Calculate spatial distance: review location vs POI location
                        review_distance = haversine_distance(
                            review_loc['latitude'], review_loc['longitude'],
                            txn_loc['latitude'], txn_loc['longitude']
                        )
                        stats['spatial_stats']['review_distance_from_poi_km'].append(review_distance)
                    else:
                        # All reviews for this POI are before this transaction - violation
                        for review_data in poi_to_review[poi_id]:
                            if review_data['timestamp'] <= txn_time:
                                stats['funnel_violations'].append({
                                    'user': user_data['user']['userId'],
                                    'poi': poi_id,
                                    'violation': 'Review before/at Transaction time'
                                })
                                break  # Only report once per transaction
    
    # Calculate summary statistics
    import numpy as np
    summary = {
        'total_users': stats['total_users'],
        'total_violations': len(stats['funnel_violations']),
        'violation_rate': len(stats['funnel_violations']) / max(1, stats['total_users']),
        'temporal_realism': {
            'view_to_txn_median_minutes': float(np.median(stats['temporal_stats']['view_to_txn_minutes'])) if stats['temporal_stats']['view_to_txn_minutes'] else 0,
            'txn_to_review_median_days': float(np.median(stats['temporal_stats']['txn_to_review_days'])) if stats['temporal_stats']['txn_to_review_days'] else 0,
        },
        'spatial_realism': {
            'view_median_distance_km': float(np.median(stats['spatial_stats']['view_distance_from_poi_km'])) if stats['spatial_stats']['view_distance_from_poi_km'] else 0,
            'review_median_distance_km': float(np.median(stats['spatial_stats']['review_distance_from_poi_km'])) if stats['spatial_stats']['review_distance_from_poi_km'] else 0,
        }
    }
    
    if verbose:
        print("\n" + "="*60)
        print("FUNNEL ORDER VALIDATION RESULTS")
        print("="*60)
        print(f"Total Users: {summary['total_users']}")
        print(f"Total Violations: {summary['total_violations']}")
        print(f"Violation Rate: {summary['violation_rate']:.2%}")
        print("\nTemporal Realism:")
        print(f"  View → Transaction: {summary['temporal_realism']['view_to_txn_median_minutes']:.1f} minutes (expect: 20-90 min, avg ~55 min)")
        print(f"  Transaction → Review: {summary['temporal_realism']['txn_to_review_median_days']:.1f} days (expect: 1-7 days)")
        print("\nSpatial Realism:")
        print(f"  View distance from POI: {summary['spatial_realism']['view_median_distance_km']:.2f} km (expect: 1-5 km)")
        print(f"  Review distance from POI: {summary['spatial_realism']['review_median_distance_km']:.2f} km (expect: 1-5 km)")
        
        if stats['funnel_violations']:
            print(f"\nFirst 5 violations:")
            for v in stats['funnel_violations'][:5]:
                print(f"  - User {v['user']}, POI {v['poi']}: {v['violation']}")
            
            # DEBUG: Count violation types
            violation_types = defaultdict(int)
            for v in stats['funnel_violations']:
                violation_types[v['violation']] += 1
            print(f"\nViolation breakdown:")
            for vtype, count in sorted(violation_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {vtype}: {count} ({100*count/len(stats['funnel_violations']):.1f}%)")
    
    return summary, stats

# ✅ ML-BASED TEMPORAL GENERATION (replaces old random logic)

def generate_ml_temporal_timestamps(visits, temporal_generator, base_date, num_days=180):
    """
    Generate timestamps using ML-learned temporal patterns.
    
    Args:
        visits: List of visit dictionaries
        temporal_generator: RealisticTimeSeriesGenerator instance
        base_date: Starting datetime
        num_days: Days to spread visits over
    
    Returns:
        dict: Mapping visit_idx -> (datetime, day_of_week, time_of_day)
    """
    num_visits = len(visits)
    
    # Generate realistic temporal sequence
    sequence = temporal_generator.generate_temporal_sequence(
        num_visits, base_date, num_days
    )
    
    # Map visit index to timestamp
    visit_timestamps = {}
    for visit_idx, dt, day_of_week in sequence:
        if visit_idx < len(visits):
            # Apply category-specific time adjustment
            category = visits[visit_idx].get('poi_category', 'Unknown')
            adjusted_dt = temporal_generator.generate_category_specific_time(
                category, dt
            )
            
            time_of_day = adjusted_dt.strftime("%I:%M %p")
            visit_timestamps[visit_idx] = (adjusted_dt, day_of_week, time_of_day)
    
    return visit_timestamps


def generate_temporal_timestamp(base_date, day_offset, time_of_day="12:00 PM", jitter_hours=0, jitter_minutes=0):
    """
    ✅ FALLBACK: Generate timestamp spread across time period (used if ML model unavailable)
    
    Args:
        base_date: Starting datetime (e.g., datetime.now() - timedelta(days=180))
        day_offset: Days to add from base_date (0 to num_days)
        time_of_day: Time string like "02:30 PM"
        jitter_hours: Random hours variation
        jitter_minutes: Random minutes variation
    
    Returns:
        ISO 8601 timestamp string
    """
    try:
        # Parse time of day
        time_obj = datetime.strptime(time_of_day, "%I:%M %p")
        
        # Calculate visit date
        visit_date = base_date + timedelta(days=day_offset)
        
        # Set time
        visit_datetime = visit_date.replace(
            hour=time_obj.hour,
            minute=time_obj.minute,
            second=random.randint(0, 59)
        )
        
        # Add jitter
        visit_datetime += timedelta(
            hours=jitter_hours,
            minutes=jitter_minutes
        )
        
        return visit_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
        
    except Exception as e:
        print(f"⚠️  Timestamp generation error: {e}")
        # Fallback
        fallback_date = base_date + timedelta(days=day_offset)
        return fallback_date.strftime("%Y-%m-%dT%H:%M:%SZ")


def distribute_visits_over_time(visits, num_days=180):
    """
    ✅ FALLBACK: Distribute visits across time period with realistic patterns (used if ML unavailable)
    
    Returns list of (visit_index, day_offset, time_of_day) tuples
    """
    n_visits = len(visits)
    
    # Calculate visit distribution
    visit_schedule = []
    
    # Strategy: Cluster some visits, spread others
    day_intervals = num_days // n_visits if n_visits > 0 else num_days
    
    for i, visit in enumerate(visits):
        # 30% chance of clustered visits (same week)
        if random.random() < 0.3 and i > 0:
            # Cluster near previous visit
            prev_day = visit_schedule[-1][1]
            day_offset = prev_day + random.randint(1, 7)
        else:
            # Spread out
            base_day = i * day_intervals
            day_offset = base_day + random.randint(0, max(1, day_intervals - 1))
        
        # Keep within bounds
        day_offset = min(day_offset, num_days - 1)
        
        # Get time from visit data
        time_of_day = visit.get("time_of_day", "12:00 PM")
        
        visit_schedule.append((i, day_offset, time_of_day))
    
    # Sort by day offset
    visit_schedule.sort(key=lambda x: x[1])
    
    return visit_schedule


def load_cluster_patterns(cluster_file_path):
    """Load cluster patterns from CSV to understand user preferences"""
    cluster_df = pd.read_csv(cluster_file_path)
    cluster_preferences = {}
    
    for _, row in cluster_df.iterrows():
        cluster_id = row['cluster']
        category_counts = row.drop('cluster').to_dict()
        total_visits = sum(category_counts.values())
        
        if total_visits == 0:
            continue
            
        preferences = {cat: count/total_visits for cat, count in category_counts.items() if count > 0}
        sorted_preferences = dict(sorted(preferences.items(), key=lambda x: x[1], reverse=True))
        
        cluster_preferences[cluster_id] = {
            'total_visits': total_visits,
            'preferences': sorted_preferences,
            'top_categories': list(sorted_preferences.keys())[:5]
        }
    
    return cluster_preferences

def analyze_cluster_category_patterns(cluster_file_path):
    """Analyze which categories are most transaction-heavy based on cluster data"""
    df = pd.read_csv(cluster_file_path)
    category_stats = {}
    
    for col in df.columns:
        if col == 'cluster':
            continue
            
        category = col.replace(' ', '_').lower()
        values = df[col].values
        
        total_occurrences = values.sum()
        avg_per_cluster = values.mean()
        max_in_cluster = values.max()
        clusters_with_category = (values > 0).sum()
        
        consistency_score = clusters_with_category / len(df)
        volume_score = total_occurrences / df.iloc[:, 1:].sum().sum() if df.iloc[:, 1:].sum().sum() > 0 else 0
        peak_score = max_in_cluster / total_occurrences if total_occurrences > 0 else 0
        
        transaction_score = (
            0.4 * volume_score +
            0.3 * consistency_score +
            0.3 * peak_score
        )
        
        category_stats[category] = {
            'total_occurrences': total_occurrences,
            'avg_per_cluster': avg_per_cluster,
            'max_in_cluster': max_in_cluster,
            'clusters_with_category': clusters_with_category,
            'consistency_score': consistency_score,
            'transaction_score': transaction_score
        }
    
    sorted_categories = sorted(category_stats.items(), 
                              key=lambda x: x[1]['transaction_score'], 
                              reverse=True)
    
    print("\n=== Category Transaction Likelihood Analysis ===")
    print("Top 15 categories most likely to generate transactions:")
    print(f"{'Category':<25} {'Total':<8} {'Clusters':<9} {'Trans Score':<12} {'Multiplier':<10}")
    print("-" * 75)
    
    multipliers = {}
    all_scores = [stats['transaction_score'] for _, stats in sorted_categories if stats['transaction_score'] > 0]
    
    if all_scores:
        mean_score = sum(all_scores) / len(all_scores)
        std_score = (sum((x - mean_score) ** 2 for x in all_scores) / len(all_scores)) ** 0.5
        
        for i, (category, stats) in enumerate(sorted_categories[:15]):
            if stats['transaction_score'] > 0:
                if std_score > 0:
                    multiplier = 1.0 + (stats['transaction_score'] - mean_score) / std_score * 0.8
                    multiplier = max(0.5, min(2.5, multiplier))
                else:
                    multiplier = 1.0
                    
                multipliers[category] = multiplier
                
                print(f"{category:<25} {stats['total_occurrences']:<8.0f} "
                      f"{stats['clusters_with_category']:<9.0f} "
                      f"{stats['transaction_score']:<12.3f} {multiplier:<10.2f}")
    
    return multipliers

def balance_category_distribution(visits, min_category_representation=0.03):
    """Identify under-represented categories"""
    category_counts = defaultdict(int)
    for visit in visits:
        category = visit.get("poi_category", "").lower().replace(" ", "_")
        category_counts[category] += 1
    
    total_visits = len(visits)
    under_represented = []
    
    for category, count in category_counts.items():
        if count / total_visits < min_category_representation and count > 0:
            under_represented.append(category)
    
    return under_represented

def assign_user_to_cluster(visits, cluster_preferences):
    """Assign user to most similar cluster based on their visit patterns"""
    user_categories = {}
    
    for visit in visits:
        category = visit.get("poi_category", "").lower().replace(" ", "_")
        user_categories[category] = user_categories.get(category, 0) + 1
    
    best_cluster = 0
    best_similarity = 0
    
    for cluster_id, cluster_data in cluster_preferences.items():
        similarity = 0
        user_total = sum(user_categories.values())
        
        if user_total == 0:
            continue
            
        for category, user_count in user_categories.items():
            if category in cluster_data['preferences']:
                user_pref = user_count / user_total
                cluster_pref = cluster_data['preferences'][category]
                similarity += min(user_pref, cluster_pref)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_cluster = cluster_id
    
    return best_cluster

def assign_funnel_interactions_with_clusters(visits, cluster_preferences, assigned_cluster, 
                                           transaction_multipliers,
                                           view_to_transaction_rate=0.2, 
                                           transaction_to_review_rate=0.4,
                                           max_category_ratio=0.30):
    """Enhanced funnel with category diversity enforcement
    
    CRITICAL: Funnel logic ensures:
    - ALL visits generate views
    - SUBSET of views convert to transactions (same event, added transaction)
    - SUBSET of transactions get reviews
    - Result: views >= transactions >= reviews (strict funnel)
    """
    n_visits = len(visits)
    all_visit_indices = set(range(n_visits))
    
    cluster_data = cluster_preferences.get(assigned_cluster, {})
    cluster_prefs = cluster_data.get('preferences', {})
    top_categories = cluster_data.get('top_categories', [])
    
    under_represented = balance_category_distribution(visits)
    
    print(f"User assigned to Cluster {assigned_cluster}")
    print(f"Cluster's top categories: {top_categories[:3]}")
    if under_represented:
        print(f"Boosting under-represented categories: {under_represented}")
    
    transaction_weights = []
    
    for i, visit in enumerate(visits):
        category = visit.get("poi_category", "").lower().replace(" ", "_")
        day_of_week = visit.get("day_of_week", "")
        time_of_day = visit.get("time_of_day", "12:00 PM")
        
        weight = 1.0
        
        # Cluster preference (reduced multipliers)
        if category in cluster_prefs:
            cluster_preference = cluster_prefs[category]
            if cluster_preference > 0.1:
                weight *= 2.0
            elif cluster_preference > 0.05:
                weight *= 1.5
            elif cluster_preference > 0.02:
                weight *= 1.2
        else:
            weight *= 0.5
        
        # Boost under-represented categories
        if category in under_represented:
            weight *= 2.5
        
        # Data-driven multipliers (capped)
        category_multiplier = transaction_multipliers.get(category, 1.0)
        weight *= min(category_multiplier, 1.8)
        
        # Time-based patterns
        try:
            hour = datetime.strptime(time_of_day, "%I:%M %p").hour
            if 12 <= hour <= 14 or 18 <= hour <= 22:
                weight *= 1.3
            elif 22 <= hour or hour <= 2:
                weight *= 1.15
        except:
            pass
        
        # Weekend boost
        if day_of_week in ["Friday", "Saturday"]:
            weight *= 1.2
        
        transaction_weights.append(weight)
    
    # Select transactions with diversity enforcement
    n_transactions = int(n_visits * view_to_transaction_rate)
    transaction_indices = set()
    transaction_categories = defaultdict(int)
    
    if n_transactions > 0:
        available_indices = list(range(n_visits))
        available_weights = transaction_weights.copy()
        attempts = 0
        max_attempts = n_transactions * 3
        
        while len(transaction_indices) < n_transactions and available_indices and attempts < max_attempts:
            attempts += 1
            
            selected_idx = random.choices(available_indices, weights=available_weights, k=1)[0]
            selected_category = visits[selected_idx].get("poi_category", "").lower().replace(" ", "_")
            
            current_category_count = transaction_categories[selected_category]
            current_total = len(transaction_indices) + 1
            
            if current_category_count / current_total > max_category_ratio:
                idx_position = available_indices.index(selected_idx)
                available_weights[idx_position] *= 0.2
                continue
            
            transaction_indices.add(selected_idx)
            transaction_categories[selected_category] += 1
            
            idx_position = available_indices.index(selected_idx)
            available_indices.pop(idx_position)
            available_weights.pop(idx_position)
    
    # Reviews selection
    review_indices = set()
    if transaction_indices:
        review_weights = []
        transaction_list = list(transaction_indices)
        
        for i in transaction_list:
            visit = visits[i]
            category = visit.get("poi_category", "").lower().replace(" ", "_")
            
            weight = 1.0
            weight *= transaction_multipliers.get(category, 1.0)
            
            if category in cluster_prefs and cluster_prefs[category] > 0.05:
                weight *= 1.3
            
            review_weights.append(weight)
        
        n_reviews = int(len(transaction_indices) * transaction_to_review_rate)
        if n_reviews > 0:
            available_indices = transaction_list.copy()
            available_weights = review_weights.copy()
            
            for _ in range(min(n_reviews, len(available_indices))):
                if not available_indices:
                    break
                
                selected_idx = random.choices(available_indices, weights=available_weights, k=1)[0]
                review_indices.add(selected_idx)
                
                idx_position = available_indices.index(selected_idx)
                available_indices.pop(idx_position)
                available_weights.pop(idx_position)
    
    return {
        "views": all_visit_indices,  # ALL visits become views
        "transactions": transaction_indices,  # Subset that converted
        "reviews": review_indices,  # Subset of transactions with reviews
        "cluster": assigned_cluster
    }

def analyze_cluster_transaction_patterns(assignments, visits, cluster_id):
    """Analyze if transactions align with cluster preferences"""
    transaction_categories = []
    for i in assignments["transactions"]:
        visit = visits[i]
        category = visit.get("poi_category", "").lower().replace(" ", "_")
        transaction_categories.append(category)
    
    category_counts = defaultdict(int)
    for cat in transaction_categories:
        category_counts[cat] += 1
    
    print(f"\n=== Cluster {cluster_id} Transaction Analysis ===")
    total_transactions = len(transaction_categories)
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        percentage = count / total_transactions * 100 if total_transactions > 0 else 0
        print(f"{cat}: {count} ({percentage:.1f}%)")
    
    return category_counts

def validate_and_print_patterns(assignments, visits):
    """Validate and print pattern analysis"""
    n_visits = len(visits)
    n_views = len(assignments["views"])
    n_transactions = len(assignments["transactions"])
    n_reviews = len(assignments["reviews"])
    
    print(f"\n=== Funnel Analysis ===")
    print(f"Total Visits: {n_visits}")
    print(f"Views: {n_views} (100%)")
    print(f"Transactions: {n_transactions} ({n_transactions/n_visits*100:.1f}%)")
    if n_transactions > 0:
        print(f"Reviews: {n_reviews} ({n_reviews/n_transactions*100:.1f}% of transactions)")
    else:
        print(f"Reviews: {n_reviews} (0% of transactions)")
    
    return assignments

def assign_funnel_interactions(visits, view_to_transaction_rate=0.2, transaction_to_review_rate=0.4):
    """Original random assignment function (fallback)
    
    CRITICAL: Ensures strict funnel hierarchy:
    - ALL visits generate views
    - Transactions are subset of views (conversion)
    - Reviews are subset of transactions
    """
    n_visits = len(visits)
    all_visit_indices = set(range(n_visits))
    
    n_transactions = int(n_visits * view_to_transaction_rate)
    transaction_indices = set(random.sample(list(all_visit_indices), min(n_transactions, len(all_visit_indices))))
    
    n_reviews = int(len(transaction_indices) * transaction_to_review_rate)
    review_indices = set(random.sample(list(transaction_indices), min(n_reviews, len(transaction_indices))))
    
    return {
        "views": all_visit_indices,  # ALL visits are views
        "transactions": transaction_indices,  # Subset that converted
        "reviews": review_indices  # Subset of transactions with reviews
    }

# -----------------------------
# Main Generation Logic
# -----------------------------

if __name__ == "__main__":
    
    print(f"Input file: {JSON_INPUT}")
    print(f"Output file: {JSON_OUTPUT}")
    print(f"Cluster params directory: {POST_SAMPLING_ANALYSIS_OUTPUT_DIR}")
    
    # ✅ ENHANCED: Train/load ML temporal model from JSON_FUNC directory
    temporal_model_dir = os.path.join(JSON_FUNC, 'temporal_models')
    os.makedirs(temporal_model_dir, exist_ok=True)
    temporal_model_path = os.path.join(temporal_model_dir, 'temporal_patterns_enhanced.pkl')
    
    print(f"Temporal model directory: {temporal_model_dir}")
    print(f"Temporal model path: {temporal_model_path}")
    
    temporal_generator = None
    use_ml_temporal = False
    
    if os.path.exists(temporal_model_path):
        print(f"\n✅ Loading trained enhanced temporal model from {temporal_model_path}")
        try:
            learner = EnhancedTemporalPatternLearner()
            learner.load(temporal_model_path)
            temporal_generator = EnhancedTimeSeriesGenerator(learner)
            use_ml_temporal = True
            print("✅ Enhanced ML-based temporal generation enabled")
        except Exception as e:
            print(f"⚠️  Failed to load enhanced temporal model: {e}")
            print("Using fallback random temporal generation")
    else:
        print(f"\n📚 Training enhanced temporal model from JSON input...")
        try:
            # Enhanced model now supports JSON input directly
            learner = train_enhanced_model(JSON_INPUT, temporal_model_path, input_format='json')
            temporal_generator = EnhancedTimeSeriesGenerator(learner)
            use_ml_temporal = True
            print("✅ Enhanced ML-based temporal generation enabled")
        except Exception as e:
            print(f"⚠️  Failed to train enhanced temporal model: {e}")
            import traceback
            traceback.print_exc()
            print("Using fallback random temporal generation")
    
    # ✅ FIX: Construct correct path to poi_category_by_cluster.csv
    cluster_file_path = os.path.join(POST_SAMPLING_ANALYSIS_OUTPUT_DIR, "poi_category_by_cluster.csv")
    
    # Normalize the path (remove ./ and extra slashes)
    cluster_file_path = os.path.normpath(cluster_file_path)
    
    print(f"Looking for cluster file at: {cluster_file_path}")
    print(f"File exists: {os.path.exists(cluster_file_path)}")
    
    # List files in the directory to help debug
    if os.path.exists(POST_SAMPLING_ANALYSIS_OUTPUT_DIR):
        print(f"\nFiles in {POST_SAMPLING_ANALYSIS_OUTPUT_DIR}:")
        for file in os.listdir(POST_SAMPLING_ANALYSIS_OUTPUT_DIR):
            print(f"  - {file}")
    else:
        print(f"\n⚠️  Directory does not exist: {POST_SAMPLING_ANALYSIS_OUTPUT_DIR}")
    
    try:
        cluster_preferences = load_cluster_patterns(cluster_file_path)
        
        print("\nAnalyzing cluster data for transaction patterns...")
        transaction_multipliers = analyze_cluster_category_patterns(cluster_file_path)
        print(f"\n✅ Loaded {len(cluster_preferences)} cluster patterns")
        print(f"✅ Generated {len(transaction_multipliers)} category multipliers")
        use_clusters = True
    except FileNotFoundError as e:
        print(f"\n❌ Cluster file not found: {e}")
        print("Using random assignment (no cluster-based logic).")
        cluster_preferences = {}
        transaction_multipliers = {}
        use_clusters = False
    except Exception as e:
        print(f"\n❌ Error loading cluster data: {e}")
        print("Using random assignment (no cluster-based logic).")
        cluster_preferences = {}
        transaction_multipliers = {}
        use_clusters = False

    # Load input data from config path
    print(f"\nLoading data from: {JSON_INPUT}")
    with open(JSON_INPUT, "r", encoding="utf-8") as f:
        checkin_data = json.load(f)

    # ✅ FIX 6: Identify popular POIs across all users for better overlap
    print("\n📊 Analyzing POI popularity for train/test overlap...")
    global_poi_counts = defaultdict(int)
    global_poi_data = {}  # Store POI metadata
    
    for user in checkin_data:
        visits = user.get("user_metadata", [])
        for v in visits:
            poi_id = v.get('poi_id', '')
            if poi_id:
                global_poi_counts[poi_id] += 1
                if poi_id not in global_poi_data:
                    global_poi_data[poi_id] = {
                        'poi_id': poi_id,
                        'poi_category': v.get('poi_category', ''),
                        'lat': v.get('lat', 0.0),
                        'lon': v.get('lon', 0.0),
                        'planning_area': v.get('planning_area', ''),
                        'day_of_week': v.get('day_of_week', 'Monday'),
                        'time_of_day': v.get('time_of_day', '12:00 PM')
                    }
    
    # Get top 200 popular POIs (these should appear in train AND test)
    popular_pois = sorted(global_poi_counts.items(), key=lambda x: x[1], reverse=True)[:200]
    popular_poi_ids = [poi_id for poi_id, _ in popular_pois]
    
    print(f"✅ Identified {len(popular_poi_ids)} popular POIs (will ensure train/test overlap)")
    print(f"   Top 5 POIs: {[f'{pid}({cnt})' for pid, cnt in popular_pois[:5]]}")

    app_profiles = []

    for user in tqdm(checkin_data, desc="Processing users"):
        user_id = str(user["user_id"])
        visits = user.get("user_metadata", [])
        if not visits or not isinstance(visits, list) or len(visits) == 0:
            continue

        # ✅ REALISTIC CAP: Limit base visits to max 120 for 180 days (~0.67 visits/day)
        # This ensures final count after adding revisits stays realistic
        # For 1 week: ~5 base visits → ~7 total after 20% revisits (realistic!)
        # For 6 months: ~120 base visits → ~140 total after 20% revisits (realistic!)
        if len(visits) > 120:
            # Randomly sample 120 visits to maintain diversity
            visits = random.sample(visits, 120)

        # ✅ FIX 1: DON'T shuffle - preserve sequential patterns
        # Original visits already have temporal ordering from data
        # random.shuffle(visits)  # ❌ REMOVED - destroys temporal order
        
        # ✅ FIX 2: Expand visits with POI repetition (10-20% revisits for realism)
        # This creates realistic interaction density and favorite POI patterns
        expanded_visits = []
        favorite_pois = []
        
        # Identify top 3-5 POIs as "favorites" based on existing patterns
        poi_counts = defaultdict(int)
        for v in visits:
            poi_id = v.get('poi_id', '')
            if poi_id:
                poi_counts[poi_id] += 1
        
        # Get existing favorites (POIs already visited multiple times)
        existing_favorites = [poi for poi, count in poi_counts.items() if count >= 2]
        
        # If user has favorites, use them; otherwise pick random 3-5
        if len(existing_favorites) >= 2:
            favorite_pois = existing_favorites[:min(5, len(existing_favorites))]
        elif len(visits) >= 5:
            # Pick 3 random POIs to be favorites
            sample_size = min(3, len(visits))
            favorite_indices = random.sample(range(len(visits)), sample_size)
            favorite_pois = [visits[i].get('poi_id', '') for i in favorite_indices]
        
        # Add original visits first (in order)
        expanded_visits.extend(visits)
        
        # ✅ FIX 7: Inject popular POIs to ensure train/test overlap
        # Each user should visit 3-5 popular POIs to create overlap
        num_popular_to_add = random.randint(3, 5)
        popular_to_inject = random.sample(popular_poi_ids, min(num_popular_to_add, len(popular_poi_ids)))
        
        for pop_poi_id in popular_to_inject:
            if pop_poi_id in global_poi_data:
                # Create visit to popular POI
                pop_visit = global_poi_data[pop_poi_id].copy()
                pop_visit['is_popular'] = True
                expanded_visits.append(pop_visit)
        
        # Add 10-20% more visits as revisits to favorites (realistic repetition)
        # This keeps total interaction count reasonable while showing loyalty patterns
        if favorite_pois:
            num_revisits = int(len(visits) * random.uniform(0.1, 0.2))
            for _ in range(num_revisits):
                # Pick a favorite POI
                fav_poi = random.choice(favorite_pois)
                # Find original visit to this POI
                original_visit = None
                for v in visits:
                    if v.get('poi_id') == fav_poi:
                        original_visit = v.copy()
                        break
                
                if original_visit:
                    # Mark as revisit for tracking
                    original_visit['is_revisit'] = True
                    expanded_visits.append(original_visit)
        
        # Use expanded visits instead of original
        visits = expanded_visits
        
        # ✅ FIX 5: Reorder visits to follow category transition patterns
        if use_ml_temporal and temporal_generator:
            visits = reorder_visits_by_category_transitions(visits, temporal_generator)
        
        if use_clusters:
            assigned_cluster = assign_user_to_cluster(visits, cluster_preferences)
            assigns = assign_funnel_interactions_with_clusters(
                visits, cluster_preferences, assigned_cluster, transaction_multipliers,
                view_to_transaction_rate=0.2, transaction_to_review_rate=0.4
            )
            
            if len(app_profiles) < 3:
                analyze_cluster_transaction_patterns(assigns, visits, assigned_cluster)
                validate_and_print_patterns(assigns, visits)
        else:
            assigns = assign_funnel_interactions(visits, view_to_transaction_rate=0.2, transaction_to_review_rate=0.4)

        user_block = {
            "userId": user_id,
            "age": random.randint(22, 74),
            "gender": random.choice(["male", "female", "other"]),
            "location": {"city": "Singapore", "country": "SG"},
            "device": {
                "platform": random.choice(["Android", "iOS"]),
                "appVersion": f"{random.randint(2,4)}.{random.randint(0,9)}.{random.randint(0,9)}"
            }
        }
        
        # ✅ FIX: Generate consistent home location for user
        user_home_lat, user_home_lon = None, None
        if visits:
            # Use first visit's planning area as approximate home location
            first_visit = visits[0]
            planning_lat = float(first_visit.get("lat", 1.3521))
            planning_lon = float(first_visit.get("lon", 103.8198))
            user_home_lat, user_home_lon = generate_user_home_location(planning_lat, planning_lon)

        interaction = {"views": [], "transactions": [], "reviews": []}

        # ✅ ML-BASED: Calculate base date (180 days ago from now)
        base_date = datetime.now() - timedelta(days=180)
        
        # ✅ FIX 3: Generate timestamps in chronological order
        if use_ml_temporal and temporal_generator:
            # Use ML model to generate realistic temporal sequence
            visit_timestamps = generate_ml_temporal_timestamps(
                visits, temporal_generator, base_date, num_days=180
            )
        else:
            # Fallback to rule-based distribution
            visit_schedule = distribute_visits_over_time(visits, num_days=180)
            visit_timestamps = {}
            for visit_idx, day_offset, time_of_day in visit_schedule:
                timestamp_str = generate_temporal_timestamp(
                    base_date, day_offset, time_of_day,
                    jitter_hours=random.randint(0, 3),
                    jitter_minutes=random.randint(0, 59)
                )
                dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
                day_of_week = dt.strftime("%A")
                visit_timestamps[visit_idx] = (dt, day_of_week, time_of_day)
        
        # ✅ FIX 4: Sort timestamps to ensure chronological order
        sorted_timestamps = sorted(visit_timestamps.items(), key=lambda x: x[1][0])
        visit_timestamps = {idx: ts_data for idx, ts_data in sorted_timestamps}

        # ✅ CRITICAL FIX: Track view objects by visit index for proper funnel pairing
        view_index_to_object = {}  # Maps visit index → generated view object

        # Generate events with ML-learned temporal distribution
        for idx in assigns["views"]:
            entry = visits[idx]
            if "lat" not in entry or "lon" not in entry:
                continue
            
            # ✅ Use ML-generated timestamp
            if idx in visit_timestamps:
                base_dt, day_of_week, time_of_day = visit_timestamps[idx]
                base_timestamp = base_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                # Fallback for missing timestamps
                day_offset = random.randint(0, 179)
                time_of_day = entry.get("time_of_day", "12:00 PM")
                base_timestamp = generate_temporal_timestamp(
                    base_date, day_offset, time_of_day,
                    jitter_hours=random.randint(0, 3),
                    jitter_minutes=random.randint(0, 59)
                )
                base_dt = datetime.strptime(base_timestamp, "%Y-%m-%dT%H:%M:%SZ")
            
            # ✅ FIX: View location from home (not random 50km jitter)
            if user_home_lat and user_home_lon:
                lat, lon = get_view_location(user_home_lat, user_home_lon, 
                                            float(entry["lat"]), float(entry["lon"]))
            else:
                lat, lon = jitter_location(float(entry["lat"]), float(entry["lon"]), distance_km=2)
            
            poi_id = entry.get("poi_id", f"POI-{idx}")
            
            if idx in assigns["transactions"]:
                duration = random.randint(60, 180)
            else:
                duration = random.randint(40, 120) if random.random() < 0.7 else random.randint(1, 19)
            
            view = {
                "timestamp": base_timestamp,
                "poiId": poi_id,
                "poiCategories": [entry["poi_category"]],
                "poiSubcategories": [],
                "duration": duration,
                "referrer": random.choice(["map", "search", "ad", "friend"]),
                "userLocation": {"latitude": lat, "longitude": lon}
            }
            interaction["views"].append(view)
            # ✅ CRITICAL: Store view object indexed by visit index for transaction pairing
            view_index_to_object[idx] = view

        # Transactions (20-90 minutes after view, with average around 55 minutes)
        # ✅ CRITICAL FIX: Use index-based pairing to ensure each transaction has its view
        # Track which indices successfully generate transactions (some may be skipped if missing lat/lon)
        txn_index_to_object = {}  # Maps visit index → generated transaction object
        
        for idx in assigns["transactions"]:
            entry = visits[idx]
            if "lat" not in entry or "lon" not in entry:
                continue  # Skip this transaction (missing data)
            
            # ✅ FIXED: Get the view object for THIS specific visit index
            if idx in view_index_to_object:
                base_view = view_index_to_object[idx]
                view_dt = datetime.strptime(base_view["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
                
                # Use a weighted distribution for more realistic timing
                # Most people take 30-60 minutes, some faster (20-30), some slower (60-90)
                minutes_delay = random.choices(
                    [random.randint(20, 30), random.randint(30, 60), random.randint(60, 90)],
                    weights=[0.2, 0.5, 0.3],  # 20% quick, 50% moderate, 30% slow
                    k=1
                )[0]
                txn_dt = view_dt + timedelta(minutes=minutes_delay)
                txn_timestamp = txn_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                # Fallback: use ML timestamps if available
                if idx in visit_timestamps:
                    base_dt, _, _ = visit_timestamps[idx]
                    txn_timestamp = base_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                else:
                    day_offset = random.randint(0, 179)
                    time_of_day = entry.get("time_of_day", "12:00 PM")
                    txn_timestamp = generate_temporal_timestamp(base_date, day_offset, time_of_day)
            
            lat, lon = float(entry["lat"]), float(entry["lon"])
            poi_id = entry.get("poi_id", f"POI-{idx}")
            
            txn = {
                "timestamp": txn_timestamp,
                "poiId": poi_id,
                "poiCategories": [entry["poi_category"]],
                "poiSubcategories": [],
                "transactionId": str(random.randint(100000, 999999)),
                "amount": random_amount(entry["poi_category"]),
                "currency": "SGD",
                "paymentMethod": random_payment(),
                "userLocation": {"latitude": lat, "longitude": lon},
                "planning_area": entry.get("planning_area", "UNKNOWN")
            }
            interaction["transactions"].append(txn)
            # ✅ CRITICAL: Store mapping ONLY for successfully generated transactions
            txn_index_to_object[idx] = txn

        # Reviews (1-7 days after transaction)
        # ✅ CRITICAL FIX: Use index-based pairing to ensure each review has its transaction
        for idx in assigns["reviews"]:
            entry = visits[idx]
            if "lat" not in entry or "lon" not in entry:
                continue
            
            # ✅ CRITICAL: Only generate review if transaction was successfully generated
            if idx not in txn_index_to_object:
                # This review index doesn't have a transaction (was skipped due to missing data)
                # Skip this review to maintain funnel integrity
                continue
            
            # ✅ FIXED: Get the transaction object for THIS specific visit index
            base_txn = txn_index_to_object[idx]
            txn_dt = datetime.strptime(base_txn["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
            
            # Reviews happen 1-7 days later (realistic reflection time)
            review_dt = txn_dt + timedelta(
                days=random.randint(1, 7),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            review_timestamp = review_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # ✅ FIX: Review location from home (not random 50km jitter)
            if user_home_lat and user_home_lon:
                lat, lon = get_review_location(user_home_lat, user_home_lon)
            else:
                lat, lon = jitter_location(float(entry["lat"]), float(entry["lon"]), distance_km=2)
            
            # Get POI ID from current entry (not from transaction loop variable)
            review_poi_id = entry.get("poi_id", f"POI-{idx}")
            
            review = {
                "timestamp": review_timestamp,
                "poiId": review_poi_id,
                "poiCategories": [entry["poi_category"]],
                "poiSubcategories": [],
                "rating": random_rating(),
                "reviewText": generate_review_text(user_id, review_poi_id, entry["poi_category"]),
                "userLocation": {"latitude": lat, "longitude": lon}
            }
            interaction["reviews"].append(review)

        # ✅ FIX 8: Sort all interactions by timestamp (chronological order)
        interaction["views"].sort(key=lambda x: x["timestamp"])
        interaction["transactions"].sort(key=lambda x: x["timestamp"])
        interaction["reviews"].sort(key=lambda x: x["timestamp"])
        
        # ✅ REMOVED: Post-generation filtering no longer needed - funnel order now correct by design
        # Index-based pairing ensures each transaction has its view, each review has its transaction

        app_profiles.append({
            "user": user_block,
            "interaction": interaction
        })

    print(f"\n✅ Generated profiles for {len(app_profiles)} users")
    
    # ✅ NEW: Validate POI overlap and repetition
    print("\n=== POI Overlap & Repetition Analysis ===")
    all_poi_visits = defaultdict(int)
    user_poi_counts = []
    revisit_counts = []
    
    for profile in app_profiles:
        user_pois = defaultdict(int)
        for view in profile["interaction"]["views"]:
            poi_id = view["poiId"]
            all_poi_visits[poi_id] += 1
            user_pois[poi_id] += 1
        
        user_poi_counts.append(len(user_pois))
        revisits = sum(1 for count in user_pois.values() if count >= 2)
        revisit_counts.append(revisits)
    
    unique_pois = len(all_poi_visits)
    avg_pois_per_user = sum(user_poi_counts) / len(user_poi_counts) if user_poi_counts else 0
    avg_revisits = sum(revisit_counts) / len(revisit_counts) if revisit_counts else 0
    
    # Calculate how many POIs are visited by multiple users
    pois_by_multiple_users = sum(1 for count in all_poi_visits.values() if count >= 2)
    overlap_rate = pois_by_multiple_users / unique_pois if unique_pois > 0 else 0
    
    print(f"Unique POIs: {unique_pois}")
    print(f"Avg POIs per user: {avg_pois_per_user:.1f}")
    print(f"Avg revisited POIs per user: {avg_revisits:.1f} ({avg_revisits/avg_pois_per_user*100:.1f}%)")
    print(f"POIs visited by 2+ users: {pois_by_multiple_users} ({overlap_rate*100:.1f}%)")
    
    # Check popular POI coverage
    popular_in_data = sum(1 for poi in popular_poi_ids if poi in all_poi_visits)
    print(f"Popular POIs in dataset: {popular_in_data}/{len(popular_poi_ids)} ({popular_in_data/len(popular_poi_ids)*100:.1f}%)")
    
    # Analyze category distribution
    all_transactions = []
    for profile in app_profiles:
        for txn in profile["interaction"]["transactions"]:
            all_transactions.append(txn["poiCategories"][0])
    
    category_counts = defaultdict(int)
    for cat in all_transactions:
        category_counts[cat] += 1
    
    print("\n=== Transaction Category Distribution ===")
    total = len(all_transactions)
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        pct = count / total * 100 if total > 0 else 0
        print(f"{cat:<30} {count:<6} ({pct:>5.1f}%)")
    
    # ✅ NEW: Validate temporal distribution
    print("\n=== Temporal Distribution Validation ===")
    all_timestamps = []
    for profile in app_profiles:
        for view in profile["interaction"]["views"]:
            all_timestamps.append(view["timestamp"])
        for txn in profile["interaction"]["transactions"]:
            all_timestamps.append(txn["timestamp"])
        for review in profile["interaction"]["reviews"]:
            all_timestamps.append(review["timestamp"])
    
    if all_timestamps:
        dates = [datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ") for ts in all_timestamps]
        dates.sort()
        
        first_event = dates[0]
        last_event = dates[-1]
        span_days = (last_event - first_event).days
        
        print(f"First event: {first_event.strftime('%Y-%m-%d %H:%M')}")
        print(f"Last event:  {last_event.strftime('%Y-%m-%d %H:%M')}")
        print(f"Time span:   {span_days} days ({span_days/30:.1f} months)")
        print(f"Total events: {len(all_timestamps):,}")
        
        # Check distribution by week
        week_counts = defaultdict(int)
        for dt in dates:
            week_key = dt.strftime("%Y-W%U")
            week_counts[week_key] += 1
        
        avg_per_week = sum(week_counts.values()) / len(week_counts) if week_counts else 0
        max_per_week = max(week_counts.values()) if week_counts else 0
        min_per_week = min(week_counts.values()) if week_counts else 0
        
        print(f"\nWeekly distribution:")
        print(f"  Avg events/week: {avg_per_week:.0f}")
        print(f"  Max events/week: {max_per_week}")
        print(f"  Min events/week: {min_per_week}")
        print(f"  Weeks covered:   {len(week_counts)}")
        
        # Validate realistic spread (not clustered)
        if span_days >= 150:  # Should be ~180 days
            print(f"✅ Temporal spread: REALISTIC ({span_days} days)")
        elif span_days >= 90:
            print(f"⚠️  Temporal spread: MODERATE ({span_days} days - expected ~180)")
        else:
            print(f"❌ Temporal spread: TOO NARROW ({span_days} days - expected ~180)")
    
    # ✅ FIX: Validate funnel order and realism before saving
    print("\n" + "="*60)
    print("VALIDATING FUNNEL ORDER & REALISM")
    print("="*60)
    summary, detailed_stats = validate_funnel_order(app_profiles, verbose=True)
    
    # Save to config path
    print(f"\nSaving to: {JSON_OUTPUT}")
    os.makedirs(os.path.dirname(JSON_OUTPUT), exist_ok=True)
    
    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(app_profiles, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved synthetic data to {JSON_OUTPUT}")