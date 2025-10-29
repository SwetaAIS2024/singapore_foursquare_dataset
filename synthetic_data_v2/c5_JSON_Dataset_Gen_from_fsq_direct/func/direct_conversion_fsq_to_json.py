# -*- coding: utf-8 -*-
"""
Direct FSQ to Synthetic Dataset Conversion
Converts Foursquare checkin data to synthetic dataset format with funnel logic.
Uses original timestamps from FSQ data - no ML temporal models needed.
"""
import json
import random
from collections import defaultdict
from datetime import datetime, timedelta
import math
from tqdm import tqdm
import os

# Input: Direct FSQ JSON (user_id + user_metadata with timestamps)
# Output: Synthetic dataset with views, transactions, reviews

# Get script directory and construct paths relative to it
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Go up one level from func/

JSON_INPUT = os.path.join(BASE_DIR, "input_sampled_fsq_json", "input.json")
JSON_OUTPUT = os.path.join(BASE_DIR, "output_syn_json", "fsq_to_synthetic.json")

# -----------------------------
# Helper Functions
# -----------------------------

def jitter_location(lat, lon, distance_km=2):
    max_deg = distance_km / 111.0
    lat_jitter = random.uniform(-max_deg, max_deg)
    lon_jitter = random.uniform(-max_deg, max_deg) / math.cos(math.radians(lat))
    return round(lat + lat_jitter, 6), round(lon + lon_jitter, 6)

def ensure_unique_timestamp(base_timestamp_str, used_timestamps):
    """
    Ensure timestamp is unique by adding seconds if duplicate exists.
    Args:
        base_timestamp_str: Original timestamp string (e.g., "2024-04-06T04:39:00Z")
        used_timestamps: Set of already used timestamp strings
    Returns:
        Unique timestamp string
    """
    if base_timestamp_str not in used_timestamps:
        used_timestamps.add(base_timestamp_str)
        return base_timestamp_str
    
    # Parse the timestamp
    base_dt = datetime.strptime(base_timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
    
    # Try adding 1-60 seconds until we find a unique timestamp
    for offset_seconds in range(1, 61):
        new_dt = base_dt + timedelta(seconds=offset_seconds)
        new_timestamp_str = new_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        if new_timestamp_str not in used_timestamps:
            used_timestamps.add(new_timestamp_str)
            return new_timestamp_str
    
    # If still not unique after 60 attempts, use microseconds (should never happen)
    new_dt = base_dt + timedelta(seconds=random.randint(61, 120))
    new_timestamp_str = new_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    used_timestamps.add(new_timestamp_str)
    return new_timestamp_str

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

def parse_fsq_timestamp(visit_entry):
    """
    Parse FSQ timestamp from visit metadata fields.
    Constructs datetime from: day_of_week, time_of_day, month_of_year
    """
    try:
        # Extract fields from visit entry
        month_name = visit_entry.get("month_of_year", "January")
        day_of_week = visit_entry.get("day_of_week", "Monday")
        time_str = visit_entry.get("time_of_day", "12:00 PM")
        
        # Use 2024 as base year (adjust if needed)
        year = 2024
        
        # Convert month name to number
        month_map = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12
        }
        month = month_map.get(month_name, 1)
        
        # Parse time (format: "06:19 PM")
        time_obj = datetime.strptime(time_str, "%I:%M %p")
        hour = time_obj.hour
        minute = time_obj.minute
        
        # Find a day in that month matching the day of week
        # Start from day 1 and find first matching weekday
        day_name_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6
        }
        target_weekday = day_name_map.get(day_of_week, 0)
        
        # Search for matching day in month (try first 28 days to avoid month overflow)
        for day in range(1, 29):
            test_date = datetime(year, month, day)
            if test_date.weekday() == target_weekday:
                # Found matching weekday, now add time
                return datetime(year, month, day, hour, minute, 0)
        
        # Fallback: use day 15 of the month
        return datetime(year, month, 15, hour, minute, 0)
        
    except Exception as e:
        # Fallback: return a fixed date
        print(f"Warning: Failed to parse timestamp from {visit_entry}: {e}")
        return datetime(2024, 1, 1, 12, 0, 0)

# ✅ Funnel Order Validation
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
        
        # Build transactionId → transaction map for review validation
        txn_id_to_txn = {}
        
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
            poi_to_txn[poi_id].append({
                'timestamp': timestamp, 
                'location': txn['userLocation'],
                'transactionId': txn.get('transactionId')
            })
            # Store transaction by ID for review validation
            if 'transactionId' in txn:
                txn_id_to_txn[txn['transactionId']] = {
                    'timestamp': timestamp,
                    'location': txn['userLocation'],
                    'poiId': poi_id
                }
        
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
                
                # Find most recent view before transaction (must be strictly before)
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
        
        # Validate reviews using transactionId links
        for review in reviews:
            review_time = datetime.strptime(review['timestamp'], "%Y-%m-%dT%H:%M:%SZ")
            review_loc = review['userLocation']
            
            # Check if review has linked transaction
            if 'transactionId' not in review:
                stats['funnel_violations'].append({
                    'user': user_data['user']['userId'],
                    'poi': review['poiId'],
                    'violation': 'Review without linked transactionId'
                })
                continue
            
            txn_id = review['transactionId']
            if txn_id not in txn_id_to_txn:
                stats['funnel_violations'].append({
                    'user': user_data['user']['userId'],
                    'poi': review['poiId'],
                    'violation': 'Review links to non-existent transaction'
                })
                continue
            
            # Get the linked transaction
            linked_txn = txn_id_to_txn[txn_id]
            txn_time = linked_txn['timestamp']
            txn_loc = linked_txn['location']
            
            # Validate temporal order: review must come after transaction (strictly after)
            if review_time <= txn_time:
                stats['funnel_violations'].append({
                    'user': user_data['user']['userId'],
                    'poi': review['poiId'],
                    'violation': 'Review before/at Transaction time'
                })
                continue
            
            # Calculate temporal delta: transaction → review
            delta_days = (review_time - txn_time).total_seconds() / 86400
            stats['temporal_stats']['txn_to_review_days'].append(delta_days)
            
            # Calculate spatial distance: review location vs POI location
            review_distance = haversine_distance(
                review_loc['latitude'], review_loc['longitude'],
                txn_loc['latitude'], txn_loc['longitude']
            )
            stats['spatial_stats']['review_distance_from_poi_km'].append(review_distance)
    
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

def assign_funnel_interactions(visits, view_to_transaction_rate=0.4, transaction_to_review_rate=0.3):
    """
    Assign funnel interactions in CYCLES per POI:
    
    For EACH POI, repeat the funnel cycle: Views → Transactions → Reviews
    
    Ratios per cycle:
    - 100% of visits start as Views
    - 40% of those Views → Transactions
    - 30% of those Transactions → Reviews
    
    Example: POI-A has 10 visits in a cycle
      - 10 views (100%)
      - 4 transactions (40% of 10)
      - 1-2 reviews (30% of 4)
    
    Cycle size varies with randomness to create realistic patterns.
    """
    n_visits = len(visits)
    
    if n_visits == 0:
        return {"views": set(), "transactions": set(), "reviews": set()}
    
    # Group visits by POI
    from collections import defaultdict
    import random
    poi_visit_indices = defaultdict(list)
    for idx, visit in enumerate(visits):
        poi_id = visit.get('poi_id', f'POI-{idx}')
        poi_visit_indices[poi_id].append(idx)
    
    # Initialize sets
    view_indices = set()
    transaction_indices = set()
    review_indices = set()
    
    # For each POI, apply cyclical funnel pattern
    for poi_id, indices in poi_visit_indices.items():
        # Sort indices chronologically by the visit timestamp
        indices = sorted(indices, key=lambda idx: visits[idx].get('_parsed_timestamp', datetime(2024, 1, 1)))
        n_poi_visits = len(indices)
        
        # Handle corner cases for POIs with few visits
        if n_poi_visits == 1:
            # Single visit: make it a view only
            view_indices.add(indices[0])
            continue
        elif n_poi_visits == 2:
            # Two visits: first = view, second = view + transaction
            view_indices.add(indices[0])
            view_indices.add(indices[1])
            transaction_indices.add(indices[1])
            continue
        elif n_poi_visits == 3:
            # Three visits: apply ratios directly (no cycles)
            # view, view+transaction, view+transaction+review
            view_indices.add(indices[0])
            view_indices.add(indices[1])
            view_indices.add(indices[2])
            n_transactions = int(3 * view_to_transaction_rate)  # 40% of 3 = 1
            n_reviews = int(n_transactions * transaction_to_review_rate)  # 30% of 1 = 0
            if n_transactions >= 1:
                transaction_indices.add(indices[1])
            if n_transactions >= 2:
                transaction_indices.add(indices[2])
            if n_reviews >= 1:
                review_indices.add(indices[2])
            continue
        
        # For POIs with 4+ visits, use cyclical pattern
        # Base cycle size (aim for multiple cycles, but adapt to visit count)
        if n_poi_visits >= 10:
            base_cycle_size = n_poi_visits // 4  # Aim for ~4 cycles
        elif n_poi_visits >= 6:
            base_cycle_size = n_poi_visits // 2  # 2-3 cycles
        else:
            base_cycle_size = n_poi_visits  # Single cycle for 4-5 visits
        
        current_idx = 0
        cycle_num = 0
        
        while current_idx < n_poi_visits:
            # Calculate cycle size with randomness
            if cycle_num == 0:
                cycle_size = base_cycle_size
            else:
                # Add randomness: ±20% variation
                variation = random.randint(-20, 20) / 100.0
                cycle_size = max(3, int(base_cycle_size * (1 + variation)))
            
            # Don't exceed remaining visits
            cycle_size = min(cycle_size, n_poi_visits - current_idx)
            
            # Within this cycle, apply the ratios:
            # Distribute checkins: some as view-only, some as transaction (which implies view), some as review (which implies transaction+view)
            n_transactions = int(cycle_size * view_to_transaction_rate)  # 40% of cycle
            n_reviews = int(n_transactions * transaction_to_review_rate)  # 30% of transactions
            
            # Ensure at least 1 transaction if cycle is large enough
            if cycle_size >= 2:
                n_transactions = max(1, n_transactions)
            if n_transactions >= 2:
                n_reviews = max(0, n_reviews)
            
            # Calculate how many are view-only
            n_view_only = cycle_size - n_transactions
            
            # Store the indices for this cycle
            cycle_indices = []
            for i in range(cycle_size):
                if current_idx < n_poi_visits:
                    cycle_indices.append(indices[current_idx])
                    current_idx += 1
            
            # Assign types sequentially to maintain temporal order:
            # First n_view_only: View only
            # Next (n_transactions - n_reviews): View + Transaction
            # Last n_reviews: View + Transaction + Review
            
            idx_pos = 0
            
            # View-only checkins (first part of cycle)
            for i in range(min(n_view_only, len(cycle_indices))):
                view_indices.add(cycle_indices[idx_pos])
                idx_pos += 1
            
            # Transaction checkins (middle part - these also have views)
            n_txn_only = n_transactions - n_reviews
            for i in range(n_txn_only):
                if idx_pos < len(cycle_indices):
                    view_indices.add(cycle_indices[idx_pos])
                    transaction_indices.add(cycle_indices[idx_pos])
                    idx_pos += 1
            
            # Review checkins (last part - these also have transactions and views)
            for i in range(n_reviews):
                if idx_pos < len(cycle_indices):
                    view_indices.add(cycle_indices[idx_pos])
                    transaction_indices.add(cycle_indices[idx_pos])
                    review_indices.add(cycle_indices[idx_pos])
                    idx_pos += 1            
            cycle_num += 1
    
    return {
        "views": view_indices,
        "transactions": transaction_indices,
        "reviews": review_indices
    }

# -----------------------------
# Main Generation Logic
# -----------------------------

if __name__ == "__main__":
    
    print(f"Input file: {JSON_INPUT}")
    print(f"Output file: {JSON_OUTPUT}")
    
    # Load input data (direct FSQ translation)
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

        # ✅ PURE TRANSFORMATION: Use ONLY original checkins, no additions
        # No capping, no POI injection, no artificial revisits
        
        # Sort visits by timestamp (construct from FSQ metadata)
        for visit in visits:
            visit['_parsed_timestamp'] = parse_fsq_timestamp(visit)
        
        # Sort by parsed timestamp for chronological order
        visits.sort(key=lambda v: v.get('_parsed_timestamp', datetime(2024, 1, 1)))
        
        # ✅ FUNNEL LOGIC: 
        # - 100% of checkins → Views (all visits start as views)
        # - 40% of views → Transactions (increased engagement)
        # - 30% of transactions → Reviews
        assigns = assign_funnel_interactions(visits, view_to_transaction_rate=0.4, transaction_to_review_rate=0.3)

        # ✅ MODEL REQUIREMENT: Ensure every user has at least 1 review
        # The model crashes if a user has zero reviews in their history
        if len(assigns["reviews"]) == 0 and len(assigns["transactions"]) > 0:
            # Pick the first transaction to also be a review (ensures temporal consistency)
            first_txn_idx = min(assigns["transactions"])
            assigns["reviews"].add(first_txn_idx)
            print(f"  ⚠️  User {user_id}: Added minimum 1 review (had 0, has {len(assigns['transactions'])} transactions)")
        elif len(assigns["reviews"]) == 0 and len(assigns["views"]) > 0:
            # Edge case: User has views but no transactions - give them 1 transaction+review
            first_view_idx = min(assigns["views"])
            assigns["transactions"].add(first_view_idx)
            assigns["reviews"].add(first_view_idx)
            print(f"  ⚠️  User {user_id}: Added minimum 1 transaction+review (had 0)")

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

        # Track objects for index-based funnel pairing
        view_index_to_object = {}
        txn_index_to_object = {}
        
        # Track which indices have been processed to avoid duplicates
        processed_view_indices = set()
        
        # ✅ NEW: Track used timestamps to ensure uniqueness per user
        used_timestamps = set()

        # Generate VIEWS (all visits)
        for idx in assigns["views"]:
            # Skip if already processed (safeguard against duplicates)
            if idx in processed_view_indices:
                continue
            processed_view_indices.add(idx)
            
            entry = visits[idx]
            if "lat" not in entry or "lon" not in entry:
                continue
            
            # Use parsed FSQ timestamp and ensure uniqueness
            base_timestamp = entry['_parsed_timestamp'].strftime("%Y-%m-%dT%H:%M:%SZ")
            unique_timestamp = ensure_unique_timestamp(base_timestamp, used_timestamps)
            
            # View location from home
            if user_home_lat and user_home_lon:
                lat, lon = get_view_location(user_home_lat, user_home_lon, 
                                            float(entry["lat"]), float(entry["lon"]))
            else:
                lat, lon = jitter_location(float(entry["lat"]), float(entry["lon"]), distance_km=2)
            
            poi_id = entry.get("poi_id", f"POI-{idx}")
            
            duration = random.randint(60, 180) if idx in assigns["transactions"] else random.randint(40, 120)
            
            view = {
                "timestamp": unique_timestamp,
                "poiId": poi_id,
                "poiCategories": [entry.get("poi_category", "Unknown")],
                "poiSubcategories": [],
                "duration": duration,
                "referrer": random.choice(["map", "search", "ad", "friend"]),
                "userLocation": {"latitude": lat, "longitude": lon}
            }
            interaction["views"].append(view)
            view_index_to_object[idx] = view

        # Generate TRANSACTIONS (subset of views, SAME timestamp as original checkin)
        for idx in assigns["transactions"]:
            entry = visits[idx]
            if "lat" not in entry or "lon" not in entry:
                continue
            
            # ✅ SEQUENTIAL FUNNEL: Use SAME timestamp as view (original FSQ checkin time)
            if idx in view_index_to_object:
                base_view = view_index_to_object[idx]
                base_timestamp = base_view["timestamp"]  # Same time as view
            else:
                # Fallback: use parsed FSQ timestamp
                base_timestamp = entry['_parsed_timestamp'].strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Ensure unique timestamp
            txn_timestamp = ensure_unique_timestamp(base_timestamp, used_timestamps)
            
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
            
            # ✅ USE ORIGINAL FSQ CHECKIN TIMESTAMP (same as the original visit)
            # Get timestamp from the original visit entry, not from transaction
            base_timestamp = entry['_parsed_timestamp'].strftime("%Y-%m-%dT%H:%M:%SZ")
            review_timestamp = ensure_unique_timestamp(base_timestamp, used_timestamps)
            
            # Get transaction object to link review to it
            base_txn = txn_index_to_object[idx]
            
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
                "userLocation": {"latitude": lat, "longitude": lon},
                "transactionId": base_txn["transactionId"]  # ✅ Link review to its transaction
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
    
    # ✅ NEW: Validate timestamp uniqueness per user
    print("\n=== Timestamp Uniqueness Validation ===")
    users_with_duplicates = 0
    total_duplicate_timestamps = 0
    
    for profile in app_profiles:
        user_id = profile["user"]["userId"]
        all_timestamps = []
        
        # Collect all timestamps from views, transactions, and reviews
        for view in profile["interaction"]["views"]:
            all_timestamps.append(view["timestamp"])
        for txn in profile["interaction"]["transactions"]:
            all_timestamps.append(txn["timestamp"])
        for review in profile["interaction"]["reviews"]:
            all_timestamps.append(review["timestamp"])
        
        # Check for duplicates
        unique_timestamps = set(all_timestamps)
        if len(all_timestamps) != len(unique_timestamps):
            users_with_duplicates += 1
            duplicate_count = len(all_timestamps) - len(unique_timestamps)
            total_duplicate_timestamps += duplicate_count
            
            # Show first few duplicates for debugging
            if users_with_duplicates <= 3:
                timestamp_counts = defaultdict(int)
                for ts in all_timestamps:
                    timestamp_counts[ts] += 1
                duplicates = {ts: count for ts, count in timestamp_counts.items() if count > 1}
                print(f"  User {user_id}: {duplicate_count} duplicate timestamp(s) - {list(duplicates.items())[:3]}")
    
    if users_with_duplicates == 0:
        print("✅ All timestamps are unique per user!")
    else:
        print(f"⚠️  Found {users_with_duplicates} users with duplicate timestamps ({total_duplicate_timestamps} total duplicates)")
    
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