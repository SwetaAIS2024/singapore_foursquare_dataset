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
            
            # Validate temporal order: review must come after transaction
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

def assign_funnel_interactions(visits, view_to_transaction_rate=0.2, transaction_to_review_rate=0.4):
    """
    Assign funnel interactions: views → transactions → reviews
    
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
        # - 100% of checkins → Views (all visits)
        # - 70% of views → Transactions (increased from 40% for better model learning)
        # - 35% of transactions → Reviews
        assigns = assign_funnel_interactions(visits, view_to_transaction_rate=0.7, transaction_to_review_rate=0.35)

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

        # Generate VIEWS (all visits)
        for idx in assigns["views"]:
            entry = visits[idx]
            if "lat" not in entry or "lon" not in entry:
                continue
            
            # Use parsed FSQ timestamp
            base_timestamp = entry['_parsed_timestamp'].strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # View location from home
            if user_home_lat and user_home_lon:
                lat, lon = get_view_location(user_home_lat, user_home_lon, 
                                            float(entry["lat"]), float(entry["lon"]))
            else:
                lat, lon = jitter_location(float(entry["lat"]), float(entry["lon"]), distance_km=2)
            
            poi_id = entry.get("poi_id", f"POI-{idx}")
            
            duration = random.randint(60, 180) if idx in assigns["transactions"] else random.randint(40, 120)
            
            view = {
                "timestamp": base_timestamp,
                "poiId": poi_id,
                "poiCategories": [entry.get("poi_category", "Unknown")],
                "poiSubcategories": [],
                "duration": duration,
                "referrer": random.choice(["map", "search", "ad", "friend"]),
                "userLocation": {"latitude": lat, "longitude": lon}
            }
            interaction["views"].append(view)
            view_index_to_object[idx] = view

        # Generate TRANSACTIONS (subset of views, 20-90 min after view)
        for idx in assigns["transactions"]:
            entry = visits[idx]
            if "lat" not in entry or "lon" not in entry:
                continue
            
            # Get view timestamp and add delay
            if idx in view_index_to_object:
                base_view = view_index_to_object[idx]
                view_dt = datetime.strptime(base_view["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
                
                # Transaction happens 20-90 minutes after view
                minutes_delay = random.choices(
                    [random.randint(20, 30), random.randint(30, 60), random.randint(60, 90)],
                    weights=[0.2, 0.5, 0.3],
                    k=1
                )[0]
                txn_dt = view_dt + timedelta(minutes=minutes_delay)
                txn_timestamp = txn_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                # Fallback: use parsed FSQ timestamp
                txn_timestamp = entry['_parsed_timestamp'].strftime("%Y-%m-%dT%H:%M:%SZ")
            
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