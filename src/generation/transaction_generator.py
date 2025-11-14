# -*- coding: utf-8 -*-
"""
Direct FSQ to Transactions Conversion
Converts Foursquare checkin data directly to transaction data only.
Views and reviews are kept as empty placeholders (0 entries).
Uses original timestamps from FSQ data - no ML temporal models needed.
"""
import json
import random
from collections import defaultdict
from datetime import datetime, timedelta
from tqdm import tqdm
import os
from typing import Any, Dict, List, Set

def safe_string(val: Any, default: str = "Unknown") -> str:
    """
    Safely convert value to string, handling NaN, None, and empty values.
    
    Parameters:
    val (Any): The value to convert to string.
    default (str): Default value if conversion fails.
    
    Returns:
    str: The converted string value or default.
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

# Input: Direct FSQ JSON (user_id + user_metadata with timestamps)
# Output: Synthetic dataset with transactions only (views and reviews as empty placeholders)

# Import paths from config module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.paths import (
    INPUT_TO_GENERATOR_FILTERED_JSON,
    INPUT_TO_GENERATOR_ALL_CATEGORIES_JSON,
    SYNTHETIC_FILTERED_JSON,
    SYNTHETIC_ALL_CATEGORIES_JSON
)


# -----------------------------
# Helper Functions
# -----------------------------

def random_payment() -> str:
    """
    Generate a random payment method.
    
    Returns:
    str: Random payment method (credit_card, mobile_wallet, or cash).
    """
    return random.choice(["credit_card", "mobile_wallet", "cash"])

def random_rating() -> float:
    """
    Generate a random rating between 3.5 and 5.0.
    
    Returns:
    float: Random rating value rounded to 1 decimal place.
    """
    return round(random.uniform(3.5, 5.0), 1)

def random_amount(category: str) -> float:
    """
    Calculate random transaction amount based on POI category.
    
    Parameters:
    category (str): POI category (e.g., 'restaurant', 'coffee shop').
    
    Returns:
    float: Random amount with ±20% variance from base price.
    """
    base = {
        "restaurant": 20,
        "nightclub": 35,
        "pub": 25,
        "coffee shop": 6,
        "dim sum restaurant": 15
    }
    return round(random.uniform(0.8, 1.2) * base.get(category.lower(), 12), 2)

def parse_fsq_timestamp(visit_entry: Dict[str, Any]) -> datetime:
    """
    Parse FSQ timestamp from ISO 8601 format.
    
    Parameters:
    visit_entry (Dict[str, Any]): FSQ check-in entry with timestamp field.
    
    Returns:
    datetime: Parsed datetime object from ISO 8601 timestamp.
    """
    try:
        # Use the timestamp field directly from input.json
        timestamp_str = visit_entry.get("timestamp", "")
        if timestamp_str:
            # Parse ISO 8601 format timestamp
            return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
        else:
            # Fallback: construct from date components (backward compatibility)
            month_name = visit_entry.get("month_of_year", "January")
            day_of_week = visit_entry.get("day_of_week", "Monday")
            time_str = visit_entry.get("time_of_day", "12:00 PM")
            
            year = 2024
            month_map = {
                "January": 1, "February": 2, "March": 3, "April": 4,
                "May": 5, "June": 6, "July": 7, "August": 8,
                "September": 9, "October": 10, "November": 11, "December": 12
            }
            month = month_map.get(month_name, 1)
            
            time_obj = datetime.strptime(time_str, "%I:%M %p")
            hour = time_obj.hour
            minute = time_obj.minute
            
            day_name_map = {
                "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                "Friday": 4, "Saturday": 5, "Sunday": 6
            }
            target_weekday = day_name_map.get(day_of_week, 0)
            
            for day in range(1, 29):
                test_date = datetime(year, month, day)
                if test_date.weekday() == target_weekday:
                    return datetime(year, month, day, hour, minute, 0)
            
            return datetime(year, month, 15, hour, minute, 0)
        
    except Exception as e:
        print(f"Warning: Failed to parse timestamp from {visit_entry}: {e}")
        return datetime(2024, 1, 1, 12, 0, 0)



def assign_transactions_only(visits: List[Dict[str, Any]]) -> Dict[str, Set[int]]:
    """
    Convert all FSQ check-ins to transaction indices.
    
    Parameters:
    visits (List[Dict[str, Any]]): List of check-in visit entries.
    
    Returns:
    Dict[str, Set[int]]: Dictionary with 'transactions' (all indices),
                         'views' (empty), and 'reviews' (empty).
    """
    n_visits = len(visits)
    
    if n_visits == 0:
        return {"views": set(), "transactions": set(), "reviews": set()}
    
    # Convert ALL visits to transactions
    transaction_indices = set(range(n_visits))
    view_indices = set()  # Empty - no views
    review_indices = set()  # Empty - no reviews
    
    return {
        "views": view_indices,
        "transactions": transaction_indices,
        "reviews": review_indices
    }

# -----------------------------
# Main Generation Logic
# -----------------------------

def process_input_to_synthetic(
    input_file: str,
    output_file: str,
    dataset_name: str
) -> List[Dict[str, Any]]:
    """
    Process input JSON and generate synthetic transaction dataset.
    
    Parameters:
    input_file (str): Path to input FSQ JSON file.
    output_file (str): Path to save synthetic output JSON.
    dataset_name (str): Dataset name for logging purposes.
    
    Returns:
    List[Dict[str, Any]]: List of user profiles with transactions.
    
    Raises:
    FileNotFoundError: If input file does not exist.
    json.JSONDecodeError: If input file is not valid JSON.
    PermissionError: If cannot write to output file.
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Load input data (direct FSQ translation)
    print(f"\nLoading data from: {input_file}")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            checkin_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: Input file not found: {input_file}")
        raise
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON in input file: {e}")
        raise
    except Exception as e:
        print(f"❌ ERROR: Failed to load input file: {e}")
        raise

    # Analyze POI popularity for statistics
    print("\n📊 Analyzing POI popularity...")
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
    
    # Get popular POIs for statistics
    popular_pois = sorted(global_poi_counts.items(), key=lambda x: x[1], reverse=True)[:200]
    popular_poi_ids = [poi_id for poi_id, _ in popular_pois]
    
    print(f"✅ Identified {len(popular_poi_ids)} popular POIs")
    print(f"   Top 5 POIs: {[f'{pid}({cnt})' for pid, cnt in popular_pois[:5]]}")

    app_profiles = []

    for user in tqdm(checkin_data, desc="Processing users"):
        user_id = str(user["user_id"])
        visits = user.get("user_metadata", [])
        if not visits or not isinstance(visits, list) or len(visits) == 0:
            continue
        
        # ✅ Skip users with no checkins
        if len(visits) < 1:
            print(f"  ⚠️  User {user_id}: Skipping user (no checkins)")
            continue

        # Direct transformation: Use original checkins as transactions
        
        # Sort visits by timestamp (construct from FSQ metadata)
        for visit in visits:
            visit['_parsed_timestamp'] = parse_fsq_timestamp(visit)
        
        # Sort by parsed timestamp for chronological order
        visits.sort(key=lambda v: v.get('_parsed_timestamp', datetime(2024, 1, 1)))
        
        # Convert all FSQ checkins to transactions only
        assigns = assign_transactions_only(visits)

        # All checkins become transactions, views and reviews remain empty

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
        
        # User location is handled per transaction (no global home location needed)

        interaction = {"views": [], "transactions": [], "reviews": []}


        
        # Generate VIEWS (placeholder - empty for direct transaction translation)
        # Views array remains empty as per requirement

        # Generate TRANSACTIONS (subset of views, SAME timestamp as original checkin)
        for idx in assigns["transactions"]:
            entry = visits[idx]
            if "lat" not in entry or "lon" not in entry:
                continue
            
            # Use exact FSQ timestamp (NO OFFSET)
            timestamp = entry['_parsed_timestamp'].strftime("%Y-%m-%dT%H:%M:%SZ")
            
            # Use exact POI location from original checkin
            try:
                lat, lon = float(entry["lat"]), float(entry["lon"])
            except (ValueError, TypeError) as e:
                print(f"  ⚠️  Warning: Invalid coordinates for user {user_id}, entry {idx}: {e}")
                continue
            
            poi_id = entry.get("poi_id", f"POI-{idx}")
            
            txn = {
                "timestamp": timestamp,
                "poiId": poi_id,
                # "poiName": safe_string(entry.get("poi_name"), "Unknown"),
                # "planning_area": entry.get("planning_area", "UNKNOWN"),
                "poiCategories": [entry["poi_category"]],
                "poiSubcategories": [],
                "transactionId": str(random.randint(100000, 999999)),
                "amount": random_amount(entry["poi_category"]),
                "currency": "SGD",
                "paymentMethod": random_payment(),
                "userLocation": {"latitude": lat, "longitude": lon},
                
            }
            interaction["transactions"].append(txn)

        # Generate REVIEWS (placeholder - empty for direct transaction translation)
        # Reviews array remains empty as per requirement

        # Sort transactions by timestamp (chronological order)
        interaction["transactions"].sort(key=lambda x: x["timestamp"])
        
        # Views and reviews arrays remain empty (placeholders)

        app_profiles.append({
            "user": user_block,
            "interaction": interaction
        })

    print(f"\n✅ Generated profiles for {len(app_profiles)} users")
    
    # Validate timestamp uniqueness per user
    print("\n=== Timestamp Uniqueness Validation ===")
    users_with_duplicates = 0
    total_duplicate_timestamps = 0
    
    for profile in app_profiles:
        user_id = profile["user"]["userId"]
        all_timestamps = []
        
        # Collect timestamps from transactions only (views and reviews are empty)
        for txn in profile["interaction"]["transactions"]:
            all_timestamps.append(txn["timestamp"])
        
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
    
    # Validate POI overlap and repetition
    print("\n=== POI Overlap & Repetition Analysis ===")
    all_poi_visits = defaultdict(int)
    user_poi_counts = []
    revisit_counts = []
    
    for profile in app_profiles:
        user_pois = defaultdict(int)
        for txn in profile["interaction"]["transactions"]:
            poi_id = txn["poiId"]
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
    
    # Validate temporal distribution
    print("\n=== Temporal Distribution Validation ===")
    all_timestamps = []
    for profile in app_profiles:
        for txn in profile["interaction"]["transactions"]:
            all_timestamps.append(txn["timestamp"])
    
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
    
    # Direct transaction conversion complete
    print("\n" + "="*60)
    print("DIRECT FSQ TO TRANSACTIONS CONVERSION COMPLETE")
    print("="*60)
    print(f"Total transactions generated: {sum(len(profile['interaction']['transactions']) for profile in app_profiles)}")
    print(f"Views: 0 (placeholder)")
    print(f"Reviews: 0 (placeholder)")
    print("✅ All FSQ checkins converted to transactions only")
    
    # Save to output file
    print(f"\nSaving to: {output_file}")
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(app_profiles, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved synthetic data to {output_file}")
    except PermissionError:
        print(f"❌ ERROR: Permission denied writing to: {output_file}")
        raise
    except Exception as e:
        print(f"❌ ERROR: Failed to save output file: {e}")
        raise
    
    return app_profiles

if __name__ == "__main__":
    
    print("="*60)
    print("SYNTHETIC DATA GENERATION - DUAL OUTPUT VERSION")
    print("="*60)
    print("Generating synthetic datasets from both input files:")
    print(f"1. Filtered categories: {INPUT_TO_GENERATOR_FILTERED_JSON}")
    print(f"2. All categories: {INPUT_TO_GENERATOR_ALL_CATEGORIES_JSON}")
    
    try:
        # Process filtered categories dataset
        print("\n" + "🎯" * 20)
        filtered_profiles = process_input_to_synthetic(
            INPUT_TO_GENERATOR_FILTERED_JSON, 
            SYNTHETIC_FILTERED_JSON, 
            "Filtered Categories Dataset"
        )
        
        # Process all categories dataset
        print("\n" + "📊" * 20)
        all_profiles = process_input_to_synthetic(
            INPUT_TO_GENERATOR_ALL_CATEGORIES_JSON, 
            SYNTHETIC_ALL_CATEGORIES_JSON, 
            "All Categories Dataset"
        )
        
        # Final summary
        print("\n" + "="*60)
        print("GENERATION COMPLETE - SUMMARY")
        print("="*60)
        print(f"📂 Output Files Generated:")
        print(f"   1. Filtered dataset: {SYNTHETIC_FILTERED_JSON}")
        print(f"      └── Users: {len(filtered_profiles)}")
        print(f"      └── Transactions: {sum(len(p['interaction']['transactions']) for p in filtered_profiles)}")
        print(f"   2. All categories dataset: {SYNTHETIC_ALL_CATEGORIES_JSON}")
        print(f"      └── Users: {len(all_profiles)}")
        print(f"      └── Transactions: {sum(len(p['interaction']['transactions']) for p in all_profiles)}")
        
        print(f"\n🎉 Successfully generated both synthetic datasets!")
        print(f"📊 Difference: {len(all_profiles) - len(filtered_profiles)} additional users in all-categories version")
        print(f"🔍 Use filtered version for focused analysis on relevant categories")
        print(f"🌐 Use all-categories version for comprehensive analysis")
        
    except FileNotFoundError as e:
        print(f"\n❌ FATAL ERROR: Required input file not found")
        print(f"   Please check that input files exist:")
        print(f"   - {INPUT_TO_GENERATOR_FILTERED_JSON}")
        print(f"   - {INPUT_TO_GENERATOR_ALL_CATEGORIES_JSON}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"\n❌ FATAL ERROR: Invalid JSON format in input file")
        print(f"   {e}")
        exit(1)
    except PermissionError as e:
        print(f"\n❌ FATAL ERROR: Permission denied")
        print(f"   Cannot write to output directory")
        exit(1)
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Process interrupted by user")
        exit(130)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: Unexpected error occurred")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)