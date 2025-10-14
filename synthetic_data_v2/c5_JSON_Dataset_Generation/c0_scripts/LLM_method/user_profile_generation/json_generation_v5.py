import json
import random
import pandas as pd 
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import math
import calendar
from datetime import datetime 
from tqdm import tqdm

from transformers import pipeline

# -----------------------------
# Helper Functions
# -----------------------------

def jitter_location(lat, lon, distance_km=2):
    max_deg = distance_km / 111.0
    lat_jitter = random.uniform(-max_deg, max_deg)
    lon_jitter = random.uniform(-max_deg, max_deg) / math.cos(math.radians(lat))
    return round(lat + lat_jitter, 6), round(lon + lon_jitter, 6)

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
    # Placeholder for LLM review generation
    return "Review text placeholder."

def generate_timestamp(day_of_week, time_of_day, month_of_year, year=2025, jitter_days=0, jitter_hours=0, jitter_minutes=0, jitter_seconds=0):
    month_num = list(calendar.month_name).index(month_of_year)
    for day in range(1, 32):
        try:
            dt = datetime(year, month_num, day)
            if dt.strftime("%A") == day_of_week:
                time_obj = datetime.strptime(time_of_day, "%I:%M %p")
                dt = dt.replace(hour=time_obj.hour, minute=time_obj.minute)
                dt += timedelta(days=jitter_days, hours=jitter_hours, minutes=jitter_minutes, seconds=jitter_seconds)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
    return f"{year}-{month_num:02d}-01T00:00:00Z"

def load_cluster_patterns(cluster_file_path):
    """Load cluster patterns from CSV to understand user preferences"""
    cluster_df = pd.read_csv(cluster_file_path)
    cluster_preferences = {}
    
    for _, row in cluster_df.iterrows():
        cluster_id = row['cluster']
        # Get top categories for this cluster (exclude cluster column)
        category_counts = row.drop('cluster').to_dict()
        
        # Calculate total visits for this cluster
        total_visits = sum(category_counts.values())
        
        if total_visits == 0:
            continue
            
        # Calculate preference weights (normalize to probabilities)
        preferences = {cat: count/total_visits for cat, count in category_counts.items() if count > 0}
        
        # Sort by preference strength
        sorted_preferences = dict(sorted(preferences.items(), key=lambda x: x[1], reverse=True))
        
        cluster_preferences[cluster_id] = {
            'total_visits': total_visits,
            'preferences': sorted_preferences,
            'top_categories': list(sorted_preferences.keys())[:5]  # Top 5 categories
        }
    
    return cluster_preferences

def analyze_cluster_category_patterns(cluster_file_path):
    """
    Analyze which categories are most transaction-heavy based on cluster data
    """
    df = pd.read_csv(cluster_file_path)
    
    # Calculate category statistics
    category_stats = {}
    
    for col in df.columns:
        if col == 'cluster':
            continue
            
        category = col.replace(' ', '_').lower()
        values = df[col].values
        
        # Calculate metrics
        total_occurrences = values.sum()
        avg_per_cluster = values.mean()
        max_in_cluster = values.max()
        clusters_with_category = (values > 0).sum()
        
        # Transaction likelihood score (higher = more likely to transact)
        # Based on: total volume, consistency across clusters, peak usage
        consistency_score = clusters_with_category / len(df)  # How many clusters use this
        volume_score = total_occurrences / df.iloc[:, 1:].sum().sum() if df.iloc[:, 1:].sum().sum() > 0 else 0
        peak_score = max_in_cluster / total_occurrences if total_occurrences > 0 else 0
        
        # Combine scores (weights can be adjusted)
        transaction_score = (
            0.4 * volume_score +      # High total volume
            0.3 * consistency_score + # Used across many clusters
            0.3 * peak_score         # High peak usage in some clusters
        )
        
        category_stats[category] = {
            'total_occurrences': total_occurrences,
            'avg_per_cluster': avg_per_cluster,
            'max_in_cluster': max_in_cluster,
            'clusters_with_category': clusters_with_category,
            'consistency_score': consistency_score,
            'transaction_score': transaction_score
        }
    
    # Sort by transaction score
    sorted_categories = sorted(category_stats.items(), 
                              key=lambda x: x[1]['transaction_score'], 
                              reverse=True)
    
    print("=== Category Transaction Likelihood Analysis ===")
    print("Top 15 categories most likely to generate transactions:")
    print(f"{'Category':<25} {'Total':<8} {'Clusters':<9} {'Trans Score':<12} {'Multiplier':<10}")
    print("-" * 75)
    
    # Create multipliers based on transaction scores
    multipliers = {}
    all_scores = [stats['transaction_score'] for _, stats in sorted_categories if stats['transaction_score'] > 0]
    
    if all_scores:
        mean_score = sum(all_scores) / len(all_scores)
        std_score = (sum((x - mean_score) ** 2 for x in all_scores) / len(all_scores)) ** 0.5
        
        for i, (category, stats) in enumerate(sorted_categories[:15]):
            if stats['transaction_score'] > 0:
                # Convert transaction score to multiplier
                if std_score > 0:
                    # Higher scores get higher multipliers
                    multiplier = 1.0 + (stats['transaction_score'] - mean_score) / std_score * 0.8
                    multiplier = max(0.5, min(2.5, multiplier))  # Cap between 0.5 and 2.5
                else:
                    multiplier = 1.0
                    
                multipliers[category] = multiplier
                
                print(f"{category:<25} {stats['total_occurrences']:<8.0f} "
                      f"{stats['clusters_with_category']:<9.0f} "
                      f"{stats['transaction_score']:<12.3f} {multiplier:<10.2f}")
    
    return multipliers

def assign_user_to_cluster(visits, cluster_preferences):
    """Assign user to most similar cluster based on their visit patterns"""
    user_categories = {}
    
    # Count user's category visits
    for visit in visits:
        category = visit.get("poi_category", "").lower().replace(" ", "_")
        user_categories[category] = user_categories.get(category, 0) + 1
    
    # Calculate similarity to each cluster
    best_cluster = 0
    best_similarity = 0
    
    for cluster_id, cluster_data in cluster_preferences.items():
        similarity = 0
        user_total = sum(user_categories.values())
        
        if user_total == 0:
            continue
            
        for category, user_count in user_categories.items():
            if category in cluster_data['preferences']:
                # Similarity based on preference alignment
                user_pref = user_count / user_total
                cluster_pref = cluster_data['preferences'][category]
                similarity += min(user_pref, cluster_pref)  # Overlap similarity
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_cluster = cluster_id
    
    return best_cluster

def assign_funnel_interactions_with_clusters(visits, cluster_preferences, assigned_cluster, 
                                           transaction_multipliers,
                                           view_to_transaction_rate=0.2, transaction_to_review_rate=0.4):
    """
    Enhanced funnel assignment using cluster-based behavioral patterns and data-driven multipliers
    """
    n_visits = len(visits)
    views_indices = set(range(n_visits))
    
    # Get cluster preferences
    cluster_data = cluster_preferences.get(assigned_cluster, {})
    cluster_prefs = cluster_data.get('preferences', {})
    top_categories = cluster_data.get('top_categories', [])
    
    print(f"User assigned to Cluster {assigned_cluster}")
    print(f"Cluster's top categories: {top_categories[:3]}")
    
    # Calculate transaction weights based on cluster preferences
    transaction_weights = []
    
    for i, visit in enumerate(visits):
        category = visit.get("poi_category", "").lower().replace(" ", "_")
        day_of_week = visit.get("day_of_week", "")
        time_of_day = visit.get("time_of_day", "12:00 PM")
        
        # Base weight
        weight = 1.0
        
        # 1. Cluster-based preference (MOST IMPORTANT)
        if category in cluster_prefs:
            cluster_preference = cluster_prefs[category]
            # Strong preference multiplier based on cluster data
            if cluster_preference > 0.1:  # High preference in cluster
                weight *= 3.0
            elif cluster_preference > 0.05:  # Medium preference
                weight *= 2.0
            elif cluster_preference > 0.02:  # Low preference
                weight *= 1.5
        else:
            # Category not preferred by this cluster type
            weight *= 0.3
        
        # 2. Data-driven transaction multipliers (REPLACES HARDCODED VALUES)
        category_multiplier = transaction_multipliers.get(category, 1.0)
        weight *= category_multiplier
        
        # 3. Time-based patterns
        try:
            hour = datetime.strptime(time_of_day, "%I:%M %p").hour
            if 12 <= hour <= 14 or 18 <= hour <= 22:  # Meal times
                weight *= 1.4
            elif 22 <= hour or hour <= 2:  # Late night
                weight *= 1.2
        except:
            pass
        
        # 4. Weekend boost
        if day_of_week in ["Friday", "Saturday"]:
            weight *= 1.3
        
        transaction_weights.append(weight)
    
    # Select transactions based on weighted probabilities
    n_transactions = int(n_visits * view_to_transaction_rate)
    transaction_indices = set()
    
    if n_transactions > 0:
        # Weighted selection without replacement
        available_indices = list(range(n_visits))
        available_weights = transaction_weights.copy()
        
        for _ in range(min(n_transactions, len(available_indices))):
            if not available_indices:
                break
            
            selected_idx = random.choices(available_indices, weights=available_weights, k=1)[0]
            transaction_indices.add(selected_idx)
            
            # Remove selected index
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
            # Use data-driven multipliers for reviews too
            weight *= transaction_multipliers.get(category, 1.0)
            
            # Cluster preference also affects review likelihood
            if category in cluster_prefs and cluster_prefs[category] > 0.05:
                weight *= 1.5
            
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
        "views": views_indices,
        "transactions": transaction_indices,
        "reviews": review_indices,
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
    """
    Original random assignment function (fallback)
    """
    n_visits = len(visits)
    
    # All visits are views
    views_indices = set(range(n_visits))
    
    # Randomly select transactions from views
    n_transactions = int(n_visits * view_to_transaction_rate)
    transaction_indices = set(random.sample(list(views_indices), min(n_transactions, len(views_indices))))
    
    # Randomly select reviews from transactions
    n_reviews = int(len(transaction_indices) * transaction_to_review_rate)
    review_indices = set(random.sample(list(transaction_indices), min(n_reviews, len(transaction_indices))))
    
    assignments = {
        "views": views_indices,
        "transactions": transaction_indices,
        "reviews": review_indices
    }
    
    return assignments

# -----------------------------
# Main Generation Logic
# -----------------------------

if __name__ == "__main__":
    
    # Load cluster preferences and calculate transaction multipliers
    cluster_file_path = "c2_Clustering_and_Analysis/c2_post_clustering_analysis/c1_output/kmeans/poi_category_by_cluster.csv"
    
    try:
        cluster_preferences = load_cluster_patterns(cluster_file_path)
        
        # Calculate data-driven transaction multipliers
        print("Analyzing cluster data for transaction patterns...")
        transaction_multipliers = analyze_cluster_category_patterns(cluster_file_path)
        print(f"\nLoaded {len(cluster_preferences)} cluster patterns")
        print(f"Generated {len(transaction_multipliers)} category multipliers")
        use_clusters = True
    except FileNotFoundError:
        print(f"Cluster file not found at {cluster_file_path}. Using random assignment.")
        cluster_preferences = {}
        transaction_multipliers = {}
        use_clusters = False
    except Exception as e:
        print(f"Error loading cluster data: {e}. Using random assignment.")
        cluster_preferences = {}
        transaction_multipliers = {}
        use_clusters = False

    # review_llm = pipeline("text2text-generation", model="google/flan-t5-small", device=0)  # CPU

    with open("c5_JSON_Dataset_Generation/c0_scripts/LLM_method/input_dataset/final_checkin_file.json", "r", encoding="utf-8") as f:
        checkin_data = json.load(f)

    with open("c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_summary_extraction/user_summaries_hf.csv", "r", encoding="utf-8") as f:
        user_summaries = json.load(f)
 
    with open("c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_profile_generation/master_poi_pool.json", "r", encoding="utf-8") as f:
        poi_master_data = json.load(f)

    app_profiles = []

    for user in tqdm(checkin_data, desc="Processing users"):
        user_id = str(user["user_id"])
        visits = user.get("user_metadata", [])
        if not visits or not isinstance(visits, list) or len(visits) == 0:
            continue

        random.shuffle(visits)
        
        if use_clusters:
            # Assign user to cluster based on their behavior
            assigned_cluster = assign_user_to_cluster(visits, cluster_preferences)
            
            # Use cluster-based funnel assignment with data-driven multipliers
            assigns = assign_funnel_interactions_with_clusters(
                visits, cluster_preferences, assigned_cluster, transaction_multipliers,
                view_to_transaction_rate=0.2, transaction_to_review_rate=0.4
            )
            
            # Optional: Analyze patterns for first few users
            if len(app_profiles) < 3:
                analyze_cluster_transaction_patterns(assigns, visits, assigned_cluster)
                validate_and_print_patterns(assigns, visits)
        else:
            # Fallback to original random assignment
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

        interaction = {"views": [], "transactions": [], "reviews": []}
        base_date = datetime(2025, 4, 1)

        # --- Build all views first (for duration split) ---
        views_temp = []
        for idx, entry in enumerate(visits):
            jitter_days = random.randint(1, 7)
            jitter_hours = random.randint(20, 23)
            jitter_minutes = random.randint(0, 59)
            jitter_seconds = random.randint(0, 59)
            if "lat" not in entry or "lon" not in entry:
                continue
            lat, lon = jitter_location(float(entry["lat"]), float(entry["lon"]), distance_km=random.uniform(2, 50))
            poi_id = entry.get("poi_id", f"POI-{idx}")
            timestamp = generate_timestamp(entry["day_of_week"], entry["time_of_day"], entry["month_of_year"],
                                          jitter_days=jitter_days, jitter_hours=jitter_hours, jitter_minutes=jitter_minutes, jitter_seconds=jitter_seconds)
            if idx in assigns["views"]:
                view = {
                    "timestamp": timestamp,
                    "poiId": poi_id,
                    "poiCategories": [entry["poi_category"]],
                    "poiSubcategories": [],
                    # "duration": to be assigned later
                    "referrer": random.choice(["map", "search", "ad", "friend"]),
                    "userLocation": {"latitude": lat, "longitude": lon}
                }
                views_temp.append(view)
        n_views = len(views_temp)
        n_high = int(n_views * 0.7)
        indices = list(range(n_views))
        random.shuffle(indices)
        for idx in indices[:n_high]:
            views_temp[idx]['duration'] = random.randint(40, 120)
        for idx in indices[n_high:]:
            views_temp[idx]['duration'] = random.randint(1, 19)
        interaction["views"].extend(views_temp)

        # --- Transactions ---
        for idx, entry in enumerate(visits):
            if "lat" not in entry or "lon" not in entry:
                continue
            # lat, lon = jitter_location(float(entry["lat"]), float(entry["lon"]), distance_km=random.uniform(2, 50))
            lat, lon = entry["lat"], entry["lon"]
            poi_id = entry.get("poi_id", f"POI-{idx}")
            timestamp = generate_timestamp(entry["day_of_week"], entry["time_of_day"], entry["month_of_year"],
                                          jitter_days=0, jitter_hours=0, jitter_minutes=0, jitter_seconds=random.randint(0, 59))    
            if idx in assigns["transactions"]:
                txn = {
                    "timestamp": timestamp,
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

        # --- Reviews ---
        for idx, entry in enumerate(visits):
            jitter_days = random.randint(1, 7)
            jitter_hours = random.randint(20, 23)
            jitter_minutes = random.randint(0, 59)
            jitter_seconds = random.randint(0, 59)
            if "lat" not in entry or "lon" not in entry:
                continue
            lat, lon = jitter_location(float(entry["lat"]), float(entry["lon"]), distance_km=random.uniform(2, 50))
            poi_id = entry.get("poi_id", f"POI-{idx}")
            timestamp = generate_timestamp(entry["day_of_week"], entry["time_of_day"], entry["month_of_year"],
                                          jitter_days=jitter_days, jitter_hours=jitter_hours, jitter_minutes=jitter_minutes, jitter_seconds=jitter_seconds)
            if idx in assigns["reviews"]:
                review = {
                    "timestamp": timestamp,
                    "poiId": poi_id,
                    "poiCategories": [entry["poi_category"]],
                    "poiSubcategories": [],
                    "rating": random_rating(),
                    "reviewText": generate_review_text(user_id, poi_id, entry["poi_category"]),
                    "userLocation": {"latitude": lat, "longitude": lon}
                }
                interaction["reviews"].append(review)

        app_profiles.append({
            "user": user_block,
            "interaction": interaction
        })

    print(f"\nGenerated profiles for {len(app_profiles)} users")
    
    # Save with cluster-based filename
    output_filename = "c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_profile_generation/app_profiles_all_users_version_5_cluster_based.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(app_profiles, f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_filename}")