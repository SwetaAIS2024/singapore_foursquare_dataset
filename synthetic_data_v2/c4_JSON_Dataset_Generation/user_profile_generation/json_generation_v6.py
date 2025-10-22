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
import os

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

def load_cluster_patterns_enhanced(cluster_file_path, poi_analysis_file_path=None):
    """Load cluster patterns from both category and POI analysis files"""
    patterns = {}
    
    # Load POI category patterns
    if os.path.exists(cluster_file_path):
        cluster_df = pd.read_csv(cluster_file_path)
        
        for _, row in cluster_df.iterrows():
            cluster_id = row['cluster']
            category_counts = row.drop('cluster').to_dict()
            total_visits = sum(category_counts.values())
            
            if total_visits == 0:
                continue
                
            category_preferences = {cat: count/total_visits for cat, count in category_counts.items() if count > 0}
            
            patterns[cluster_id] = {
                'total_visits': total_visits,
                'category_preferences': dict(sorted(category_preferences.items(), key=lambda x: x[1], reverse=True)),
                'top_categories': list(sorted(category_preferences.items(), key=lambda x: x[1], reverse=True))[:5]
            }
    
    # Load POI ID patterns if available
    if poi_analysis_file_path and os.path.exists(poi_analysis_file_path):
        print(f"[INFO] Loading POI patterns from: {poi_analysis_file_path}")
        poi_df = pd.read_csv(poi_analysis_file_path)
        
        for cluster_id in patterns:
            cluster_poi_data = poi_df[poi_df['cluster'] == cluster_id]
            
            if not cluster_poi_data.empty:
                # Get top POIs for this cluster
                top_pois = cluster_poi_data.nlargest(20, 'visit_count')  # Top 20 POIs
                poi_preferences = {}
                total_poi_visits = cluster_poi_data['visit_count'].sum()
                
                for _, poi_row in top_pois.iterrows():
                    poi_id = str(poi_row['place_id'])  # Ensure string
                    visit_count = poi_row['visit_count']
                    poi_preferences[poi_id] = visit_count / total_poi_visits if total_poi_visits > 0 else 0
                
                patterns[cluster_id]['poi_preferences'] = poi_preferences
                patterns[cluster_id]['top_pois'] = list(poi_preferences.keys())[:10]  # Top 10 POIs
                
                print(f"[INFO] Cluster {cluster_id}: {len(poi_preferences)} POI preferences loaded")
            else:
                patterns[cluster_id]['poi_preferences'] = {}
                patterns[cluster_id]['top_pois'] = []
    else:
        print("[WARNING] POI analysis file not found - using category-only patterns")
        # Add empty POI preferences for all clusters
        for cluster_id in patterns:
            patterns[cluster_id]['poi_preferences'] = {}
            patterns[cluster_id]['top_pois'] = []
    
    return patterns

def analyze_cluster_category_and_poi_patterns(cluster_category_file, poi_analysis_file=None):
    """
    Enhanced analysis including both category and POI patterns
    """
    # Analyze categories (existing logic)
    category_multipliers = analyze_cluster_category_patterns(cluster_category_file)
    
    # Analyze POI patterns if file available
    poi_multipliers = {}
    
    if poi_analysis_file and os.path.exists(poi_analysis_file):
        try:
            print(f"[INFO] Analyzing POI patterns from: {poi_analysis_file}")
            poi_df = pd.read_csv(poi_analysis_file)
            
            # Calculate POI-level transaction likelihood
            poi_stats = {}
            
            for poi_id in poi_df['place_id'].unique():
                poi_data = poi_df[poi_df['place_id'] == poi_id]
                
                total_visits = poi_data['visit_count'].sum()
                clusters_with_poi = len(poi_data)
                max_visits_in_cluster = poi_data['visit_count'].max()
                
                # POI transaction score
                volume_score = total_visits / poi_df['visit_count'].sum()
                consistency_score = clusters_with_poi / poi_df['cluster'].nunique()
                peak_score = max_visits_in_cluster / total_visits if total_visits > 0 else 0
                
                transaction_score = (
                    0.4 * volume_score +
                    0.3 * consistency_score +
                    0.3 * peak_score
                )
                
                poi_stats[str(poi_id)] = {  # Ensure string key
                    'transaction_score': transaction_score,
                    'total_visits': total_visits
                }
            
            # Convert to multipliers
            all_poi_scores = [stats['transaction_score'] for stats in poi_stats.values() if stats['transaction_score'] > 0]
            
            if all_poi_scores:
                mean_score = sum(all_poi_scores) / len(all_poi_scores)
                std_score = (sum((x - mean_score) ** 2 for x in all_poi_scores) / len(all_poi_scores)) ** 0.5
                
                for poi_id, stats in poi_stats.items():
                    if stats['transaction_score'] > 0 and std_score > 0:
                        multiplier = 1.0 + (stats['transaction_score'] - mean_score) / std_score * 0.6
                        multiplier = max(0.6, min(2.0, multiplier))
                        poi_multipliers[poi_id] = multiplier
            
            print(f"\n=== POI Transaction Likelihood Analysis ===")
            print(f"Analyzed {len(poi_multipliers)} POIs with transaction multipliers")
            
            # Show top POIs
            sorted_pois = sorted(poi_multipliers.items(), key=lambda x: x[1], reverse=True)[:10]
            print("Top 10 POIs most likely to generate transactions:")
            print(f"{'POI ID':<15} {'Multiplier':<10} {'Total Visits':<12}")
            print("-" * 40)
            for poi_id, multiplier in sorted_pois:
                total_visits = poi_stats[poi_id]['total_visits']
                print(f"{poi_id:<15} {multiplier:<10.2f} {total_visits:<12.0f}")
        
        except Exception as e:
            print(f"Error analyzing POI patterns: {e}")
    else:
        print("[WARNING] POI analysis file not provided - using category-only multipliers")
    
    return category_multipliers, poi_multipliers

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
    
    print("\n=== Category Transaction Likelihood Analysis ===")
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

def assign_user_to_cluster_enhanced(visits, cluster_preferences):
    """Enhanced user-to-cluster assignment using both categories and POI IDs"""
    user_categories = {}
    user_pois = {}
    
    # Count user's category and POI visits
    for visit in visits:
        category = visit.get("poi_category", "").lower().replace(" ", "_")
        poi_id = str(visit.get("poi_id", ""))
        
        user_categories[category] = user_categories.get(category, 0) + 1
        if poi_id:
            user_pois[poi_id] = user_pois.get(poi_id, 0) + 1
    
    # Calculate similarity to each cluster
    best_cluster = 0
    best_similarity = 0
    
    for cluster_id, cluster_data in cluster_preferences.items():
        category_similarity = 0
        poi_similarity = 0
        
        user_total = sum(user_categories.values())
        if user_total == 0:
            continue
        
        # Category-based similarity
        category_prefs = cluster_data.get('category_preferences', {})
        for category, user_count in user_categories.items():
            if category in category_prefs:
                user_pref = user_count / user_total
                cluster_pref = category_prefs[category]
                category_similarity += min(user_pref, cluster_pref)
        
        # POI-based similarity (if available)
        poi_prefs = cluster_data.get('poi_preferences', {})
        if poi_prefs and user_pois:
            poi_total = sum(user_pois.values())
            for poi_id, user_count in user_pois.items():
                if poi_id in poi_prefs:
                    user_poi_pref = user_count / poi_total
                    cluster_poi_pref = poi_prefs[poi_id]
                    poi_similarity += min(user_poi_pref, cluster_poi_pref)
        
        # Combined similarity (weighted)
        total_similarity = 0.6 * category_similarity + 0.4 * poi_similarity
        
        if total_similarity > best_similarity:
            best_similarity = total_similarity
            best_cluster = cluster_id
    
    return best_cluster

def assign_funnel_interactions_with_clusters_enhanced(visits, cluster_preferences, assigned_cluster, 
                                                    category_multipliers, poi_multipliers,
                                                    view_to_transaction_rate=0.2, transaction_to_review_rate=0.4):
    """
    Enhanced funnel assignment using both category and POI-level patterns
    """
    n_visits = len(visits)
    views_indices = set(range(n_visits))
    
    # Get cluster preferences
    cluster_data = cluster_preferences.get(assigned_cluster, {})
    category_prefs = cluster_data.get('category_preferences', {})
    poi_prefs = cluster_data.get('poi_preferences', {})
    top_categories = [cat for cat, _ in cluster_data.get('top_categories', [])][:3]
    top_pois = cluster_data.get('top_pois', [])[:5]
    
    print(f"User assigned to Cluster {assigned_cluster}")
    print(f"Cluster's top categories: {top_categories}")
    print(f"Cluster's top POIs: {top_pois}")
    
    # Calculate enhanced transaction weights
    transaction_weights = []
    
    for i, visit in enumerate(visits):
        category = visit.get("poi_category", "").lower().replace(" ", "_")
        poi_id = str(visit.get("poi_id", ""))
        day_of_week = visit.get("day_of_week", "")
        time_of_day = visit.get("time_of_day", "12:00 PM")
        
        # Base weight
        weight = 1.0
        
        # 1. Category-based cluster preference
        if category in category_prefs:
            cluster_preference = category_prefs[category]
            if cluster_preference > 0.1:
                weight *= 3.0
            elif cluster_preference > 0.05:
                weight *= 2.0
            elif cluster_preference > 0.02:
                weight *= 1.5
        else:
            weight *= 0.3
        
        # 2. POI-based cluster preference (NEW!)
        if poi_id and poi_id in poi_prefs:
            poi_cluster_preference = poi_prefs[poi_id]
            if poi_cluster_preference > 0.05:  # High POI preference
                weight *= 2.5
            elif poi_cluster_preference > 0.02:  # Medium POI preference
                weight *= 2.0
            elif poi_cluster_preference > 0.01:  # Low POI preference
                weight *= 1.5
            
            print(f"  POI {poi_id} has cluster preference: {poi_cluster_preference:.3f} -> weight boost")
        
        # 3. Data-driven category multipliers
        category_multiplier = category_multipliers.get(category, 1.0)
        weight *= category_multiplier
        
        # 4. Data-driven POI multipliers (NEW!)
        if poi_id and poi_id in poi_multipliers:
            poi_multiplier = poi_multipliers[poi_id]
            weight *= poi_multiplier
            print(f"  POI {poi_id} has global multiplier: {poi_multiplier:.2f}")
        
        # 5. Time-based patterns
        try:
            hour = datetime.strptime(time_of_day, "%I:%M %p").hour
            if 12 <= hour <= 14 or 18 <= hour <= 22:
                weight *= 1.4
            elif 22 <= hour or hour <= 2:
                weight *= 1.2
        except:
            pass
        
        # 6. Weekend boost
        if day_of_week in ["Friday", "Saturday"]:
            weight *= 1.3
        
        transaction_weights.append(weight)
    
    # Select transactions (same logic as before)
    n_transactions = int(n_visits * view_to_transaction_rate)
    transaction_indices = set()
    
    if n_transactions > 0:
        available_indices = list(range(n_visits))
        available_weights = transaction_weights.copy()
        
        for _ in range(min(n_transactions, len(available_indices))):
            if not available_indices:
                break
            
            selected_idx = random.choices(available_indices, weights=available_weights, k=1)[0]
            transaction_indices.add(selected_idx)
            
            idx_position = available_indices.index(selected_idx)
            available_indices.pop(idx_position)
            available_weights.pop(idx_position)
    
    # Reviews selection (enhanced with POI preferences)
    review_indices = set()
    if transaction_indices:
        review_weights = []
        transaction_list = list(transaction_indices)
        
        for i in transaction_list:
            visit = visits[i]
            category = visit.get("poi_category", "").lower().replace(" ", "_")
            poi_id = str(visit.get("poi_id", ""))
            
            weight = 1.0
            
            # Category multipliers
            weight *= category_multipliers.get(category, 1.0)
            
            # POI multipliers for reviews
            if poi_id and poi_id in poi_multipliers:
                weight *= poi_multipliers[poi_id] * 0.8  # Slightly lower than transaction
            
            # Cluster preferences
            if category in category_prefs and category_prefs[category] > 0.05:
                weight *= 1.5
            
            if poi_id and poi_id in poi_prefs and poi_prefs[poi_id] > 0.02:
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
        "views": views_indices,
        "transactions": transaction_indices,
        "reviews": review_indices,
        "cluster": assigned_cluster
    }

def analyze_cluster_transaction_patterns(assignments, visits, cluster_id):
    """Analyze if transactions align with cluster preferences"""
    transaction_categories = []
    transaction_pois = []
    
    for i in assignments["transactions"]:
        visit = visits[i]
        category = visit.get("poi_category", "").lower().replace(" ", "_")
        poi_id = str(visit.get("poi_id", ""))
        transaction_categories.append(category)
        if poi_id:
            transaction_pois.append(poi_id)
    
    category_counts = defaultdict(int)
    poi_counts = defaultdict(int)
    
    for cat in transaction_categories:
        category_counts[cat] += 1
    
    for poi in transaction_pois:
        poi_counts[poi] += 1
    
    print(f"\n=== Cluster {cluster_id} Transaction Analysis ===")
    
    # Category analysis
    total_transactions = len(transaction_categories)
    print("Top Transaction Categories:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        percentage = count / total_transactions * 100 if total_transactions > 0 else 0
        print(f"  {cat}: {count} ({percentage:.1f}%)")
    
    # POI analysis
    print("Top Transaction POIs:")
    for poi, count in sorted(poi_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        percentage = count / total_transactions * 100 if total_transactions > 0 else 0
        print(f"  {poi}: {count} ({percentage:.1f}%)")
    
    return category_counts, poi_counts

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
    
    # File paths for both category and POI analysis
    cluster_category_file = "c2_Clustering_and_Analysis/post_clustering_analysis/output/kmeans/poi_category_by_cluster.csv"
    poi_analysis_file = "c2_Clustering_and_Analysis/post_clustering_analysis/output/kmeans/top_pois_by_cluster.csv"  # NEW!
    
    try:
        # Load enhanced cluster patterns (both categories and POIs)
        cluster_preferences = load_cluster_patterns_enhanced(cluster_category_file, poi_analysis_file)
        
        # Calculate enhanced multipliers
        print("Analyzing cluster data for category and POI transaction patterns...")
        category_multipliers, poi_multipliers = analyze_cluster_category_and_poi_patterns(
            cluster_category_file, poi_analysis_file
        )
        
        print(f"\nLoaded {len(cluster_preferences)} cluster patterns")
        print(f"Generated {len(category_multipliers)} category multipliers")
        print(f"Generated {len(poi_multipliers)} POI multipliers")
        use_clusters = True
        
    except FileNotFoundError as e:
        print(f"Cluster files not found: {e}. Using random assignment.")
        cluster_preferences = {}
        category_multipliers = {}
        poi_multipliers = {}
        use_clusters = False
    except Exception as e:
        print(f"Error loading cluster data: {e}. Using random assignment.")
        cluster_preferences = {}
        category_multipliers = {}
        poi_multipliers = {}
        use_clusters = False

    # review_llm = pipeline("text2text-generation", model="google/flan-t5-small", device=0)  # CPU

    # Load input data
    with open("c5_JSON_Dataset_Generation/c0_scripts/LLM_method/input_dataset/final_checkin_file.json", "r", encoding="utf-8") as f:
        checkin_data = json.load(f)

    try:
        with open("c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_summary_extraction/user_summaries_hf.csv", "r", encoding="utf-8") as f:
            user_summaries = json.load(f)
    except:
        print("User summaries file not found - continuing without summaries")
        user_summaries = {}

    try:
        with open("c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_profile_generation/master_poi_pool.json", "r", encoding="utf-8") as f:
            poi_master_data = json.load(f)
    except:
        print("POI master data not found - continuing without master POI data")
        poi_master_data = {}

    app_profiles = []

    for user in tqdm(checkin_data, desc="Processing users"):
        user_id = str(user["user_id"])
        visits = user.get("user_metadata", [])
        if not visits or not isinstance(visits, list) or len(visits) == 0:
            continue

        random.shuffle(visits)
        
        if use_clusters:
            # Enhanced cluster assignment using both categories and POIs
            assigned_cluster = assign_user_to_cluster_enhanced(visits, cluster_preferences)
            
            # Enhanced funnel assignment with POI-level analysis
            assigns = assign_funnel_interactions_with_clusters_enhanced(
                visits, cluster_preferences, assigned_cluster, 
                category_multipliers, poi_multipliers,
                view_to_transaction_rate=0.2, transaction_to_review_rate=0.4
            )
            
            # Analysis for first few users
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
            poi_id = str(entry.get("poi_id", f"POI-{idx}"))  # Ensure string
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
            poi_id = str(entry.get("poi_id", f"POI-{idx}"))  # Ensure string
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
            poi_id = str(entry.get("poi_id", f"POI-{idx}"))  # Ensure string
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
    
    # Save with enhanced filename
    output_filename = "c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_profile_generation/app_profiles_all_users_version_6_enhanced_poi_category.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(app_profiles, f, ensure_ascii=False, indent=2)

    print(f"Saved enhanced profiles to {output_filename}")