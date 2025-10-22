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

# Seven-bin Venn allocation helpers
VENN_PROPS = {
    "V_only": 0.45,
    "T_only": 0.10,
    "R_only": 0.02,
    "VT_only": 0.25,
    "VR_only": 0.03,
    "TR_only": 0.05,
    "VTR": 0.10
}

def allocate_bins(N, props=VENN_PROPS):
    exact = {k: N*v for k,v in props.items()}
    base = {k: int(exact[k]) for k in props}
    missing = N - sum(base.values())
    rema = sorted(((exact[k]-base[k], k) for k in props), reverse=True)
    for i in range(missing):
        base[rema[i][1]] += 1
    return base

def assign_interactions(checkins, bins):
    random.shuffle(checkins)
    idx = 0
    assignments = {"views": set(), "transactions": set(), "reviews": set()}
    for bin_name, count in bins.items():
        for _ in range(count):
            if idx >= len(checkins):
                break
            cid = idx
            if "V" in bin_name: assignments["views"].add(cid)
            if "T" in bin_name: assignments["transactions"].add(cid)
            if "R" in bin_name: assignments["reviews"].add(cid)
            idx += 1
    return assignments

# -----------------------------
# Main Generation Logic
# -----------------------------

if __name__ == "__main__":

    review_llm = pipeline("text2text-generation", model="google/flan-t5-small", device=0)  # CPU

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
        n = len(visits)
        bins = allocate_bins(n)
        assigns = assign_interactions(visits, bins)

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

    with open("c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_profile_generation/app_profiles_all_users_version_4.json", "w", encoding="utf-8") as f:
        json.dump(app_profiles, f, ensure_ascii=False, indent=2)