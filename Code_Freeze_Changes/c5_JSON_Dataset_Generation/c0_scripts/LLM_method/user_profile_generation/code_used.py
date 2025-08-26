import json
import random
import pandas as pd 
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import math
import calendar
from datetime import datetime 

from transformers import pipeline

# Initialize a small LLM for text generation
review_llm = pipeline("text2text-generation", model="google/flan-t5-small", device=0)  # CPU

# -----------------------------
# Load Inputs
# -----------------------------

with open("c5_JSON_Dataset_Generation/c0_scripts/LLM_method/input_dataset/final_checkin_file.json", "r", encoding="utf-8") as f:
    checkin_data = json.load(f)

# with open("c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_summary_extraction/user_summaries_hf.csv", "r", encoding="utf-8") as f:
#     summary_df = pd.read_csv(f)
#     user_summaries = dict(zip(summary_df["user_id"].astype(str), summary_df["summary"]))
with open("c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_summary_extraction/user_summaries_hf.csv", "r", encoding="utf-8") as f:
    user_summaries = json.load(f)
# -----------------------------
# Helper Functions
# -----------------------------

def jitter_location(lat, lon, distance_km=2):
    # 1 degree latitude ≈ 111 km
    max_deg = distance_km / 111.0
    lat_jitter = random.uniform(-max_deg, max_deg)
    # Longitude degrees depend on latitude
    lon_jitter = random.uniform(-max_deg, max_deg) / math.cos(math.radians(lat))
    loca = round(lat + lat_jitter, 6), round(lon + lon_jitter, 6)
    return loca

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

# def generate_review_text(user_id, poi_id, category):
#     base_summary = user_summaries.get(str(user_id), "")
#     words = base_summary.split()
#     random.shuffle(words)
#     snippet = " ".join(words[:random.randint(18, 24)]) if words else "Nothing specific, but a decent experience."
#     return f"Visited this {category.lower()} — {snippet}"

def generate_review_text(user_id, poi_id, category):
    """
    Generate a unique review using the user summary and POI category with a small LLM.
    """
    base_summary = user_summaries.get(str(user_id), "")
    prompt = (
        f"User summary: {base_summary}\n"
        f"POI category: {category}\n"
        "Write a short, unique review for this POI in the user's style."
    )
    # Use the LLM to generate the review
    result = review_llm(prompt, max_length=48, min_length=12, do_sample=True)[0]['generated_text']
    return result

def generate_unique_poi_id(lat, lon):
    hash_input = f"{lat:.6f}_{lon:.6f}"
    hash_digest = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
    return f"POI-ID-{hash_digest}"

def generate_timestamp(day_of_week, time_of_day, month_of_year, year=2025):
        # Map month name to number
    month_num = list(calendar.month_name).index(month_of_year)
    # Find the first day in the month that matches the day_of_week
    for day in range(1, 32):
        try:
            dt = datetime(year, month_num, day)
            if dt.strftime("%A") == day_of_week:
                # Parse time_of_day (e.g., "08:00 PM")
                time_obj = datetime.strptime(time_of_day, "%I:%M %p")
                dt = dt.replace(hour=time_obj.hour, minute=time_obj.minute)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
    # Fallback if not found
    final_timestamp = f"{year}-{month_num:02d}-01T00:00:00Z"
    return final_timestamp

# -----------------------------
# Main Generation Logic
# -----------------------------

users_grouped = defaultdict(list)
for entry in checkin_data:
    users_grouped[str(entry["user_id"])].append(entry)

app_profiles = []

for user_id, visits in users_grouped.items():

    random.shuffle(visits)
    n = len(visits)
    n70 = int(n * 0.7)
    n20 = int(n * 0.2)
    n10 = n - n70 - n20  # ensures all visits are included

    group_70 = visits[:n70]
    group_20 = visits[n70:n70+n20]
    group_10 = visits[n70+n20:]

    # Create User Block
    user_block = {
        "userId": user_id,
        "age": random.randint(22, 45),
        "gender": random.choice(["male", "female", "other"]),
        "location": {"city": "Singapore", "country": "SG"},
        "device": {
            "platform": random.choice(["Android", "iOS"]),
            "appVersion": f"{random.randint(2,4)}.{random.randint(0,9)}.{random.randint(0,9)}"
        }
    }

    interaction = {"views": [], "transactions": [], "reviews": []}
    pois = {}
    base_date = datetime(2025, 4, 1)

    # views block 70 percent
    for idx, entry in enumerate(group_70):
        # poi_id = f"POI-{entry['poi_category'].lower()}-{entry['planning_area'].upper()}-{idx:03}"
        if "lat" not in entry or "lon" not in entry:
            continue  # skip entries without lat/lon
        lat, lon = jitter_location(float(entry["lat"]), float(entry["lon"]), distance_km=random.uniform(2, 50)) # add jitter for the views location 
        poi_id = generate_unique_poi_id(lat, lon)
        # timestamp = (base_date + timedelta(days=idx)).strftime("%Y-%m-%dT12:00:00Z")
        timestamp = generate_timestamp(entry["day_of_week"], entry["time_of_day"], entry["month_of_year"])
        # View
        view = {
            "timestamp": timestamp,
            "poiId": poi_id,
            "poiCategories": [entry["poi_category"]],
            "poiSubcategories": [],
            "duration": random.randint(5, 50),
            "referrer": random.choice(["map", "search", "ad", "friend"]),
            "userLocation": {"latitude": lat, "longitude": lon}
        }
        interaction["views"].append(view)
    
    # transactions block 
    for idx, entry in enumerate(group_20, start=n70):
        # poi_id = f"POI-{entry['poi_category'].lower()}-{entry['planning_area'].upper()}-{idx:03}"
        # lat, lon = jitter_location(entry["lat"], entry["lon"]) # donot add jitter for the transaction interaction
        if "lat" not in entry or "lon" not in entry:
            continue  # skip entries without lat/lon
        lat, lon = entry["lat"], entry["lon"]
        poi_id = generate_unique_poi_id(lat, lon)
        # timestamp = (base_date + timedelta(days=idx)).strftime("%Y-%m-%dT12:00:00Z")
        timestamp = generate_timestamp(entry["day_of_week"], entry["time_of_day"], entry["month_of_year"])
        # Transaction
        txn = {
            "timestamp": timestamp,
            "poiId": poi_id,
            "poiCategories": [entry["poi_category"]],
            "poiSubcategories": [],
            "transactionId": str(random.randint(100000, 999999)),
            "amount": random_amount(entry["poi_category"]),
            "currency": "SGD",
            "paymentMethod": random_payment(),
            "userLocation": {"latitude": lat, "longitude": lon}
        }
        interaction["transactions"].append(txn)

    # reviews block 
    for idx, entry in enumerate(group_10, start=n70+n20):
        # poi_id = f"POI-{entry['poi_category'].lower()}-{entry['planning_area'].upper()}-{idx:03}"
        if "lat" not in entry or "lon" not in entry:
            continue  # skip entries without lat/lon
        lat, lon = jitter_location(float(entry["lat"]), float(entry["lon"]), distance_km=random.uniform(2, 80)) # add jitter for the reviews location
        poi_id = generate_unique_poi_id(lat, lon)
        # timestamp = (base_date + timedelta(days=idx)).strftime("%Y-%m-%dT12:00:00Z")
        timestamp = generate_timestamp(entry["day_of_week"], entry["time_of_day"], entry["month_of_year"])
        # Review 
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
        "interaction": interaction,
        "pois": list(pois.values())
    })

# -----------------------------
# Save Output
# -----------------------------
with open("c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_profile_generation/app_profiles_all_users_file_with_changes.json", "w", encoding="utf-8") as f:
    json.dump(app_profiles, f, ensure_ascii=False, indent=2)