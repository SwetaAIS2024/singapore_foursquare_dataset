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
    lat = float(lat)
    lon = float(lon)
    hash_input = f"{lat:.6f}_{lon:.6f}"
    hash_digest = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
    return f"POI-ID-{hash_digest}"

def build_poi_id_pool():
    poi_id_pool = defaultdict(list)
    for user in checkin_data:
        visits = user.get("user_metadata", [])
        for entry in visits:
            if "lat" in entry and "lon" in entry:
                lat, lon = float(entry["lat"]), float(entry["lon"])
                poi_id = generate_unique_poi_id(lat, lon)
                poi_id_pool[poi_id].append((lat, lon))
    return poi_id_pool


def generate_timestamp(day_of_week, time_of_day, month_of_year, year=2025, jitter_days=0, jitter_hours=0, jitter_minutes=0, jitter_seconds=0):
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
                # Apply jitter to timestamp
                dt += timedelta(days=jitter_days, hours=jitter_hours, minutes=jitter_minutes, seconds=jitter_seconds)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
    # Fallback if not found
    final_timestamp = f"{year}-{month_num:02d}-01T00:00:00Z"
    return final_timestamp

def generate_poi_name(category, area):
    return f"{category.title()} in {area.title()}"

def random_station():
    stations = [
        "Dhoby Ghaut", "Orchard", "Bugis", "Marina Bay", "Raffles Place",
        "Tanjong Pagar", "Chinatown", "Little India", "Clarke Quay", "Esplanade"
    ]
    return random.choice(stations)

# -----------------------------
# Main Generation Logic
# -----------------------------

if __name__ == "__main__":

    # Initialize a small LLM for text generation
    review_llm = pipeline("text2text-generation", model="google/flan-t5-small", device=0)  # CPU

    # -----------------------------
    # Load Inputs
    # -----------------------------

    with open("c5_JSON_Dataset_Generation/c0_scripts/LLM_method/input_dataset/final_checkin_file.json", "r", encoding="utf-8") as f:
        checkin_data = json.load(f)

    with open("c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_summary_extraction/user_summaries_hf.csv", "r", encoding="utf-8") as f:
        user_summaries = json.load(f)
 
    app_profiles = []

    for user in tqdm(checkin_data, desc="Processing users"):
        user_id = str(user["user_id"])
        visits = user.get("user_metadata", [])
        if not visits or not isinstance(visits, list):
            print(f"Skipping user {user_id}: no valid check-ins")
            continue

        random.shuffle(visits)
        n = len(visits)
        n70 = int(n * 0.7)
        n20 = int(n * 0.2)
        n10 = n - n70 - n20  # ensures all visits are included

        group_70 = visits[:n70]
        group_20 = visits[n70:n70+n20]
        group_10 = visits[n70+n20:]

        # USER BLOCK 

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
        jitter_days = random.randint(1, 7)
        jitter_hours = random.randint(20, 23)
        jitter_minutes = random.randint(0, 59)
        jitter_seconds = random.randint(0, 59)

        # INTERACTIONS

        # views block 70 percent
        for idx, entry in enumerate(group_70):
            # poi_id = f"POI-{entry['poi_category'].lower()}-{entry['planning_area'].upper()}-{idx:03}"
            if "lat" not in entry or "lon" not in entry:
                continue  # skip entries without lat/lon
            lat, lon = jitter_location(float(entry["lat"]), float(entry["lon"]), distance_km=random.uniform(2, 50)) # add jitter for the views location 
            poi_id = generate_unique_poi_id(lat, lon)
            # timestamp = (base_date + timedelta(days=idx)).strftime("%Y-%m-%dT12:00:00Z")
            timestamp = generate_timestamp(entry["day_of_week"], entry["time_of_day"], entry["month_of_year"], jitter_days=jitter_days, jitter_hours=jitter_hours, jitter_minutes=jitter_minutes)
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
                "userLocation": {"latitude": lat, "longitude": lon},
                "planning_area": entry.get("planning_area", "UNKNOWN")
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
        
        # POIs
        # pois will be where the transactions took place
        # locations will be same as the transactions interactions
        pois = []
        for txn in interaction["transactions"]:
            poi_id = txn["poiId"]
            cat = txn.get("poiCategories", ["restaurant"])[0]
            area = txn.get("planning_area", "OTHERS")

            poi = {
                "poiId": poi_id,
                "name": generate_poi_name(cat, area),
                "categories": [cat],
                "subcategories": [cat.lower()],
                "location": {
                    "latitude": txn["userLocation"]["latitude"],
                    "longitude": txn["userLocation"]["longitude"],
                    "address": f"{area.title()}, Singapore"
                },
                "nearestStation": random_station(),
                "rating": round(random.uniform(3.5, 5.0), 1),
                "deal": {
                    "dealId": f"DEAL-{user_id}-{idx+1}",
                    "discount": f"{random.choice([10, 15, 20, 25])}% off",
                    "validUntil": (datetime.now() + timedelta(days=random.randint(10, 60))).strftime("%Y-%m-%d")
                }
            }
            pois.append(poi)
            
        # print("User:", user_block)
        # print("Interaction:", interaction)
        # print("POIs:", pois)

        app_profiles.append({
            "user": user_block,
            "interaction": interaction,
            "pois": pois
        })

    # -----------------------------
    # Save Output
    # -----------------------------
    with open("c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_profile_generation/app_profiles_all_users_file_with_changes.json", "w", encoding="utf-8") as f:
        json.dump(app_profiles, f, ensure_ascii=False, indent=2)