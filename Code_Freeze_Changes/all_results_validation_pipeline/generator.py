import pandas as pd
import numpy as np
import json
import random
from collections import defaultdict
from datetime import datetime, timedelta

# --------------- CONFIG ---------------
NUM_SYNTH_USERS = 50
SYNTH_TXNS_PER_USER = None  # None = sample by real user distribution

random.seed(42)
np.random.seed(42)

# ----------- 1. LOAD DATA -------------
real_df = pd.read_csv("singapore_checkins_filtered_with_locations_coord_truncated.txt", sep="\t", header=None,
                      names=["user_id", "place_id", "timestamp", "unknown", "latitude", "longitude"])
cat_map = pd.read_csv("sg_place_id_to_category.csv")
rel_poi = pd.read_excel("Relevant_POI_category.xlsx")

# Map POIs to categories (and filter)
real_merged = pd.merge(real_df, cat_map, left_on="place_id", right_on="place_id", how="left")
real_merged = real_merged.rename(columns={"category": "poi_category"})
relevant_cats = rel_poi[rel_poi.iloc[:, 2].str.lower() == 'yes']['POI Category in Singapore'].str.strip().tolist()
real_merged = real_merged[real_merged['poi_category'].isin(relevant_cats)]

# Build POI info dict for lookup
poi_info = {}
for _, row in real_merged.iterrows():
    pid = row['place_id']
    if pid not in poi_info:
        poi_info[pid] = {
            "categories": [row['poi_category']],
            "lat": row['latitude'],
            "lon": row['longitude']
        }

# ----------- 2. EMPIRICAL DISTRIBUTIONS -------------
# (a) POI selection distribution
poi_probs = real_merged['place_id'].value_counts(normalize=True)
poi_list = poi_probs.index.tolist()
poi_weights = poi_probs.values

# (b) Hour-of-day and day-of-week
real_hours = pd.to_datetime(real_merged['timestamp']).dt.hour
real_dow = pd.to_datetime(real_merged['timestamp']).dt.dayofweek

# (c) Transaction counts per user
txn_counts = real_merged['user_id'].value_counts().values
if SYNTH_TXNS_PER_USER is None:
    txn_dist = txn_counts
else:
    txn_dist = [SYNTH_TXNS_PER_USER] * NUM_SYNTH_USERS

# ----------- 3. USER/DEVICE ATTRIBUTES --------------
def random_gender():
    return random.choice(["male", "female"])

def random_age():
    return random.randint(18, 65)

def random_platform():
    return random.choice(["iOS", "Android"])

def random_app_version():
    major = random.choice([2, 3])
    minor = random.randint(0, 5)
    patch = random.randint(0, 9)
    return f"{major}.{minor}.{patch}"

# ----------- 4. SYNTHETIC USER PROFILE GENERATION -----------
def random_timestamp():
    base_date = datetime(2025, 4, 1)
    rand_day = random.randint(0, 29)
    rand_hour = np.random.choice(real_hours)
    rand_minute = random.randint(0, 59)
    rand_second = random.randint(0, 59)
    ts = base_date + timedelta(days=rand_day, hours=rand_hour, minutes=rand_minute, seconds=rand_second)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

def random_referrer():
    return random.choice([
        "journey_planner_recommendations",
        "nearby_promotion",
        "search_results",
        "homepage",
        "event_ad"
    ])

def random_review_text():
    texts = [
        "Great spot, would recommend!", "Service could be better.", "Loved the atmosphere!",
        "Food was decent, not amazing.", "Would visit again.", "Nice for a quick stop."
    ]
    return random.choice(texts)

synthetic_profiles = []

for uid in range(NUM_SYNTH_USERS):
    user_id = str(100000 + uid)
    age = random_age()
    gender = random_gender()
    platform = random_platform()
    app_version = random_app_version()

    n_txn = int(np.random.choice(txn_dist))  # Number of transactions for this user
    if n_txn < 1:
        n_txn = 1

    transactions = []
    views = []
    reviews = []
    pois = []
    used_pois = set()

    for _ in range(n_txn):
        # POI sampling
        poi_idx = np.random.choice(len(poi_list), p=poi_weights)
        poi_id = poi_list[poi_idx]
        category = poi_info[poi_id]["categories"]
        subcategory = [category[0].split()[0].lower()]  # basic mapping
        lat, lon = poi_info[poi_id]["lat"], poi_info[poi_id]["lon"]

        # Add POI if not already
        if poi_id not in used_pois:
            pois.append({
                "poiId": poi_id,
                "name": f"POI-{poi_id[-3:]}",
                "categories": category,
                "subcategories": subcategory,
                "location": {
                    "latitude": lat,
                    "longitude": lon,
                    "address": f"{random.randint(100,999)} Orchard Road, Singapore"
                },
                "nearestStation": {
                    "stationName": "Somerset MRT",
                    "stationCode": "NS23",
                    "coordinates": {"latitude": 1.3009, "longitude": 103.8390}
                },
                "rating": round(random.uniform(3.0, 5.0), 1),
                "deal": {
                    "dealId": f"DEAL-{random.randint(100,999)}",
                    "discount": f"{random.choice([10, 15, 20, 25])}% off",
                    "validUntil": "2025-04-30"
                }
            })
            used_pois.add(poi_id)

        ts = random_timestamp()
        # Transaction
        txn_id = f"TXN-{random.randint(100000, 999999)}"
        transactions.append({
            "timestamp": ts,
            "poiId": poi_id,
            "poiCategories": category,
            "poiSubcategories": subcategory,
            "transactionId": txn_id,
            "amount": round(random.uniform(5, 100), 2),
            "currency": "SGD",
            "paymentMethod": random.choice(["credit_card", "cash", "ewallet"]),
            "userLocation": {"latitude": lat, "longitude": lon}
        })
        # View (randomized fields)
        views.append({
            "timestamp": ts,
            "poiId": poi_id,
            "poiCategories": category,
            "poiSubcategories": subcategory,
            "duration": random.randint(1, 12),
            "referrer": random_referrer(),
            "userLocation": {"latitude": lat, "longitude": lon}
        })
        # Review (randomized fields)
        reviews.append({
            "timestamp": ts,
            "poiId": poi_id,
            "poiCategories": category,
            "poiSubcategories": subcategory,
            "rating": round(random.uniform(3.0, 5.0), 1),
            "reviewText": random_review_text(),
            "userLocation": {"latitude": lat, "longitude": lon}
        })

    user_profile = {
        "user": {
            "userId": user_id,
            "age": age,
            "gender": gender,
            "location": {"city": "Singapore", "country": "SG"},
            "device": {"platform": platform, "appVersion": app_version}
        },
        "interaction": {
            "views": views,
            "transactions": transactions,
            "reviews": reviews
        },
        "pois": pois
    }
    synthetic_profiles.append(user_profile)

# ------- 5. SAVE TO JSON -------
with open("singapore_synthetic_user_profiles_full.json", "w") as f:
    json.dump(synthetic_profiles, f, indent=2)
print("Synthetic user dataset generated and saved as 'singapore_synthetic_user_profiles_full.json'.")
