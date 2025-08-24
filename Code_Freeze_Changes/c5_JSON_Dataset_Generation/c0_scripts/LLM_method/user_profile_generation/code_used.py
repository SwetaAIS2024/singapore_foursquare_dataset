import json
import random
from collections import defaultdict
from datetime import datetime, timedelta

# -----------------------------
# Load Inputs
# -----------------------------

with open("final_checkin_file.json", "r", encoding="utf-8") as f:
    checkin_data = json.load(f)

with open("user_summaries_hf.csv", "r", encoding="utf-8") as f:
    import pandas as pd
    summary_df = pd.read_csv(f)
    user_summaries = dict(zip(summary_df["user_id"].astype(str), summary_df["summary"]))

# -----------------------------
# Helper Functions
# -----------------------------

def jitter_location(lat, lon):
    return round(lat + random.uniform(-0.003, 0.003), 6), round(lon + random.uniform(-0.003, 0.003), 6)

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
    base_summary = user_summaries.get(str(user_id), "")
    words = base_summary.split()
    random.shuffle(words)
    snippet = " ".join(words[:random.randint(18, 24)]) if words else "Nothing specific, but a decent experience."
    return f"Visited this {category.lower()} — {snippet}"

# -----------------------------
# Main Generation Logic
# -----------------------------

users_grouped = defaultdict(list)
for entry in checkin_data:
    users_grouped[str(entry["user_id"])].append(entry)

app_profiles = []

for user_id, visits in users_grouped.items():
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

    for idx, entry in enumerate(visits):
        poi_id = f"POI-{entry['poi_category'].lower()}-{entry['planning_area'].upper()}-{idx:03}"
        lat, lon = jitter_location(entry["lat"], entry["lon"])
        timestamp = (base_date + timedelta(days=idx)).strftime("%Y-%m-%dT12:00:00Z")

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

        # Transaction (only for ~50% of views)
        if random.random() < 0.5:
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

            # Review (1:1 with transaction)
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

        # POI (ensure uniqueness)
        if poi_id not in pois:
            pois[poi_id] = {
                "poiId": poi_id,
                "name": f"{entry['poi_category']} - {entry['planning_area']}",
                "categories": [entry["poi_category"]],
                "subcategories": [],
                "location": {
                    "latitude": lat,
                    "longitude": lon,
                    "address": f"Somewhere in {entry['planning_area']}, SG"
                },
                "nearestStation": {
                    "stationName": random.choice(["Orchard", "City Hall", "Bugis", "Raffles Place", "Dhoby Ghaut"]),
                    "stationCode": f"NS{random.randint(1, 28)}",
                    "coordinates": {"latitude": lat, "longitude": lon}
                },
                "rating": random_rating(),
                "deal": {
                    "dealId": f"DEAL-{random.randint(1000,9999)}",
                    "discount": random.choice(["10% off", "Buy 1 Get 1", "Free dessert"]),
                    "validUntil": "2025-12-31"
                }
            }

    app_profiles.append({
        "user": user_block,
        "interaction": interaction,
        "pois": list(pois.values())
    })

# -----------------------------
# Save Output
# -----------------------------
with open("app_profiles_all_users_NO_LLM_reviews_unique_per_poi.json", "w", encoding="utf-8") as f:
    json.dump(app_profiles, f, ensure_ascii=False, indent=2)