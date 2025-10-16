import pandas as pd
import json
import random
from datetime import datetime, timedelta
from synthetic_data_v2.c0_Configuration.config_paths import JSON_OUTPUT_DIR, POI_UNIQUE_ID_MAPPING

# --- Load your mock JSON schema ---
# For this template, we assume you have a Python dict called `json_schema`
# You can load it from a file or define it inline

# --- Load sampled FSQ dataset ---
fsq_df = pd.read_csv("sampled_FSQ_dataset.csv")  # Update path as needed

# --- Helper functions for generative fields ---
def generate_view_time():
    # Example: random time in the last month
    dt = datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def generate_review_text():
    # Example: placeholder or call to LLM
    return random.choice([
        "Great experience!",
        "Nice place, will visit again.",
        "Service could be better.",
        "Loved the ambiance."
    ])

def generate_device_info():
    # Example: random device info
    return {
        "platform": random.choice(["iOS", "Android"]),
        "appVersion": f"{random.randint(1,3)}.{random.randint(0,9)}.{random.randint(0,9)}"
    }

def generate_transaction_id(idx):
    # Ensures unique transaction ID per checkin
    return f"TXN-{100000 + idx}"

def get_poi_id(place_id):
    # Load the mapping from place_id to poiId
    mapping_df = pd.read_csv(POI_UNIQUE_ID_MAPPING)
    mapping_dict = dict(zip(mapping_df['place_id'], mapping_df['poiId']))
    return mapping_dict.get(place_id, f"POI-{random.randint(1000, 9999)}")  # Fallback if not found

# --- Main mapping and JSON generation ---
for idx, row in fsq_df.iterrows():
    # Direct fields from CSV
    user_json = {
        "userId": row["user_id"],
        "age": row.get("age", random.randint(18, 60)),  # If not in CSV, generate
        "gender": row.get("gender", random.choice(["male", "female"])),
        "location": {
            "city": "Singapore",
            "country": "SG"
        },
        "device": generate_device_info()
    }

    transaction_id = generate_transaction_id(idx)
    poi_id = get_poi_id(row["place_id"])  # Function to map place_id to poiId using POI_UNIQUE_ID_MAPPING

    # Interactions (example for views, transactions, reviews)
    interaction_json = {
        "views": [{
            "timestamp": generate_view_time(),
            "poiId": poi_id,
            "poiCategories": [row.get("poi_category", "")],
            "poiSubcategories": [row.get("poi_subcategory", "")],
            "duration": random.randint(1, 60),
            "referrer": row.get("referrer", "app"),
            "userLocation": {
                "latitude": row["lat"],
                "longitude": row["lon"]
            }
        }],
        "transactions": [{
            "timestamp": generate_view_time(),
            "poiId": poi_id,
            "poiCategories": [row.get("poi_category", "")],
            "poiSubcategories": [row.get("poi_subcategory", "")],
            "transactionId": transaction_id,
            "amount": row.get("amount", round(random.uniform(5, 100), 2)),
            "currency": row.get("currency", "SGD"),
            "paymentMethod": row.get("payment_method", random.choice(["credit_card", "cash"])),
            "userLocation": {
                "latitude": row["lat"],
                "longitude": row["lon"]
            }
        }],
        "reviews": [{
            "timestamp": generate_view_time(),
            "poiId": poi_id,
            "poiCategories": [row.get("poi_category", "")],
            "poiSubcategories": [row.get("poi_subcategory", "")],
            "rating": row.get("rating", round(random.uniform(1, 5), 1)),
            "reviewText": generate_review_text(),
            "userLocation": {
                "latitude": row["lat"],
                "longitude": row["lon"]
            }
        }]
    }

    # POIs (can be expanded as needed)
    pois_json = [{
        "poiId": poi_id,
        "name": row.get("poi_name", "Unknown POI"),
        "categories": [row.get("poi_category", "")],
        "subcategories": [row.get("poi_subcategory", "")],
        "location": {
            "latitude": row["lat"],
            "longitude": row["lon"],
            "address": row.get("address", "Unknown Address")
        },
        "nearestStation": {
            "stationName": row.get("station_name", "Unknown Station"),
            "stationCode": row.get("station_code", "NS00"),
            "coordinates": {
                "latitude": row.get("station_lat", ""),
                "longitude": row.get("station_lon", "")
            }
        },
        "rating": row.get("rating", round(random.uniform(1, 5), 1)),
        "deal": {
            "dealId": row.get("deal_id", f"DEAL-{random.randint(100,999)}"),
            "discount": row.get("discount", "10% off"),
            "validUntil": row.get("valid_until", "2025-12-31")
        }
    }]

    # Assemble full JSON
    full_json = {
        "user": user_json,
        "interaction": interaction_json,
        "pois": pois_json
    }

    # Save to file
    with open(f"user_{idx}_interaction.json", "w") as f:
        json.dump(full_json, f, indent=2)