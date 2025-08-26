import json
from datetime import datetime, timedelta
import random

def fake_latlon(planning_area):
    # Dummy lat/lon generator for demo purposes
    base = {
        "ORCHARD": (1.3048, 103.8318),
        "ROCHOR": (1.3070, 103.8520),
        "SINGAPORE RIVER": (1.2906, 103.8465),
        "GEYLANG": (1.3163, 103.8874),
        "TAMPINES": (1.3496, 103.9568),
        "DOWNTOWN CORE": (1.2830, 103.8510),
        "KALLANG": (1.3122, 103.8660),
        "BUKIT MERAH": (1.2826, 103.8185),
        "CLEMENTI": (1.3151, 103.7658),
        "OTHERS": (1.3521, 103.8198)
    }
    return base.get(planning_area.upper(), base["OTHERS"])

def random_station():
    stations = [
        {"stationName": "Somerset MRT", "stationCode": "NS23", "coordinates": {"latitude": 1.3009, "longitude": 103.8390}},
        {"stationName": "Dhoby Ghaut MRT", "stationCode": "NS24", "coordinates": {"latitude": 1.2988, "longitude": 103.8456}},
        {"stationName": "Bugis MRT", "stationCode": "DT14", "coordinates": {"latitude": 1.3000, "longitude": 103.8565}},
        {"stationName": "Tampines MRT", "stationCode": "EW2", "coordinates": {"latitude": 1.3530, "longitude": 103.9457}},
    ]
    return random.choice(stations)

def generate_profile(user_id, checkins, user_info=None):
    # user_info: dict with keys age, gender, city, country, platform, appVersion
    if user_info is None:
        user_info = {
            "age": random.randint(20, 40),
            "gender": random.choice(["male", "female"]),
            "city": "Singapore",
            "country": "SG",
            "platform": "iOS",
            "appVersion": "2.5.1"
        }
    # Generate POIs
    pois = []
    poi_ids = {}
    for idx, ci in enumerate(checkins):
        cat = ci.get("poi_category", "restaurant")
        area = ci.get("planning_area", "OTHERS")
        name = f"{cat.title()} in {area.title()}"
        poi_id = f"POI-{user_id}-{idx+1}"
        lat, lon = fake_latlon(area)
        poi = {
            "poiId": poi_id,
            "name": name,
            "categories": [cat],
            "subcategories": [cat.lower()],
            "location": {
                "latitude": lat,
                "longitude": lon,
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
        poi_ids[(cat, area)] = poi_id

    # Generate interactions
    views, transactions, reviews = [], [], []
    for idx, ci in enumerate(checkins):
        cat = ci.get("poi_category", "restaurant")
        area = ci.get("planning_area", "OTHERS")
        poi_id = poi_ids[(cat, area)]
        timestamp = datetime.now() - timedelta(days=random.randint(1, 90), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        lat, lon = fake_latlon(area)
        # Views
        views.append({
            "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "poiId": poi_id,
            "poiCategories": [cat],
            "poiSubcategories": [cat.lower()],
            "duration": random.randint(5, 30),
            "referrer": random.choice(["journey_planner_recommendations", "search", "map"]),
            "userLocation": {"latitude": lat + random.uniform(-0.001, 0.001), "longitude": lon + random.uniform(-0.001, 0.001)}
        })
        # Transactions (simulate for some)
        if random.random() < 0.5:
            transactions.append({
                "timestamp": (timestamp + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "poiId": poi_id,
                "poiCategories": [cat],
                "poiSubcategories": [cat.lower()],
                "transactionId": f"TXN-{user_id}-{idx+1}",
                "amount": round(random.uniform(10, 50), 2),
                "currency": "SGD",
                "paymentMethod": random.choice(["credit_card", "cash", "mobile_pay"]),
                "userLocation": {"latitude": lat + random.uniform(-0.001, 0.001), "longitude": lon + random.uniform(-0.001, 0.001)}
            })
        # Reviews (simulate for some)
        if random.random() < 0.4:
            reviews.append({
                "timestamp": (timestamp + timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "poiId": poi_id,
                "poiCategories": [cat],
                "poiSubcategories": [cat.lower()],
                "rating": round(random.uniform(3.5, 5.0), 1),
                "reviewText": random.choice([
                    "Great place, would visit again!",
                    "Enjoyed the atmosphere and service.",
                    "Food was delicious and staff were friendly.",
                    "Had a wonderful experience here."
                ]),
                "userLocation": {"latitude": lat + random.uniform(-0.001, 0.001), "longitude": lon + random.uniform(-0.001, 0.001)}
            })

    profile = {
        "user": {
            "userId": str(user_id),
            "age": user_info["age"],
            "gender": user_info["gender"],
            "location": {
                "city": user_info["city"],
                "country": user_info["country"]
            },
            "device": {
                "platform": user_info["platform"],
                "appVersion": user_info["appVersion"]
            }
        },
        "interaction": {
            "views": views,
            "transactions": transactions,
            "reviews": reviews
        },
        "pois": pois
    }
    return profile

# Example usage:
if __name__ == "__main__":
    # Example input: dict of user_id -> list of checkins
    users_checkins = {
        "203821": [
            {"poi_category": "Nightclub", "planning_area": "ORCHARD"},
            {"poi_category": "Pub", "planning_area": "DOWNTOWN CORE"},
            {"poi_category": "Dim Sum Restaurant", "planning_area": "GEYLANG"},
            # ... more check-ins ...
        ],
        "244491": [
            {"poi_category": "Coffee Shop", "planning_area": "TAMPINES"},
            {"poi_category": "Department Store", "planning_area": "ORCHARD"},
            # ... more check-ins ...
        ]
        # Add more users as needed
    }

    all_profiles = []
    for user_id, checkins in users_checkins.items():
        profile = generate_profile(user_id, checkins)
        all_profiles.append(profile)

    # Write to file
    with open("user_profiles.json", "w") as f:
        json.dump(all_profiles, f, indent=4)