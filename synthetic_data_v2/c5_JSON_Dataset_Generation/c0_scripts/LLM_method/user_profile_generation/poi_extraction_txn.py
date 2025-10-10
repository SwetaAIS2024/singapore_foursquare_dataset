import json

input_file = "c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_profile_generation/app_profiles_all_users_version_4.json"
output_file = "c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_profile_generation/unique_pois_from_all_interactions.json"

unique_pois = {}

def safe_float(val):
    try:
        return float(val)
    except Exception:
        return None

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for entry in data:
    if not isinstance(entry, dict):
        continue
    interaction = entry.get("interaction", {})
    transactions = interaction.get("transactions", [])
    for txn in transactions:
        poi_id = txn.get("poiId")
        if not poi_id:
            continue  # skip if no poiId
        if poi_id in unique_pois:
            continue  # already added

        # Compose POI details
        poi_categories = txn.get("poiCategories", [])
        planning_area = txn.get("planning_area") or txn.get("planningArea") or ""
        user_location = txn.get("userLocation", {})
        lat = user_location.get("latitude")
        lon = user_location.get("longitude")
        # Convert to float if possible
        lat = safe_float(lat)
        lon = safe_float(lon)
        user_location_out = {}
        if lat is not None and lon is not None:
            user_location_out = {"latitude": lat, "longitude": lon}

        unique_pois[poi_id] = {
            "poiId": poi_id,
            "poiCategories": poi_categories,
            "planningArea": planning_area,
            "userLocation": user_location_out
        }

# Convert to list
unique_pois_list = list(unique_pois.values())

# Save to file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(unique_pois_list, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(unique_pois_list)} unique POIs to {output_file}")