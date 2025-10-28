import json
import os


# Get script directory and construct paths relative to it
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Go up to c5_JSON_Dataset_Gen_from_fsq_direct/

input_file = os.path.join(BASE_DIR, "input_sampled_fsq_json", "input.json")
output_file = os.path.join(SCRIPT_DIR, "all_pois.json")

# code
unique_pois = {}

def safe_float(val):
    try:
        return float(val)
    except Exception:
        return None

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for user in data:
    user_metadata = user.get("user_metadata", [])
    for entry in user_metadata:
        poi_id = entry.get("poi_id")
        if not poi_id or poi_id in unique_pois:
            continue
        poi_category = entry.get("poi_category")
        planning_area = entry.get("planning_area")
        lat = safe_float(entry.get("lat"))
        lon = safe_float(entry.get("lon"))
        if None in (poi_category, planning_area, lat, lon):
            continue
        unique_pois[poi_id] = {
            "poiId": poi_id,
            "poiCategories": [poi_category],
            "planningArea": planning_area,
            "userLocation": {
                "latitude": lat,
                "longitude": lon
            }
        }

unique_pois_list = list(unique_pois.values())


with open(output_file, "w", encoding="utf-8") as f:
    json.dump(unique_pois_list, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(unique_pois_list)} unique POIs to {output_file}")