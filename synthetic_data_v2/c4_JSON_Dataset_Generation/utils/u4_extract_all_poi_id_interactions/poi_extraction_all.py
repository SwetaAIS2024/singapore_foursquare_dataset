import json
from c0_Configuration.config_paths import JSON_INPUT, ALL_POI_ID_INTERACTIONS

input_file = JSON_INPUT
output_file = ALL_POI_ID_INTERACTIONS

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