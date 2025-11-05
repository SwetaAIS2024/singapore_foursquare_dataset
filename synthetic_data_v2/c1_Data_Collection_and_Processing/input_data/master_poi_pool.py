import pandas as pd
import json
import hashlib
from synthetic_data_v2.c0_Configuration.config_paths import CHECKINS_PATH, PLACE_ID_POI_CAT

# File paths
# checkin_file = CHECKINS_PATH
checkin_file = "c1_Data_Collection_and_Processing/c0_original/sampled_FSQ_dataset_with_planning_area.txt"
place_cat_file = PLACE_ID_POI_CAT
output_csv = "c1_Data_Collection_and_Processing/c0_original/master_poi_pool.csv"
output_json = "c1_Data_Collection_and_Processing/c0_original/master_poi_pool.json"

# 1. Load check-in data
# cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon']

cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon', 'cluster_id', 'sampled_count', 'planning_area']
checkins = pd.read_csv(checkin_file, sep='\t', names=cols, header=None)

# 2. Load place_id to category mapping
place_cat = pd.read_csv(place_cat_file, names=['place_id', 'category'], header=0)

# 3. Merge on place_id
merged = pd.merge(checkins, place_cat, on='place_id', how='left')

# 4. Drop rows with missing category (optional, but recommended)
merged = merged.dropna(subset=['category'])

# 5. Drop duplicates to get unique POIs (by place_id, category, lat, lon)
master_poi_pool = merged.drop_duplicates(subset=['place_id', 'category', 'lat', 'lon'])

# 6. Save to CSV
master_poi_pool.to_csv(output_csv, index=False)

# 7. Save to JSON in the requested format
poi_json_list = []
for _, row in master_poi_pool.iterrows():
    # Create hash from place_id
    hash_digest = hashlib.sha256(str(row['place_id']).encode()).hexdigest()[:8]
    
    poi_json_list.append({
        "poiId": f"POI-ID-{hash_digest}",
        # "poiId": row['place_id'],
        "poiCategories": [row['category']],
        "planningArea": None if pd.isna(row['planning_area']) else row['planning_area'],
        "userLocation": {
            "latitude": float(row['lat']),
            "longitude": float(row['lon'])
        }
    })

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(poi_json_list, f, indent=2, ensure_ascii=False)

print(f"Master POI pool saved to {output_csv} (CSV) and {output_json} (JSON) with {len(master_poi_pool)} unique POIs.")