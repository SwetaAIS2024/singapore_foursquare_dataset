import pandas as pd
import json
from collections import defaultdict

# Load synthetic checkins
df = pd.read_csv('analysis_older_dataset/Final_code/syn_org_eval/synthetic_checkins.txt')

# Load POI category mapping
cat_map = pd.read_csv('analysis_older_dataset/Final_code/clustering/sg_place_id_to_category.csv')
cat_dict = dict(zip(cat_map['place_id'], cat_map['category']))

def get_user_profile(user_df, cat_dict):
    user_id = user_df['synthetic_user_id'].iloc[0]
    profile = {
        "user": {
            "userId": user_id,
            "age": None,
            "gender": None,
            "location": {"city": "Singapore", "country": "SG"},
            "device": {"platform": None, "appVersion": None}
        },
        "interaction": {"views": []},
        "pois": []
    }
    pois = {}
    for _, row in user_df.iterrows():
        poi_id = row['place_id']
        cat = cat_dict.get(poi_id, "Unknown")
        view = {
            "timestamp": pd.to_datetime(row['datetime'], errors='coerce').isoformat() if pd.notnull(row['datetime']) else None,
            "poiId": poi_id,
            "poiCategories": [cat],
            "poiSubcategories": [],
            "duration": None,
            "referrer": None,
            "userLocation": {"latitude": float(row['lat']), "longitude": float(row['lon'])}
        }
        profile["interaction"]["views"].append(view)
        if poi_id not in pois:
            pois[poi_id] = {
                "poiId": poi_id,
                "name": None,
                "categories": [cat],
                "subcategories": [],
                "location": {"latitude": float(row['lat']), "longitude": float(row['lon']), "address": None},
                "nearestStation": None,
                "rating": None,
                "deal": None
            }
    profile["pois"] = list(pois.values())
    return profile

# Generate profiles for 1000 different users
for i, (user_id, user_df) in enumerate(df.groupby('synthetic_user_id')):
    profile = get_user_profile(user_df, cat_dict)
    with open(f'analysis_older_dataset/Final_code/json_dataset_generation/user_profile_{user_id}.json', 'w') as f:
        json.dump(profile, f, indent=2)
    if i >= 999:
        break
