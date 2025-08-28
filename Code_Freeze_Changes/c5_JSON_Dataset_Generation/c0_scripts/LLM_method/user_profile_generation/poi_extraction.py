import json

input_file = "c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_profile_generation/app_profiles_all_users_file_with_changes.json"
output_file = "c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_profile_generation/unique_pois_from_all_interactions.json"

unique_pois = {}

def make_hashable(poi_id, categories, location):
    # Convert dict to tuple for hashing
    return (
        poi_id,
        tuple(categories) if categories else (),
        tuple(sorted(location.items())) if location else ()
    )

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for user_obj in data:
    # Collect all views from both locations
    all_views = []
    # Top-level views
    if "views" in user_obj and isinstance(user_obj["views"], list):
        all_views.extend(user_obj["views"])
    # Views inside interaction
    if "interaction" in user_obj and "views" in user_obj["interaction"] and isinstance(user_obj["interaction"]["views"], list):
        all_views.extend(user_obj["interaction"]["views"])
    # Extract unique POIs from all views
    for view in all_views:
        poi_id = view.get("poiId")
        poi_categories = view.get("poiCategories")
        user_location = view.get("userLocation")
        if poi_id and poi_categories and user_location:
            key = make_hashable(poi_id, poi_categories, user_location)
            if key not in unique_pois:
                unique_pois[key] = {
                    "poiId": poi_id,
                    "poiCategories": poi_categories,
                    "userLocation": user_location
                }

# Convert to list
unique_pois_list = list(unique_pois.values())

# Save to file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(unique_pois_list, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(unique_pois_list)} unique POIs to {output_file}")