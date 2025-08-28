import json
import random

# Load your JSON file
with open("c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_profile_generation/app_profiles_all_users_file_with_changes.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for user in data:
    views = user.get('interaction', {}).get('views', [])
    n_views = len(views)
    n_high = int(n_views * 0.7)
    n_low = n_views - n_high

    # Shuffle indices to randomize which views get which durations
    indices = list(range(n_views))
    random.shuffle(indices)

    # Assign high durations (40-120s) to 70%
    for idx in indices[:n_high]:
        views[idx]['duration'] = random.randint(40, 120)

    # Assign low durations (<20s) to the rest
    for idx in indices[n_high:]:
        views[idx]['duration'] = random.randint(1, 19)

    # Save back the modified views
    user['interaction']['views'] = views

# Save the corrected data back to a new JSON file
with open("c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_profile_generation/app_profiles_all_users_version_1.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)