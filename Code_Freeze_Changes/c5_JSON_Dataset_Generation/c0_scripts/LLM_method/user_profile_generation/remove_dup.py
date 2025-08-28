import json

# Load your JSON file
with open('c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_profile_generation/app_profiles_all_users_file_with_changes.json', 'r') as f:
    data = json.load(f)

# Remove the top-level 'views' key for each user object
for user_obj in data:
    if 'views' in user_obj:
        del user_obj['views']

# Save the cleaned JSON back to file
with open('c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_profile_generation/app_profiles_all_users_version_2.json', "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)