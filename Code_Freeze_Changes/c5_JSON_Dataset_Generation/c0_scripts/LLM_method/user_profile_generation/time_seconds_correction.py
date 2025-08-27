import json
from datetime import datetime, timedelta
import random

def random_jitter_timestamp(ts):
    # Parse the existing timestamp
    dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    # Add random jitter (example: 1-7 days, 1-7 hours)
    dt += timedelta(seconds=random.randint(0, 59))
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

with open("c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_profile_generation/app_profiles_all_users_file_with_changes.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for user in data:
    for interaction_type in ["views", "transactions", "reviews"]:
        for interaction in user.get("interaction", {}).get(interaction_type, []):
            old_ts = interaction.get("timestamp")
            if old_ts:
                interaction["timestamp"] = random_jitter_timestamp(old_ts)

with open("c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_profile_generation/app_profiles_all_users_file_with_changes.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)