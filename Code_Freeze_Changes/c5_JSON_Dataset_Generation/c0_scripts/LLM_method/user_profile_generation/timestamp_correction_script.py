import json
from datetime import datetime

# Helper to get month number from name
MONTHS = {m: i for i, m in enumerate([
    '', 'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])}

def get_month_number(month_name):
    return MONTHS.get(month_name, 0)

# Load check-in data
with open('C:\\Users\\admin\\Desktop\\sweta\\MPS_syn_data_gen\\singapore_foursquare_dataset\\Code_Freeze_Changes\\c5_JSON_Dataset_Generation\\c0_scripts\\LLM_method\\input_dataset\\final_checkin_file.json', 'r', encoding='utf-8') as f:
    checkin_data = json.load(f)

# Build a lookup: user_id -> list of metadata
checkin_lookup = {}
for entry in checkin_data:
    user_id = str(entry['user_id'])
    checkin_lookup[user_id] = entry['user_metadata']

# Load app profiles data
with open('C:\\Users\\admin\\Desktop\\sweta\\MPS_syn_data_gen\\singapore_foursquare_dataset\\Code_Freeze_Changes\\c5_JSON_Dataset_Generation\\c0_scripts\\LLM_method\\user_profile_generation\\app_profiles_all_users_LLM_reviews_unique_per_poi.json', 'r', encoding='utf-8') as f:
    app_profiles = json.load(f)

def find_time(user_id, poi_category, planning_area, date_str):
    """Finds the time_of_day for a given user, poi_category, planning_area, and date."""
    metas = checkin_lookup.get(user_id, [])
    # Parse date
    dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    month = dt.strftime("%B")
    day_of_week = dt.strftime("%A")
    for meta in metas:
        if (meta.get('poi_category', '').lower() == poi_category.lower() and
            meta.get('planning_area', '').upper() == planning_area.upper() and
            meta.get('day_of_week', '') == day_of_week and
            meta.get('month_of_year', '') == month):
            return meta.get('time_of_day')
    return None

def update_section(section, user_id):
    for item in section:
        if 'timestamp' in item and 'poiCategories' in item and 'poiId' in item:
            date_str = item['timestamp']
            # Extract category and planning area from POI ID
            # Example: POI-nightclub-ORCHARD-000
            poi_id_parts = item['poiId'].split('-')
            if len(poi_id_parts) >= 3:
                planning_area = poi_id_parts[-2]
                poi_category = item['poiCategories'][0]
                time_of_day = find_time(user_id, poi_category, planning_area, date_str)
                if time_of_day:
                    # Build new timestamp
                    dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
                    # Convert time_of_day to 24h
                    t = datetime.strptime(time_of_day, "%I:%M %p")
                    dt = dt.replace(hour=t.hour, minute=t.minute)
                    item['timestamp'] = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

for user_profile in app_profiles:
    user_id = str(user_profile['user']['userId'])
    for section_name in ['views', 'transactions', 'reviews']:
        section = user_profile['interaction'].get(section_name, [])
        update_section(section, user_id)

# Save updated profiles
with open('C:\\Users\\admin\\Desktop\\sweta\\MPS_syn_data_gen\\singapore_foursquare_dataset\\Code_Freeze_Changes\\c5_JSON_Dataset_Generation\\c0_scripts\\LLM_method\\user_profile_generation\\app_profiles_all_users_LLM_reviews_unique_per_poi.updated.json', 'w', encoding='utf-8') as f:
    json.dump(app_profiles, f, indent=2)