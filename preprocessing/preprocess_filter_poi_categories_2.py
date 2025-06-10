"""
Preprocessing script to filter out check-ins for specified POI categories.
- Loads singapore_checkins.txt
- Loads POI ID to category mapping from sg_poi_id_name.txt
- Removes all check-ins where the POI category is in the filter list
- Writes a filtered check-ins file (e.g., singapore_checkins_filtered.txt)
- The filter list can be edited in the script
"""
import os

# Config
CHECKINS_FILE = 'preprocessing/singapore_checkins.txt'
POI_ID_NAME_FILE = 'preprocessing/sg_poi_id_name.txt'
OUTPUT_FILE = 'preprocessing/singapore_checkins_filtered.txt'
USER_CHECKINS_FILE = 'preprocessing/sg_checkins_per_user.txt'
MIN_USER_CHECKINS = 40

# List of POI categories to filter out (edit as needed)
FILTER_POI_CATEGORIES = [
    'Residential Building (Apartment / Condo)',
    'Home (private)',
    'Housing Development',
    'Office',
    'Building',
    'College Classroom',
    'Road',
    # Add more categories as needed
]

# Load POI ID to category mapping
poi_id_to_cat = {}
with open(POI_ID_NAME_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            poi_id_to_cat[parts[0]] = parts[1]

# Load valid users
valid_users = set()
with open(USER_CHECKINS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        user, count = line.strip().split('\t')
        if int(count) >= MIN_USER_CHECKINS:
            valid_users.add(user)

# Filter check-ins
with open(CHECKINS_FILE, 'r', encoding='utf-8') as fin, open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
    removed = 0
    kept = 0
    for line in fin:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        user_id = parts[0]
        venue_id = parts[1]
        cat = poi_id_to_cat.get(venue_id, None)
        if cat in FILTER_POI_CATEGORIES:
            removed += 1
            continue
        if user_id not in valid_users:
            removed += 1
            continue
        fout.write(line)
        kept += 1
print(f"Done. Filtered check-ins written to {OUTPUT_FILE}")
print(f"Check-ins removed: {removed}")
print(f"Check-ins kept: {kept}")
