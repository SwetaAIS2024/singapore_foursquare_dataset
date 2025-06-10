"""
Adds latitude and longitude to each check-in in singapore_checkins_filtered.txt using POI info from dataset_TIST2015_POIs.txt (SG only).
Output: singapore_checkins_filtered_with_locations_coord.txt
"""
import os

POI_FILE = "original_dataset_truncated/dataset_TIST2015_POIs.txt"
CHECKINS_FILE = "singapore_checkins_filtered.txt"
OUTPUT_FILE = "singapore_checkins_filtered_with_locations_coord.txt"

# 1. Build venue_id -> (lat, lon) for SG
venue_to_latlon = {}
with open(POI_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 5:
            continue
        venue_id, lat, lon, category, country = parts[:5]
        if country == 'SG':
            try:
                venue_to_latlon[venue_id] = (lat, lon)
            except Exception:
                continue

# 2. Process check-ins and add lat/lon
with open(CHECKINS_FILE, 'r', encoding='utf-8') as fin, open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
    for line in fin:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        user_id, venue_id = parts[0], parts[1]
        rest = parts[2:]
        latlon = venue_to_latlon.get(venue_id)
        if latlon:
            fout.write('\t'.join([user_id, venue_id] + rest + [latlon[0], latlon[1]]) + '\n')
print(f"Done. Output written to {OUTPUT_FILE}")
