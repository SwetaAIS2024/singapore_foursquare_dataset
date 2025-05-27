# Script to analyze Singapore check-ins
# - Number of check-ins per user
# - Number of check-ins per place (venue)
# - Additional stats: most active user, most popular place, average check-ins per user/place

from collections import Counter
import matplotlib.pyplot as plt
import datetime
import numpy as np

SG_CHECKINS_FILE = 'singapore_checkins.txt'

user_counter = Counter()
place_counter = Counter()

with open(SG_CHECKINS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        user_id = parts[0]
        venue_id = parts[1]
        user_counter[user_id] += 1
        place_counter[venue_id] += 1

# Basic stats
num_users = len(user_counter)
num_places = len(place_counter)
total_checkins = sum(user_counter.values())
avg_checkins_per_user = total_checkins / num_users if num_users else 0
avg_checkins_per_place = total_checkins / num_places if num_places else 0
most_active_user, max_user_checkins = user_counter.most_common(1)[0]
most_popular_place, max_place_checkins = place_counter.most_common(1)[0]

print(f"Total check-ins: {total_checkins}")
print(f"Number of unique users: {num_users}")
print(f"Number of unique places: {num_places}")
print(f"Average check-ins per user: {avg_checkins_per_user:.2f}")
print(f"Average check-ins per place: {avg_checkins_per_place:.2f}")
print(f"Most active user: {most_active_user} with {max_user_checkins} check-ins")
print(f"Most popular place: {most_popular_place} with {max_place_checkins} check-ins")

# Save summary stats to a file
with open('singapore_analysis_foursquare.txt', 'w', encoding='utf-8') as out:
    out.write(f"Total check-ins: {total_checkins}\n")
    out.write(f"Number of unique users: {num_users}\n")
    out.write(f"Number of unique places: {num_places}\n")
    out.write(f"Average check-ins per user: {avg_checkins_per_user:.2f}\n")
    out.write(f"Average check-ins per place: {avg_checkins_per_place:.2f}\n")
    out.write(f"Most active user: {most_active_user} with {max_user_checkins} check-ins\n")
    out.write(f"Most popular place: {most_popular_place} with {max_place_checkins} check-ins\n")

# Save distributions to files
with open('sg_checkins_per_user.txt', 'w', encoding='utf-8') as out:
    for user, count in user_counter.most_common():
        out.write(f"{user}\t{count}\n")
with open('sg_checkins_per_place.txt', 'w', encoding='utf-8') as out:
    for place, count in place_counter.most_common():
        out.write(f"{place}\t{count}\n")

# --- Distribution of check-ins over time (all check-ins) ---
hour_counter = Counter()
with open(SG_CHECKINS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue
        raw_time = parts[2]
        try:
            dt = datetime.datetime.strptime(raw_time, '%a %b %d %H:%M:%S %z %Y')
            # Convert to Singapore local time (UTC+8)
            dt_sgt = dt.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
            hour_counter[dt_sgt.hour] += 1
        except Exception:
            continue

hours = list(range(24))
counts = [hour_counter.get(h, 0) for h in hours]
plt.figure(figsize=(10, 5))
plt.bar(hours, counts, color='skyblue')
plt.xlabel('Hour of Day (SGT)')
plt.ylabel('Number of Check-ins')
plt.title('Distribution of Singapore Check-ins by Hour (All POIs, SGT)')
plt.xticks(hours)
plt.tight_layout()
plt.savefig('sg_checkins_by_hour_all.png')
plt.close()

# --- Load POI names for Singapore POIs ---
poi_id_to_name = {}
with open('sg_poi_id_name.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        poi_id, poi_name = parts[0], parts[1]
        poi_id_to_name[poi_id] = poi_name

# --- Top 5 POIs: hourly check-in distribution with POI name in title ---
top5_places = [place for place, _ in place_counter.most_common(5)]
for place_id in top5_places:
    place_hour_counter = Counter()
    with open(SG_CHECKINS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            if parts[1] != place_id:
                continue
            raw_time = parts[2]
            try:
                dt = datetime.datetime.strptime(raw_time, '%a %b %d %H:%M:%S %z %Y')
                # Convert to Singapore local time (UTC+8)
                dt_sgt = dt.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
                place_hour_counter[dt_sgt.hour] += 1
            except Exception:
                continue
    place_counts = [place_hour_counter.get(h, 0) for h in hours]
    poi_name = poi_id_to_name.get(place_id, 'Unknown')
    plt.figure(figsize=(10, 5))
    plt.bar(hours, place_counts, color='orange')
    plt.xlabel('Hour of Day (SGT)')
    plt.ylabel('Number of Check-ins')
    plt.title(f'Check-ins by Hour for POI {place_id} ({poi_name}), SGT')
    plt.xticks(hours)
    plt.tight_layout()
    plt.savefig(f'sg_checkins_by_hour_{place_id}.png')
    plt.close()

# --- Plot hourly check-in distribution for top 5 categories ---
top_categories = [
    'Home (private)',
    'Residential Building (Apartment / Condo)',
    'Chinese Restaurant',
    'Bus Station',
    'Office'
]

# Build category to POI IDs mapping
category_to_poi_ids = {cat: set() for cat in top_categories}
with open('sg_poi_id_name.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        poi_id, poi_name = parts[0], parts[1]
        if poi_name in category_to_poi_ids:
            category_to_poi_ids[poi_name].add(poi_id)

# Aggregate check-ins for each category by hour
for cat in top_categories:
    cat_hour_counter = Counter()
    poi_ids = category_to_poi_ids[cat]
    with open(SG_CHECKINS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            venue_id = parts[1]
            if venue_id not in poi_ids:
                continue
            raw_time = parts[2]
            try:
                dt = datetime.datetime.strptime(raw_time, '%a %b %d %H:%M:%S %z %Y')
                # Convert to Singapore local time (UTC+8)
                dt_sgt = dt.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
                cat_hour_counter[dt_sgt.hour] += 1
            except Exception:
                continue
    cat_counts = [cat_hour_counter.get(h, 0) for h in hours]
    plt.figure(figsize=(10, 5))
    plt.bar(hours, cat_counts, color=np.random.rand(3,))
    plt.xlabel('Hour of Day (SGT)')
    plt.ylabel('Number of Check-ins')
    plt.title(f'Check-ins by Hour for Category: {cat} (SGT)')
    plt.xticks(hours)
    plt.tight_layout()
    plt.savefig(f'sg_checkins_by_hour_cat_{cat.replace(" ", "_").replace("/", "_")}.png')
    plt.close()
