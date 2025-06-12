# Script to generate timeline graphs for top 10 subcategories in each POI category
# For each main category, finds top 10 subcategories (by check-ins),
# then creates a check-ins vs timeline graph for each subcategory.
# Graphs are saved in subfolders by main category.

import os
import matplotlib.pyplot as plt
import datetime
from collections import Counter, defaultdict
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POI_FILE = os.path.join(BASE_DIR, '..', '..', 'main_files_fsq', 'dataset_TIST2015_POIs.txt')
CHECKINS_FILE = os.path.join(BASE_DIR, '..', 'singapore_checkins.txt')
# Change to correct relative path from current script location
if not os.path.exists(POI_FILE):
    POI_FILE = os.path.join(BASE_DIR, '..', '..', 'main_files_fsq', 'dataset_TIST2015_POIs.txt')
if not os.path.exists(CHECKINS_FILE):
    CHECKINS_FILE = os.path.join(BASE_DIR, '..', '..', 'singapore_checkins.txt')

# Step 1: Build POI ID to (main_cat, sub_cat) mapping
poi_id_to_cat = {}
cat_to_subcats = defaultdict(set)

with open(POI_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 5:
            continue
        poi_id = parts[0]
        sub_cat = parts[3]
        main_cat = sub_cat.split(' / ')[0] if ' / ' in sub_cat else sub_cat
        poi_id_to_cat[poi_id] = (main_cat, sub_cat)
        cat_to_subcats[main_cat].add(sub_cat)

# Step 2: Count check-ins per subcategory
subcat_checkin_counter = Counter()
with open(CHECKINS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        poi_id = parts[1]
        if poi_id in poi_id_to_cat:
            main_cat, sub_cat = poi_id_to_cat[poi_id]
            subcat_checkin_counter[(main_cat, sub_cat)] += 1

# Step 3: For each main category, get top 10 subcategories
maincat_to_top_subcats = {}
for main_cat, subcats in cat_to_subcats.items():
    subcat_counts = [(sub_cat, subcat_checkin_counter[(main_cat, sub_cat)]) for sub_cat in subcats]
    subcat_counts.sort(key=lambda x: x[1], reverse=True)
    maincat_to_top_subcats[main_cat] = [sub_cat for sub_cat, _ in subcat_counts[:10]]

# Step 4: For each subcategory, collect check-in times
subcat_to_times = defaultdict(list)
with open(CHECKINS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue
        poi_id = parts[1]
        raw_time = parts[2]
        if poi_id in poi_id_to_cat:
            main_cat, sub_cat = poi_id_to_cat[poi_id]
            if sub_cat in maincat_to_top_subcats[main_cat]:
                try:
                    dt = datetime.datetime.strptime(raw_time, '%a %b %d %H:%M:%S %z %Y')
                    # Convert to Singapore local time (UTC+8)
                    dt_sgt = dt.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
                    subcat_to_times[(main_cat, sub_cat)].append(dt_sgt)
                except Exception:
                    continue

# Only generate timeline and 24-hour plots for top 20 main categories

# Get top 20 main categories by total check-ins
maincat_total = Counter()
for (main_cat, sub_cat), count in subcat_checkin_counter.items():
    maincat_total[main_cat] += count

top20_maincats = [cat for cat, _ in maincat_total.most_common(20)]

# Step 5: Plot timeline and 24-hour distribution for each subcategory (top 20 main categories only)
for main_cat in tqdm(top20_maincats, desc='Main categories'):
    for sub_cat in maincat_to_top_subcats.get(main_cat, []):
        times = subcat_to_times.get((main_cat, sub_cat), [])
        if not times:
            continue
        times.sort()
        # Timeline plot (by day)
        day_counter = Counter([dt.date() for dt in times])
        days = sorted(day_counter)
        counts = [day_counter[day] for day in days]
        folder = f'{main_cat.replace("/", "_").replace(" ", "_")}'
        out_dir = os.path.join('sg_category_timeline_analysis', folder)
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(14, 5))
        plt.plot(days, counts, marker='o')
        plt.xlabel('Date')
        plt.ylabel('Number of Check-ins')
        plt.title(f'Check-ins Over Time: {sub_cat} (Top 10 in {main_cat}, SGT)')
        plt.tight_layout()
        fname = f'{sub_cat.replace("/", "_").replace(" ", "_")}_timeline.png'
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()
        # 24-hour plot
        hour_counter = Counter([dt.hour for dt in times])
        hours = list(range(24))
        hour_counts = [hour_counter.get(h, 0) for h in hours]
        plt.figure(figsize=(10, 5))
        plt.bar(hours, hour_counts, color='teal')
        plt.xlabel('Hour of Day (SGT)')
        plt.ylabel('Number of Check-ins')
        plt.title(f'Check-ins by Hour: {sub_cat} (Top 10 in {main_cat}, SGT)')
        plt.xticks(hours)
        plt.tight_layout()
        fname = f'{sub_cat.replace("/", "_").replace(" ", "_")}_hourly.png'
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()
print('Timeline and 24-hour distribution plots generated for top 20 main categories.')
