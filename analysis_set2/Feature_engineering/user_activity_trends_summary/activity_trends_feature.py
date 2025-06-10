"""
Extracts temporal activity trends for each user from singapore_checkins.txt.
Outputs: For each user, a CSV with activity counts per hour, per day, per month.
"""
import os
import datetime
import csv
from collections import defaultdict
import pandas as pd
import numpy as np
from calendar import day_name

CHECKINS_FILE = 'singapore_checkins_filtered.txt'
OUTPUT_DIR = 'analysis_set2/Feature_engineering/user_activity_trends'
OUTPUT_DIR_SUMMARY = 'analysis_set2/Feature_engineering/user_activity_trends_summary'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# user -> {(year, month, day, hour): count}
user_time_counts = defaultdict(lambda: defaultdict(int))

with open(CHECKINS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue
        user_id = parts[0]
        raw_time = parts[2]
        try:
            dt = datetime.datetime.strptime(raw_time, '%a %b %d %H:%M:%S %z %Y')
            dt_sgt = dt.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
            key = (dt_sgt.year, dt_sgt.month, dt_sgt.day, dt_sgt.hour)
            user_time_counts[user_id][key] += 1
        except Exception:
            continue

# --- Only include users with >=42 average check-ins ---
# Load users with >=42 check-ins
valid_users = set()
with open('preprocessing/sg_checkins_per_user.txt', 'r', encoding='utf-8') as f:
    for line in f:
        user, count = line.strip().split('\t')
        if int(count) >= 42:
            valid_users.add(user)

# Write per-user CSVs (only for valid users)
for user, time_counts in user_time_counts.items():
    if user not in valid_users:
        continue
    out_path = os.path.join(OUTPUT_DIR, f'{user}_activity_trend.csv')
    with open(out_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['year', 'month', 'day', 'hour', 'count'])
        for (year, month, day, hour), count in sorted(time_counts.items()):
            writer.writerow([year, month, day, hour, count])
print(f"Done. Per-user activity trend CSVs written to {OUTPUT_DIR}/ (users with >=42 check-ins)")

# Write a single CSV: one row per user, with user_id and total check-ins per hour, per day, per month
summary_path = os.path.join(OUTPUT_DIR_SUMMARY, 'user_hour_day_month_summary.csv')
summary_rows = []
for user, time_counts in user_time_counts.items():
    if user not in valid_users:
        continue
    # Aggregate per hour (0-23), per day (YYYY-MM-DD), per month (YYYY-MM)
    hour_counts = [0]*24
    day_counts = defaultdict(int)
    month_counts = defaultdict(int)
    for (year, month, day, hour), count in time_counts.items():
        hour_counts[hour] += count
        day_counts[f"{year:04d}-{month:02d}-{day:02d}"] += count
        month_counts[f"{year:04d}-{month:02d}"] += count
    row = {
        'user_id': user,
        **{f'hour_{h}': hour_counts[h] for h in range(24)},
        **{f'day_{d}': v for d, v in day_counts.items()},
        **{f'month_{m}': v for m, v in month_counts.items()}
    }
    summary_rows.append(row)
# Save as CSV
pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
print(f"User summary written to {summary_path}")

# Analyze trends for each user and write a summary
def hour_range_str(start, end):
    return f"{start:02d}:00-{end:02d}:00"

def get_peak_hour_range(hour_counts):
    # Find the 2-hour window with the highest sum
    max_sum = 0
    max_range = (0, 1)
    for i in range(23):
        s = hour_counts[i] + hour_counts[i+1]
        if s > max_sum:
            max_sum = s
            max_range = (i, i+1)
    return max_range, max_sum

def get_peak_day(day_counts):
    # day_counts: dict of YYYY-MM-DD -> count
    # Aggregate by weekday
    weekday_counts = [0]*7
    for d, v in day_counts.items():
        dt = datetime.datetime.strptime(d, "%Y-%m-%d")
        weekday_counts[dt.weekday()] += v
    max_idx = int(np.argmax(weekday_counts))
    return day_name[max_idx], weekday_counts[max_idx]

def get_peak_month(month_counts):
    # month_counts: dict of YYYY-MM -> count
    if not month_counts:
        return None, 0
    max_month = max(month_counts, key=month_counts.get)
    return max_month, month_counts[max_month]

# Load POI ID to category mapping
poi_id_to_cat = {}
with open('preprocessing/sg_poi_id_name.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            poi_id_to_cat[parts[0]] = parts[1]

# user -> list of (dt_sgt, venue_id)
user_checkins = defaultdict(list)
with open(CHECKINS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue
        user_id = parts[0]
        venue_id = parts[1]
        raw_time = parts[2]
        try:
            dt = datetime.datetime.strptime(raw_time, '%a %b %d %H:%M:%S %z %Y')
            dt_sgt = dt.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
            user_checkins[user_id].append((dt_sgt, venue_id))
        except Exception:
            continue

trend_summary_path = os.path.join(OUTPUT_DIR_SUMMARY, 'user_activity_trend_summary.txt')
with open(trend_summary_path, 'w', encoding='utf-8') as f:
    for user, time_counts in user_time_counts.items():
        if user not in valid_users:
            continue
        hour_counts = [0]*24
        day_counts = defaultdict(int)
        month_counts = defaultdict(int)
        for (year, month, day, hour), count in time_counts.items():
            hour_counts[hour] += count
            day_counts[f"{year:04d}-{month:02d}-{day:02d}"] += count
            month_counts[f"{year:04d}-{month:02d}"] += count
        # Hourly trend
        (h1, h2), max_hour_sum = get_peak_hour_range(hour_counts)
        # POI types in peak hour range
        hour_poi_counter = defaultdict(int)
        for dt, vid in user_checkins[user]:
            if h1 <= dt.hour <= h2:
                cat = poi_id_to_cat.get(vid, 'Unknown')
                hour_poi_counter[cat] += 1
        top_hour_poi = sorted(hour_poi_counter.items(), key=lambda x: -x[1])[:3]
        # Daily trend
        peak_day, peak_day_count = get_peak_day(day_counts)
        # POI types in peak day
        day_poi_counter = defaultdict(int)
        for dt, vid in user_checkins[user]:
            if day_name[dt.weekday()] == peak_day:
                cat = poi_id_to_cat.get(vid, 'Unknown')
                day_poi_counter[cat] += 1
        top_day_poi = sorted(day_poi_counter.items(), key=lambda x: -x[1])[:3]
        # Monthly trend
        peak_month, peak_month_count = get_peak_month(month_counts)
        # POI types in peak month
        month_poi_counter = defaultdict(int)
        for dt, vid in user_checkins[user]:
            if f"{dt.year:04d}-{dt.month:02d}" == peak_month:
                cat = poi_id_to_cat.get(vid, 'Unknown')
                month_poi_counter[cat] += 1
        top_month_poi = sorted(month_poi_counter.items(), key=lambda x: -x[1])[:3]
        f.write(f"User: {user}\n")
        f.write(f"  Peak activity hours: {hour_range_str(h1, h2)} (Total check-ins: {max_hour_sum})\n")
        f.write(f"    Top POI types: {', '.join([f'{cat} ({cnt})' for cat, cnt in top_hour_poi])}\n")
        f.write(f"  Peak day of week: {peak_day} (Total check-ins: {peak_day_count})\n")
        f.write(f"    Top POI types: {', '.join([f'{cat} ({cnt})' for cat, cnt in top_day_poi])}\n")
        f.write(f"  Peak month: {peak_month} (Total check-ins: {peak_month_count})\n")
        f.write(f"    Top POI types: {', '.join([f'{cat} ({cnt})' for cat, cnt in top_month_poi])}\n\n")
print(f"User trend summary written to {trend_summary_path}")

# --- CSV summary for user activity trends with POI encoding ---
# 1. Collect POI category counts for valid users
poi_cat_counter = defaultdict(int)
for user, checkins in user_checkins.items():
    if user not in valid_users:
        continue
    for dt, vid in checkins:
        cat = poi_id_to_cat.get(vid, 'Unknown')
        poi_cat_counter[cat] += 1
# 2. Select top 50 POI categories
all_poi_cats = [cat for cat, _ in sorted(poi_cat_counter.items(), key=lambda x: -x[1])[:50]]

# 3. For each user, create a row with encoded POI counts for peak hour, day, and month (only top 50 types)
encoding_csv_path = os.path.join(OUTPUT_DIR_SUMMARY, 'user_activity_trend_encoded.csv')
with open(encoding_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
    fieldnames = [
        'user_id',
        'peak_hour_range', 'peak_hour_checkins',
        'peak_day_of_week', 'peak_day_checkins',
        'peak_month', 'peak_month_checkins'
    ]
    # Add POI columns for each bin (only top 50)
    fieldnames += [f'peak_hour_POI_{cat}' for cat in all_poi_cats]
    fieldnames += [f'peak_day_POI_{cat}' for cat in all_poi_cats]
    fieldnames += [f'peak_month_POI_{cat}' for cat in all_poi_cats]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for user, time_counts in user_time_counts.items():
        if user not in valid_users:
            continue
        hour_counts = [0]*24
        day_counts = defaultdict(int)
        month_counts = defaultdict(int)
        for (year, month, day, hour), count in time_counts.items():
            hour_counts[hour] += count
            day_counts[f"{year:04d}-{month:02d}-{day:02d}"] += count
            month_counts[f"{year:04d}-{month:02d}"] += count
        # Hourly trend
        (h1, h2), max_hour_sum = get_peak_hour_range(hour_counts)
        hour_poi_counter = defaultdict(int)
        for dt, vid in user_checkins[user]:
            if h1 <= dt.hour <= h2:
                cat = poi_id_to_cat.get(vid, 'Unknown')
                if cat in all_poi_cats:
                    hour_poi_counter[cat] += 1
        # Daily trend
        peak_day, peak_day_count = get_peak_day(day_counts)
        day_poi_counter = defaultdict(int)
        for dt, vid in user_checkins[user]:
            if day_name[dt.weekday()] == peak_day:
                cat = poi_id_to_cat.get(vid, 'Unknown')
                if cat in all_poi_cats:
                    day_poi_counter[cat] += 1
        # Monthly trend
        peak_month, peak_month_count = get_peak_month(month_counts)
        month_poi_counter = defaultdict(int)
        for dt, vid in user_checkins[user]:
            if f"{dt.year:04d}-{dt.month:02d}" == peak_month:
                cat = poi_id_to_cat.get(vid, 'Unknown')
                if cat in all_poi_cats:
                    month_poi_counter[cat] += 1
        row = {
            'user_id': user,
            'peak_hour_range': f"{hour_range_str(h1, h2)}",
            'peak_hour_checkins': max_hour_sum,
            'peak_day_of_week': peak_day,
            'peak_day_checkins': peak_day_count,
            'peak_month': peak_month,
            'peak_month_checkins': peak_month_count
        }
        # POI encoding for each bin (only top 50)
        for cat in all_poi_cats:
            row[f'peak_hour_POI_{cat}'] = hour_poi_counter.get(cat, 0)
            row[f'peak_day_POI_{cat}'] = day_poi_counter.get(cat, 0)
            row[f'peak_month_POI_{cat}'] = month_poi_counter.get(cat, 0)
        writer.writerow(row)
print(f"User activity trend encoded CSV written to {encoding_csv_path}")