# Script to generate per-day trajectory plots for top 15 users (>1000 check-ins)
# Each user's plots are saved in their own folder under sg_trajectory_analysis_top_15_users_1k_checkins

import os
import matplotlib.pyplot as plt
import datetime
from collections import defaultdict

SG_CHECKINS_FILE = 'singapore_checkins.txt'
USER_COUNTS_FILE = 'sg_checkins_per_user.txt'
OUTPUT_BASE = 'sg_trajectory_analysis_top_15_users_1k_checkins'

# Step 1: Get top 15 users with >1000 check-ins
user_counts = []
with open(USER_COUNTS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        user, count = line.strip().split('\t')
        count = int(count)
        if count > 1000:
            user_counts.append((user, count))
        if len(user_counts) >= 15:
            break

top_users = [user for user, _ in user_counts]

# Step 2: Collect all check-ins for top users, grouped by user and day
user_day_checkins = {user: defaultdict(list) for user in top_users}
with open(SG_CHECKINS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue
        user_id, venue_id, raw_time = parts[0], parts[1], parts[2]
        if user_id not in top_users:
            continue
        try:
            dt = datetime.datetime.strptime(raw_time, '%a %b %d %H:%M:%S %z %Y')
            dt_sgt = dt.astimezone(datetime.timezone(datetime.timedelta(hours=8)))
            day = dt_sgt.date()
            user_day_checkins[user_id][day].append((dt_sgt, venue_id))
        except Exception:
            continue

# Step 2.1: Build venue_id to category mapping for Singapore POIs
venue_to_cat = {}
with open('sg_poi_id_name.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        venue_to_cat[parts[0]] = parts[1]

# Step 3: Plot per-day trajectories for each user
os.makedirs(OUTPUT_BASE, exist_ok=True)
for user in top_users:
    user_dir = os.path.join(OUTPUT_BASE, user)
    os.makedirs(user_dir, exist_ok=True)
    for day, checkins in user_day_checkins[user].items():
        if len(checkins) <= 2:
            continue  # Only analyze days with more than 2 check-ins
        checkins.sort()  # sort by datetime
        times = [dt.strftime('%H:%M') for dt, vid in checkins]  # 24-hour format
        categories = [venue_to_cat[vid] if vid in venue_to_cat else 'Unknown' for dt, vid in checkins]
        plt.figure(figsize=(10, 2))
        plt.plot(times, categories, marker='o', linestyle='-')
        plt.xlabel('Time (SGT, 24h)')
        plt.ylabel('POI Category')
        plt.title(f'User {user} Trajectory on {day} (n={len(checkins)})')
        plt.tight_layout()
        fname = f'{day}_trajectory.png'
        plt.savefig(os.path.join(user_dir, fname))
        plt.close()

# Step 4: Create histograms for each user (day, noon, evening)
HIST_BASE = 'histograms'
os.makedirs(HIST_BASE, exist_ok=True)
time_bins = {
    'day': (6, 12),      # 06:00 to 11:59
    'noon': (12, 18),    # 12:00 to 17:59
    'evening': (18, 24)  # 18:00 to 23:59
}
for user in top_users:
    hist_dir = os.path.join(HIST_BASE, user)
    os.makedirs(hist_dir, exist_ok=True)
    # Collect all check-ins for this user, grouped by category and time bin
    bin_cat_counts = {bin_name: defaultdict(list) for bin_name in time_bins}
    for day, checkins in user_day_checkins[user].items():
        for dt, vid in checkins:
            hour = dt.hour
            cat = venue_to_cat[vid] if vid in venue_to_cat else 'Unknown'
            for bin_name, (start, end) in time_bins.items():
                if start <= hour < end:
                    bin_cat_counts[bin_name][cat].append(day)
    # For each bin, plot histogram of POI type vs avg check count per day
    for bin_name in time_bins:
        cat_avg = {}
        for cat, days in bin_cat_counts[bin_name].items():
            # Average check-ins per day for this category in this bin
            day_counts = defaultdict(int)
            for d in days:
                day_counts[d] += 1
            if day_counts:
                avg = sum(day_counts.values()) / len(day_counts)
                cat_avg[cat] = avg
        if not cat_avg:
            continue
        plt.figure(figsize=(10, 4))
        cats = list(cat_avg.keys())
        avgs = [cat_avg[c] for c in cats]
        plt.barh(cats, avgs, color='skyblue')
        plt.xlabel('Avg Check-ins per Day')
        plt.ylabel('POI Category')
        plt.title(f'User {user} - {bin_name.capitalize()} Histogram')
        plt.tight_layout()
        plt.savefig(os.path.join(hist_dir, f'{bin_name}_hist.png'))
        plt.close()
        # Save histogram data as CSV for summary script
        import csv
        csv_path = os.path.join(hist_dir, f'{bin_name}_hist.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['POI Category', 'Avg Check-ins per Day'])
            for c in cats:
                writer.writerow([c, cat_avg[c]])

print('Trajectory plots and histograms generated for top 15 users.')
