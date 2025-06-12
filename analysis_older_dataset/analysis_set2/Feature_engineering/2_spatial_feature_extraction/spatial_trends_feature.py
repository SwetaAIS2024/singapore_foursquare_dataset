"""
Extracts grid-based spatial features for each user from the filtered check-ins file.
Features computed per user:
- main_grid_cell: the grid cell with the most check-ins
- num_unique_grid_cells: number of unique grid cells visited
- fraction_in_main_grid_cell: fraction of check-ins in the main grid cell
- grid_entropy: entropy of the user's check-in distribution across grid cells

Assumes the check-in file format: user_id, venue_id, latitude, longitude, ... (tab-separated)
"""
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import math

# Config
CHECKINS_FILE = 'singapore_checkins_filtered_with_locations_coord.txt'  # updated input file
OUTPUT_CSV = 'analysis_set2/Feature_engineering/2_spatial_feature_extraction/user_spatial_grid_features.csv'
GRID_SIZE = 10  # 10x10 grid

# 1. Read check-ins and collect all lat/lon
user_checkins = defaultdict(list)
all_lats = []
all_lons = []
with open(CHECKINS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 6:
            continue
        user_id = parts[0]
        try:
            lat = float(parts[4])
            lon = float(parts[5])
        except Exception:
            continue
        user_checkins[user_id].append((lat, lon))
        all_lats.append(lat)
        all_lons.append(lon)

# 2. Define grid boundaries
min_lat, max_lat = min(all_lats), max(all_lats)
min_lon, max_lon = min(all_lons), max(all_lons)
lat_step = (max_lat - min_lat) / GRID_SIZE
lon_step = (max_lon - min_lon) / GRID_SIZE

def get_grid_cell(lat, lon):
    i = int((lat - min_lat) / lat_step)
    j = int((lon - min_lon) / lon_step)
    # Clamp to grid
    i = min(max(i, 0), GRID_SIZE-1)
    j = min(max(j, 0), GRID_SIZE-1)
    return f"cell_{i}_{j}"

# 3. Compute features per user
rows = []
for user, coords in user_checkins.items():
    if not coords:
        continue
    grid_cells = [get_grid_cell(lat, lon) for lat, lon in coords]
    total = len(grid_cells)
    cell_counts = Counter(grid_cells)
    main_cell, main_count = cell_counts.most_common(1)[0]
    num_unique = len(cell_counts)
    if num_unique <= 1:
        continue  # Skip users with only one unique grid cell
    fraction_main = main_count / total
    # Entropy
    probs = np.array(list(cell_counts.values())) / total
    grid_entropy = -np.sum(probs * np.log2(probs))
    rows.append({
        'user_id': user,
        'main_grid_cell': main_cell,
        'num_unique_grid_cells': num_unique,
        'fraction_in_main_grid_cell': fraction_main,
        'grid_entropy': grid_entropy
    })

# 4. Save to CSV
pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
print(f"Spatial grid features written to {OUTPUT_CSV}")
