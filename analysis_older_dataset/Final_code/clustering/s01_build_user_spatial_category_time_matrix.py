import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from s00_config_paths import CHECKINS_PATH, CATEGORIES_XLSX, MATRIX_PATH, PLACE_ID_POI_CAT

# --- Parameters ---
def main(eps_km=0.5, min_samples=10, n_time_bins=168):
    # Load check-in data (with lat/lon)
    cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon']
    df = pd.read_csv(CHECKINS_PATH, sep='\t', names=cols)

    # --- Step 1: Spatial clustering of POIs ---
    df = df.dropna(subset=['lat', 'lon'])
    coords = df[['lat', 'lon']].to_numpy()
    coords_rad = np.radians(coords)
    kms_per_radian = 6371.0088
    epsilon = eps_km / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(coords_rad)
    df['spatial_cluster'] = db.labels_

    # Remap DBSCAN cluster labels to consecutive indices (ignore noise)
    valid_spatial_clusters = sorted([c for c in set(df['spatial_cluster']) if c != -1])
    spatial_cluster_map = {old: new for new, old in enumerate(valid_spatial_clusters)}
    df = df[df['spatial_cluster'] != -1]  # Remove noise
    df['spatial_cluster'] = df['spatial_cluster'].map(spatial_cluster_map)

    # --- Step 2: Load relevant POI categories ---
    relevant_cats_df = pd.read_excel(CATEGORIES_XLSX)
    cat_col = 'POI Category in Singapore'
    yes_col = 'Relevant to use case '
    relevant_categories = [cat for cat, flag in zip(relevant_cats_df[cat_col], relevant_cats_df[yes_col]) if str(flag).strip().lower() == 'yes']
    relevant_categories = [cat.strip().lower() for cat in relevant_categories if cat and str(cat).strip()]
    relevant_categories = list(dict.fromkeys(relevant_categories))
    ordered_categories = [cat.title() for cat in relevant_categories]
    cat2idx = {cat: i for i, cat in enumerate(ordered_categories)}

    # --- Step 3: Prepare time features ---
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    if 'category' not in df.columns:
        try:
            place_cat = pd.read_csv(PLACE_ID_POI_CAT)
            df = df.merge(place_cat[['place_id', 'category']], on='place_id', how='left')
        except Exception as e:
            print('Warning: Could not merge category info:', e)
            df['category'] = ''
    df = df.dropna(subset=['datetime', 'category', 'spatial_cluster'])
    df['hour_of_week'] = df['datetime'].dt.dayofweek * 24 + df['datetime'].dt.hour
    df['category'] = df['category'].astype(str).str.strip().str.lower()
    df = df[df['category'].isin(relevant_categories)]

    # --- Step 4: Build 4D matrix: user x spatial_cluster x category x time ---
    # Fixed user, cluster, and category lists for consistent shape
    n_users = 2000
    n_clusters = 50
    n_categories = 180
    n_time_bins = 168

    # Get top 2000 users by check-in count AFTER all filtering
    user_counts = df['user_id'].value_counts().head(n_users)
    users = user_counts.index.tolist()
    user_idx = {u: i for i, u in enumerate(users)}

    # Use spatial cluster labels 0 to min(49, actual number of clusters-1)
    spatial_clusters = list(range(min(n_clusters, len(valid_spatial_clusters))))
    spatial_cluster_idx = {c: i for i, c in enumerate(spatial_clusters)}

    # Use the first 180 categories from ordered_categories
    ordered_categories = ordered_categories[:n_categories]
    cat2idx = {cat: i for i, cat in enumerate(ordered_categories)}

    print("First user in matrix:", users[0] if users else None)
    print("Check-ins for this user after filtering:", df[df['user_id'] == users[0]].shape[0] if users else 0)
    print("First 10 user IDs in filtered data:", df['user_id'].unique()[:10])
    print("First 10 user IDs in matrix:", users[:10])

    matrix = np.zeros((n_users, n_clusters, n_categories, n_time_bins), dtype=np.float32)

    for _, row in df.iterrows():
        u = row['user_id']
        sc = row['spatial_cluster']
        cat = row['category'].title()
        t = int(row['hour_of_week'])
        if u in user_idx and sc in spatial_cluster_idx and cat in cat2idx and 0 <= t < n_time_bins:
            matrix[user_idx[u], spatial_cluster_idx[sc], cat2idx[cat], t] += 1

    # Show which users have nonzero data
    nonzero_users = np.where(matrix.sum(axis=(1,2,3)) > 0)[0]
    print("Indices of users with nonzero data:", nonzero_users)
    if len(nonzero_users) > 0:
        print("First nonzero user index:", nonzero_users[0], "User ID:", users[nonzero_users[0]])

    # Save user ID mapping for downstream use
    np.save(MATRIX_PATH.replace('.npy', '_user_ids.npy'), np.array(users))
    # Save as npy
    np.save(MATRIX_PATH, matrix)
    print(f"Saved user x spatial_cluster x category x time matrix to {MATRIX_PATH} with shape {matrix.shape}")

if __name__ == "__main__":
    # Set DBSCAN parameters here if you want to change them
    main(eps_km=0.5, min_samples=10)
