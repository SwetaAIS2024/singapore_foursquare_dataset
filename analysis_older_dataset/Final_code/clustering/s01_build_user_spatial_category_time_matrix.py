import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from s00_config_paths import CHECKINS_PATH, CATEGORIES_XLSX, MATRIX_PATH, PLACE_ID_POI_CAT
import argparse
import csv

# --- Parameters ---
def main(eps_km=0.5, min_samples=10, n_time_bins=168,
         input_path=None, placeid_to_cat=None, output_matrix=None,
         output_user_list=None, output_cluster_list=None, output_cat_list=None, output_timebin_list=None,
         n_users=2000, n_spatial_clusters=50, n_categories=180):
    try:
        cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon']
        if input_path:
            # Try to auto-detect delimiter and header
            with open(input_path, 'r') as f:
                sample = f.read(2048)
                f.seek(0)
                dialect = csv.Sniffer().sniff(sample)
                has_header = csv.Sniffer().has_header(sample)
            if has_header:
                df = pd.read_csv(input_path, sep=dialect.delimiter)
            else:
                df = pd.read_csv(input_path, sep=dialect.delimiter, names=cols)
        else:
            df = pd.read_csv(CHECKINS_PATH, sep='\t', names=cols)
    except Exception as e:
        print(f"[ERROR] Failed to read input file: {e}")
        import traceback; traceback.print_exc()
        return 1
    try:
        # --- Step 1: Spatial clustering of POIs ---
        if not set(['lat', 'lon']).issubset(df.columns):
            raise ValueError(f"Input file missing required columns: {set(['lat', 'lon']) - set(df.columns)}. Columns present: {df.columns.tolist()}")
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
        # Use user-supplied or default values for shape
        # Get top n_users users by check-in count AFTER all filtering
        user_counts = df['user_id'].value_counts().head(n_users)
        users = user_counts.index.tolist()
        user_idx = {u: i for i, u in enumerate(users)}
        # Use spatial cluster labels 0 to min(n_spatial_clusters-1, actual number of clusters-1)
        spatial_clusters = list(range(min(n_spatial_clusters, len(valid_spatial_clusters))))
        spatial_cluster_idx = {c: i for i, c in enumerate(spatial_clusters)}
        # Use the first n_categories from ordered_categories
        ordered_categories = ordered_categories[:n_categories]
        cat2idx = {cat: i for i, cat in enumerate(ordered_categories)}

        print("First user in matrix:", users[0] if users else None)
        print("Check-ins for this user after filtering:", df[df['user_id'] == users[0]].shape[0] if users else 0)
        print("First 10 user IDs in filtered data:", df['user_id'].unique()[:10])
        print("First 10 user IDs in matrix:", users[:10])

        matrix = np.zeros((n_users, n_spatial_clusters, n_categories, n_time_bins), dtype=np.float32)

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
        if output_matrix:
            np.save(output_matrix.replace('.npy', '_user_ids.npy'), np.array(users))
            np.save(output_matrix, matrix)
            print(f"Saved user x spatial_cluster x category x time matrix to {output_matrix} with shape {matrix.shape}")
        else:
            np.save(MATRIX_PATH.replace('.npy', '_user_ids.npy'), np.array(users))
            np.save(MATRIX_PATH, matrix)
            print(f"Saved user x spatial_cluster x category x time matrix to {MATRIX_PATH} with shape {matrix.shape}")
        # Save lists to provided paths or default
        if output_user_list:
            with open(output_user_list, 'w') as f:
                for u in users:
                    f.write(f"{u}\n")
        else:
            with open('user_list.txt', 'w') as f:
                for u in users:
                    f.write(f"{u}\n")
        if output_cluster_list:
            with open(output_cluster_list, 'w') as f:
                for c in spatial_clusters:
                    f.write(f"{c}\n")
        else:
            with open('spatial_cluster_list.txt', 'w') as f:
                for c in spatial_clusters:
                    f.write(f"{c}\n")
        if output_cat_list:
            with open(output_cat_list, 'w') as f:
                for cat in ordered_categories:
                    f.write(f"{cat}\n")
        else:
            with open('poi_cat_list.txt', 'w') as f:
                for cat in ordered_categories:
                    f.write(f"{cat}\n")
        if output_timebin_list:
            with open(output_timebin_list, 'w') as f:
                for t in range(n_time_bins):
                    f.write(f"{t}\n")
        else:
            with open('timebin_list.txt', 'w') as f:
                for t in range(n_time_bins):
                    f.write(f"{t}\n")
        print("Saved user_list, spatial_cluster_list, poi_cat_list, timebin_list")
    except Exception as e:
        print(f"[ERROR] Matrix creation failed: {e}")
        import traceback; traceback.print_exc()
        return 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--placeid_to_cat', type=str, default=None)
    parser.add_argument('--output_matrix', type=str, default=None)
    parser.add_argument('--output_user_list', type=str, default=None)
    parser.add_argument('--output_cluster_list', type=str, default=None)
    parser.add_argument('--output_cat_list', type=str, default=None)
    parser.add_argument('--output_timebin_list', type=str, default=None)
    parser.add_argument('--eps_km', type=float, default=0.5)
    parser.add_argument('--min_samples', type=int, default=10)
    parser.add_argument('--n_users', type=int, default=2000)
    parser.add_argument('--n_spatial_clusters', type=int, default=50)
    parser.add_argument('--n_categories', type=int, default=180)
    args = parser.parse_args()
    main(eps_km=args.eps_km, min_samples=args.min_samples,
         input_path=args.input, placeid_to_cat=args.placeid_to_cat,
         output_matrix=args.output_matrix,
         output_user_list=args.output_user_list,
         output_cluster_list=args.output_cluster_list,
         output_cat_list=args.output_cat_list,
         output_timebin_list=args.output_timebin_list,
         n_users=args.n_users, n_spatial_clusters=args.n_spatial_clusters, n_categories=args.n_categories)
