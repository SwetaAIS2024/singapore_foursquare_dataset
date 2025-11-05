import os
import json
from scipy import sparse
import numpy as np
import sys
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MaxAbsScaler
from collections import Counter
from sklearn.preprocessing import KBinsDiscretizer, MaxAbsScaler
from c0_Configuration.config_paths import CHECKINS_PATH, CATEGORIES_XLSX, MATRIX_PATH, PLACE_ID_POI_CAT, FINAL_INPUT_DATASET, CONFIG_PATH 
# Set the output directory for matrix batches and metadata
OUTPUT_DIR = FINAL_INPUT_DATASET

def save_matrix_and_metadata(matrix, metadata, base_path):
    """Save the quantized matrix and its metadata."""
    matrix_path = os.path.join(base_path, 'user_spatial_category_time_matrix.npz')
    metadata_path = os.path.join(base_path, 'matrix_metadata.json')
    sparse.save_npz(matrix_path, matrix)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    # Write index files as plain text, one value per line
    base_name = os.path.join(base_path, 'matrix')
    user_list_path = f'{base_name}_user_list.txt'
    cluster_list_path = f'{base_name}_spatial_cluster_list.txt'
    category_list_path = f'{base_name}_poi_cat_list.txt'
    timebin_list_path = f'{base_name}_timebin_list.txt'
    with open(user_list_path, 'w') as f:
        for u in metadata['user_ids']:
            f.write(f"{u}\n")
    with open(cluster_list_path, 'w') as f:
        for c in metadata['spatial_clusters']:
            f.write(f"{c}\n")
    with open(category_list_path, 'w') as f:
        for cat in metadata['categories']:
            f.write(f"{cat}\n")
    with open(timebin_list_path, 'w') as f:
        for t in range(metadata['n_time_slots']):
            f.write(f"{t}\n")

def load_matrix_and_metadata(base_path):
    matrix_path = os.path.join(base_path, 'user_spatial_category_time_matrix.npz')
    metadata_path = os.path.join(base_path, 'matrix_metadata.json')
    matrix = sparse.load_npz(matrix_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return matrix, metadata


def load_config():
    """Load configuration from c0_Configuration/config.json"""
    config_path = CONFIG_PATH
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"[ERROR] Config file not found: {config_path}", file=sys.stderr)
        return {}
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in config file: {e}", file=sys.stderr)
        return {}


def build_user_spatial_category_time_matrix_batchwise( df, user_ids, spatial_clusters, categories, poi_ids,n_time_slots=168, batch_size=500, output_dir=None, n_quantization_bins=64):
    n_users = len(user_ids)
    n_spatial_clusters = len(spatial_clusters)
    n_categories = len(categories)
    n_pois = len(poi_ids)
    
    matrix_shape = (n_users, n_spatial_clusters, n_categories, n_pois, n_time_slots)

    #mapping from original ids to indices
    spatial_to_idx = {sc: idx for idx, sc in enumerate(spatial_clusters)}
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    poi_to_idx = {poi: idx for idx, poi in enumerate(poi_ids)}

    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    batch_files = []
    all_nonzero_data = []  # Collect nonzero data for global quantization

    # --- First pass: Build batches and collect nonzero data ---
    for batch_start in range(0, n_users, batch_size):
        batch_end = min(batch_start + batch_size, n_users)
        batch_users = user_ids[batch_start:batch_end]

        # fix for the WRONG user label file 
        batch_file_prefix = os.path.join(output_dir, f"user_spatial_category_time_matrix_batch_{batch_start}_{batch_end-1}")
        user_id_file = batch_file_prefix + "_user_ids.npy"
        np.save(user_id_file, np.array(batch_users))

        batch_user_idx = {u: i for i, u in enumerate(batch_users)}
        batch_df = df[df['user_id'].isin(batch_users)]
        if batch_df.empty:
            continue

        key_counter = Counter()
        for _, row in batch_df.iterrows():
            u = row['user_id']
            sc = row['spatial_cluster']
            cat = row['category'].title()
            poi = row['place_id']
            t = int(row['hour_of_week'])
            if (u in batch_user_idx and 
                sc in spatial_to_idx and 
                cat in category_to_idx and 
                poi in poi_to_idx and
                0 <= t < n_time_slots):
                key = (batch_user_idx[u], spatial_to_idx[sc], category_to_idx[cat], poi_to_idx[poi], t)
                key_counter[key] += 1


        rows, cols, data = [], [], []
        for (u_idx, sc_idx, cat_idx, poi_idx, t), count in key_counter.items():
            flat_idx = (
                u_idx * (n_spatial_clusters * n_categories * n_pois * n_time_slots) +
                sc_idx * (n_categories * n_pois * n_time_slots) +
                cat_idx * (n_pois * n_time_slots) +
                poi_idx * n_time_slots +
                t
            )
            rows.append(flat_idx)
            cols.append(0)
            data.append(count)

        total_size = batch_size * n_spatial_clusters * n_categories * n_pois * n_time_slots
        batch_matrix = sparse.coo_matrix((data, (rows, cols)), shape=(total_size, 1), dtype=np.float32).tocsr()
        batch_matrix = batch_matrix.reshape(batch_size, n_spatial_clusters * n_categories * n_pois * n_time_slots).tocsr()

        # Collect nonzero data for global quantization
        all_nonzero_data.extend(batch_matrix.data.tolist())
        batch_file = os.path.join(output_dir, f"user_spatial_category_time_matrix_batch_{batch_start}_{batch_end-1}.npz")
        sparse.save_npz(batch_file, batch_matrix)
        batch_files.append(batch_file)

    # --- Second pass: Fit KBinsDiscretizer globally ---
    # all_nonzero_data = np.array(all_nonzero_data).reshape(-1, 1)
    all_nonzero_data = np.log1p(np.array(all_nonzero_data)).reshape(-1, 1)
    # discretizer = KBinsDiscretizer(n_bins=n_quantization_bins, encode='ordinal', strategy='quantile')
    discretizer = KBinsDiscretizer(n_bins=n_quantization_bins, encode='ordinal', strategy='uniform')
    discretizer.fit(all_nonzero_data)

    # --- Third pass: Quantize, normalize, and save each batch ---
    for batch_file in batch_files:
        batch_matrix = sparse.load_npz(batch_file)
        original_sum = batch_matrix.sum()
        if batch_matrix.nnz > 0:
            #batch_data_log = np.log1p(batch_matrix.data.reshape(-1, 1)) # do not use the log transformation, else the individual 
            #cluster distribution will always be normal distribution if log norm is used
            batch_data_reshaped = batch_matrix.data.reshape(-1, 1) 
            quantized_data = discretizer.transform(batch_data_reshaped).astype(np.uint8).flatten()
            batch_matrix.data = quantized_data
        quantized_sum = batch_matrix.sum()
        scaler_quant_matrix = MaxAbsScaler()
        quantized_matrix_norm = scaler_quant_matrix.fit_transform(batch_matrix)
        norm_sum = quantized_matrix_norm.sum()
        print(f"[VALIDATION] {batch_file}: Original sum={original_sum}, Quantized sum={quantized_sum}, Normalized sum={norm_sum}")
        # Validation check for data loss
        if quantized_sum < 0.5 * original_sum:
            print(f"[WARNING] Significant data loss detected in {batch_file}: Quantized sum is less than 50% of original sum.")
        sparse.save_npz(batch_file, quantized_matrix_norm)

    metadata = {
        'shape': matrix_shape,
        'user_ids': user_ids,
        'spatial_clusters': spatial_clusters,
        'categories': categories,
        'poi_ids': poi_ids,
        'n_time_slots': n_time_slots,
        'batch_size': batch_size,
        'batch_files': batch_files,
        'quantization_bins': n_quantization_bins
    }

    # debug print for metadata contents
    for k, v in metadata.items():
        print(f"{k}: {type(v)}")
        if isinstance(v, list) and len(v) > 0:
            print(f"  First element type: {type(v[0])}")
    

    with open(os.path.join(output_dir, 'matrix_metadata.json'), 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    return batch_files, metadata


def main_matrix_build(eps_km, min_samples, n_time_bins, n_users, n_spatial_clusters, n_categories, n_pois, n_quantization_bins, batch_size):
    
    cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon']
    df = pd.read_csv(CHECKINS_PATH, sep='\t', names=cols)
    df = df.dropna(subset=['lat', 'lon'])
    
    # SPATIAL CLUSTERING
    coords = df[['lat', 'lon']].to_numpy()
    kms_per_radian = 6371.0088
    epsilon = eps_km / kms_per_radian
    print(f"[INFO] Using eps_km={eps_km} for DBSCAN clustering")
    print(f"[INFO] Using epsilon={epsilon} radians for DBSCAN clustering")
    
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(coords)
    df['spatial_cluster'] = db.labels_
    print("Dataset with the spatial clustering labels : ", df.head())
    
    valid_spatial_clusters = sorted([c for c in set(df['spatial_cluster']) if c != -1]) 
    spatial_cluster_map = {old: new for new, old in enumerate(valid_spatial_clusters)}
    df = df[df['spatial_cluster'] != -1]
    df['spatial_cluster'] = df['spatial_cluster'].map(spatial_cluster_map)

    unique_spatial_clusters = df['spatial_cluster'].unique()
    print(f"[INFO] Number of unique spatial clusters: {len(unique_spatial_clusters)}")
    print(f"[INFO] Unique spatial clusters: {unique_spatial_clusters}")    

    checkins_per_cluster = df['spatial_cluster'].value_counts().sort_index()
    print("\n[INFO] Check-ins per spatial cluster:")
    for cluster_id, count in checkins_per_cluster.items():
        print(f"Spatial Cluster {cluster_id}: {count} check-ins")

    users_per_cluster = df.groupby('spatial_cluster')['user_id'].nunique()
    print("\n[INFO] Unique users per spatial cluster:")
    for cluster_id, user_count in users_per_cluster.items():
        print(f"Spatial Cluster {cluster_id}: {user_count} users")

    # VALID CATEGORIES 
    relevant_cats_df = pd.read_excel(CATEGORIES_XLSX)
    cat_col = 'POI Category in Singapore'
    yes_col = 'Relevant to use case '
    relevant_categories = [cat.strip().lower() for cat, flag in zip(relevant_cats_df[cat_col], relevant_cats_df[yes_col]) if str(flag).strip().lower() == 'yes' and cat and str(cat).strip()]
    relevant_categories = list(dict.fromkeys(relevant_categories))
    ordered_categories = [cat.title() for cat in relevant_categories]
    print(f"[INFO] Number of relevant categories: {len(relevant_categories)}")
    print(f"[INFO] Example relevant categories: {relevant_categories[:10]}")


    # CRITICAL FIX: limit the categories after initial filtering
    if len(relevant_categories) > n_categories:
        updated_relevant_categories = relevant_categories[:n_categories]  # Take only first n_categories
        print(f"[INFO] Reduced categories from {len(relevant_categories)} to {len(updated_relevant_categories)}")
    else:
        updated_relevant_categories = relevant_categories
        print(f"[INFO] Using all {len(updated_relevant_categories)} categories as they are within the limit")


    ordered_categories = [cat.title() for cat in updated_relevant_categories]
    print(f"[INFO] Number of categories to use: {len(ordered_categories)}")
    print(f"[INFO] Categories: {ordered_categories}")

    # DATETIME PARSING 
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # OTHER PREPROCESSING
    if 'category' not in df.columns:
        place_cat = pd.read_csv(PLACE_ID_POI_CAT)
        df = df.merge(place_cat[['place_id', 'category']], on='place_id', how='left')
    df = df.dropna(subset=['datetime', 'category', 'spatial_cluster'])
    df['hour_of_week'] = df['datetime'].dt.dayofweek * 24 + df['datetime'].dt.hour
    df['category'] = df['category'].astype(str).str.strip().str.lower()
    
    # REDUCE DIMENSIONS TO AVOID MEMORY OVERFLOW
    # Limit categories first
    # ordered_categories = ordered_categories[:n_categories]  #redundant due to earlier limit
    print(f"[INFO] Limited to {len(ordered_categories)} categories")
    
    # Filter by relevant categories BEFORE user/cluster selection
    df_filtered = df[df['category'].isin([cat.lower() for cat in ordered_categories])]
    print(f"[INFO] Dataset size after category filtering: {len(df_filtered)}")
    
    # LIMIT POIs TO MOST POPULAR ONES PER CATEGORY
        # LIMIT POIs TO MOST POPULAR ONES PER CATEGORY
    max_pois_per_category = 10  # REDUCE from 50 to 10
    max_total_pois = n_pois        # ADD hard limit on total POIs
    selected_poi_ids = []
    
    print(f"[INFO] Selecting max {max_pois_per_category} POIs per category")
    
    for category in ordered_categories:
        cat_lower = category.lower()
        cat_df = df_filtered[df_filtered['category'] == cat_lower]
        if not cat_df.empty:
            # Select top POIs by check-in frequency for this category
            top_pois = cat_df['place_id'].value_counts().head(max_pois_per_category).index.tolist()
            selected_poi_ids.extend(top_pois)
            print(f"  {category}: {len(top_pois)} POIs selected")
    
    selected_poi_ids = list(dict.fromkeys(selected_poi_ids))  # Remove duplicates
    
    # Apply hard limit on total POIs
    if len(selected_poi_ids) > max_total_pois:
        print(f"[INFO] Too many POIs ({len(selected_poi_ids)}), limiting to {max_total_pois}")
        selected_poi_ids = selected_poi_ids[:max_total_pois]
    
    print(f"[INFO] Limited to {len(selected_poi_ids)} POIs (max {max_pois_per_category} per category)")
    
    # Filter dataframe to only include selected POIs
    df_filtered = df_filtered[df_filtered['place_id'].isin(selected_poi_ids)]
    print(f"[INFO] Dataset size after POI filtering: {len(df_filtered)}")

    # CLUSTER FILTERING AND USER SAMPLING (on filtered data)
    min_cluster_size = 20    # Reduced for better balance
    max_cluster_size = 100   # Reduced to limit matrix size
    max_users_per_cluster = 20  # Reduced to limit matrix size

    # Filter clusters by size (on filtered data)
    cluster_sizes = df_filtered['spatial_cluster'].value_counts()
    filtered_clusters = cluster_sizes[(cluster_sizes >= min_cluster_size) & (cluster_sizes <= max_cluster_size)]
    top_clusters = filtered_clusters.head(n_spatial_clusters).index.tolist()
    
    # Stratified user sampling
    users = []
    final_spatial_clusters = []
    for cluster in top_clusters:
        cluster_user_counts = df_filtered[df_filtered['spatial_cluster'] == cluster]['user_id'].value_counts()
        if len(cluster_user_counts) >= min_cluster_size:
            cluster_users = cluster_user_counts.head(max_users_per_cluster).index.tolist()
            users.extend(cluster_users)
            final_spatial_clusters.append(cluster)
    
    users = list(dict.fromkeys(users))[:n_users]  # Deduplicate and cap total users

    # if len(users) < n_users:
    #     print(f"[WARNING] Only {len(users)} users found after filtering, less than requested {n_users}.")
    #     all_unique_users = df_filtered['user_id'].unique()
    #     remaining_users = [u for u in all_unique_users if u not in users]
    #     users.extend(remaining_users[:n_users - len(users)])
    
    # users = users[:n_users]
    # users = [int(u) for u in users[:n_users]]
    # spatial_clusters = final_spatial_clusters

    if len(users) < n_users:
        print(f"[INFO] Found {len(users)} quality users from cluster sampling (requested {n_users})")
        print(f"[INFO] Using {len(users)} users to keep matrix manageable")
    else:
        users = users[:n_users]  # Cap at n_users if we have more than needed
        print(f"[INFO] Capped to {n_users} users from {len(users)} available")
    
    users = [int(u) for u in users]
    spatial_clusters = final_spatial_clusters

    # FINAL FILTERING - Keep only data for selected users, clusters, and POIs
    df_final = df_filtered[
        (df_filtered['user_id'].isin(users)) &
        (df_filtered['spatial_cluster'].isin(spatial_clusters)) &
        (df_filtered['place_id'].isin(selected_poi_ids))
    ]
    
    print(f"\n[INFO] FINAL MATRIX DIMENSIONS:")
    print(f"  Users: {len(users)}")
    print(f"  Spatial Clusters: {len(spatial_clusters)}")
    print(f"  Categories: {len(ordered_categories)}")
    print(f"  POIs: {len(selected_poi_ids)}")
    print(f"  Time slots: {n_time_bins}")
    print(f"  Final dataset size: {len(df_final)}")
    
    # MEMORY CHECK
    total_elements = len(users) * len(spatial_clusters) * len(ordered_categories) * len(selected_poi_ids) * n_time_bins
    memory_gb = (total_elements * 4) / (1024**3)  # 4 bytes per float32
    print(f"  Total matrix elements: {total_elements:,}")
    print(f"  Estimated memory: {memory_gb:.2f} GB")
    
    if memory_gb > 500:  # REDUCE from 16 to 4 GB
        print(f"[ERROR] Matrix too large ({memory_gb:.2f} GB). Reduce dimensions further.")
        print(f"[SUGGESTION] Current: {len(users)}×{len(spatial_clusters)}×{len(ordered_categories)}×{len(selected_poi_ids)}×{n_time_bins}")
        print(f"[SUGGESTION] Try: users={len(users)//2}, categories={len(ordered_categories)//2}, POIs={len(selected_poi_ids)//2}")
        return 1

    # BUILDING THE BATCHWISE MATRIX
    batch_files, metadata = build_user_spatial_category_time_matrix_batchwise(
        df=df_final,  # Use final filtered dataframe
        user_ids=users,
        spatial_clusters=spatial_clusters,
        categories=ordered_categories,
        poi_ids=selected_poi_ids,
        n_time_slots=n_time_bins,
        batch_size=batch_size,
        output_dir=OUTPUT_DIR,
        n_quantization_bins=n_quantization_bins
    )

    print("[INFO] All batches processed.")
    print(f"[INFO] Matrix shape: {metadata['shape']}")
    print(f"[INFO] Number of batches: {len(batch_files)}")
    print(f"[INFO] Example batch file: {batch_files[0] if batch_files else None}")
    
    return 0