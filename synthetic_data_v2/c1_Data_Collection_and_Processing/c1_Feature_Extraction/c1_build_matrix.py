import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MaxAbsScaler
import os
from collections import Counter
from scipy import sparse


from sklearn.preprocessing import KBinsDiscretizer, MaxAbsScaler
from c0_Configuration.s00_config_paths import CHECKINS_PATH, CATEGORIES_XLSX, MATRIX_PATH, PLACE_ID_POI_CAT, FINAL_INPUT_DATASET

# Set the output directory for matrix batches and metadata
OUTPUT_DIR = FINAL_INPUT_DATASET

def build_user_spatial_category_time_matrix_batchwise( df, user_ids, spatial_clusters, categories, n_time_slots=168, batch_size=500, output_dir=None, n_quantization_bins=64):
    n_users = len(user_ids)
    n_spatial_clusters = len(spatial_clusters)
    n_categories = len(categories)
    matrix_shape = (n_users, n_spatial_clusters, n_categories, n_time_slots)
    spatial_to_idx = {sc: idx for idx, sc in enumerate(spatial_clusters)}
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}

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
            t = int(row['hour_of_week'])
            if u in batch_user_idx and sc in spatial_to_idx and cat in category_to_idx and 0 <= t < n_time_slots:
                key = (batch_user_idx[u], spatial_to_idx[sc], category_to_idx[cat], t)
                key_counter[key] += 1


        rows, cols, data = [], [], []
        for (u_idx, sc_idx, cat_idx, t), count in key_counter.items():
            flat_idx = (
                u_idx * (n_spatial_clusters * n_categories * n_time_slots) +
                sc_idx * (n_categories * n_time_slots) +
                cat_idx * n_time_slots +
                t
            )
            rows.append(flat_idx)
            cols.append(0)
            data.append(count)

        total_size = batch_size * n_spatial_clusters * n_categories * n_time_slots
        batch_matrix = sparse.coo_matrix((data, (rows, cols)), shape=(total_size, 1), dtype=np.float32).tocsr()
        batch_matrix = batch_matrix.reshape(batch_size, n_spatial_clusters * n_categories * n_time_slots).tocsr()

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



def main_matrix_build(eps_km, min_samples, n_time_bins, n_users, n_spatial_clusters, n_categories, n_quantization_bins, batch_size):
    
    cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon']
    df = pd.read_csv(CHECKINS_PATH, sep='\t', names=cols)
    df = df.dropna(subset=['lat', 'lon'])
    
    # SPATIAL CLUSTERING
    coords = df[['lat', 'lon']].to_numpy()
    # coords_rad = np.radians(coords) #converting the degrees to radians
    kms_per_radian = 6371.0088 #DBSCAN uses the haversine distance
    # which is in radians, so we need to convert the km to radians
    # by dividing by the kms_per_radian which is the radius of the Earth in km
    epsilon = eps_km / kms_per_radian # this is a param for DBSCAN, it is the maximum distance 
    print(f"[INFO] Using eps_km={eps_km} for DBSCAN clustering")
    print(f"[INFO] Using epsilon={epsilon} radians for DBSCAN clustering")
    # between two samples for them to be considered as in the same neighborhood
    # DBSCAN clustering on the coordinates in radians
    #db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(coords_rad)
    # db = DBSCAN(eps=0.0001  , min_samples=min_samples).fit(coords_rad) #- OOM error
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(coords) #- OOM error
    df['spatial_cluster'] = db.labels_
    print("Dataset with the spatial clustering labels : ", df.head())
    # find the unique spatial cluster labels, excluding the -1 labels which represents the noise
    valid_spatial_clusters = sorted([c for c in set(df['spatial_cluster']) if c != -1]) 
    # mapping the original spatial cluster labels to a new set of consecutive labels starting from 0
    spatial_cluster_map = {old: new for new, old in enumerate(valid_spatial_clusters)}
    # removing the rows with the noisy labels or -1 spatial labels 
    df = df[df['spatial_cluster'] != -1]
    # relabeling the spatial clusters using the new mapping creted above - spatial_cluster_map
    df['spatial_cluster'] = df['spatial_cluster'].map(spatial_cluster_map)

    # VALID SPATIAL CLUSTERS - checking the number of unique spatial clusters
    unique_spatial_clusters = df['spatial_cluster'].unique()
    print(f"[INFO] Number of unique spatial clusters: {len(unique_spatial_clusters)}")
    print(f"[INFO] Unique spatial clusters: {unique_spatial_clusters}")    

    # Check number of check-ins per spatial cluster
    checkins_per_cluster = df['spatial_cluster'].value_counts().sort_index()
    print("\n[INFO] Check-ins per spatial cluster:")
    for cluster_id, count in checkins_per_cluster.items():
        print(f"Spatial Cluster {cluster_id}: {count} check-ins")

    # Check number of unique users per spatial cluster
    users_per_cluster = df.groupby('spatial_cluster')['user_id'].nunique()
    print("\n[INFO] Unique users per spatial cluster:")
    for cluster_id, user_count in users_per_cluster.items():
        print(f"Spatial Cluster {cluster_id}: {user_count} users")

    # VALID CATEGORIES 
    relevant_cats_df = pd.read_excel(CATEGORIES_XLSX)
    cat_col = 'POI Category in Singapore'
    yes_col = 'Relevant to use case '
    # Filtering the relevant categories based on the yes_col
    relevant_categories = [cat.strip().lower() for cat, flag in zip(relevant_cats_df[cat_col], relevant_cats_df[yes_col]) if str(flag).strip().lower() == 'yes' and cat and str(cat).strip()]
    relevant_categories = list(dict.fromkeys(relevant_categories))
    ordered_categories = [cat.title() for cat in relevant_categories]
    # Print diagnostics for debugging
    print(f"[INFO] Number of relevant categories: {len(relevant_categories)}")
    print(f"[INFO] Example relevant categories: {relevant_categories[:10]}")

    # DATETIME PARSING 
    #print('[DEBUG] Sample datetime before parsing:', df['datetime'].head().tolist())
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    #print('[DEBUG] Sample datetime after parsing:', df['datetime'].head().tolist())

    # OTHER PREPROCESSING
    if 'category' not in df.columns:
        place_cat = pd.read_csv(PLACE_ID_POI_CAT)
        df = df.merge(place_cat[['place_id', 'category']], on='place_id', how='left')
    df = df.dropna(subset=['datetime', 'category', 'spatial_cluster'])
    df['hour_of_week'] = df['datetime'].dt.dayofweek * 24 + df['datetime'].dt.hour
    df['category'] = df['category'].astype(str).str.strip().str.lower()
    
    # FINAL FEATURES FOR THE MATRIX 
    
    # THIS APPROACH IS SELECTING THE TOP N USERS AND SPATIAL CLUSTERS BASED ON THE NUMBER OF CHECK-INS
    # bUT THE DATASET IS NOT BALANCED 
    # df = df[df['category'].isin(relevant_categories)]
    # user_counts = df['user_id'].value_counts().head(n_users) # select the top n_users with most no of checkins
    # users = user_counts.index.tolist()
    # #spatial_clusters = list(range(min(n_spatial_clusters, len(valid_spatial_clusters))))
    # top_clusters = df['spatial_cluster'].value_counts().head(n_spatial_clusters).index.tolist()
    # spatial_clusters = top_clusters
    # print("\n[INFO] Check-ins per selected spatial cluster:")
    # for cluster_id in spatial_clusters:
    #     count = df[df['spatial_cluster'] == cluster_id].shape[0]
    #     print(f"Spatial Cluster {cluster_id}: {count} check-ins")

    # # TO MAKE THE DATASET BALANCED
    # # Spatial Cluster filtering and stratified user sampling 
    
# --- Cluster filtering and stratified user sampling ---

    min_cluster_size = 60    # Example: clusters must have at least 60 users
    max_cluster_size = 500   # Example: clusters must have at most 300 users
    max_users_per_cluster = 100  # Max users to sample per cluster
    #n_users = 2000  # Or your config value

    # 1. Filter clusters by size
    cluster_sizes = df['spatial_cluster'].value_counts()
    filtered_clusters = cluster_sizes[(cluster_sizes >= min_cluster_size) & (cluster_sizes <= max_cluster_size)]
    top_clusters = filtered_clusters.head(n_spatial_clusters).index.tolist()
    
    # 2. Stratified user sampling
    users = []
    final_spatial_clusters = []
    for cluster in top_clusters:
        cluster_user_counts = df[df['spatial_cluster'] == cluster]['user_id'].value_counts()
        if len(cluster_user_counts) >= min_cluster_size:
            cluster_users = cluster_user_counts.head(max_users_per_cluster).index.tolist()
            users.extend(cluster_users)
            final_spatial_clusters.append(cluster)
    users = list(dict.fromkeys(users))[:n_users]  # Deduplicate and cap total users

    if len(users) < n_users:
        print(f"[WARNING] Only {len(users)} users found after filtering, less than requested {n_users}.")
        all_unique_users = df['user_id'].unique()
        remaining_users = [u for u in all_unique_users if u not in users]
        users.extend(remaining_users[:n_users - len(users)])  # Fill up to n_users if possible
    
    users = users[:n_users]  # Ensure we only take the first n_users
    users = [int(u) for u in users[:n_users]] # FIX for the json serialization issue with the user ids
    spatial_clusters = final_spatial_clusters     # Only clusters meeting criteria

    print(f"[INFO] Selected {len(spatial_clusters)} clusters and {len(users)} users after balancing.")

    print(f"[INFO] Selected {len(users)} users: {users[:10]}...")  # Show first 10 users
    print(f"[INFO] Selected {len(spatial_clusters)} spatial clusters: {spatial_clusters}")

    ordered_categories = ordered_categories[:n_categories]

    # BAUIDING THE BATCHWISE MATRIX USING THE USERS, SPATIAL CLUSTERS, CATEGORIES AND THE TIME SLOTS 
    batch_files, metadata = build_user_spatial_category_time_matrix_batchwise(
        df=df,
        user_ids=users,
        spatial_clusters=spatial_clusters,
        categories=ordered_categories,
        n_time_slots=n_time_bins,
        batch_size=batch_size, # if no batching is needed, then set the batch_size to n_users
        output_dir=OUTPUT_DIR,
        n_quantization_bins=n_quantization_bins
    )

    print("[INFO] All batches processed.")
    print(f"[INFO] Matrix shape: {metadata['shape']}")
    print(f"[INFO] Number of batches: {len(batch_files)}")
    print(f"[INFO] Example batch file: {batch_files[0] if batch_files else None}")
    print(f"[INFO] Metadata file: {os.path.join(os.path.dirname(MATRIX_PATH), 'matrix_metadata.json')}")
    
    # # Save the metadata and index files - dummycode for saving the full matrix,
    #  here also need to add the logic for converting the batches of matrices to a sngle matrix
    # print("[INFO] Saving metadata and index files...")
    # dummy_matrix = sparse.csr_matrix((0, 0))  # Empty matrix, won't be saved
    # save_matrix_and_metadata(dummy_matrix, metadata, OUTPUT_DIR)
    return 0


