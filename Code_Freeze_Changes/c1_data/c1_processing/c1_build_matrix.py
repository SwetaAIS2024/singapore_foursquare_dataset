import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.cluster import DBSCAN
import os
from collections import Counter
from scipy import sparse
from .c2_quantization import quantize_sparse_matrix
from .c3_utils import save_matrix_and_metadata
from c0_config.s00_config_paths import CHECKINS_PATH, CATEGORIES_XLSX, MATRIX_PATH, PLACE_ID_POI_CAT, FINAL_INPUT_DATASET

# Set the output directory for matrix batches and metadata
OUTPUT_DIR = FINAL_INPUT_DATASET

def build_user_spatial_category_time_matrix_batchwise(df, user_ids, spatial_clusters, categories, n_time_slots=168, batch_size=100, output_dir=None, n_quantization_bins=8):
    n_users = len(user_ids)
    n_spatial_clusters = len(spatial_clusters)
    n_categories = len(categories)
    # matrix_shape
    matrix_shape = (n_users, n_spatial_clusters, n_categories, n_time_slots)
    # mapping user_ids, spatial_clusters, categories to indices
    #user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)} - not really needed as we will use the user_ids directly
    spatial_to_idx = {sc: idx for idx, sc in enumerate(spatial_clusters)}
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    # Always use the correct output directory
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    batch_files = []
    for batch_start in range(0, n_users, batch_size):
        batch_end = min(batch_start + batch_size, n_users)
        batch_users = user_ids[batch_start:batch_end]
        batch_user_idx = {u: i for i, u in enumerate(batch_users)}
        batch_df = df[df['user_id'].isin(batch_users)]
        if batch_df.empty:
            continue
        
        # counting the evets 
        # counts the occurrences of each (user_id, spatial_cluster, category, hour_of_week) tuple
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
            flat_idx = (u_idx * (n_spatial_clusters * n_categories * n_time_slots) +
                        sc_idx * (n_categories * n_time_slots) +
                        cat_idx * n_time_slots +
                        t)
            rows.append(flat_idx)
            cols.append(0) # this is always 0 since the matrix is a column vector at this point
            data.append(count)
        
        total_size = batch_size * n_spatial_clusters * n_categories * n_time_slots
        # creating a sparse matrix in COO format 
        batch_matrix = sparse.coo_matrix((data, (rows, cols)), shape=(total_size, 1), dtype=np.float32).tocsr()
        batch_matrix = batch_matrix.reshape(batch_size, n_spatial_clusters * n_categories * n_time_slots).tocsr()
        
        # quantizing the sparse matrix into discrete bins 
        quantized_matrix, quantization_metadata = quantize_sparse_matrix(batch_matrix, n_bins=n_quantization_bins)
        batch_file = os.path.join(output_dir, f"user_spatial_category_time_matrix_batch_{batch_start}_{batch_end-1}.npz")
        sparse.save_npz(batch_file, quantized_matrix)
        batch_files.append(batch_file)
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
    with open(os.path.join(output_dir, 'matrix_metadata.json'), 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    return batch_files, metadata

def main(eps_km=0.5, min_samples=10, n_time_bins=168, n_users=2000, n_spatial_clusters=50, n_categories=180, n_quantization_bins=8):
    cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon']
    df = pd.read_csv(CHECKINS_PATH, sep='\t', names=cols)
    df = df.dropna(subset=['lat', 'lon'])
    
    # SPATIAL CLUSTERING
    coords = df[['lat', 'lon']].to_numpy()
    coords_rad = np.radians(coords) #converting the degrees to radians
    kms_per_radian = 6371.0088 #DBSCAN uses the haversine distance
    # which is in radians, so we need to convert the km to radians
    # by dividing by the kms_per_radian which is the radius of the Earth in km
    epsilon = eps_km / kms_per_radian # this is a param for DBSCAN, it is the maximum distance 
    # between two samples for them to be considered as in the same neighborhood
    # DBSCAN clustering on the coordinates in radians
    db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(coords_rad)
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

    # VALID CATEGORIES 
    relevant_cats_df = pd.read_excel(CATEGORIES_XLSX)
    cat_col = 'POI Category in Singapore'
    yes_col = 'Relevant to use case '
    # Filtering the relevant categories based on the yes_col
    relevant_categories = [cat for cat, flag in zip(relevant_cats_df[cat_col], relevant_cats_df[yes_col]) if str(flag).strip().lower() == 'yes']
    relevant_categories = [cat.strip().lower() for cat in relevant_categories if cat and str(cat).strip()]
    relevant_categories = list(dict.fromkeys(relevant_categories))
    ordered_categories = [cat.title() for cat in relevant_categories]

    # DATETIME PARSING 
    print('[DEBUG] Sample datetime before parsing:', df['datetime'].head().tolist())
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    print('[DEBUG] Sample datetime after parsing:', df['datetime'].head().tolist())

    # OTHER PREPROCESSING
    if 'category' not in df.columns:
        place_cat = pd.read_csv(PLACE_ID_POI_CAT)
        df = df.merge(place_cat[['place_id', 'category']], on='place_id', how='left')
    df = df.dropna(subset=['datetime', 'category', 'spatial_cluster'])
    df['hour_of_week'] = df['datetime'].dt.dayofweek * 24 + df['datetime'].dt.hour
    df['category'] = df['category'].astype(str).str.strip().str.lower()
    
    # FINAL FEATURES FOR THE MATRIX 
    df = df[df['category'].isin(relevant_categories)]
    user_counts = df['user_id'].value_counts().head(n_users) # select the top n_users with most no of checkins
    users = user_counts.index.tolist()
    spatial_clusters = list(range(min(n_spatial_clusters, len(valid_spatial_clusters))))
    ordered_categories = ordered_categories[:n_categories]

    # BAUIDING THE BATCHWISE MATRIX USING THE USERS, SPATIAL CLUSTERS, CATEGORIES AND THE TIME SLOTS 
    batch_files, metadata = build_user_spatial_category_time_matrix_batchwise(
        df=df,
        user_ids=users,
        spatial_clusters=spatial_clusters,
        categories=ordered_categories,
        n_time_slots=n_time_bins,
        batch_size=100, # if no batching is needed, then set the batch_size to n_users
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


