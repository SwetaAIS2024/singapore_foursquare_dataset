import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import KBinsDiscretizer
from scipy import sparse
from s00_config_paths import CHECKINS_PATH, CATEGORIES_XLSX, MATRIX_PATH, PLACE_ID_POI_CAT
import json
import os

def quantize_sparse_matrix(matrix, n_bins=256, strategy='quantile'):
    """
    Quantize a matrix to uint8 format using KBinsDiscretizer or ordinal encoding if unique values are few.
    Returns the quantized matrix and quantization parameters for reconstruction.
    """
    # Ensure input is CSR matrix
    if not sparse.isspmatrix_csr(matrix):
        print("Converting input matrix to CSR format...")
        matrix = matrix.tocsr()
    
    # Get non-zero elements
    data = matrix.data
    
    if len(data) == 0:
        return matrix, {'bin_edges': None, 'min_val': 0, 'max_val': 0}
    
    # Reshape for sklearn
    data = data.reshape(-1, 1)
    
    # Find the unique values in the data
    unique_values = np.unique(data)
    n_unique = len(unique_values)
    
    # If only one unique value, just cast to uint8 and skip binning
    if n_unique == 1:
        quantized_data = data.astype(np.uint8)
        quantized_matrix = sparse.csr_matrix((quantized_data.ravel(), matrix.indices, matrix.indptr),
                                             shape=matrix.shape, dtype=np.uint8)
        metadata = {
            'bin_edges': [unique_values[0]],
            'min_val': float(data.min()),
            'max_val': float(data.max())
        }
        print(f"[QUANTIZE] Only one unique value ({unique_values[0]}), skipping binning.")
        return quantized_matrix, metadata
    
    # If all values are small consecutive integers and n_unique <= n_bins, use ordinal encoding
    if n_unique <= n_bins and np.all(np.diff(unique_values) == 1):
        value_to_bin = {v: i for i, v in enumerate(unique_values)}
        quantized_data = np.vectorize(value_to_bin.get)(data.ravel()).astype(np.uint8)
        quantized_matrix = sparse.csr_matrix((quantized_data, matrix.indices, matrix.indptr),
                                             shape=matrix.shape, dtype=np.uint8)
        metadata = {
            'bin_edges': unique_values.tolist(),
            'min_val': float(data.min()),
            'max_val': float(data.max()),
            'encoding': 'ordinal'
        }
        print(f"[QUANTIZE] Used ordinal encoding for {n_unique} unique integer values.")
        return quantized_matrix, metadata
    
    # Otherwise, use KBinsDiscretizer
    actual_bins = max(2, min(n_bins, n_unique))
    print(f"[QUANTIZE] Quantizing with {actual_bins} bins (requested: {n_bins}, unique: {n_unique})")
    discretizer = KBinsDiscretizer(n_bins=actual_bins, encode='ordinal', strategy=strategy)
    quantized_data = discretizer.fit_transform(data)
    quantized_data = quantized_data.astype(np.uint8)
    quantized_matrix = sparse.csr_matrix((quantized_data.ravel(), matrix.indices, matrix.indptr),
                                       shape=matrix.shape, dtype=np.uint8)
    metadata = {
        'bin_edges': discretizer.bin_edges_[0].tolist(),
        'min_val': float(data.min()),
        'max_val': float(data.max()),
        'encoding': 'binned'
    }
    return quantized_matrix, metadata

# --- Parameters ---
def main(eps_km=0.5, min_samples=10, n_time_bins=168,
         placeid_to_cat=None, output_matrix=None,
         output_user_list=None, output_cluster_list=None, output_cat_list=None, output_timebin_list=None,
         n_users=2000, n_spatial_clusters=50, n_categories=180,
         n_quantization_bins=8):
    try:
        cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon']
        # Remove input_path support: always use CHECKINS_PATH
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
        print('[DEBUG] After dropna lat/lon:', len(df))
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
        print('[DEBUG] After DBSCAN noise removal:', len(df))
        df['spatial_cluster'] = df['spatial_cluster'].map(spatial_cluster_map)

        # --- Step 2: Load relevant POI categories ---
        relevant_cats_df = pd.read_excel(CATEGORIES_XLSX)
        cat_col = 'POI Category in Singapore'
        yes_col = 'Relevant to use case '
        relevant_categories = [cat for cat, flag in zip(relevant_cats_df[cat_col], relevant_cats_df[yes_col]) if str(flag).strip().lower() == 'yes']
        relevant_categories = [cat.strip().lower() for cat in relevant_categories if cat and str(cat).strip()]
        relevant_categories = list(dict.fromkeys(relevant_categories))
        ordered_categories = [cat.title() for cat in relevant_categories]
        # cat2idx = {cat: i for i, cat in enumerate(ordered_categories)}

        # --- Step 3: Prepare time features ---
        print('[DEBUG] Sample datetime before parsing:', df['datetime'].head().tolist())
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        print('[DEBUG] Sample datetime after parsing:', df['datetime'].head().tolist())
        if 'category' not in df.columns:
            try:
                place_cat = pd.read_csv(PLACE_ID_POI_CAT)
                df = df.merge(place_cat[['place_id', 'category']], on='place_id', how='left')
            except Exception as e:
                print('Warning: Could not merge category info:', e)
                df['category'] = ''
        print('[DEBUG] Sample place_id/category after merge:', df[['place_id', 'category']].head().to_dict())
        print('[DEBUG] Nulls in datetime:', df['datetime'].isnull().sum(), 'Nulls in category:', df['category'].isnull().sum(), 'Nulls in spatial_cluster:', df['spatial_cluster'].isnull().sum())
        df = df.dropna(subset=['datetime', 'category', 'spatial_cluster'])
        print('[DEBUG] After dropna datetime/category/cluster:', len(df))
        df['hour_of_week'] = df['datetime'].dt.dayofweek * 24 + df['datetime'].dt.hour
        df['category'] = df['category'].astype(str).str.strip().str.lower()
        df = df[df['category'].isin(relevant_categories)]
        print('[DEBUG] After category filtering:', len(df))

        # --- Step 4: Build 4D matrix: user x spatial_cluster x category x time ---
        # Use user-supplied or default values for shape
        # Get top n_users users by check-in count AFTER all filtering
        user_counts = df['user_id'].value_counts().head(n_users)
        users = user_counts.index.tolist()
        user_idx = {u: i for i, u in enumerate(users)}
        spatial_clusters = list(range(min(n_spatial_clusters, len(valid_spatial_clusters))))
        spatial_cluster_idx = {c: i for i, c in enumerate(spatial_clusters)}
        ordered_categories = ordered_categories[:n_categories]
        # cat2idx = {cat: i for i, cat in enumerate(ordered_categories)}

        print("First user in matrix:", users[0] if users else None)
        print("Check-ins for this user after filtering:", df[df['user_id'] == users[0]].shape[0] if users else 0)
        print("First 10 user IDs in filtered data:", df['user_id'].unique()[:10])
        print("First 10 user IDs in matrix:", users[:10])

        print("[INFO] Building matrix in batches...")
        batch_files, metadata = build_user_spatial_category_time_matrix_batchwise(
            df=df,
            user_ids=users,
            spatial_clusters=spatial_clusters,
            categories=ordered_categories,
            n_time_slots=n_time_bins,
            batch_size=100,
            output_dir=os.path.dirname(MATRIX_PATH),
            n_quantization_bins=n_quantization_bins
        )
        print("[INFO] All batches processed.")
        print(f"[INFO] Matrix shape: {metadata['shape']}")
        print(f"[INFO] Number of batches: {len(batch_files)}")
        print(f"[INFO] Example batch file: {batch_files[0] if batch_files else None}")
        print(f"[INFO] Metadata file: {os.path.join(os.path.dirname(MATRIX_PATH), 'matrix_metadata.json')}")
        return 0
    except Exception as e:
        print(f"[ERROR] Matrix creation failed: {e}")
        import traceback; traceback.print_exc()
        return 2

def build_user_spatial_category_time_matrix(df, user_ids, spatial_clusters, categories, n_time_slots=168):
    """
    Build a 4D sparse matrix of user activity patterns (user x spatial_cluster x category x time)
    with quantization to reduce storage requirements.
    """
    n_users = len(user_ids)
    n_spatial_clusters = len(spatial_clusters)
    n_categories = len(categories)
    
    # Create dictionaries for faster lookup
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    spatial_to_idx = {sc: idx for idx, sc in enumerate(spatial_clusters)}
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    # Initialize sparse matrix
    matrix_shape = (n_users, n_spatial_clusters, n_categories, n_time_slots)
    total_size = np.prod(matrix_shape)
    print(f"Matrix shape: {matrix_shape}, Total size: {total_size} elements")
    
    # Use COO format for efficient construction
    rows, cols = [], []
    data = []
    
    # Process data in chunks to handle large datasets
    chunk_size = 1000000
    for chunk_start in range(0, len(df), chunk_size):
        chunk = df.iloc[chunk_start:chunk_start + chunk_size]
        
        for _, row in chunk.iterrows():
            user_idx = user_to_idx.get(row['user_id'])
            spatial_idx = spatial_to_idx.get(row['spatial_cluster'])
            category_idx = category_to_idx.get(row['category'].title())  # Match the title case used in main()
            time_slot = int(row['hour_of_week'])  # Changed from 'hour' to 'hour_of_week'
            
            if all(x is not None for x in [user_idx, spatial_idx, category_idx]) and 0 <= time_slot < n_time_slots:
                # Calculate flat index for sparse matrix
                flat_idx = (user_idx * (n_spatial_clusters * n_categories * n_time_slots) +
                          spatial_idx * (n_categories * n_time_slots) +
                          category_idx * n_time_slots +
                          time_slot)
                rows.append(flat_idx)
                cols.append(0)  # Single column as we'll reshape later
                data.append(1)
    
    # Create sparse matrix in COO format first (efficient for construction)
    print("Creating initial sparse matrix...")
    coo_matrix = sparse.coo_matrix((data, (rows, cols)), 
                                shape=(total_size, 1),
                                dtype=np.float32)
    
    # Store the shape information for metadata
    matrix_shape = (n_users, n_spatial_clusters, n_categories, n_time_slots)
    
    # Convert to CSR format
    print("Converting to CSR format...")
    matrix = coo_matrix.tocsr()
    
    # Calculate original size before quantization (using CSR format)
    print("Calculating matrix sizes...")
    original_size = matrix.data.nbytes + matrix.indptr.nbytes + matrix.indices.nbytes
    
    # Reshape to 2D (this returns COO, so convert again to CSR after)
    matrix = matrix.reshape(n_users, n_spatial_clusters * n_categories * n_time_slots).tocsr()
    
    # Quantize the matrix
    print("Quantizing matrix...")
    quantized_matrix, quantization_metadata = quantize_sparse_matrix(matrix)
    
    # Ensure quantized matrix is in CSR format for size calculation
    if not sparse.isspmatrix_csr(quantized_matrix):
        quantized_matrix = quantized_matrix.tocsr()
    
    # Calculate compressed size
    compressed_size = quantized_matrix.data.nbytes + quantized_matrix.indptr.nbytes + quantized_matrix.indices.nbytes
    compression_ratio = original_size / compressed_size
    
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Original size: {original_size / 1e6:.2f} MB")
    print(f"Compressed size: {compressed_size / 1e6:.2f} MB")
    
    # Save metadata
    metadata = {
        'shape': matrix_shape,
        'quantization': quantization_metadata,
        'compression_ratio': compression_ratio,
        'user_ids': user_ids,  # Already a list
        'spatial_clusters': spatial_clusters,  # Already a list
        'categories': categories,  # Already a list
        'n_time_slots': n_time_slots
    }
    
    return quantized_matrix, metadata

def save_matrix_and_metadata(matrix, metadata, base_path):
    """Save the quantized matrix and its metadata."""
    matrix_path = os.path.join(base_path, 'user_spatial_category_time_matrix.npz')
    metadata_path = os.path.join(base_path, 'matrix_metadata.json')
    
    # Save sparse matrix
    sparse.save_npz(matrix_path, matrix)
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved matrix to: {matrix_path}")
    print(f"Saved metadata to: {metadata_path}")
    
    # Write index files as plain text, one value per line
    base_name = os.path.join(base_path, 'matrix')
    user_list_path = f'{base_name}_user_list.txt'
    cluster_list_path = f'{base_name}_spatial_cluster_list.txt'
    category_list_path = f'{base_name}_poi_cat_list.txt'
    timebin_list_path = f'{base_name}_timebin_list.txt'
    
    with open(user_list_path, 'w') as f:
        for u in metadata['user_ids']:
            f.write(f"{u}\n")
    print(f"[INFO] Wrote user list to {user_list_path} ({len(metadata['user_ids'])} users)")

    with open(cluster_list_path, 'w') as f:
        for c in metadata['spatial_clusters']:
            f.write(f"{c}\n")
    print(f"[INFO] Wrote cluster list to {cluster_list_path} ({len(metadata['spatial_clusters'])} clusters)")

    with open(category_list_path, 'w') as f:
        for cat in metadata['categories']:
            f.write(f"{cat}\n")
    print(f"[INFO] Wrote category list to {category_list_path} ({len(metadata['categories'])} categories)")

    with open(timebin_list_path, 'w') as f:
        for t in range(metadata['n_time_slots']):
            f.write(f"{t}\n")
    print(f"[INFO] Wrote timebin list to {timebin_list_path} ({metadata['n_time_slots']} timebins)")

def load_matrix_and_metadata(base_path):
    """Load the quantized matrix and its metadata."""
    matrix_path = os.path.join(base_path, 'user_spatial_category_time_matrix.npz')
    metadata_path = os.path.join(base_path, 'matrix_metadata.json')
    
    matrix = sparse.load_npz(matrix_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return matrix, metadata

def build_user_spatial_category_time_matrix_batchwise(df, user_ids, spatial_clusters, categories, n_time_slots=168, batch_size=100, output_dir=None, n_quantization_bins=8):
    """
    Build a 4D sparse matrix in user batches, quantize each batch, and save to disk.
    """
    n_users = len(user_ids)
    n_spatial_clusters = len(spatial_clusters)
    n_categories = len(categories)
    matrix_shape = (n_users, n_spatial_clusters, n_categories, n_time_slots)
    print(f"[BATCH] Matrix shape: {matrix_shape}, batch size: {batch_size}")

    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    spatial_to_idx = {sc: idx for idx, sc in enumerate(spatial_clusters)}
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}

    if output_dir is None:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    batch_files = []
    for batch_start in range(0, n_users, batch_size):
        batch_end = min(batch_start + batch_size, n_users)
        batch_users = user_ids[batch_start:batch_end]
        batch_user_idx = {u: i for i, u in enumerate(batch_users)}
        print(f"[BATCH] Processing users {batch_start} to {batch_end-1}...")

        # Filter df for this batch
        batch_df = df[df['user_id'].isin(batch_users)]
        if batch_df.empty:
            print(f"[BATCH] No data for users {batch_start} to {batch_end-1}, skipping.")
            continue

        # Build sparse matrix for this batch using counts
        from collections import Counter
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
            cols.append(0)
            data.append(count)
        total_size = batch_size * n_spatial_clusters * n_categories * n_time_slots
        batch_matrix = sparse.coo_matrix((data, (rows, cols)), shape=(total_size, 1), dtype=np.float32).tocsr()
        batch_matrix = batch_matrix.reshape(batch_size, n_spatial_clusters * n_categories * n_time_slots).tocsr()

        # Debug print: unique values before quantization
        print(f"[DEBUG] Batch {batch_start}-{batch_end}: unique values before quantization:", np.unique(batch_matrix.data))
        # Quantize to uint8 immediately
        quantized_matrix, quantization_metadata = quantize_sparse_matrix(batch_matrix, n_bins=n_quantization_bins)
        # Debug print: unique values after quantization
        print(f"[DEBUG] Batch {batch_start}-{batch_end}: unique values after quantization:", np.unique(quantized_matrix.data))
        print(f"[DEBUG] Batch {batch_start}-{batch_end}: Nonzero elements after quantization: {quantized_matrix.nnz}")

        # Save batch
        batch_file = os.path.join(output_dir, f"user_spatial_category_time_matrix_batch_{batch_start}_{batch_end-1}.npz")
        sparse.save_npz(batch_file, quantized_matrix)
        batch_files.append(batch_file)
        print(f"[BATCH] Saved quantized batch to {batch_file}")

    # Save metadata for all batches
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
        json.dump(metadata, f, indent=2)
    print(f"[BATCH] Saved metadata to {os.path.join(output_dir, 'matrix_metadata.json')}")
    
    # Write index files as plain text, one value per line
    base_name = os.path.join(output_dir, 'matrix')
    user_list_path = f'{base_name}_user_list.txt'
    cluster_list_path = f'{base_name}_spatial_cluster_list.txt'
    category_list_path = f'{base_name}_poi_cat_list.txt'
    timebin_list_path = f'{base_name}_timebin_list.txt'
    
    with open(user_list_path, 'w') as f:
        for u in metadata['user_ids']:
            f.write(f"{u}\n")
    print(f"[INFO] Wrote user list to {user_list_path} ({len(metadata['user_ids'])} users)")

    with open(cluster_list_path, 'w') as f:
        for c in metadata['spatial_clusters']:
            f.write(f"{c}\n")
    print(f"[INFO] Wrote cluster list to {cluster_list_path} ({len(metadata['spatial_clusters'])} clusters)")

    with open(category_list_path, 'w') as f:
        for cat in metadata['categories']:
            f.write(f"{cat}\n")
    print(f"[INFO] Wrote category list to {category_list_path} ({len(metadata['categories'])} categories)")

    with open(timebin_list_path, 'w') as f:
        for t in range(metadata['n_time_slots']):
            f.write(f"{t}\n")
    print(f"[INFO] Wrote timebin list to {timebin_list_path} ({metadata['n_time_slots']} timebins)")
    
    return batch_files, metadata

# --- Streamlit UI ---
if __name__ == "__main__":
    print("[INFO] Starting matrix building process...")
    # Adjust parameters for more spatial clusters:
    # - Smaller eps_km = smaller clusters = more clusters
    # - Smaller min_samples = easier to form clusters
    result = main(eps_km=0.3, min_samples=5)  # Try these parameters for more clusters
    if result in [1, 2]:
        print("[ERROR] Matrix building failed")
    else:
        print("[INFO] Matrix building completed successfully")
