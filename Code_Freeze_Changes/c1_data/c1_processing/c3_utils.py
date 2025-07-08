import os
import json
from scipy import sparse
import numpy as np

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
