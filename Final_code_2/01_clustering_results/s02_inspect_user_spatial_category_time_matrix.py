import pandas as pd
import numpy as np
from s00_config_paths import MATRIX_PATH
from scipy import sparse
import os
import json 
import glob

if __name__ == "__main__":
    # Load metadata for batch file info
    meta_path = os.path.join(os.path.dirname(MATRIX_PATH), "matrix_metadata.json")
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    batch_files = metadata.get("batch_files", [])
    shape = metadata["shape"]

    all_stats = []
    batch_files = sorted(glob.glob('user_spatial_category_time_matrix_batch_*.npz'))
    print(f"Found {len(batch_files)} batch files.")

    for batch_file in batch_files:
        print(f"\nInspecting {batch_file}:")
        mat = sparse.load_npz(batch_file)
        print(f"  Shape: {mat.shape}")
        print(f"  Nonzero elements: {mat.nnz}")
        unique_vals = np.unique(mat.data)
        print(f"  Unique values: {unique_vals}")
        if len(unique_vals) == 1 and unique_vals[0] == 0:
            print("  [WARNING] All quantized values are zero!")
        else:
            print(f"  Value counts: {np.unique(mat.data, return_counts=True)}")
        stats = {
            "batch_file": os.path.basename(batch_file),
            "n_users_in_batch": mat.shape[0],
            "total_nonzero": int(mat.nnz),
            "max_value": mat.max(),
            "min_value": float(mat.min())
        }
        all_stats.append(stats)
        print(stats)

    # Optionally, save summary
    df = pd.DataFrame(all_stats)
    output = os.path.join(os.path.dirname(MATRIX_PATH), "matrix_batches_summary.csv")
    df.to_csv(output, index=False)
    print(f"Batch summary written to {output}")
