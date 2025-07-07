import numpy as np
import os
import json
import glob
from scipy import sparse

if __name__ == "__main__":
    # Use config_paths.py for paths
    from s00_config_paths import MATRIX_PATH
    meta_path = os.path.join(os.path.dirname(MATRIX_PATH), "matrix_metadata.json")
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    batch_files = metadata.get("batch_files", [])
    print(f"Found {len(batch_files)} batch files.")

    for batch_file in batch_files:
        print(f"\nNormalizing {batch_file} ...")
        mat = sparse.load_npz(batch_file)
        # Normalize each user row
        row_sums = np.array(mat.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        mat_norm = mat.multiply(1 / row_sums[:, None])
        # Save normalized batch
        out_file = batch_file.replace('.npz', '_normalized.npz')
        sparse.save_npz(out_file, mat_norm)
        print(f"Saved normalized batch to {out_file}")
        # Optional: print stats
        print(f"  Max value: {mat_norm.max()}, Min value: {mat_norm.min()}")
        print(f"  Nonzero elements: {mat_norm.nnz}")
