import numpy as np
import os
import json
from scipy import sparse

if __name__ == "__main__":
    from s00_config_paths import MATRIX_PATH
    meta_path = os.path.join(os.path.dirname(MATRIX_PATH), "matrix_metadata.json")
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    # Get normalized batch files
    batch_files = [f.replace('.npz', '_normalized.npz') for f in metadata.get("batch_files", [])]
    print(f"Found {len(batch_files)} normalized batch files.")

    for batch_file in batch_files:
        print(f"\nFlattening {batch_file} ...")
        mat = sparse.load_npz(batch_file)
        dense_mat = mat.toarray()
        # Debug: print feature variance
        feature_var = np.var(dense_mat, axis=0)
        print("Batch feature variance (mean, min, max):", np.mean(feature_var), np.min(feature_var), np.max(feature_var))
        # Save as dense .npy (if fits in memory)
        out_file = batch_file.replace('_normalized.npz', '_flattened.npy')
        np.save(out_file, dense_mat)
        print(f"Saved flattened batch to {out_file}")
        print(f"  Shape: {mat.shape}, Nonzero: {mat.nnz}")
