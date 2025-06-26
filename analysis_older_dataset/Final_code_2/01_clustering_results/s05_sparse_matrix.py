import numpy as np
from scipy import sparse
from s00_config_paths import MATRIX_PATH

# Load the flattened user vectors
MATRIX_NORM_FLAT_PATH = MATRIX_PATH.replace('.npy', '_normalized_flattened.npy')
user_vectors = np.load(MATRIX_NORM_FLAT_PATH)
print(f"Loaded flattened user vectors with shape: {user_vectors.shape}")

# Convert to sparse matrix (CSR format)
user_vectors_sparse = sparse.csr_matrix(user_vectors)
print(f"Converted to sparse matrix. Shape: {user_vectors_sparse.shape}")
num_total = user_vectors_sparse.shape[0] * user_vectors_sparse.shape[1]
num_nonzero = user_vectors_sparse.count_nonzero()
sparsity = 1.0 - (num_nonzero / num_total)
print(f"Nonzero entries: {num_nonzero}, Total entries: {num_total}")
print(f"Sparsity (fraction of zero entries): {sparsity:.6f}")

# Save the sparse matrix in .npz format
sparse_path = MATRIX_NORM_FLAT_PATH.replace('.npy', '_sparse.npz')
sparse.save_npz(sparse_path, user_vectors_sparse)
print(f"Saved sparse user vectors to {sparse_path}")

# Optional: Inspect the first user's nonzero entries
user_idx = 0
row = user_vectors_sparse.getrow(user_idx)
indices = row.indices
values = row.data
print(f"Nonzero entries for user {user_idx} in sparse vector:")
for idx, val in zip(indices, values):
    print(f"  Index {idx}, Value: {val:.4f}")
