import numpy as np
from s00_config_paths import MATRIX_PATH

# Load the normalized matrix (user x spatial_cluster x category x hour)
MATRIX_NORM_PATH = MATRIX_PATH.replace('.npy', '_normalized.npy')
matrix = np.load(MATRIX_NORM_PATH)
print(f"Loaded normalized matrix with shape: {matrix.shape}")

# Flatten each user's 3D matrix (spatial_cluster x category x hour) to 1D vector
n_users = matrix.shape[0]
user_vectors = matrix.reshape(n_users, -1)
print(f"Flattened user vectors shape: {user_vectors.shape}")

# Save the stacked user vectors
flattened_path = MATRIX_NORM_PATH.replace('.npy', '_flattened.npy')
np.save(flattened_path, user_vectors)
print(f"Saved flattened user vectors to {flattened_path}")

# Compute and print sparsity
num_total = user_vectors.size
num_nonzero = np.count_nonzero(user_vectors)
sparsity = 1.0 - (num_nonzero / num_total)
print(f"Sparsity of flattened user vectors: {sparsity:.6f} (fraction of zero entries)")

# Optional: Inspect the first user's vector (nonzero entries)
user_idx = 0
nonzero = np.nonzero(user_vectors[user_idx])[0]
print(f"Nonzero entries for user {user_idx} in flattened vector:")
for idx in nonzero:
    print(f"  Index {idx}, Value: {user_vectors[user_idx, idx]:.4f}")
