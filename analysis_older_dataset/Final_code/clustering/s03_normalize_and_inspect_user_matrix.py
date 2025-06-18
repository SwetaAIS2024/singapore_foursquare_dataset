import numpy as np
from s00_config_paths import MATRIX_PATH

# Load the matrix
matrix = np.load(MATRIX_PATH)
print(f"Loaded matrix with shape: {matrix.shape}")

# Per-user normalization: divide each user's matrix by their total check-ins (across all spatial clusters, categories, and time)
user_sums = matrix.sum(axis=(1,2,3), keepdims=True)
print("Calculated per-user sums for normalization.")
print(f"User sums: {user_sums}")
user_sums[user_sums == 0] = 1  # Avoid division by zero
matrix_norm = matrix / user_sums
print("Per-user normalization complete.")

# Inspection logic: print nonzero entries for the first user (user index 0)
user_idx = 0
nonzero = np.nonzero(matrix_norm[user_idx])
print(f"Nonzero entries for user {user_idx} (normalized):")
for spatial_cluster, cat, t in zip(*nonzero):
    val = matrix_norm[user_idx, spatial_cluster, cat, t]
    if val > 0:
        print(f"  Spatial Cluster {spatial_cluster}, Category {cat}, Hour {t}, Value: {val:.4f}")

# Optionally, save the normalized matrix
np.save(MATRIX_PATH.replace('.npy', '_normalized.npy'), matrix_norm)
print(f"Saved normalized matrix to {MATRIX_PATH.replace('.npy', '_normalized.npy')}")
