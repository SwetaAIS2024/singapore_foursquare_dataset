import numpy as np
from s00_config_paths import MATRIX_PATH

matrix = np.load(MATRIX_PATH)

print(f"Matrix shape: {matrix.shape}")
print(f"Matrix dtype: {matrix.dtype}")

# Show some basic stats
print(f"Total nonzero entries: {np.count_nonzero(matrix)}")
print(f"Max value: {matrix.max()}")
print(f"Min value: {matrix.min()}")

# Show a sample user, cluster, category
n_users, n_clusters, n_categories, n_time = matrix.shape
print("\nSample (user 0, all clusters, all categories, all time bins):")
print(matrix[0])

# Optionally, show nonzero slices for the first user
nonzero_clusters = np.where(matrix[0].sum(axis=(1,2)) > 0)[0]
print(f"\nUser 0 has nonzero data in clusters: {nonzero_clusters}")

for c in nonzero_clusters[:3]:
    print(f"\nUser 0, Cluster {c}, nonzero category/time bins:")
    nonzero_cats, nonzero_times = np.where(matrix[0, c] > 0)
    for cat, t in zip(nonzero_cats, nonzero_times):
        print(f"  Category {cat}, Hour {t}, Value: {matrix[0, c, cat, t]}")
