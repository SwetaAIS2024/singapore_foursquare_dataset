import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from s00_config_paths import MATRIX_PATH
import gc

# Load cluster labels and user vectors
labels = np.load(MATRIX_PATH.replace('.npy', '_user_cluster_labels.npy'))
user_vectors = np.load(MATRIX_PATH.replace('.npy', '_normalized_flattened.npy'))

# Load the original user list (from the matrix-building step)
# This assumes you saved the user list as a .npy or .txt file; if not, you should do so in s01
try:
    users = np.load(MATRIX_PATH.replace('.npy', '_user_ids.npy'), allow_pickle=True)
except FileNotFoundError:
    print("User ID mapping file not found. Please save user IDs during matrix construction for accurate mapping.")
    users = np.arange(user_vectors.shape[0])

# 1. 2D Scatter plot of clusters (using PCA for visualization)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
user_vec_2d = pca.fit_transform(user_vectors)
plt.figure(figsize=(10,7))
scatter = plt.scatter(user_vec_2d[:,0], user_vec_2d[:,1], c=labels, cmap='tab10', alpha=0.7)
plt.title('User Clusters (PCA 2D)')
plt.xlabel('PC1')
plt.ylabel('PC2')
cbar = plt.colorbar(scatter, ticks=np.unique(labels))
cbar.set_label('Cluster')
plt.tight_layout()
plt.savefig(MATRIX_PATH.replace('.npy', '_user_cluster_scatter.png'))
plt.close()
print('Saved cluster scatter plot.')

# 2. Cluster-wise POI details (top POI categories per cluster)
# Load POI category names
try:
    import openpyxl
    cats_df = pd.read_excel('analysis_older_dataset/Final_code/Relevant_POI_category.xlsx')
    cat_col = 'POI Category in Singapore'
    poi_categories = [c for c in cats_df[cat_col] if pd.notnull(c)]
except Exception:
    poi_categories = [f'Cat_{i}' for i in range(user_vectors.shape[1]//(50*168))]

n_clusters = len(np.unique(labels))
n_spatial = 50
n_cat = 180
n_time = 168

# 3. Heatmap: For each cluster, use the user nearest to the cluster centroid for all analysis
for cl in range(n_clusters):
    idxs = np.where(labels == cl)[0]
    if len(idxs) == 0:
        continue
    # Select 10 random users from the cluster (or all if <10)
    if len(idxs) > 10:
        rep_idxs = np.random.choice(idxs, 10, replace=False)
    else:
        rep_idxs = idxs
    # Aggregate their vectors for the heatmap
    rep_vectors = user_vectors[rep_idxs].astype(np.float32)
    agg_vector = rep_vectors.mean(axis=0)
    agg_3d = agg_vector.reshape((n_spatial, n_cat, n_time))
    agg_cat_time = agg_3d.sum(axis=0)  # sum over spatial clusters -> (n_cat, n_time)
    plt.figure(figsize=(16,8))
    sns.heatmap(agg_cat_time, cmap='viridis', cbar=True)
    plt.title(f'Cluster {cl} 10 Random Users: POI Category vs Hour Heatmap (summed over spatial clusters)')
    plt.xlabel('Hour of Week')
    plt.ylabel('POI Category')
    plt.yticks(ticks=np.arange(n_cat)+0.5, labels=poi_categories[:n_cat], fontsize=6)
    plt.tight_layout()
    plt.savefig(MATRIX_PATH.replace('.npy', f'_cluster{cl}_10randuser_cat_time_heatmap.png'))
    plt.close()
    print(f'Saved 10-random-user heatmap for cluster {cl}')
    print(f'Cluster {cl} random user indices: {rep_idxs}, user_ids: {[users[i] for i in rep_idxs]}')
    del rep_vectors, agg_vector, agg_3d, agg_cat_time
    gc.collect()

# 4. (Optional) Save user-cluster mapping as CSV
user_cluster_df = pd.DataFrame({'user_id': users, 'cluster': labels})
user_cluster_df.to_csv(MATRIX_PATH.replace('.npy', '_user_cluster_mapping.csv'), index=False)
print('Saved user-cluster mapping CSV.')

# 5. (Optional) Save top POI categories for the representative user of each cluster
user_matrix = np.load(MATRIX_PATH.replace('.npy', '_normalized.npy'), mmap_mode='r')
for cl in range(n_clusters):
    idxs = np.where(labels == cl)[0]
    if len(idxs) == 0:
        continue
    # Select 10 random users from the cluster (or all if <10)
    if len(idxs) > 10:
        rep_idxs = np.random.choice(idxs, 10, replace=False)
    else:
        rep_idxs = idxs
    # Aggregate their matrices for top POI categories
    rep_matrices = user_matrix[rep_idxs]  # shape: (n_users, n_spatial, n_cat, n_time)
    agg = rep_matrices.sum(axis=0) / len(rep_idxs)  # mean over users
    top_cats = np.argsort(agg.sum(axis=1))[::-1][:10]
    top_cats = np.array(top_cats).flatten().tolist()  # Ensure it's a flat list of ints
    print(f'Cluster {cl} 10 random users top POI categories: {[poi_categories[int(i)] for i in top_cats]}')
    del rep_matrices, agg, top_cats
    gc.collect()
del user_matrix
gc.collect()

# 6. Save cluster, user_ids, and top POI categories for the 10 random users of each cluster
import csv
output_rows = []
user_matrix = np.load(MATRIX_PATH.replace('.npy', '_normalized.npy'), mmap_mode='r')
user_id_map_path = MATRIX_PATH.replace('.npy', '_user_ids.npy')
user_id_map = np.load(user_id_map_path, allow_pickle=True)
for cl in range(n_clusters):
    idxs = np.where(labels == cl)[0]
    if len(idxs) == 0:
        continue
    if len(idxs) > 10:
        rep_idxs = np.random.choice(idxs, 10, replace=False)
    else:
        rep_idxs = idxs
    rep_user_ids = [user_id_map[i] for i in rep_idxs]
    rep_matrices = user_matrix[rep_idxs]
    agg = rep_matrices.sum(axis=0) / len(rep_idxs)
    top_cats = np.argsort(agg.sum(axis=1))[::-1][:10]
    top_cats = np.array(top_cats).flatten().tolist()  # Ensure it's a flat list of ints
    top_cat_names = [poi_categories[int(i)] for i in top_cats] 
    output_rows.append({
        'cluster': cl,
        'user_ids': ', '.join(map(str, rep_user_ids)),
        'top_poi_categories': ', '.join(top_cat_names)
    })
    del rep_matrices, agg, top_cats, top_cat_names
    gc.collect()
del user_matrix
gc.collect()

output_csv_path = MATRIX_PATH.replace('.npy', '_cluster_top_poi_categories_with_10users.csv')
with open(output_csv_path, 'w', newline='') as csvfile:
    fieldnames = ['cluster', 'user_ids', 'top_poi_categories']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in output_rows:
        writer.writerow(row)
print(f'Saved cluster top POI categories with 10 user details to {output_csv_path}')
