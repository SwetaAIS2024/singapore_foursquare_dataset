import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Paths
pca_path = 'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_180/Dimension_reduction_graph_construction/embeddings/user_pca_embeddings.csv'
cluster_path = 'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_180/Clustering_Profiling/user_louvain_clusters.csv'

# Load data
pca_df = pd.read_csv(pca_path, index_col=0)
clu_df = pd.read_csv(cluster_path)

# Merge on user_id (index in PCA, user_id col in clusters)
pca_df = pca_df.reset_index().rename(columns={'index': 'user_id'})
merged = pd.merge(pca_df, clu_df, on='user_id')

# Normalize PCA components 0, 1, 2 (same as in scatter script)
scaler = MinMaxScaler()
merged[['0', '1', '2']] = scaler.fit_transform(merged[['0', '1', '2']])

# Compute centroids
grouped = merged.groupby('cluster')[['0', '1', '2']].mean().reset_index()

# For each centroid, find nearest user (Euclidean distance)
results = []
for _, row in grouped.iterrows():
    cluster = row['cluster']
    centroid = row[['0', '1', '2']].values.astype(float)
    cluster_users = merged[merged['cluster'] == cluster].copy()
    user_vecs = cluster_users[['0', '1', '2']].values.astype(float)
    dists = np.linalg.norm(user_vecs - centroid, axis=1)
    min_idx = np.argmin(dists)
    nearest_user_id = cluster_users.iloc[min_idx]['user_id']
    results.append({
        'cluster': cluster,
        'centroid_pca0': centroid[0],
        'centroid_pca1': centroid[1],
        'centroid_pca2': centroid[2],
        'nearest_user_id': nearest_user_id,
        'distance': dists[min_idx]
    })

out_df = pd.DataFrame(results)
out_path = 'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_180/Clustering_Profiling/cluster_centroid_nearest_user.csv'
out_df.to_csv(out_path, index=False)
print(f'Saved cluster centroid to nearest user mapping to {out_path}')
