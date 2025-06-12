"""
Performs basic clustering of users based on their activity trends using the encoded activity trend CSV.
- Loads user_activity_trend_encoded.csv
- Uses only the POI columns (all peak_hour_POI_*, peak_day_POI_*, peak_month_POI_*) for clustering
- Standardizes features
- Runs KMeans clustering (default: 4 clusters, configurable)
- Outputs a CSV with user_id and assigned cluster
- Prints basic cluster statistics
- Plots user clusters in 2D PCA space
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os

# Config
INPUT_CSV = 'analysis_set2/Feature_engineering/user_activity_trends_summary/user_activity_trend_encoded.csv'
OUTPUT_CSV = 'analysis_set2/Feature_engineering/basic_clustering_activity_trends/user_clusters.csv'
N_CLUSTERS = 7  # Change as needed

# Load data
print(f"Loading {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV)

# Identify POI columns (all columns starting with 'peak_hour_POI_', 'peak_day_POI_', 'peak_month_POI_')
poi_cols = [col for col in df.columns if col.startswith('peak_hour_POI_') or col.startswith('peak_day_POI_') or col.startswith('peak_month_POI_')]

# Extract features for clustering
X = df[poi_cols].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- KMeans ---
print(f"Clustering users into {N_CLUSTERS} clusters using KMeans ...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['cluster'] = clusters

# Output user_id and cluster assignment
df[['user_id', 'cluster']].to_csv(OUTPUT_CSV, index=False)
print(f"KMeans cluster assignments written to {OUTPUT_CSV}")

# Print basic cluster statistics
print("\nKMeans Cluster sizes:")
print(df['cluster'].value_counts().sort_index())

print("\nKMeans Cluster centroids (in standardized feature space):")
print(pd.DataFrame(kmeans.cluster_centers_, columns=poi_cols))

# Plot KMeans clusters using PCA (2D projection)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for cluster_id in range(N_CLUSTERS):
    mask = df['cluster'] == cluster_id
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {cluster_id}', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('User Clusters by Activity Trends (PCA projection) - KMeans')
plt.legend()
plt.tight_layout()
plot_path = os.path.join(os.path.dirname(OUTPUT_CSV), 'user_clusters_pca_plot_kmeans.png')
plt.savefig(plot_path)
plt.show()
print(f"KMeans cluster plot saved to {plot_path}")

# --- Agglomerative Clustering ---
agglo_dir = os.path.join(os.path.dirname(OUTPUT_CSV), 'agglomerative')
os.makedirs(agglo_dir, exist_ok=True)
print(f"Clustering users into {N_CLUSTERS} clusters using Agglomerative Clustering ...")
agglo = AgglomerativeClustering(n_clusters=N_CLUSTERS)
agglo_labels = agglo.fit_predict(X_scaled)
df['agglo_cluster'] = agglo_labels
df[['user_id', 'agglo_cluster']].to_csv(os.path.join(agglo_dir, 'user_clusters_agglomerative.csv'), index=False)
# Plot
plt.figure(figsize=(8, 6))
for cluster_id in np.unique(agglo_labels):
    mask = agglo_labels == cluster_id
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {cluster_id}', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Agglomerative Clustering (PCA projection)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(agglo_dir, 'user_clusters_pca_plot_agglomerative.png'))
plt.close()

# --- Gaussian Mixture Model ---
gmm_dir = os.path.join(os.path.dirname(OUTPUT_CSV), 'gmm')
os.makedirs(gmm_dir, exist_ok=True)
print(f"Clustering users into {N_CLUSTERS} clusters using Gaussian Mixture Model ...")
gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
df['gmm_cluster'] = gmm_labels
df[['user_id', 'gmm_cluster']].to_csv(os.path.join(gmm_dir, 'user_clusters_gmm.csv'), index=False)
# Plot
plt.figure(figsize=(8, 6))
for cluster_id in np.unique(gmm_labels):
    mask = gmm_labels == cluster_id
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {cluster_id}', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Gaussian Mixture Model Clustering (PCA projection)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(gmm_dir, 'user_clusters_pca_plot_gmm.png'))
plt.close()

# --- DBSCAN ---
dbscan_dir = os.path.join(os.path.dirname(OUTPUT_CSV), 'dbscan')
os.makedirs(dbscan_dir, exist_ok=True)
print(f"Clustering users using DBSCAN ...")
dbscan = DBSCAN(eps=2, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
df['dbscan_cluster'] = dbscan_labels
df[['user_id', 'dbscan_cluster']].to_csv(os.path.join(dbscan_dir, 'user_clusters_dbscan.csv'), index=False)
# Plot
plt.figure(figsize=(8, 6))
for cluster_id in np.unique(dbscan_labels):
    mask = dbscan_labels == cluster_id
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {cluster_id}', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('DBSCAN Clustering (PCA projection)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(dbscan_dir, 'user_clusters_pca_plot_dbscan.png'))
plt.close()

# --- Silhouette Scores ---
try:
    print(f"KMeans silhouette: {silhouette_score(X_scaled, clusters):.3f}")
    print(f"Agglomerative silhouette: {silhouette_score(X_scaled, agglo_labels):.3f}")
    print(f"GMM silhouette: {silhouette_score(X_scaled, gmm_labels):.3f}")
    # DBSCAN may have -1 labels (noise), so only if more than 1 cluster
    if len(set(dbscan_labels)) > 1 and -1 not in set(dbscan_labels):
        print(f"DBSCAN silhouette: {silhouette_score(X_scaled, dbscan_labels):.3f}")
    else:
        print("DBSCAN silhouette: Not applicable (noise or single cluster)")
except Exception as e:
    print(f"Silhouette score error: {e}")

# --- Dominant and Auxiliary POI Types ---
# Determine dominant POI type and up to 5 auxiliary POI types for each user (across all POI columns)
dominant_poi_types = []
aux_poi_types = []
poi_type_names = [col.replace('peak_hour_POI_', '').replace('peak_day_POI_', '').replace('peak_month_POI_', '') for col in poi_cols]
unique_poi_types = sorted(set(poi_type_names))
for idx, row in df[poi_cols].iterrows():
    # Sum all columns for each POI type
    poi_type_totals = {cat: 0 for cat in unique_poi_types}
    for col, val in row.items():
        cat = col.replace('peak_hour_POI_', '').replace('peak_day_POI_', '').replace('peak_month_POI_', '')
        poi_type_totals[cat] += val
    # Sort POI types by total descending
    sorted_types = sorted(poi_type_totals.items(), key=lambda x: -x[1])
    dominant_cat = sorted_types[0][0]
    aux_cats = [cat for cat, cnt in sorted_types[1:6] if cnt > 0]
    dominant_poi_types.append(dominant_cat)
    aux_poi_types.append(';'.join(aux_cats))
df['dominant_poi_type'] = dominant_poi_types
df['aux_poi_types'] = aux_poi_types

# Plot clusters with dominant POI type as marker/annotation
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# Assign a color to each cluster and a marker to each POI type (limit to top 10 POI types for clarity)
top_poi_types = pd.Series(dominant_poi_types).value_counts().index[:10].tolist()
poi_type_to_marker = {cat: m for cat, m in zip(top_poi_types, ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>'])}
def marker_for_poi(cat):
    return poi_type_to_marker.get(cat, '.')

plt.figure(figsize=(10, 8))
colors = cm.get_cmap('tab10', N_CLUSTERS)
for cluster_id in range(N_CLUSTERS):
    mask = (df['cluster'] == cluster_id)
    for poi_cat in top_poi_types:
        submask = mask & (df['dominant_poi_type'] == poi_cat)
        plt.scatter(X_pca[submask, 0], X_pca[submask, 1],
                    label=f'Cluster {cluster_id}, {poi_cat}',
                    alpha=0.7, marker=poi_type_to_marker[poi_cat],
                    color=colors(cluster_id))
# Add legend for POI types
legend_elements = [Line2D([0], [0], marker=poi_type_to_marker[cat], color='w', label=cat,
                          markerfacecolor='gray', markersize=10) for cat in top_poi_types]
plt.legend(handles=legend_elements, title='Dominant POI Type (Top 10)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('User Clusters by Activity Trends (PCA)\nMarker = Dominant POI Type (Top 10)')
plt.tight_layout()
plot_path2 = os.path.join(os.path.dirname(OUTPUT_CSV), 'user_clusters_pca_dominant_poi_plot.png')
plt.savefig(plot_path2)
plt.show()
print(f"Cluster+POI plot saved to {plot_path2}")

# Save user_id, cluster, dominant_poi_type, and aux_poi_types
out_csv = os.path.join(os.path.dirname(OUTPUT_CSV), 'user_cluster_dominant_poi.csv')
df[['user_id', 'cluster', 'dominant_poi_type', 'aux_poi_types']].to_csv(out_csv, index=False)
