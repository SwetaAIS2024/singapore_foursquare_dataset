"""
Performs clustering on combined spatial-temporal features for users.
- Loads combined_spatial_temporal_features.csv
- Selects only numeric features for clustering (excluding user_id and categorical/text columns)
- Standardizes features
- Runs KMeans, Agglomerative, GMM, DBSCAN, and HDBSCAN clustering (default: 3 clusters, configurable)
- Saves cluster assignments and PCA plots in clustering_overall/
- Prints silhouette scores for each method
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import os

# Config
INPUT_CSV = 'analysis_set2/Feature_engineering/spatial_temporal_activity_trends/combined_spatial_temporal_features.csv'
OUTPUT_DIR = 'analysis_set2/Feature_engineering/spatial_temporal_activity_trends/clustering_overall'
N_CLUSTERS = 5  # Change as needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print(f"Loading {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV)

# Select only numeric features for clustering (exclude user_id and categorical/text columns)
exclude_cols = ['user_id', 'peak_hour_range', 'peak_day_of_week', 'peak_month', 'dominant_poi_type', 'aux_poi_types', 'main_grid_cell']
feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
X = df[feature_cols].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
df['kmeans_cluster'] = kmeans_labels

# Agglomerative clustering
agglo = AgglomerativeClustering(n_clusters=N_CLUSTERS)
agglo_labels = agglo.fit_predict(X_scaled)
df['agglo_cluster'] = agglo_labels

# GMM clustering
gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
df['gmm_cluster'] = gmm_labels

# Save cluster assignments
df[['user_id', 'kmeans_cluster', 'agglo_cluster', 'gmm_cluster']].to_csv(os.path.join(OUTPUT_DIR, 'user_clusters_overall.csv'), index=False)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot KMeans clusters
plt.figure(figsize=(8, 6))
for cluster_id in range(N_CLUSTERS):
    mask = kmeans_labels == cluster_id
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'KMeans {cluster_id}', alpha=0.7)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('KMeans Clusters (PCA projection)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'kmeans_clusters_pca.png'))
plt.close()

# Plot Agglomerative clusters
plt.figure(figsize=(8, 6))
for cluster_id in range(N_CLUSTERS):
    mask = agglo_labels == cluster_id
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Agglo {cluster_id}', alpha=0.7)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Agglomerative Clusters (PCA projection)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'agglo_clusters_pca.png'))
plt.close()

# Plot GMM clusters
plt.figure(figsize=(8, 6))
for cluster_id in range(N_CLUSTERS):
    mask = gmm_labels == cluster_id
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'GMM {cluster_id}', alpha=0.7)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('GMM Clusters (PCA projection)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'gmm_clusters_pca.png'))
plt.close()

# t-SNE visualization
print("Running t-SNE for visualization...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
for cluster_id in range(N_CLUSTERS):
    mask = kmeans_labels == cluster_id
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=f'KMeans {cluster_id}', alpha=0.7)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('KMeans Clusters (t-SNE projection)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'kmeans_clusters_tsne.png'))
plt.close()

# Compute silhouette score for t-SNE projection (optional, for completeness)
try:
    tsne_silhouette = silhouette_score(X_tsne, kmeans_labels)
    print(f"KMeans silhouette (t-SNE space): {tsne_silhouette:.3f}")
except Exception as e:
    print(f"Could not compute t-SNE silhouette: {e}")

# UMAP visualization
print("Running UMAP for visualization...")
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
for cluster_id in range(N_CLUSTERS):
    mask = kmeans_labels == cluster_id
    plt.scatter(X_umap[mask, 0], X_umap[mask, 1], label=f'KMeans {cluster_id}', alpha=0.7)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('KMeans Clusters (UMAP projection)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'kmeans_clusters_umap.png'))
plt.close()

# Compute silhouette score for UMAP projection (optional, for completeness)
try:
    umap_silhouette = silhouette_score(X_umap, kmeans_labels)
    print(f"KMeans silhouette (UMAP space): {umap_silhouette:.3f}")
except Exception as e:
    print(f"Could not compute UMAP silhouette: {e}")

# DBSCAN clustering
from sklearn.cluster import DBSCAN
print("Running DBSCAN clustering...")
dbscan = DBSCAN(eps=0.8, min_samples=5)  # You may want to tune eps/min_samples
try:
    dbscan_labels = dbscan.fit_predict(X_scaled)
    df['dbscan_cluster'] = dbscan_labels
    n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    print(f"DBSCAN found {n_dbscan_clusters} clusters (excluding noise)")
    # Save DBSCAN cluster assignments
    df[['user_id', 'dbscan_cluster']].to_csv(os.path.join(OUTPUT_DIR, 'user_clusters_dbscan.csv'), index=False)
    # Plot DBSCAN clusters (PCA)
    plt.figure(figsize=(8, 6))
    for cluster_id in set(dbscan_labels):
        mask = dbscan_labels == cluster_id
        label = f'DBSCAN {cluster_id}' if cluster_id != -1 else 'Noise'
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.7)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('DBSCAN Clusters (PCA projection)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dbscan_clusters_pca.png'))
    plt.close()
    # Silhouette score (excluding noise)
    if n_dbscan_clusters > 1 and (dbscan_labels != -1).sum() > 0:
        dbscan_silhouette = silhouette_score(X_scaled[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1])
        print(f"DBSCAN silhouette: {dbscan_silhouette:.3f}")
    else:
        print("DBSCAN silhouette: Not enough clusters/noise for silhouette score.")
except Exception as e:
    print(f"DBSCAN failed: {e}")

# HDBSCAN clustering
try:
    import hdbscan
    print("Running HDBSCAN clustering...")
    hdb = hdbscan.HDBSCAN(min_cluster_size=5)
    hdb_labels = hdb.fit_predict(X_scaled)
    df['hdbscan_cluster'] = hdb_labels
    n_hdbscan_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    print(f"HDBSCAN found {n_hdbscan_clusters} clusters (excluding noise)")
    # Save HDBSCAN cluster assignments
    df[['user_id', 'hdbscan_cluster']].to_csv(os.path.join(OUTPUT_DIR, 'user_clusters_hdbscan.csv'), index=False)
    # Plot HDBSCAN clusters (PCA)
    plt.figure(figsize=(8, 6))
    for cluster_id in set(hdb_labels):
        mask = hdb_labels == cluster_id
        label = f'HDBSCAN {cluster_id}' if cluster_id != -1 else 'Noise'
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.7)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('HDBSCAN Clusters (PCA projection)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hdbscan_clusters_pca.png'))
    plt.close()
    # Silhouette score (excluding noise)
    if n_hdbscan_clusters > 1 and (hdb_labels != -1).sum() > 0:
        hdbscan_silhouette = silhouette_score(X_scaled[hdb_labels != -1], hdb_labels[hdb_labels != -1])
        print(f"HDBSCAN silhouette: {hdbscan_silhouette:.3f}")
    else:
        print("HDBSCAN silhouette: Not enough clusters/noise for silhouette score.")
except ImportError:
    print("hdbscan package not installed. Skipping HDBSCAN clustering.")
except Exception as e:
    print(f"HDBSCAN failed: {e}")

# Print silhouette scores
print(f"KMeans silhouette: {silhouette_score(X_scaled, kmeans_labels):.3f}")
print(f"Agglomerative silhouette: {silhouette_score(X_scaled, agglo_labels):.3f}")
print(f"GMM silhouette: {silhouette_score(X_scaled, gmm_labels):.3f}")
print(f"Cluster assignments and plots saved in {OUTPUT_DIR}/")
