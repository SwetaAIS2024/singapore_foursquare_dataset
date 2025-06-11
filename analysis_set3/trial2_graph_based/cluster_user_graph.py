import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_selection import VarianceThreshold
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 1. Load the feature file
csv_path = "analysis_set3/trial2_graph_based/user_graph_features_rootcat.csv"  # Use root-category features with distance
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Feature file not found: {csv_path}")
df = pd.read_csv(csv_path)

# 2. Initial analysis
print("Shape:", df.shape)
print("Columns:", df.columns[:10], "...")  # print first 10 columns
print("Sample:\n", df.head())
print("Feature types:\n", df.dtypes.value_counts())
print("Basic stats:\n", df.describe().T.head(10))

# 3. Prepare for clustering
# Drop user_id (not a feature)
features = df.drop(columns=['user_id'])

# Optionally, drop columns with zero variance (all zeros)
features = features.loc[:, features.std() > 0]

# If distance-based features exist, print them for confirmation
distance_cols = [col for col in features.columns if 'distance' in col or 'weight' in col]
if distance_cols:
    print('Distance-based features found:', distance_cols)
    print('Sample distance feature values:\n', features[distance_cols].head())
else:
    print('No distance-based features found in the feature file.')

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 4. Dimensionality reduction for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

# UMAP (if available)
try:
    import umap
    umap_available = True
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
except ImportError:
    umap_available = False

# Feature Selection (VarianceThreshold)
selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X_scaled)
# For visualization, reduce to 2D with PCA after selection
pca_sel = PCA(n_components=2)
X_pca_sel = pca_sel.fit_transform(X_selected)

# --- Autoencoder for 2D embedding ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, code_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, code_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        code = self.encoder(x)
        recon = self.decoder(code)
        return code, recon

# Prepare data for torch
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
input_dim = X_tensor.shape[1]
autoencoder = Autoencoder(input_dim)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Train autoencoder
n_epochs = 100
for epoch in range(n_epochs):
    optimizer.zero_grad()
    code, recon = autoencoder(X_tensor)
    loss = loss_fn(recon, X_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Autoencoder epoch {epoch}, loss: {loss.item():.4f}")

with torch.no_grad():
    X_autoenc = autoencoder.encoder(X_tensor).numpy()

N_CLUSTERS = 7  # Set number of clusters for KMeans and others

# 5. Clustering (KMeans, k=4 as example)
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)

# Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=N_CLUSTERS)
labels_agg = agg.fit_predict(X_scaled)

# DBSCAN
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=2, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)

# Gaussian Mixture Model
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=42)
labels_gmm = gmm.fit_predict(X_scaled)

# Print mean feature values for each cluster (for interpretation)
for name, labels in zip(
    ['KMeans', 'Agglomerative', 'DBSCAN', 'GMM'],
    [labels_kmeans, labels_agg, labels_dbscan, labels_gmm]
):
    print(f"\n{name} cluster means:")
    print(pd.DataFrame(X_scaled, columns=features.columns).groupby(labels).mean())

# 6. Visualize clusters
methods = [
    (X_pca, 'PCA'),
    (X_tsne, 't-SNE'),
    (X_pca_sel, 'VarianceThreshold+PCA')
]
if umap_available:
    methods.insert(2, (X_umap, 'UMAP'))

clusterings = [
    (labels_kmeans, 'KMeans'),
    (labels_agg, 'Agglomerative'),
    (labels_dbscan, 'DBSCAN'),
    (labels_gmm, 'GMM')
]

# Add autoencoder embedding to methods
methods.append((X_autoenc, 'Autoencoder'))

# Fix: set subplot rows to match number of methods
fig, axes = plt.subplots(len(methods), 4, figsize=(24, 4 * len(methods)))
axes = axes.reshape(-1, 4)

for row, (X_emb, emb_name) in enumerate(methods):
    for col, (labels, clust_name) in enumerate(clusterings):
        ax = axes[row, col]
        ax.scatter(X_emb[:,0], X_emb[:,1], c=labels, cmap='tab10', s=10)
        ax.set_title(f"{clust_name} ({emb_name})")
        ax.set_xlabel(f"{emb_name} 1")
        ax.set_ylabel(f"{emb_name} 2")

plt.tight_layout()
os.makedirs("analysis_set3/trial2_graph_based/plots_clustering", exist_ok=True)
plt.savefig("analysis_set3/trial2_graph_based/plots_clustering/user_clustering_comparison.png")
plt.show()

# 7. Evaluate clustering results and save to text file
results = []
for (X_emb, emb_name) in methods:
    for (labels, clust_name) in clusterings:
        # Some metrics require at least 2 clusters and no all-noise
        valid = (len(set(labels)) > 1) and (len(set(labels)) < len(labels))
        if valid:
            try:
                sil = silhouette_score(X_emb, labels)
            except Exception:
                sil = 'NA'
            try:
                ch = calinski_harabasz_score(X_emb, labels)
            except Exception:
                ch = 'NA'
            try:
                db = davies_bouldin_score(X_emb, labels)
            except Exception:
                db = 'NA'
        else:
            sil = ch = db = 'NA'
        results.append(f"{emb_name}\t{clust_name}\tSilhouette: {sil}\tCalinski-Harabasz: {ch}\tDavies-Bouldin: {db}")

os.makedirs("analysis_set3/trial2_graph_based/plots_clustering", exist_ok=True)
eval_path = "analysis_set3/trial2_graph_based/plots_clustering/user_clustering_evaluation.txt"
with open(eval_path, "w") as f:
    f.write("Embedding\tClustering\tSilhouette\tCalinski-Harabasz\tDavies-Bouldin\n")
    for line in results:
        f.write(line + "\n")
print(f"Clustering evaluation results saved to {eval_path}")

# 8. Save cluster assignments to CSV for interpretability
# We'll use the best clustering (PCA+KMeans) as default, but add all clusterings as columns
cluster_assignments = pd.DataFrame({'user_id': df['user_id']})
cluster_assignments['PCA_KMeans'] = labels_kmeans
cluster_assignments['PCA_Agglomerative'] = labels_agg
cluster_assignments['PCA_DBSCAN'] = labels_dbscan
cluster_assignments['PCA_GMM'] = labels_gmm
# Optionally, add more clusterings if needed

# Merge with original features for interpretability
features_with_clusters = pd.concat([df, cluster_assignments.drop(columns=['user_id'])], axis=1)
features_with_clusters.to_csv("analysis_set3/trial2_graph_based/plots_clustering/user_graph_features_with_clusters.csv", index=False)
print("Saved user features with cluster assignments to user_graph_features_with_clusters.csv")

# 9. Interpretability tests: print and save mean feature values for each cluster (PCA+KMeans)
interpret_path = "analysis_set3/trial2_graph_based/plots_clustering/user_clustering_interpretability.txt"
with open(interpret_path, "w") as f:
    for cluster in sorted(cluster_assignments['PCA_KMeans'].unique()):
        f.write(f"\nCluster {cluster} (PCA+KMeans):\n")
        means = features[cluster_assignments['PCA_KMeans'] == cluster].mean()
        f.write(means.to_string())
        f.write("\n")
        print(f"\nCluster {cluster} (PCA+KMeans):\n", means)
print(f"Interpretability results saved to {interpret_path}")

# 10. Extract cluster centroids for best method (PCA+KMeans) and create user profiles
centroids = kmeans.cluster_centers_  # In standardized feature space
# Inverse transform to get centroids in original feature space
centroids_orig = scaler.inverse_transform(centroids)
centroids_df = pd.DataFrame(centroids_orig, columns=features.columns)
centroids_df.index.name = 'Cluster'
centroids_df.to_csv("analysis_set3/trial2_graph_based/plots_clustering/user_cluster_centroids_PCA_KMeans.csv")
print("Saved cluster centroids to user_cluster_centroids_PCA_KMeans.csv")

# Create a user profile summary for each cluster
profile_path = "analysis_set3/trial2_graph_based/plots_clustering/user_cluster_profiles_PCA_KMeans.txt"
with open(profile_path, "w") as f:
    for cluster, row in centroids_df.iterrows():
        f.write(f"\nCluster {cluster} profile (PCA+KMeans):\n")
        # Show top 3 root categories for this cluster
        top_cats = row.sort_values(ascending=False).head(3)
        for cat, val in top_cats.items():
            f.write(f"  {cat}: {val:.2f}\n")
        f.write("\nFull centroid:\n")
        f.write(row.to_string())
        f.write("\n")
        print(f"\nCluster {cluster} profile (PCA+KMeans):\n", top_cats)
print(f"User profile summaries saved to {profile_path}")