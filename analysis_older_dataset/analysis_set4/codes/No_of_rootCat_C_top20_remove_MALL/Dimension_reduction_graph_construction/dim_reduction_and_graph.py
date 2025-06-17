import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load user [C x 168] matrices
data = np.load('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20_remove_MALL/C_time_matrix/user_category_time_matrices.npy', allow_pickle=True).item()
user_ids = list(data.keys())
user_matrices = np.array([data[uid].flatten() for uid in user_ids])

print(f'Loaded {len(user_ids)} users with feature vectors of length {user_matrices.shape[1]}')

# Standardize features before PCA
scaler = StandardScaler()
user_matrices_scaled = scaler.fit_transform(user_matrices)

# Dimensionality reduction (PCA to 50D)
pca = PCA(n_components=50, random_state=42)
user_embeds = pca.fit_transform(user_matrices_scaled)
print(f'PCA reduced to shape: {user_embeds.shape}')

# Save embeddings
pd.DataFrame(user_embeds, index=user_ids).to_csv('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20_remove_MALL/Dimension_reduction_graph_construction/embeddings/user_pca_embeddings.csv')

# Similarity graph construction (cosine similarity)
sim_matrix = cosine_similarity(user_embeds)

# Build kNN graph (k=10)
k = 10
G = nx.Graph()
for i, uid in enumerate(user_ids):
    G.add_node(uid)
    # Get top k neighbors (excluding self)
    sim_scores = sim_matrix[i]
    top_k_idx = np.argsort(sim_scores)[-k-1:-1][::-1]
    for j in top_k_idx:
        neighbor = user_ids[j]
        weight = sim_scores[j]
        G.add_edge(uid, neighbor, weight=weight)

# Save the graph using nx.write_gpickle (for older networkx) or nx.write_gpickle (for newer)
os.makedirs('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20_remove_MALL/Dimension_reduction_graph_construction/graph', exist_ok=True)
with open('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20_remove_MALL/Dimension_reduction_graph_construction/graph/user_similarity_knn_graph.gpickle', 'wb') as f:
    pickle.dump(G, f)
print('Saved user similarity kNN graph and PCA embeddings.')

# 2D Graph visualization (optional, for a small sample)
sample_nodes = list(G.nodes)[:2000]
subG = G.subgraph(sample_nodes)
pos = nx.spring_layout(subG, seed=42)
plt.figure(figsize=(12, 10))
nx.draw_networkx_nodes(subG, pos, node_size=2000, node_color='blue', alpha=0.7)
nx.draw_networkx_edges(subG, pos, alpha=0.3)
plt.title('User Similarity kNN Subgraph (2000 users, 2D)')
plt.axis('off')
plt.tight_layout()
plt.savefig('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20_remove_MALL/Dimension_reduction_graph_construction/graph/user_similarity_knn_subgraph_2d.png')
plt.show()

# 3D Graph visualization (optional, for a small sample)
from mpl_toolkits.mplot3d import Axes3D
# Use PCA to reduce node features to 3D for layout
node_features = np.array([user_embeds[user_ids.index(n)] for n in sample_nodes])
from sklearn.decomposition import PCA as PCA3D
pca3d = PCA3D(n_components=3)
node_pos_3d = pca3d.fit_transform(node_features)
pos_3d = {n: node_pos_3d[i] for i, n in enumerate(sample_nodes)}

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
# Draw nodes
xs = node_pos_3d[:, 0]
ys = node_pos_3d[:, 2]
zs = node_pos_3d[:, 1]
ax.scatter(xs, ys, zs, s=2000, c='blue', alpha=0.7)
# Draw edges
for u, v in subG.edges():
    x = [pos_3d[u][0], pos_3d[v][0]]
    y = [pos_3d[u][2], pos_3d[v][2]]
    z = [pos_3d[u][1], pos_3d[v][1]]
    ax.plot(x, y, z, c='gray', alpha=0.3)
ax.set_title('User Similarity kNN Subgraph (2000 users, 3D)')
plt.tight_layout()
plt.savefig('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20_remove_MALL/Dimension_reduction_graph_construction/graph/user_similarity_knn_subgraph_3d.png')
plt.show()
