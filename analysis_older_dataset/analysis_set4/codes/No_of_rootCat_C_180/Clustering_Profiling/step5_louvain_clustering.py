import pickle
import networkx as nx
import community as community_louvain
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load the user similarity graph
graph_path = 'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_180/Dimension_reduction_graph_construction/graph/user_similarity_knn_graph.gpickle'
with open(graph_path, 'rb') as f:
    G = pickle.load(f)

# Louvain clustering
partition = community_louvain.best_partition(G, weight='weight', resolution=2.0)

# Save cluster assignments
user_clusters = pd.DataFrame(list(partition.items()), columns=['user_id', 'cluster'])
user_clusters.to_csv('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_180/Clustering_Profiling/user_louvain_clusters.csv', index=False)
print('Saved Louvain user clusters to analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_180/Clustering_Profiling/user_louvain_clusters.csv')

# After saving cluster assignments, create a scatter plot of user_id vs cluster
plt.figure(figsize=(12, 6))
user_clusters_sorted = user_clusters.sort_values('cluster')
plt.scatter(range(len(user_clusters_sorted)), user_clusters_sorted['cluster'], c=user_clusters_sorted['cluster'], cmap='tab20', s=10)
plt.xlabel('User Index (sorted by cluster)')
plt.ylabel('Cluster ID')
plt.title('User Cluster Assignments (Louvain)')
plt.tight_layout()
plt.savefig('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_180/Clustering_Profiling/user_louvain_clusters_scatter.png')
plt.close()
print('Saved user cluster scatter plot to .../user_louvain_clusters_scatter.png')
