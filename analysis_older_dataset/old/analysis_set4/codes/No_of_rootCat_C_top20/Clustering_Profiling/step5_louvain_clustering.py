import pickle
import networkx as nx
import community as community_louvain
import numpy as np
import pandas as pd
import os

# Load the user similarity graph
graph_path = 'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Dimension_reduction_graph_construction/graph/user_similarity_knn_graph.gpickle'
with open(graph_path, 'rb') as f:
    G = pickle.load(f)

# Louvain clustering
partition = community_louvain.best_partition(G, weight='weight', resolution=2.0)

# Save cluster assignments
user_clusters = pd.DataFrame(list(partition.items()), columns=['user_id', 'cluster'])
user_clusters.to_csv('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Clustering_Profiling/user_louvain_clusters.csv', index=False)
print('Saved Louvain user clusters to analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Clustering_Profiling/user_louvain_clusters.csv')
