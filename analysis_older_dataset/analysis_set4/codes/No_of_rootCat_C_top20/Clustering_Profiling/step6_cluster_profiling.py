import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load user cluster assignments
clusters = pd.read_csv('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Clustering_Profiling/user_louvain_clusters.csv')
# Load user [C x 168] matrices
user_matrices = np.load('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/C_time_matrix/user_category_time_matrices.npy', allow_pickle=True).item()
# Load category labels
cat_labels = []
with open('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/C_time_matrix/user_category_labels.txt') as f:
    for line in f:
        idx, cat = line.strip().split('\t')
        cat_labels.append(cat)
C = len(cat_labels)

# Prepare output dir
os.makedirs('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Clustering_Profiling/cluster_profiles', exist_ok=True)

# For each cluster, aggregate and profile
for cluster_id in sorted(clusters['cluster'].unique()):
    user_ids = clusters[clusters['cluster'] == cluster_id]['user_id']
    mats = [user_matrices[uid] for uid in user_ids if uid in user_matrices]
    if not mats:
        continue
    mats = np.stack(mats)
    mean_mat = mats.mean(axis=0)  # [C x 168]
    # Save mean matrix
    np.save(f'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Clustering_Profiling/cluster_profiles/cluster_{cluster_id}_mean_matrix.npy', mean_mat)
    # Heatmap
    plt.figure(figsize=(18, 8))
    sns.heatmap(mean_mat, yticklabels=cat_labels, cmap='viridis')
    plt.title(f'Cluster {cluster_id} - POI Category vs Hour-of-Week Heatmap')
    plt.xlabel('Hour of Week (0=Mon 0:00)')
    plt.ylabel('POI Root Category')
    plt.tight_layout()
    plt.savefig(f'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Clustering_Profiling/cluster_profiles/cluster_{cluster_id}_heatmap.png')
    plt.close()
    # POI Preferences
    cat_pref = mean_mat.sum(axis=1)
    top_cats = [cat_labels[i] for i in np.argsort(cat_pref)[::-1][:5]]
    # Temporal Signature
    time_profile = mean_mat.sum(axis=0)
    plt.figure(figsize=(16, 4))
    plt.plot(time_profile)
    plt.title(f'Cluster {cluster_id} - Hourly Activity Profile')
    plt.xlabel('Hour of Week (0=Mon 0:00)')
    plt.ylabel('Avg. Activity')
    plt.tight_layout()
    plt.savefig(f'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Clustering_Profiling/cluster_profiles/cluster_{cluster_id}_time_profile.png')
    plt.close()
    # Save summary
    with open(f'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Clustering_Profiling/cluster_profiles/cluster_{cluster_id}_summary.txt', 'w') as f:
        f.write(f'Cluster {cluster_id}\n')
        f.write(f'Num users: {len(user_ids)}\n')
        f.write(f'Top POI categories: {top_cats}\n')
print('Cluster profiling complete. See analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Clustering_Profiling/cluster_profiles/')
