import numpy as np
import pandas as pd
import os

# Load user cluster assignments
cluster_path = 'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/k_means/user_kmeans_clusters.csv'
user_clusters = pd.read_csv(cluster_path, index_col=0)

# Load user [C x 168] matrices
matrices_path = 'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/C_time_matrix/user_category_time_matrices.npy'
user_matrices = np.load(matrices_path, allow_pickle=True).item()

# Load category labels
cat_labels_path = 'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/C_time_matrix/user_category_labels.txt'
cat_labels = []
with open(cat_labels_path) as f:
    for line in f:
        idx, cat = line.strip().split('\t')
        cat_labels.append(cat)

# Prepare output directory
profile_dir = 'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/k_means/cluster_profiles'
os.makedirs(profile_dir, exist_ok=True)

summary = {}
for cluster_id in sorted(user_clusters['cluster'].unique()):
    user_ids = user_clusters[user_clusters['cluster'] == cluster_id].index
    mats = [user_matrices[uid] for uid in user_ids if uid in user_matrices]
    if not mats:
        continue
    mats = np.stack(mats)
    mean_mat = mats.mean(axis=0)  # [C x 168]
    # Save mean matrix
    np.save(os.path.join(profile_dir, f'cluster_{cluster_id}_mean_matrix.npy'), mean_mat)
    # POI Preferences (top 5 by sum)
    cat_pref = mean_mat.sum(axis=1)
    top_cats = [cat_labels[i] for i in np.argsort(cat_pref)[::-1][:5]]
    # Temporal Signature
    time_profile = mean_mat.sum(axis=0)
    # Save summary
    summary[cluster_id] = {
        'num_users': len(user_ids),
        'top_categories': top_cats,
        'peak_hour': int(np.argmax(time_profile)),
        'peak_hour_activity': float(np.max(time_profile)),
    }
    # Save a simple text summary
    with open(os.path.join(profile_dir, f'cluster_{cluster_id}_summary.txt'), 'w') as f:
        f.write(f'Cluster {cluster_id}\n')
        f.write(f'Num users: {len(user_ids)}\n')
        f.write(f'Top POI categories: {top_cats}\n')
        f.write(f'Peak hour of week: {np.argmax(time_profile)} (activity={np.max(time_profile):.4f})\n')

# Save all cluster summaries as CSV
summary_df = pd.DataFrame.from_dict(summary, orient='index')
summary_df.to_csv(os.path.join(profile_dir, 'kmeans_cluster_summary.csv'))
print('Cluster interpretation complete. See cluster_profiles/ for summaries and mean matrices.')
