import numpy as np
import pandas as pd
import os

# Load category labels
cat_labels = []
with open('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_180/C_time_matrix/user_category_labels.txt') as f:
    for line in f:
        idx, cat = line.strip().split('\t')
        cat_labels.append(cat)

# For each cluster, load mean matrix and extract time-wise POI category mapping
profile_dir = 'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_180/Clustering_Profiling/cluster_profiles'
output = {}
for fname in os.listdir(profile_dir):
    if fname.endswith('_mean_matrix.npy'):
        cluster_id = fname.split('_')[1]
        mean_mat = np.load(os.path.join(profile_dir, fname))
        # For each hour, get top POI category
        hour_to_cat = {}
        for hour in range(168):
            cat_idx = np.argmax(mean_mat[:, hour])
            hour_to_cat[hour] = cat_labels[cat_idx]
        output[cluster_id] = hour_to_cat
# Save as CSV for inspection
with open('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_180/Clustering_Profiling/cluster_timewise_category.csv', 'w') as f:
    f.write('cluster_id,hour,top_category\n')
    for cluster_id, hour_map in output.items():
        for hour, cat in hour_map.items():
            f.write(f'{cluster_id},{hour},{cat}\n')
print('Saved time-wise POI category mapping for each cluster to analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_180/Clustering_Profiling/cluster_timewise_category.csv')
