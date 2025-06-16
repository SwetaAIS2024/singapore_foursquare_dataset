import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing cluster mean matrices
profile_dir = 'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Clustering_Profiling/cluster_profiles'
# Category labels
cat_labels_path = 'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/C_time_matrix/user_category_labels.txt'
cat_labels = []
with open(cat_labels_path) as f:
    for line in f:
        idx, cat = line.strip().split('\t')
        cat_labels.append(cat)

# Output summary
summary = []

for fname in sorted(os.listdir(profile_dir)):
    if fname.endswith('_mean_matrix.npy'):
        cluster_id = int(fname.split('_')[1])
        mean_mat = np.load(os.path.join(profile_dir, fname))  # shape: [C, 168]
        # Find the peak value and its location
        peak_val = np.max(mean_mat)
        peak_cat_idx, peak_hour = np.unravel_index(np.argmax(mean_mat), mean_mat.shape)
        peak_cat = cat_labels[peak_cat_idx]
        day = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][peak_hour//24]
        hour = peak_hour%24
        peak_time = f"{day} {hour:02d}:00"
        # Simple interpretation
        interpretation = f"Cluster {cluster_id}: Highest activity for '{peak_cat}' at {peak_time} (value={peak_val:.4f})"
        summary.append({
            'cluster': cluster_id,
            'peak_value': float(peak_val),
            'peak_category': peak_cat,
            'peak_time': peak_time,
            'interpretation': interpretation
        })

# Sort summary by cluster number
summary = sorted(summary, key=lambda x: x['cluster'])

# Save summary as CSV and TXT
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(profile_dir, 'cluster_heatmap_peak_summary.csv'), index=False)
with open(os.path.join(profile_dir, 'cluster_heatmap_peak_summary.txt'), 'w') as f:
    for row in summary:
        f.write(row['interpretation'] + '\n')
print('Peak value analysis complete. See cluster_profiles/ for summary.')

# --- Combine all cluster heatmaps into a single image, sorted by cluster number ---
heatmaps = []
cluster_ids = []
for fname in os.listdir(profile_dir):
    if fname.endswith('_mean_matrix.npy'):
        cluster_id = int(fname.split('_')[1])
        mean_mat = np.load(os.path.join(profile_dir, fname))  # shape: [C, 168]
        heatmaps.append((cluster_id, mean_mat))

# Sort heatmaps by cluster_id (ascending)
heatmaps = sorted(heatmaps, key=lambda x: x[0])
cluster_ids = [cid for cid, _ in heatmaps]
heatmaps_only = [mat for _, mat in heatmaps]

if heatmaps_only:
    n_clusters = len(heatmaps_only)
    fig, axes = plt.subplots(1, n_clusters, figsize=(4*n_clusters, 6), squeeze=False)
    for i, (mat, cid) in enumerate(zip(heatmaps_only, cluster_ids)):
        ax = axes[0, i]
        im = ax.imshow(mat, aspect='auto', cmap='viridis')
        ax.set_title(f'Cluster {cid}')
        ax.set_xlabel('Hour (0-167)')
        ax.set_ylabel('Category')
        ax.set_yticks(range(len(cat_labels)))
        ax.set_yticklabels(cat_labels, fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='Mean Activity')
    plt.tight_layout()
    out_path = os.path.join(profile_dir, 'all_clusters_heatmap.png')
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'Combined heatmap saved to {out_path}')
else:
    print('No cluster heatmaps found to combine.')
