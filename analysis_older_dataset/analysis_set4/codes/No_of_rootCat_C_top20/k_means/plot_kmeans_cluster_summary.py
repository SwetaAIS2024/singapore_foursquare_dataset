import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
import matplotlib.cm as cm

# Load summary CSV
df = pd.read_csv('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/k_means/cluster_profiles/kmeans_cluster_summary.csv', index_col=0)

# Function to convert peak hour (0-167) to day and time (e.g., 'Tue 00:00')
def peak_hour_to_day_time(peak_hour):
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_idx = int(peak_hour) // 24
    hour = int(peak_hour) % 24
    return f"{days[day_idx]} {hour:02d}:00"

# Plot: Number of users per cluster
plt.figure(figsize=(10, 5))
plt.bar(df.index.astype(str), df['num_users'], color='skyblue')
plt.xlabel('Cluster')
plt.ylabel('Number of Users')
plt.title('Number of Users per KMeans Cluster')
plt.tight_layout()
plt.savefig('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/k_means/cluster_profiles/kmeans_cluster_size.png')
plt.show()

# Plot: Peak hour of week per cluster
plt.figure(figsize=(10, 5))
plt.bar(df.index.astype(str), df['peak_hour'], color='orange')
plt.xlabel('Cluster')
plt.ylabel('Peak Hour of Week')
plt.title('Peak Hour of Week per KMeans Cluster')
plt.tight_layout()
plt.savefig('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/k_means/cluster_profiles/kmeans_cluster_peak_hour.png')
plt.show()

# Plot: Top POI categories per cluster (bar for each cluster, top 1 category)
top_cat = df['top_categories'].apply(lambda x: ast.literal_eval(x)[0] if pd.notnull(x) else '')
plt.figure(figsize=(12, 6))
plt.bar(df.index.astype(str), top_cat)
plt.xlabel('Cluster')
plt.ylabel('Top POI Category')
plt.title('Top POI Category per KMeans Cluster')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/k_means/cluster_profiles/kmeans_cluster_top_poi.png')
plt.show()

# Plot: Peak hour activity per cluster
plt.figure(figsize=(10, 5))
plt.bar(df.index.astype(str), df['peak_hour_activity'], color='green')
plt.xlabel('Cluster')
plt.ylabel('Peak Hour Activity')
plt.title('Peak Hour Activity per KMeans Cluster')
plt.tight_layout()
plt.savefig('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/k_means/cluster_profiles/kmeans_cluster_peak_hour_activity.png')
plt.show()

# Combined plot: Cluster size (bar) with top POI category (text label, horizontal)
plt.figure(figsize=(14, 7))
bars = plt.bar(df.index.astype(str), df['num_users'], color='skyblue')
plt.xlabel('Cluster')
plt.ylabel('Number of Users')
plt.title('Cluster Size and Top POI Category per KMeans Cluster')
plt.xticks(rotation=45)
# Add top POI category as text label above each bar (horizontal)
for i, rect in enumerate(bars):
    top_poi = ast.literal_eval(df.iloc[i]['top_categories'])[0] if pd.notnull(df.iloc[i]['top_categories']) else ''
    plt.text(rect.get_x() + rect.get_width()/2, rect.get_height() + max(df['num_users'])*0.01, top_poi, ha='center', va='bottom', fontsize=10, rotation=0, color='darkblue', clip_on=True)
plt.tight_layout()
plt.savefig('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/k_means/cluster_profiles/kmeans_cluster_size_top_poi_combined.png')
plt.show()

# Combined plot: Cluster size (bar) with top POI category (text label, horizontal) and temporal component (peak hour as color)
norm = plt.Normalize(df['peak_hour'].min(), df['peak_hour'].max())
cmap = plt.colormaps['viridis']
bar_colors = cmap(norm(df['peak_hour']))

fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(df.index.astype(str), df['num_users'], color=bar_colors)
ax.set_xlabel('Cluster')
ax.set_ylabel('Number of Users')
ax.set_title('Cluster Size, Top POI, and Temporal (Peak Hour) per KMeans Cluster')
ax.set_xticks(np.arange(len(df.index)))
ax.set_xticklabels(df.index.astype(str), rotation=45)

# Add top POI category as text label above each bar (horizontal)
for i, rect in enumerate(bars):
    top_poi = ast.literal_eval(df.iloc[i]['top_categories'])[0] if pd.notnull(df.iloc[i]['top_categories']) else ''
    peak_hour = df.iloc[i]['peak_hour']
    peak_hr_str = peak_hour_to_day_time(peak_hour)
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + max(df['num_users'])*0.01, f"{top_poi}\n({peak_hr_str})", ha='center', va='bottom', fontsize=10, rotation=0, color='darkblue', clip_on=True)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.01)
cbar.set_label('Peak Hour of Week')
fig.tight_layout()
fig.savefig('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/k_means/cluster_profiles/kmeans_cluster_size_top_poi_temporal_combined.png')
plt.show()

print('Plots saved to k_means/cluster_profiles/.')
