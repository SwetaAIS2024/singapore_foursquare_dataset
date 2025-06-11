import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load cluster centroids
centroids_path = 'analysis_set3/trial2_graph_based/plots_clustering/user_cluster_centroids_PCA_KMeans.csv'
df = pd.read_csv(centroids_path)

# Plot all 9 root categories as y-axis in a single 3D plot, using category index
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

root_categories = [
    'degree_Arts & Entertainment',
    'degree_College & University',
    'degree_Food',
    'degree_Great Outdoors',
    'degree_Nightlife Spot',
    'degree_Professional & Other Places',
    'degree_Residence',
    'degree_Shop & Service',
    'degree_Travel & Transport'
]

for i, cat in enumerate(root_categories):
    ax.scatter(
        df['mean_distance'],
        [i]*len(df),  # y-axis is category index
        df[cat],
        c=df['Cluster'],
        cmap='tab10',
        s=100,
        edgecolor='k',
        alpha=0.9,
        label=cat.replace('degree_', '') if i == 0 else None  # only label first for legend
    )

ax.set_xlabel('Mean Distance (km)')
ax.set_ylabel('POI Root Category')
ax.set_zlabel('Degree (Category)')
ax.set_yticks(list(range(len(root_categories))))
ax.set_yticklabels([cat.replace('degree_', '') for cat in root_categories], rotation=30)
plt.title('Cluster Centroids: Spatial, POI Root Category, and Activity')
plt.tight_layout()
plt.show()

# If you have a temporal feature like mean_time_between_checkins, you can plot it as well:
if 'mean_time_between_checkins' in df.columns:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i, cat in enumerate(root_categories):
        ax.scatter(
            df['mean_distance'],
            [i]*len(df),
            df['mean_time_between_checkins'],
            c=df['Cluster'],
            cmap='tab10',
            s=100,
            edgecolor='k',
            alpha=0.9,
            label=cat.replace('degree_', '') if i == 0 else None
        )
    ax.set_xlabel('Mean Distance (km)')
    ax.set_ylabel('POI Root Category')
    ax.set_zlabel('Mean Time Between Check-ins (h)')
    ax.set_yticks(list(range(len(root_categories))))
    ax.set_yticklabels([cat.replace('degree_', '') for cat in root_categories], rotation=30)
    plt.title('Cluster Centroids: Spatial, POI Root Category, and Temporal (Mean Time Between Check-ins)')
    plt.tight_layout()
    plt.show()
