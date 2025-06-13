import pandas as pd

# Load cluster assignments
clusters = pd.read_csv('../poi_clusters.csv')

# Load places and categories
places = pd.read_csv('../../latest_dataset/places_sg.csv')
categories = pd.read_csv('../../latest_dataset/categories_sg.csv')

# Merge cluster info with places and categories
merged = clusters.merge(places, on='fsq_place_id', how='left')
merged = merged.merge(categories[['category_id', 'category_name']], left_on='primary_category', right_on='category_id', how='left')

# Cluster size summary
cluster_counts = merged['cluster'].value_counts().sort_index()
print('POI count per cluster:')
print(cluster_counts)

# Top categories per cluster
with open('cluster_top_categories.txt', 'w') as f:
    for cluster_id in sorted(merged['cluster'].unique()):
        f.write(f'\n--- Cluster {cluster_id} ---\n')
        sub = merged[merged['cluster'] == cluster_id]
        top_cats = sub['category_name'].value_counts().head(10)
        f.write('Top categories:\n')
        f.write(top_cats.to_string())
        f.write('\n')
        f.write('Sample POI names:\n')
        f.write(sub['name'].dropna().head(10).to_string())
        f.write('\n')
print('Cluster analysis complete. Results saved to cluster_top_categories.txt')
