import pandas as pd

# Load all cluster assignments
clusters = pd.read_csv('poi_clusters.csv')

# Count POIs in each cluster
print('POI count per cluster:')
print(clusters['cluster'].value_counts())

# Get all POIs in cluster 6
cluster6_ids = clusters[clusters['cluster'] == 6]['fsq_place_id']

# Load places and categories
places = pd.read_csv('analysis_new_dataset_12thJune/latest_dataset/places_sg.csv')
categories = pd.read_csv('analysis_new_dataset_12thJune/latest_dataset/categories_sg.csv')

# Merge to get readable category names
cluster6_places = places[places['fsq_place_id'].isin(cluster6_ids)]
cluster6_places = cluster6_places.merge(categories[['category_id', 'category_name']], left_on='primary_category', right_on='category_id', how='left')

# Show top categories in cluster 6
print('\nTop primary categories in cluster 6:')
print(cluster6_places['category_name'].value_counts().head(20))

# Show some sample POI names
print('\nSample POI names in cluster 6:')
print(cluster6_places['name'].head(20))
