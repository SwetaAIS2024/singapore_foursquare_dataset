import pandas as pd

# Load a sample of POI IDs from cluster 6
with open('cluster6_sample.txt') as f:
    poi_ids = [line.strip() for line in f if line.strip()]

# Load features and places data
features = pd.read_csv('poi_spatial_semantic_features.csv')
places = pd.read_csv('analysis_new_dataset_12thJune/latest_dataset/places_sg.csv')
categories = pd.read_csv('analysis_new_dataset_12thJune/latest_dataset/categories_sg.csv')

# Filter for cluster 6 POIs
cluster6_features = features[features['fsq_place_id'].isin(poi_ids)]
cluster6_places = places[places['fsq_place_id'].isin(poi_ids)]

# Merge to get readable category names
cluster6_places = cluster6_places.merge(categories[['category_id', 'category_name']], left_on='primary_category', right_on='category_id', how='left')

# Show top categories in cluster 6
print('Top primary categories in cluster 6:')
print(cluster6_places['category_name'].value_counts().head(10))

# Show some sample POI names
print('\nSample POI names in cluster 6:')
print(cluster6_places['name'].head(10))
