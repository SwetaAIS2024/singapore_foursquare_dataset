import pandas as pd
import numpy as np

# Path to your POI CSV file
csv_path = 'analysis_new_dataset_12thJune/latest_dataset/places_sg.csv'  # Update if needed

df = pd.read_csv(csv_path)

# 1. Extract semantic features: one-hot or label encoding for primary_category
category_counts = df['primary_category'].value_counts()
top_categories = category_counts.head(20).index.tolist()
df['primary_category_top'] = df['primary_category'].where(df['primary_category'].isin(top_categories), 'Other')
category_dummies = pd.get_dummies(df['primary_category_top'], prefix='cat')

# 2. Extract spatial features: use latitude and longitude directly
spatial_features = df[['latitude', 'longitude']]

# 3. Combine features for clustering
features = pd.concat([category_dummies, spatial_features], axis=1)
features['fsq_place_id'] = df['fsq_place_id']

# 4. Save features for clustering
features.to_csv('poi_spatial_semantic_features.csv', index=False)
print('POI spatial and semantic features saved to poi_spatial_semantic_features.csv')

# 5. Analyze the created features file
features_summary_path = 'poi_spatial_semantic_features_summary.txt'
with open(features_summary_path, 'w') as f:
    f.write('--- POI Spatial & Semantic Features Summary ---\n')
    f.write(f'Number of POIs: {len(features)}\n')
    f.write(f'Columns: {list(features.columns)}\n')
    f.write(f'First 5 rows:\n{features.head()}\n')
    f.write(f'Column types:\n{features.dtypes}\n')
    f.write(f'Null values per column:\n{features.isnull().sum()}\n')
    # Category columns summary
    cat_cols = [col for col in features.columns if col.startswith('cat_')]
    f.write(f'Category columns: {cat_cols}\n')
    f.write(f'POIs per top category (sum of one-hot columns):\n')
    for col in cat_cols:
        f.write(f'  {col}: {features[col].sum()}\n')
    # Latitude/Longitude summary
    f.write(f'Latitude: min={features["latitude"].min()}, max={features["latitude"].max()}, mean={features["latitude"].mean()}\n')
    f.write(f'Longitude: min={features["longitude"].min()}, max={features["longitude"].max()}, mean={features["longitude"].mean()}\n')
print(f'Feature summary saved to {features_summary_path}')
