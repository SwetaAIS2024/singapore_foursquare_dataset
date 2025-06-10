"""
Combines temporal and spatial trend features for each user.
- Loads temporal features from user_activity_trend_encoded.csv
- Loads spatial features from user_spatial_grid_features.csv
- Merges on user_id (inner join: only users present in both)
- Saves the combined features to spatial_temporal_activity_trends/combined_spatial_temporal_features.csv
"""
import pandas as pd
import os

# Paths
TEMPORAL_CSV = 'analysis_set2/Feature_engineering/user_activity_trends_summary/user_activity_trend_encoded.csv'
SPATIAL_CSV = 'analysis_set2/Feature_engineering/2_spatial_feature_extraction/user_spatial_grid_features.csv'
OUTPUT_DIR = 'analysis_set2/Feature_engineering/spatial_temporal_activity_trends'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'combined_spatial_temporal_features.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print(f"Loading temporal features from {TEMPORAL_CSV}")
df_temporal = pd.read_csv(TEMPORAL_CSV)
print(f"Loading spatial features from {SPATIAL_CSV}")
df_spatial = pd.read_csv(SPATIAL_CSV)

# Merge on user_id (inner join)
df_combined = pd.merge(df_temporal, df_spatial, on='user_id', how='inner')
print(f"Combined shape: {df_combined.shape}")

# Save
df_combined.to_csv(OUTPUT_CSV, index=False)
print(f"Combined spatial-temporal features written to {OUTPUT_CSV}")
