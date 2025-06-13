import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

# Load check-in data
cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon']
# sample: 
# 21418	4b138667f964a5208b9723e3	Tue Apr 03 18:19:55 +0000 2012	480	1.359951	103.884701
checkins = pd.read_csv('analysis_older_dataset/analysis_set4/codes/singapore_checkins_filtered_with_locations_coord.txt', sep='\t', names=cols)

print(checkins.head())

# Load POI to category mapping
place_cat = pd.read_csv('analysis_older_dataset/analysis_set4/codes/sg_place_id_to_category.csv')

# Load category to root mapping
cat_root = pd.read_csv('analysis_older_dataset/analysis_set4/codes/poi_category_to_root_mapping.csv')

# Merge to get root category for each check-in
checkins = checkins.merge(place_cat, on='place_id', how='left')
checkins = checkins.merge(cat_root, on='category', how='left')

# Prepare time features
checkins['datetime'] = pd.to_datetime(checkins['datetime'], errors='coerce')
checkins = checkins.dropna(subset=['datetime', 'root_category'])
checkins['hour_of_week'] = checkins['datetime'].dt.dayofweek * 24 + checkins['datetime'].dt.hour

# Get all root categories
root_categories = sorted(checkins['root_category'].unique())
print (f'Found {len(root_categories)} unique root categories.')
print(root_categories)
cat2idx = {cat: i for i, cat in enumerate(root_categories)}

# Build [C x 168] matrix for each user
user_matrices = {}
for user, group in checkins.groupby('user_id'):
    mat = np.zeros((len(root_categories), 168))
    for _, row in group.iterrows():
        cidx = cat2idx[row['root_category']]
        hidx = int(row['hour_of_week'])
        if 0 <= hidx < 168:
            mat[cidx, hidx] += 1
    # Normalize
    total = mat.sum()
    if total > 0:
        mat = mat / total
    user_matrices[user] = mat

# Save user matrices for next steps
np.save('analysis_older_dataset/analysis_set4/codes/C_time_matrix/user_category_time_matrices.npy', user_matrices)
with open('analysis_older_dataset/analysis_set4/codes/C_time_matrix/user_category_labels.txt', 'w') as f:
    for i, cat in enumerate(root_categories):
        f.write(f'{i}\t{cat}\n')
print('Saved user [C x 168] matrices and category labels.')
