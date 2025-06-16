import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

# Load check-in data
cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon']
# sample: 
# 21418	4b138667f964a5208b9723e3	Tue Apr 03 18:19:55 +0000 2012	480	1.359951	103.884701
checkins = pd.read_csv('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/singapore_checkins_filtered_with_locations_coord.txt', sep='\t', names=cols)

print(checkins.head())

# Load POI to category mapping
place_cat = pd.read_csv('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/sg_place_id_to_category.csv')

# Merge to get 'category' for each check-in
checkins = checkins.merge(place_cat, on='place_id', how='left')

# Prepare time features
checkins['datetime'] = pd.to_datetime(checkins['datetime'], errors='coerce')
checkins = checkins.dropna(subset=['datetime', 'category'])
checkins['hour_of_week'] = checkins['datetime'].dt.dayofweek * 24 + checkins['datetime'].dt.hour

# Restrict to top 2000 users by check-in count
user_counts = checkins['user_id'].value_counts().head(2000)
top_users = set(user_counts.index)
checkins = checkins[checkins['user_id'].isin(top_users)]
print(f'Using top {len(top_users)} users by check-in count.')

# Get top 50 categories by user check-in counts (not global counts)
user_cat_counts = checkins.groupby(['user_id', 'category']).size().reset_index(name='count')
top_cats_by_user = user_cat_counts.groupby('category')['count'].sum().sort_values(ascending=False).head(20)
top_categories = top_cats_by_user.index.tolist()
print(f'Using top 20 categories by user check-in counts: {top_categories}')
cat2idx = {cat: i for i, cat in enumerate(top_categories)}

# Build [C x 168] matrix for each user (C = 20)
user_matrices = {}
for user, group in checkins.groupby('user_id'):
    mat = np.zeros((len(top_categories), 168))
    for _, row in group.iterrows():
        cat = row['category']
        hidx = int(row['hour_of_week'])
        if cat in cat2idx and 0 <= hidx < 168:
            cidx = cat2idx[cat]
            mat[cidx, hidx] += 1
    # Normalize
    total = mat.sum()
    if total > 0:
        mat = mat / total
    user_matrices[user] = mat

# Save user matrices for next steps
np.save('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/C_time_matrix/user_category_time_matrices.npy', user_matrices)
with open('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/C_time_matrix/user_category_labels.txt', 'w') as f:
    for i, cat in enumerate(top_categories):
        f.write(f'{i}\t{cat}\n')
print('Saved user [C x 168] matrices and category labels (Top 20 by user check-in counts).')

# Display the first three rows of the first user's matrix for inspection
if user_matrices:
    first_user = next(iter(user_matrices))
    print(f"First user: {first_user}")
    print("First three rows of the matrix (shape {}):".format(user_matrices[first_user].shape))
    print(user_matrices[first_user][:3])
else:
    print("No user matrices found.")

# Display the number of users and the final matrix dimension
print(f"Number of users in matrix: {len(user_matrices)}")
if user_matrices:
    matrix_shape = next(iter(user_matrices.values())).shape
    print(f"Shape of each user matrix: {matrix_shape}")
else:
    print("No user matrices found.")
