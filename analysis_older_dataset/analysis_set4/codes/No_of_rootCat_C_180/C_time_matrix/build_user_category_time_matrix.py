import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

# Load check-in data
cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon']
# sample: 
# 21418	4b138667f964a5208b9723e3	Tue Apr 03 18:19:55 +0000 2012	480	1.359951	103.884701
checkins = pd.read_csv('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_180/singapore_checkins_filtered_with_locations_coord.txt', sep='\t', names=cols)

print(checkins.head())

# Load POI to category mapping
place_cat = pd.read_csv('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_180/sg_place_id_to_category.csv')

# Merge to get 'category' for each check-in
checkins = checkins.merge(place_cat, on='place_id', how='left')

# Prepare time features
checkins['datetime'] = pd.to_datetime(checkins['datetime'], errors='coerce')
checkins = checkins.dropna(subset=['datetime', 'category'])
checkins['hour_of_week'] = checkins['datetime'].dt.dayofweek * 24 + checkins['datetime'].dt.hour

# Restrict to top 2000 users by check-in count
user_counts = checkins['user_id'].value_counts().head(2003) # plus 3 because of some users with no check-ins in the top 2000
top_users = set(user_counts.index)
checkins = checkins[checkins['user_id'].isin(top_users)]
print(f'Using top {len(top_users)} users by check-in count.')

# Remove all check-ins where the category is 'Mall' (case-insensitive, strip whitespace)
checkins['category'] = checkins['category'].astype(str).str.strip().str.lower()
checkins = checkins[checkins['category'] != 'mall']
print(f"Removed 'Mall' check-ins (case-insensitive, trimmed). Remaining check-ins: {len(checkins)}")

# Load relevant POI categories from Excel
relevant_cats_xlsx = 'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_180/C_time_matrix/Relevant_POI_category.xlsx'
relevant_cats_df = pd.read_excel(relevant_cats_xlsx)
# Use correct column names from Excel
cat_col = 'POI Category in Singapore'
yes_col = 'Relevant to use case '
# Only keep categories with 'Yes' in the yes_col
relevant_categories = [cat for cat, flag in zip(relevant_cats_df[cat_col], relevant_cats_df[yes_col]) if str(flag).strip().lower() == 'yes']
relevant_categories = [cat.strip().lower() for cat in relevant_categories if cat and str(cat).strip()]
relevant_categories = list(dict.fromkeys(relevant_categories))  # Removes duplicates, keeps order

# Standardize check-in categories
checkins['category'] = checkins['category'].astype(str).str.strip().str.lower()

# Filter check-ins to only those in the relevant categories
checkins = checkins[checkins['category'].isin(relevant_categories)]
print(f"Filtered to relevant {len(relevant_categories)} POI categories (with 'Yes' flag). Remaining check-ins: {len(checkins)}")

# Use the categories in the order from the Excel file (title case for output)
ordered_categories = [cat.title() for cat in relevant_categories]
cat2idx = {cat: i for i, cat in enumerate(ordered_categories)}

# Build [C x 168] matrix for each user (C = 180)
user_matrices = {}
for user, group in checkins.groupby('user_id'):
    mat = np.zeros((len(ordered_categories), 168))
    for _, row in group.iterrows():
        cat = row['category'].title()
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
np.save('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_180/C_time_matrix/user_category_time_matrices.npy', user_matrices)
with open('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_180/C_time_matrix/user_category_labels.txt', 'w') as f:
    for i, cat in enumerate(ordered_categories):
        f.write(f'{i}\t{cat}\n')
print('Saved user [C x 168] matrices and category labels (Relevant 180 POI categories).')

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
