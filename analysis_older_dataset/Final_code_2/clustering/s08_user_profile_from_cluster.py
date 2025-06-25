import pandas as pd
from s00_config_paths import CHECKINS_PATH
import numpy as np

# Load the representative users for each cluster
rep_users_df = pd.read_csv("analysis_older_dataset/Final_code/matrix_output_cluster_top_poi_categories_with_user.csv")
rep_user_ids = rep_users_df['user_id'].astype(str).tolist()

# Load the original check-in dataset
cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon']
df = pd.read_csv(CHECKINS_PATH, sep='\t', names=cols, dtype={'user_id': str})

# Filter for check-ins of the representative users
df_rep = df[df['user_id'].isin(rep_user_ids)]

# Add cluster info to each row
user2cluster = dict(zip(rep_users_df['user_id'].astype(str), rep_users_df['cluster']))
df_rep.loc[:, 'cluster'] = df_rep['user_id'].map(user2cluster)

if df_rep.empty:
    print("Warning: No check-ins found for the representative users. Check user ID mapping and types.")

# Save to CSV
out_path = "analysis_older_dataset/Final_code/cluster_representative_users_checkins.csv"
df_rep.to_csv(out_path, index=False)
print(f"Saved check-in details for representative users to {out_path}")
