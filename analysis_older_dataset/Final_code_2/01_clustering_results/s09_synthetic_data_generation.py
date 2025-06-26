import pandas as pd
import numpy as np
from s00_config_paths import CHECKINS_PATH

# Load the 10 representative users per cluster
rep_users_df = pd.read_csv("analysis_older_dataset/Final_code/matrix_output_cluster_top_poi_categories_with_10users.csv")

# Flatten user_ids column to get all unique user IDs
all_user_ids = set()
for user_id_str in rep_users_df['user_ids']:
    for uid in user_id_str.split(','):
        all_user_ids.add(uid.strip())

# Load the original check-in dataset
cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon']
df = pd.read_csv(CHECKINS_PATH, sep='\t', names=cols, dtype={'user_id': str})

# Filter for check-ins of the selected users
df_rep = df[df['user_id'].isin(all_user_ids)].copy()

# Generate synthetic user IDs (randomized, not in original set)
unique_synth_ids = [f'synth_{i}' for i in range(len(all_user_ids))]
np.random.shuffle(unique_synth_ids)  # Randomize the order
user_id_map = dict(zip(sorted(all_user_ids), unique_synth_ids))
df_rep['synthetic_user_id'] = df_rep['user_id'].map(user_id_map)

# Optionally, randomize timestamps or locations for more synthetic realism
# For now, keep original check-in structure but with synthetic user IDs

# Save synthetic check-in data
out_path = "analysis_older_dataset/Final_code/synthetic_checkins_from_10users.csv"
df_rep.to_csv(out_path, index=False)
print(f"Saved synthetic check-in data for 10 representative users per cluster to {out_path}")
