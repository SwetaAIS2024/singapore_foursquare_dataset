import pandas as pd
import numpy as np
import pickle
import json
import os 
import glob
from scipy import stats
from scipy.stats import truncnorm
from c0_Configuration.s00_config_paths import (
    ORIGINAL_DATA_PATH,
    CLUSTER_LABELS_PATH,
    OUTPUT_SAMPLED_FSQ_PATH, 
    CLUSTER_DISTRIBUTION_SUMMARY_PATH,
    CLUSTER_DISTRIBUTION_PARAMS_PATH
)
from c1_Data_Collection_and_Processing.c1_Feature_Extraction.c2_utils import load_config
TOTAL_USERS_FOR_SAMPLING = 1000 # Total number of users to consider for sampling

def load_data():
    cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon']
    df = pd.read_csv(ORIGINAL_DATA_PATH, sep='\t', names=cols)
    cluster_labels = pd.read_csv(CLUSTER_LABELS_PATH, sep='\t', header=None, names=['user_id', 'cluster_id'])
    print("Number of unique users in cluster labels:", cluster_labels['user_id'].nunique())
    df = df.merge(cluster_labels, on='user_id', how='inner')
    return df

def load_cluster_distributions():
    # Load cluster summary CSV
    summary_path = CLUSTER_DISTRIBUTION_SUMMARY_PATH
    cluster_summary = pd.read_csv(summary_path)
    # Build a dictionary: {cluster_id: best_distribution}
    cluster_distributions = {}
    for _, row in cluster_summary.iterrows():
        cluster_id = int(row['Cluster'])
        dist_name = row['Best_Distribution']
        # You may need to load parameters from another source if not present in the CSV
        if dist_name != "N/A":
            cluster_distributions[cluster_id] = dist_name
    return cluster_distributions

def load_cluster_params():
    params_dir = CLUSTER_DISTRIBUTION_PARAMS_PATH
    cluster_params = {}
    # Assuming params_dir contains folders for each cluster
    for cluster_json in glob.glob(os.path.join(params_dir, 'cluster_*', 'fitted_parameters.json')):
        cluster_id_str = os.path.basename(os.path.dirname(cluster_json))
        cluster_id = int(cluster_id_str.replace('cluster_', ''))
        with open(cluster_json, 'r') as f:
            params_data = json.load(f)
        for dim, info in params_data.items():
            dist_name = info['distribution']
            params = [float(p) for p in info['parameters']]
            cluster_params[cluster_id] = (dist_name, params)
            break
    return cluster_params

def sample_from_distribution(dist_name, params, size, lower_bound=0.1, upper_bound=0.9):
    dist = getattr(stats, dist_name)
    # Check scale parameter (usually last or second-to-last param)
    scale = params[-1]
    if scale <= 0:
        print(f"[ERROR] Invalid scale parameter ({scale}) for distribution '{dist_name}'. Skipping sampling.")
        return np.array([])
    try:
        lwr = dist.ppf(lower_bound, *params)
        upr = dist.ppf(upper_bound, *params)
        if dist_name == "norm":
            mean, std = params
            a, b = (lwr - mean) / std, (upr - mean) / std
            return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
        else:
            samples = []
            while len(samples) < size:
                s = dist.rvs(*params, size=size)
                s = s[(s >= lwr) & (s <= upr)]
                samples.extend(s.tolist())
                samples = samples[:size]
            return np.array(samples)
    except Exception as e:
        print(f"[ERROR] Sampling failed for distribution '{dist_name}' with params {params}: {e}")
        return np.array([])

def sample_clusters(df, cluster_params):
    sampled_rows = []
    for cluster_id, group in df.groupby('cluster_id'):
        user_ids = group['user_id'].unique()
        n_samples = len(group)
        if cluster_id in cluster_params:
            dist_name, params = cluster_params[cluster_id]
            sampled_counts = sample_from_distribution(dist_name, params, n_samples)
            if sampled_counts.size == n_samples and not np.any(np.isnan(sampled_counts)):
                print(f"[SUCCESS] Distribution sampling for cluster {cluster_id} ({dist_name})")
            else:
                print(f"[WARN] Sampling failed for cluster {cluster_id}, using soft fallback.")
                sampled_counts = soft_sample(group, n_samples)
        else:
            print(f"[WARN] No params for cluster {cluster_id}, using soft fallback.")
            sampled_counts = soft_sample(group, n_samples)
        sampled_group = group.copy()
        sampled_group['sampled_count'] = np.round(sampled_counts).astype(int)
        sampled_rows.append(sampled_group)
    sampled_df = pd.concat(sampled_rows, ignore_index=True) if sampled_rows else pd.DataFrame()
    return sampled_df

def soft_sample(group, n_samples):
    # Sample uniformly from the mid 80% of the original counts
    counts = group['count'] if 'count' in group.columns else np.ones(len(group))
    lower = np.percentile(counts, 10)
    upper = np.percentile(counts, 90)
    mid_counts = counts[(counts >= lower) & (counts <= upper)]
    if len(mid_counts) == 0:
        mid_counts = counts
    return np.random.choice(mid_counts, size=n_samples, replace=True)

def save_sampled_dataset(sampled_df):
    sampled_df.to_csv(OUTPUT_SAMPLED_FSQ_PATH, sep='\t', index=False)
    print(f"Sampled dataset saved to {OUTPUT_SAMPLED_FSQ_PATH}")

def compute_samples_per_cluster(df):
    total_users = TOTAL_USERS_FOR_SAMPLING
    cluster_sizes = df['cluster_id'].value_counts().sort_index()
    total = cluster_sizes.sum()
    samples_per_cluster = {}
    for cluster_id, count in cluster_sizes.items():
        n_samples = max(1, int(round((count / total) * total_users)))
        samples_per_cluster[cluster_id] = n_samples
    # Adjust for rounding errors
    diff = total_users - sum(samples_per_cluster.values())
    if diff != 0:
        largest_cluster = cluster_sizes.idxmax()
        samples_per_cluster[largest_cluster] += diff
    print("Sample allocation per cluster:", samples_per_cluster)
    return samples_per_cluster

def sample_unique_users(df):
    total_users = TOTAL_USERS_FOR_SAMPLING
    # Get user_ids per cluster
    cluster_user_ids = df.groupby('cluster_id')['user_id'].unique()
    cluster_sizes = cluster_user_ids.apply(len)
    total = cluster_sizes.sum()
    users_per_cluster = {}
    for cluster_id, count in cluster_sizes.items():
        n_users = max(1, int(round((count / total) * total_users)))
        users_per_cluster[cluster_id] = n_users
    # Adjust for rounding errors
    diff = total_users - sum(users_per_cluster.values())
    if diff != 0:
        largest_cluster = cluster_sizes.idxmax()
        users_per_cluster[largest_cluster] += diff
    print("User allocation per cluster:", users_per_cluster)
    # Sample user_ids
    sampled_user_ids = []
    for cluster_id, n_users in users_per_cluster.items():
        user_ids = cluster_user_ids[cluster_id]
        sampled = np.random.choice(user_ids, size=min(n_users, len(user_ids)), replace=False)
        sampled_user_ids.extend(sampled)
    return set(sampled_user_ids)
    

def main():
    df = load_data()
    sampled_user_ids = sample_unique_users(df)
    df_sampled_users = df[df['user_id'].isin(sampled_user_ids)].copy()
    cluster_dist_parameters = load_cluster_params()
    sampled_df = sample_clusters(df_sampled_users, cluster_dist_parameters)
    print("Number of unique users in sampled dataset:", sampled_df['user_id'].nunique())
    print("Total samples in sampled dataset:", len(sampled_df))
    save_sampled_dataset(sampled_df)

if __name__ == "__main__":
    main()