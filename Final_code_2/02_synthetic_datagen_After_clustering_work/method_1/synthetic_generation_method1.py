"""
Method 1: Synthetic Data Generation via Manual Distribution Fitting
Author: [Your Name]
Date: [Today's Date]

This script implements the high-level plan for generating synthetic data from clustered Foursquare data by fitting base distributions to each cluster and sampling accordingly.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import ks_2samp, anderson_ksamp, wasserstein_distance

# 1. Load clustered data (replace with your actual data loading code)
# Static paths for input files
CHECKINS_PATH = '/home/rubesh/Desktop/sweta/google_review_collection_POI_task/singapore_foursquare_dataset_analysis/analysis_older_dataset/Final_code_2/01_clustering_results/singapore_checkins_filtered_with_locations_coord.txt'
CLUSTER_MAP_PATH = '/home/rubesh/Desktop/sweta/google_review_collection_POI_task/singapore_foursquare_dataset_analysis/analysis_older_dataset/Final_code_2/01_clustering_results/matrix_output_user_cluster_mapping.csv'

# Load user to cluster mapping
cluster_map_df = pd.read_csv(CLUSTER_MAP_PATH)
user_id_to_cluster = dict(zip(cluster_map_df['user_id'], cluster_map_df['cluster']))

# Load checkins data (no headers, tab-separated)
col_names = ['user_id', 'place_id', 'datetime', 'timezone_offset', 'latitude', 'longitude']
checkins_df = pd.read_csv(CHECKINS_PATH, sep='\t', header=None, names=col_names)

# Filter to users present in cluster mapping
checkins_df = checkins_df[checkins_df['user_id'].isin(user_id_to_cluster)]

# Aggregate user vectors (example: group by user_id and aggregate features)
# You may need to adjust this part based on your feature engineering
user_features = checkins_df.groupby('user_id').mean(numeric_only=True)  # or sum, or custom aggregation
user_ids = user_features.index.values
user_vectors = user_features.values

# Map each user to their cluster
cluster_labels = np.array([user_id_to_cluster[uid] for uid in user_ids])

# --- Parameters ---
K_MIN = 10  # Minimum points to fit a distribution
BASE_DISTRIBUTIONS = ['norm', 'expon', 'beta', 'gamma', 'lognorm']
N_SYNTH = 100  # Number of synthetic samples per cluster (user-defined)

# --- Helper Functions ---
def compute_archetype(cluster_data):
    """Compute the centroid (archetype) of the cluster."""
    return np.mean(cluster_data, axis=0)

def compute_geometry(cluster_data):
    """Compute geometric parameters: covariance, principal axes, value range."""
    cov = np.cov(cluster_data, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    min_vals = np.min(cluster_data, axis=0)
    max_vals = np.max(cluster_data, axis=0)
    return {'cov': cov, 'eigvals': eigvals, 'eigvecs': eigvecs, 'min': min_vals, 'max': max_vals}

def select_base_distribution(archetype, geometry):
    """Dummy G function: select base distribution based on skewness/kurtosis/variance."""
    # Example: use normal if low skew/kurtosis, else lognorm/gamma
    # Replace with your own logic
    return 'norm'

def fit_distribution(data, dist_name):
    dist = getattr(stats, dist_name)
    params = dist.fit(data)
    return dist, params

def sample_from_distribution(dist, params, n, min_val, max_val):
    samples = dist.rvs(*params, size=n)
    # Clip to observed range
    return np.clip(samples, min_val, max_val)

def evaluate_fit(real, synth):
    ks_stat, ks_p = ks_2samp(real, synth)
    ad_stat, _, _ = anderson_ksamp([real, synth])
    emd = wasserstein_distance(real, synth)
    return {'ks': ks_stat, 'ks_p': ks_p, 'ad': ad_stat, 'emd': emd}

# --- Main Synthetic Data Generation ---
synthetic_data = []
eval_results = []
for cl in np.unique(cluster_labels):
    cluster_data = user_vectors[cluster_labels == cl]
    if len(cluster_data) < K_MIN:
        # Too few points: random sampling
        synth = cluster_data[np.random.choice(len(cluster_data), N_SYNTH, replace=True)]
        synthetic_data.append(synth)
        continue
    # 1. Archetype & geometry
    archetype = compute_archetype(cluster_data)
    geometry = compute_geometry(cluster_data)
    # 2. Select base distribution
    dist_name = select_base_distribution(archetype, geometry)
    # 3. Fit distribution (univariate for each feature)
    synth_cluster = []
    for i in range(cluster_data.shape[1]):
        real_feat = cluster_data[:, i]
        dist, params = fit_distribution(real_feat, dist_name)
        synth_feat = sample_from_distribution(dist, params, N_SYNTH, geometry['min'][i], geometry['max'][i])
        synth_cluster.append(synth_feat)
        # 4. Evaluate fit
        eval_results.append({'cluster': cl, 'feature': i, **evaluate_fit(real_feat, synth_feat)})
    synth_cluster = np.stack(synth_cluster, axis=1)
    synthetic_data.append(synth_cluster)

synthetic_data = np.vstack(synthetic_data)


# Save synthetic data and evaluation results
np.save('synthetic_data_method1.npy', synthetic_data)
pd.DataFrame(eval_results).to_csv('synthetic_eval_method1.csv', index=False)

print('Synthetic data generation (Method 1) complete.')
