"""
Method 1: Cluster-wise Distribution Fitting and Synthetic Data Generation
Author: [Your Name]
Date: 2025-06-27

This script implements the updated Method 1 for synthetic data generation:
- For each cluster, project data onto PC1 (or another 1D feature)
- Fit multiple candidate distributions using `fitter`
- Select the best-fitting distribution by statistical criteria
- Sample synthetic data from the best fit
- Evaluate the fit and compare real vs synthetic data

Requirements:
- numpy, pandas, matplotlib, seaborn, scikit-learn, fitter

Usage:
- Place this script in the desired folder and update the data loading section as needed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from fitter import Fitter
from scipy.stats import ks_2samp, anderson_ksamp, wasserstein_distance
import seaborn as sns

# --- Parameters ---
DATA_PATH = 'your_data.csv'  # Update with your data file
CLUSTER_LABELS_PATH = 'your_cluster_labels.csv'  # Update with your cluster label file
N_SYNTH = 100  # Number of synthetic samples per cluster
K_MIN = 10     # Minimum points to fit a distribution
CANDIDATE_DISTS = ['norm', 'gamma', 'beta', 'lognorm', 'expon']

# --- Data Loading (update as needed) ---
# Example: load data and cluster labels
# data = pd.read_csv(DATA_PATH)
# cluster_labels = pd.read_csv(CLUSTER_LABELS_PATH)['cluster'].values
# For demonstration, generate dummy data:
np.random.seed(42)
data = np.random.randn(500, 10)
cluster_labels = np.random.randint(0, 5, size=500)

# --- PCA Projection ---
pca = PCA(n_components=1)
data_pc1 = pca.fit_transform(data).flatten()  # 1D projection for all points

# --- Main Loop ---
results = []
sampled_pc1_data = []
for cl in np.unique(cluster_labels):
    idxs = np.where(cluster_labels == cl)[0]
    if len(idxs) < K_MIN:
        # Too few points: fallback to random sampling, do not generate synthetic data
        sampled_pc1 = data_pc1[idxs]  # Just use the available points as-is
        sampled_pc1_data.append(sampled_pc1)
        continue
    # 1. Extract PC1 values for this cluster
    cluster_pc1 = data_pc1[idxs]
    # 2. Fit candidate distributions
    f = Fitter(cluster_pc1, distributions=CANDIDATE_DISTS)
    f.fit()
    best_dist = list(f.get_best().keys())[0]
    best_params = list(f.get_best().values())[0]
    # Check that all best_params are numeric (float or int)
    if not all(isinstance(p, (float, int, np.floating, np.integer)) for p in best_params):
        print(f"Skipping cluster {cl} due to non-numeric fit params: {best_params}")
        sampled_pc1 = cluster_pc1  # fallback: just use the original points
        sampled_pc1_data.append(sampled_pc1)
        continue
    # 3. Sample original points according to the best fit distribution's PDF
    dist_obj = getattr(__import__('scipy.stats', fromlist=[best_dist]), best_dist)
    pdf_vals = dist_obj.pdf(cluster_pc1, *best_params)
    # Normalize to get probabilities
    prob = pdf_vals / np.sum(pdf_vals)
    sampled_pc1 = np.random.choice(cluster_pc1, N_SYNTH, replace=True, p=prob)
    # 4. Evaluate fit
    ks_stat, ks_p = ks_2samp(cluster_pc1, sampled_pc1)
    ad_stat, _, _ = anderson_ksamp([cluster_pc1, sampled_pc1])
    emd = wasserstein_distance(cluster_pc1, sampled_pc1)
    results.append({'cluster': cl, 'best_dist': best_dist, 'ks': ks_stat, 'ks_p': ks_p, 'ad': ad_stat, 'emd': emd})
    sampled_pc1_data.append(sampled_pc1)

# --- Combine and Save Results ---
sampled_pc1_data = np.concatenate(sampled_pc1_data)
pd.DataFrame(results).to_csv('synthetic_fit_results.csv', index=False)
np.save('analysis_older_dataset/Final_code_2/02_synthetic_datagen_After_clustering_work/method_1/synthetic_data_method1_pc1.npy', sampled_pc1_data)

# --- Visualization Example ---
plt.figure(figsize=(8, 4))
sns.histplot(data_pc1, color='blue', label='Real PC1', kde=True, stat='density', bins=30)
plt.legend()
plt.title('Real PC1 Distribution')
plt.xlabel('PC1')
plt.ylabel('Density')
plt.savefig('analysis_older_dataset/Final_code_2/02_synthetic_datagen_After_clustering_work/method_1/real_pc1_hist.png')
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(sampled_pc1_data, color='orange', label='Sampled PC1', kde=True, stat='density', bins=30)
plt.legend()
plt.title('Sampled PC1 Distribution')
plt.xlabel('PC1')
plt.ylabel('Density')
plt.savefig('analysis_older_dataset/Final_code_2/02_synthetic_datagen_After_clustering_work/method_1/sampled_pc1_hist.png')
plt.show()

print('PC1 data sampling and evaluation complete.')
