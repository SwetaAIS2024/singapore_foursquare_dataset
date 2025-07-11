"""
Post-Sampling Analysis: Compare synthetic and real user data

This script evaluates the quality of synthetic user samples by comparing them to the original user data.
It performs:
- Per-feature distribution comparison (histograms, KS test, Wasserstein distance)
- PCA visualization of real vs synthetic users
- Classifier-based distinguishability test (optional)

Usage:
- Update REAL_DATA_PATH and SYNTHETIC_DATA_PATH as needed.
- Run from the workspace root or as a module.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from c0_config.s00_config_paths import REAL_USERS_NPY_PATH, SYNTHETIC_USERS_NPY_PATH, SYNTHETIC_USERS_CSV_PATH, POST_SAMPLING_ANALYSIS_OUTPUT_DIR, FINAL_INPUT_DATASET, DIMRED_MODEL_PATH
import joblib

# --- Config ---
REAL_DATA_PATH = REAL_USERS_NPY_PATH
SYNTHETIC_DATA_PATH = SYNTHETIC_USERS_NPY_PATH
SYNTHETIC_CSV_PATH = SYNTHETIC_USERS_CSV_PATH
OUTPUT_DIR = POST_SAMPLING_ANALYSIS_OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)
N_FEATURES_TO_PLOT = 10  # Number of features to plot/analyze

# --- Load Data ---
# Load real user vectors from all batch .npz files listed in matrix_metadata.json
import json as js
meta_path = os.path.join(FINAL_INPUT_DATASET, 'matrix_metadata.json')
with open(meta_path, 'r') as f:
    metadata = js.load(f)
batch_files = [os.path.join(FINAL_INPUT_DATASET, os.path.basename(f)) for f in metadata.get('batch_files', [])]
real_users = []
import scipy.sparse
for batch_file in batch_files:
    if not os.path.exists(batch_file):
        continue
    batch_vectors = scipy.sparse.load_npz(batch_file).toarray()
    # Flatten each user vector (row) to 1D, matching the original SVD input
    batch_vectors_flat = batch_vectors.reshape(batch_vectors.shape[0], -1)
    real_users.append(batch_vectors_flat)
real_users = np.vstack(real_users)
print(f"[DEBUG] Shape of real_users before SVD: {real_users.shape}")
# --- Project real users into reduced feature space ---
try:
    dimred_bundle = joblib.load(DIMRED_MODEL_PATH)
    if isinstance(dimred_bundle, dict):
        svd = dimred_bundle['svd']
        scaler = dimred_bundle['scaler']
        selector = dimred_bundle['selector']
        print(f"[DEBUG] SVD expects input with {svd.components_.shape[1]} features.")
        if real_users.shape[1] != svd.components_.shape[1]:
            raise ValueError(f"[ERROR] Feature mismatch: real_users has {real_users.shape[1]} features, SVD expects {svd.components_.shape[1]}")
        real_users_reduced = selector.transform(scaler.transform(svd.transform(real_users)))
    else:
        real_users_reduced = dimred_bundle.transform(real_users)
except Exception as e:
    print(f"[ERROR] Failed to load or apply dimensionality reduction model: {e}")
    print("Proceeding with original real user features (may cause dimension mismatch!)")
    real_users_reduced = real_users

synth_users = np.load(SYNTHETIC_DATA_PATH)

# --- Per-feature Distribution Comparison ---
ks_stats = []
wasserstein_stats = []
for i in range(min(N_FEATURES_TO_PLOT, real_users_reduced.shape[1], synth_users.shape[1])):
    real_feat = real_users_reduced[:, i]
    synth_feat = synth_users[:, i]
    ks_stat, ks_p = ks_2samp(real_feat, synth_feat)
    wd = wasserstein_distance(real_feat, synth_feat)
    ks_stats.append((i, ks_stat, ks_p))
    wasserstein_stats.append((i, wd))
    # Plot
    plt.figure(figsize=(6, 3))
    sns.kdeplot(real_feat, label='Real', color='blue')
    sns.kdeplot(synth_feat, label='Synthetic', color='orange')
    plt.title(f'Feature {i} Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'feature_{i}_kde.png'))
    plt.close()

# Save stats
pd.DataFrame(ks_stats, columns=['Feature', 'KS_stat', 'KS_p']).to_csv(os.path.join(OUTPUT_DIR, 'ks_stats.csv'), index=False)
pd.DataFrame(wasserstein_stats, columns=['Feature', 'Wasserstein']).to_csv(os.path.join(OUTPUT_DIR, 'wasserstein_stats.csv'), index=False)

# --- PCA Visualization ---
all_data = np.vstack([real_users_reduced, synth_users])
labels = np.array([0]*len(real_users_reduced) + [1]*len(synth_users))
pca = PCA(n_components=2)
all_pca = pca.fit_transform(all_data)
plt.figure(figsize=(6, 5))
plt.scatter(all_pca[labels==0, 0], all_pca[labels==0, 1], alpha=0.5, label='Real', s=10)
plt.scatter(all_pca[labels==1, 0], all_pca[labels==1, 1], alpha=0.5, label='Synthetic', s=10)
plt.legend()
plt.title('PCA: Real vs Synthetic Users')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_real_vs_synth.png'))
plt.close()

# --- Classifier-based Evaluation (Optional) ---
try:
    X = all_data
    y = labels
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    y_pred = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred)
    with open(os.path.join(OUTPUT_DIR, 'classifier_auc.txt'), 'w') as f:
        f.write(f'RandomForest AUC (real vs synthetic): {auc:.4f}\n')
    print(f'RandomForest AUC (real vs synthetic): {auc:.4f}')
except Exception as e:
    print(f'[WARNING] Classifier-based evaluation failed: {e}')

print('Post-sampling analysis complete. See output plots and stats in:', OUTPUT_DIR)
