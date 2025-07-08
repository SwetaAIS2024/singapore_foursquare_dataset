# Singapore Foursquare Dataset Analysis

This project performs user clustering and distribution analysis on Foursquare check-in data for Singapore, followed by synthetic user sampling based on fitted distributions.

---

## Pipeline Overview

### 1. Configuration

- All key parameters (number of users, clusters, SVD components, thresholds, etc.) are set in `c0_config/config.json`.
- Example config options:
  - `n_users`, `n_clusters`, `svd_components`, `clustering_algo`, `distfit_scaled_rss_threshold`, etc.

### 2. Data Preparation

- User vectors are loaded from sparse `.npz` batch files in `c1_data/c2_output/`.
- Zero columns (features) are removed.
- Dimensionality reduction is performed using TruncatedSVD (default: 100 components).
- Features are scaled (StandardScaler) and filtered for high variance.
- Final processed data shape is typically `(n_users, n_features)` (e.g., 2000 x 100).

### 3. Clustering

- User vectors are clustered using the selected algorithm (`kmeans`, `agg`, `dbscan`, or `gmm`), as specified in the config.
- Cluster labels are saved for each user in `c2_clustering/c1_output/{algo}/user_cluster_labels.npy`.
- Clustering output is stored in `c2_clustering/c0_clustering/`.

### 4. Individual Cluster Analysis

- For each cluster:
  - If the cluster is small, random sampling is used for synthetic data generation.
  - For larger clusters:
    - Each significant dimension (above a variance threshold) is analyzed.
    - Distribution fitting is performed using `distfit` on each dimension.
      - For AGG clusters, if the cluster is very large, a random sample of users is used for fitting (default: 50).
      - Stricter range and outlier checks are applied to avoid memory issues.
      - Only appropriate distributions are fitted based on data properties (e.g., positive-only).
      - The number of histogram bins is reduced (default: 5) to prevent OOM errors.
    - Fit results, statistics, and plots are saved per cluster.
- Results are saved as JSON in `c3_individual_cluster_analysis/cluster_analysis/{algo}/`.

### 5. Summary and Visualization

- Summary plots and tables are generated:
  - Cluster sizes
  - Number of successfully fitted dimensions per cluster
  - Best-fitting distributions and fit quality
- Summaries are saved as CSV and PNG files.

### 6. Synthetic User Sampling

- Scripts in `c4_sampling/c0_sampling_scripts/` use the fitted distributions per cluster to sample synthetic users.
- Sampling is proportional to cluster size, with a total of 100 users sampled across all clusters.
- Sampling uses the best-fit distribution parameters for each cluster/dimension, or falls back to random sampling if needed.

---

## Key Files & Directories

- `c0_config/config.json` — Main configuration file.
- `c2_clustering/c0_clustering/` — Clustering outputs.
- `c3_individual_cluster_analysis/` — Per-cluster distribution analysis and plots.
- `c4_sampling/c0_sampling_scripts/` — Scripts for synthetic user sampling.
- `README.md` — This documentation.

---

## Notes

- The pipeline is robust to memory issues by limiting cluster sample sizes, reducing histogram bins, and skipping problematic dimensions.
- All major steps print informative logs and warnings for traceability.
- The code is modular and can be extended for new clustering or sampling strategies.

---
