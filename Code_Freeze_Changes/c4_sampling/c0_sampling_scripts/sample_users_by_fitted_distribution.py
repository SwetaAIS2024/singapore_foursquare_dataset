import os
import json
import numpy as np
from scipy.stats import norm, gamma, beta, lognorm, expon, uniform, cauchy, laplace, rayleigh, genextreme

# Map distfit names to scipy distributions
SCIPY_DISTRIBUTIONS = {
    'norm': norm,
    'gamma': gamma,
    'beta': beta,
    'lognorm': lognorm,
    'expon': expon,
    'uniform': uniform,
    'cauchy': cauchy,
    'laplace': laplace,
    'rayleigh': rayleigh,
    'genextreme': genextreme
}

def load_cluster_analysis(analysis_dir):
    """Load all cluster analysis results from a single all_clusters_analysis.json file if present, else fallback to per-cluster files."""
    all_clusters_path = os.path.join(analysis_dir, "all_clusters_analysis.json")
    if os.path.exists(all_clusters_path):
        with open(all_clusters_path, 'r') as f:
            clusters = json.load(f)
        return clusters
    # Fallback to per-cluster files
    clusters = []
    for fname in os.listdir(analysis_dir):
        if fname.startswith('cluster_') and fname.endswith('analysis_results.json'):
            with open(os.path.join(analysis_dir, fname), 'r') as f:
                clusters.append(json.load(f))
    return clusters

def proportional_allocation(cluster_sizes, total_samples=100):
    """Allocate samples to clusters proportionally to their size."""
    total = sum(cluster_sizes)
    raw_alloc = [size / total * total_samples for size in cluster_sizes]
    alloc = [int(np.floor(x)) for x in raw_alloc]
    # Distribute remaining samples
    remainder = total_samples - sum(alloc)
    if remainder > 0:
        # Assign extra samples to clusters with largest remainder
        remainders = np.array(raw_alloc) - np.array(alloc)
        for i in np.argsort(-remainders)[:remainder]:
            alloc[i] += 1
    return alloc

def sample_from_distribution(distribution, params, n, random_state=None):
    """Sample n values from a fitted distribution (scipy)."""
    if distribution not in SCIPY_DISTRIBUTIONS:
        raise ValueError(f"Distribution {distribution} not supported.")
    dist = SCIPY_DISTRIBUTIONS[distribution]
    # Params may be dict or list
    if isinstance(params, dict):
        # Try to unpack as shape, loc, scale
        shape = []
        loc = params.get('loc', 0)
        scale = params.get('scale', 1)
        for k in params:
            if k not in ('loc', 'scale'):
                shape.append(params[k])
        args = tuple(shape) + (loc, scale)
    else:
        args = tuple(params)
    rng = np.random.default_rng(random_state)
    return dist.rvs(*args, size=n, random_state=rng)

def main():
    # User config
    algo = 'kmeans'  # Change as needed
    analysis_dir = f"c2_clustering/c3_individual_cluster_analysis/cluster_analysis/{algo}"
    # Ensure output directory exists (robust to working directory)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "../c1_sampling_outputs"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Load cluster analysis
    clusters = load_cluster_analysis(analysis_dir)
    clusters = [c for c in clusters if c['method'] == 'distribution_fitting']
    cluster_sizes = [c['size'] for c in clusters]
    alloc = proportional_allocation(cluster_sizes, total_samples=100)
    print(f"Sample allocation per cluster: {alloc}")
    # Load user vectors and labels
    # (Assume same logic as in s07_individual_cluster_analysis.py)
    from c0_config.s00_config_paths import CLUSTER_OUTPUT_DIR, FINAL_INPUT_DATASET
    import scipy.sparse
    import json as js
    meta_path = os.path.join(FINAL_INPUT_DATASET, "matrix_metadata.json")
    with open(meta_path, "r") as f:
        metadata = js.load(f)
    batch_files = [os.path.join(FINAL_INPUT_DATASET, os.path.basename(f)) for f in metadata.get("batch_files", [])]
    user_vectors = []
    for batch_file in batch_files:
        if not os.path.exists(batch_file):
            continue
        batch_vectors = scipy.sparse.load_npz(batch_file).toarray()
        user_vectors.append(batch_vectors)
    user_vectors = np.vstack(user_vectors)
    # Remove zero columns as in clustering
    nonzero_cols = np.any(user_vectors != 0, axis=0)
    user_vectors = user_vectors[:, nonzero_cols]
    # Load cluster labels
    labels = np.load(os.path.join(CLUSTER_OUTPUT_DIR, algo, "user_cluster_labels.npy"))
    synthetic_users = []  # Collect synthetic user vectors here
    # For each cluster, sample users
    sampled_indices = []
    for c, n_samples in zip(clusters, alloc):
        cluster_id = c['cluster_id']
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        if n_samples == 0 or len(cluster_indices) == 0:
            continue
        # If distribution fit info is available, sample synthetic users
        if c['dimensions']:
            dim_samples = []
            for dim in c['dimensions']:
                dist_name = dim['distribution']['name']
                params = dim['distribution']['parameters']
                try:
                    dim_sample = sample_from_distribution(dist_name, params, n_samples)
                except Exception as e:
                    print(f"[WARNING] Could not sample from {dist_name} for cluster {cluster_id}: {e}")
                    # Fallback: sample from empirical data
                    dim_data = user_vectors[cluster_indices, dim['dimension']]
                    dim_sample = np.random.choice(dim_data, size=n_samples, replace=True)
                dim_samples.append(dim_sample)
            # Stack to shape (n_samples, n_dims)
            cluster_synth = np.column_stack(dim_samples)
            synthetic_users.append(cluster_synth)
            # Find closest real users to synthetic samples (optional, else just keep synthetic)
            # Here, we just select random real users for now
            chosen = np.random.choice(cluster_indices, size=n_samples, replace=False) if len(cluster_indices) >= n_samples else np.random.choice(cluster_indices, size=n_samples, replace=True)
        else:
            # Fallback: random sampling from cluster
            chosen = np.random.choice(cluster_indices, size=n_samples, replace=False) if len(cluster_indices) >= n_samples else np.random.choice(cluster_indices, size=n_samples, replace=True)
            sampled_indices.extend(chosen.tolist())
    # Save sampled user indices
    output_path = os.path.join(OUTPUT_DIR, f"sampled_user_indices_{algo}.npy")
    np.save(output_path, np.array(sampled_indices))
    print(f"Saved sampled user indices to {output_path}")
    # Save synthetic users as CSV if any were generated
    if synthetic_users:
        all_synth = np.vstack(synthetic_users)
        synth_csv_path = os.path.join(OUTPUT_DIR, f"sampled_synthetic_users_{algo}.csv")
        # Add dummy user IDs as first column
        n_samples = all_synth.shape[0]
        user_ids = np.arange(1, n_samples + 1)
        all_synth_with_ids = np.column_stack((user_ids, all_synth))
        n_features = all_synth.shape[1]
        header = 'UserID,' + ','.join([f'Feature_{i}' for i in range(n_features)])
        np.savetxt(synth_csv_path, all_synth_with_ids, delimiter=",", fmt="%d" + ",%.6f" * n_features, header=header, comments='')
        print(f"Saved synthetic user feature matrix to {synth_csv_path}")

if __name__ == "__main__":
    main()
