import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance, entropy
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic, get_column_plot
from sdv.metadata import SingleTableMetadata
import matplotlib
from sdmetrics.visualization import get_column_plot as sdmetrics_get_column_plot
from c0_Configuration.s00_config_paths import (
    CHECKINS_PATH,
    CATEGORIES_XLSX,
    PLACE_ID_POI_CAT,
    SDV_OUTPUT,
    SDV_POST_SYNTHESIS_ANALYSIS_OUTPUT_DIR
)

matplotlib.use('Agg')  # Use non-interactive backend for file saving

# Paths
FSQ_PATH = CHECKINS_PATH
POI_CATEGORIES_PATH = CATEGORIES_XLSX
PLACEID_TO_CAT_PATH = PLACE_ID_POI_CAT
SYNTH_PATH = SDV_OUTPUT
RESULTS_DIR = SDV_POST_SYNTHESIS_ANALYSIS_OUTPUT_DIR 



# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load original FSQ dataset
cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon']
df = pd.read_csv(FSQ_PATH, sep='\t', header=0, names=cols)

# Load synthetic dataset
synth_df = pd.read_csv(SYNTH_PATH, sep='\t')

# --- Debug: Print columns in synthetic dataset ---
print("Synthetic dataset columns:", synth_df.columns.tolist())

# --- Fix column names in synthetic dataset if needed ---
# Try to find the correct column name for place_id
if 'place_id' not in synth_df.columns:
    # Try lower-case or other variants
    for col in synth_df.columns:
        if col.lower().replace(' ', '').replace('-', '').replace('_', '') == 'placeid':
            synth_df.rename(columns={col: 'place_id'}, inplace=True)
            print(f"Renamed column {col} to 'place_id'")
            break

# Map place_id to POI category for both real and synthetic datasets
placeid_to_cat = pd.read_csv(PLACEID_TO_CAT_PATH)
placeid_to_cat_dict = dict(zip(placeid_to_cat['place_id'], placeid_to_cat['category']))
df['POI category'] = df['place_id'].map(placeid_to_cat_dict)
synth_df['POI category'] = synth_df['place_id'].map(placeid_to_cat_dict)

# Load relevant POI categories (marked 'yes')
relevant_cats_df = pd.read_excel(POI_CATEGORIES_PATH)
cat_col = 'POI Category in Singapore'
yes_col = 'Relevant to use case '
relevant_categories = [cat.strip().lower() for cat, flag in zip(relevant_cats_df[cat_col], relevant_cats_df[yes_col]) if str(flag).strip().lower() == 'yes' and cat and str(cat).strip()]
relevant_categories = list(dict.fromkeys(relevant_categories))
df['POI category'] = df['POI category'].astype(str).str.strip().str.lower()
synth_df['POI category'] = synth_df['POI category'].astype(str).str.strip().str.lower()
df = df[df['POI category'].isin(relevant_categories)]
synth_df = synth_df[synth_df['POI category'].isin(relevant_categories)]

# Harmonize columns for comparison
common_cols = [col for col in synth_df.columns if col in df.columns]
if 'POI category' not in common_cols:
    common_cols.append('POI category')
real_eval_df = df[common_cols].copy()
synth_eval_df = synth_df[common_cols].copy()

# --- Ensure POI category column name matches exactly ---
for df_ in [real_eval_df, synth_eval_df]:
    if "POI_category" in df_.columns:
        df_.rename(columns={"POI_category": "POI category"}, inplace=True)
    if "poi_category" in df_.columns:
        df_.rename(columns={"poi_category": "POI category"}, inplace=True)

# Compare distributions of numerical features
numerical_cols = [col for col in common_cols if pd.api.types.is_numeric_dtype(real_eval_df[col])]
results = {}
for col in numerical_cols:
    real = real_eval_df[col]
    synth = synth_eval_df[col]
    ks_stat, ks_p = ks_2samp(real, synth)
    wd = wasserstein_distance(real, synth)
    results[col] = {'ks_stat': ks_stat, 'ks_p': ks_p, 'wasserstein': wd}
    plt.figure()
    plt.hist(real, bins=30, alpha=0.5, label='Real')
    plt.hist(synth, bins=30, alpha=0.5, label='Synthetic')
    plt.title(f"Distribution of {col}")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f"dist_{col}.png"))
    plt.close()

# Compare POI category frequencies
if 'POI category' in real_eval_df.columns and 'POI category' in synth_eval_df.columns:
    real_cat_freq = real_eval_df['POI category'].value_counts(normalize=True)
    synth_cat_freq = synth_eval_df['POI category'].value_counts(normalize=True)
    cat_compare = pd.DataFrame({'real': real_cat_freq, 'synthetic': synth_cat_freq}).fillna(0)
    # select the top 10 POI categories for better visualization
    top10 = cat_compare.sort_values('real', ascending=False).head(10)
    top10.plot.bar(figsize=(12,6))
    # cat_compare.plot.bar(figsize=(12,6))
    plt.title('top 10 POI Category Frequency: Real vs Synthetic')
    plt.savefig(os.path.join(RESULTS_DIR, 'top_10_poi_category_freq_comparison.png'))
    plt.close()

# SDV evaluation: compare synthetic and real data
# For SDV, columns must match and types must be compatible
# We'll use only the columns present in both datasets
common_cols = [col for col in synth_df.columns if col in df.columns]
real_eval_df = df[common_cols].copy()
synth_eval_df = synth_df[common_cols].copy()

# Build SDV metadata for evaluation using actual column names
numerical_cols = [col for col in common_cols if col != "POI category"]
meta_dict = {
    "columns": {
        **{col: {"sdtype": "numerical"} for col in numerical_cols},
        "POI category": {"sdtype": "categorical"}
    },
    "primary_key": None
}
metadata = SingleTableMetadata.load_from_dict(meta_dict)

# 1. SDV Diagnostic
print("\n=== SDV Diagnostic Report ===")
diagnostic_report = run_diagnostic(
    real_data=real_eval_df,
    synthetic_data=synth_eval_df,
    metadata=metadata)
print("Diagnostic Score:", diagnostic_report.get_score())
# Print all available property names for diagnostic_report
print("Available Diagnostic Properties:", diagnostic_report.get_properties())
# Print details for each property (skip 'Property' if present, only use valid names)
for prop in diagnostic_report.get_properties():
    try:
        details = diagnostic_report.get_details(prop)
        print(f"Diagnostic Details for {prop}:", details)
    except Exception as e:
        print(f"[ERROR] Could not get details for property '{prop}': {e}")
with open(os.path.join(RESULTS_DIR, "sdv_diagnostic_report.txt"), "w") as f:
    f.write(f"Diagnostic Score: {diagnostic_report.get_score()}\n")
    f.write("Available Diagnostic Properties: " + str(diagnostic_report.get_properties()) + "\n")
    for prop in diagnostic_report.get_properties():
        try:
            details = diagnostic_report.get_details(prop)
            f.write(f"Diagnostic Details for {prop}: {details}\n")
        except Exception as e:
            f.write(f"[ERROR] Could not get details for property '{prop}': {e}\n")
# 2. SDV Visualization (first categorical/numerical column)
for col in common_cols:
    try:
        plot_result = get_column_plot(
            real_data=real_eval_df,
            synthetic_data=synth_eval_df,
            metadata=metadata,
            column_name=col
        )
        # Handle both Figure and (Figure, Axes) return types
        fig = plot_result[0] if isinstance(plot_result, (tuple, list)) else plot_result
        if hasattr(fig, 'savefig'):
            plot_path = os.path.join(RESULTS_DIR, f"sdv_builtin_colplot_{col}.png")
            fig.savefig(plot_path)
            print(f"[INFO] Saved SDV built-in plot: {plot_path}")
            plt.close(fig)
        else:
            print(f"[WARNING] Could not save plot for column {col}: Not a matplotlib Figure.")
    except Exception as e:
        print(f"[ERROR] SDV built-in plot for column {col} failed: {e}")

# SDV evaluation: compare synthetic and real data
# For SDV, columns must match and types must be compatible
# We'll use only the columns present in both datasets
common_cols = [col for col in synth_df.columns if col in df.columns]
real_eval_df = df[common_cols].copy()
synth_eval_df = synth_df[common_cols].copy()

# Build SDV metadata for evaluation using actual column names
numerical_cols = [col for col in common_cols if col != "POI category"]
meta_dict = {
    "columns": {
        **{col: {"sdtype": "numerical"} for col in numerical_cols},
        "POI category": {"sdtype": "categorical"}
    },
    "primary_key": None
}
metadata = SingleTableMetadata.load_from_dict(meta_dict)

# Evaluate using SDV's built-in metrics
sdv_report = evaluate_quality(synth_eval_df, real_eval_df, metadata)
# Print SDV report details (overall score and per-metric details)
print("\n=== SDV Overall Quality Report ===")
print("Overall Score:", sdv_report.get_score())
print("Column Shapes:", sdv_report.get_details('Column Shapes'))
print("Column Pair Trends:", sdv_report.get_details('Column Pair Trends'))

# Save SDV report details to text file
with open(os.path.join(RESULTS_DIR, "sdv_quality_report.txt"), "w") as f:
    f.write(f"Overall Score: {sdv_report.get_score()}\n")
    f.write("Column Shapes:\n" + str(sdv_report.get_details('Column Shapes')) + "\n")
    f.write("Column Pair Trends:\n" + str(sdv_report.get_details('Column Pair Trends')) + "\n")

# --- Save SDV quality report details as CSV/tabular format ---
col_shapes = sdv_report.get_details('Column Shapes')
col_pair_trends = sdv_report.get_details('Column Pair Trends')

# Convert to DataFrame if possible and save as CSV
try:
    if isinstance(col_shapes, dict):
        col_shapes_df = pd.DataFrame.from_dict(col_shapes, orient='index')
    else:
        col_shapes_df = pd.DataFrame(col_shapes)
    col_shapes_df.to_csv(os.path.join(RESULTS_DIR, 'sdv_quality_column_shapes.csv'))
except Exception as e:
    print(f"[ERROR] Could not save Column Shapes as CSV: {e}")

try:
    if isinstance(col_pair_trends, dict):
        col_pair_trends_df = pd.DataFrame.from_dict(col_pair_trends, orient='index')
    else:
        col_pair_trends_df = pd.DataFrame(col_pair_trends)
    col_pair_trends_df.to_csv(os.path.join(RESULTS_DIR, 'sdv_quality_column_pair_trends.csv'))
except Exception as e:
    print(f"[ERROR] Could not save Column Pair Trends as CSV: {e}")

# SDV column-wise plots (saved for first 5 columns)
print(f"[INFO] Saving plots to: {RESULTS_DIR}")
for col in common_cols[:5]:
    try:
        print(f"[INFO] Plotting column: {col}")
        if pd.api.types.is_numeric_dtype(real_eval_df[col]) and pd.api.types.is_numeric_dtype(synth_eval_df[col]):
            plt.figure()
            plt.hist(real_eval_df[col].dropna(), bins=30, alpha=0.5, label='Real')
            plt.hist(synth_eval_df[col].dropna(), bins=30, alpha=0.5, label='Synthetic')
            plt.title(f"SDV Column Plot: {col}")
            plt.legend()
            plot_path = os.path.join(RESULTS_DIR, f"sdv_colplot_{col}.png")
            plt.savefig(plot_path)
            print(f"[INFO] Saved plot: {plot_path}")
            plt.close()
        elif pd.api.types.is_object_dtype(real_eval_df[col]) or pd.api.types.is_categorical_dtype(real_eval_df[col]):
            plt.figure()
            real_counts = real_eval_df[col].value_counts(normalize=True)
            synth_counts = synth_eval_df[col].value_counts(normalize=True)
            plot_df = pd.DataFrame({'Real': real_counts, 'Synthetic': synth_counts}).fillna(0)
            plot_df.plot.bar(alpha=0.7)
            plt.title(f"SDV Category Plot: {col}")
            plt.ylabel('Frequency')
            plot_path = os.path.join(RESULTS_DIR, f"sdv_colplot_{col}.png")
            plt.savefig(plot_path)
            print(f"[INFO] Saved plot: {plot_path}")
            plt.close()
    except Exception as e:
        print(f"[ERROR] Plot for column {col} failed: {e}")

# --- SDMetrics Visualization: get_column_plot ---
# Save SDMetrics plots for all columns in common_cols (distplot for numericals, bar for categoricals)
for col in common_cols:
    try:
        if pd.api.types.is_numeric_dtype(real_eval_df[col]) and pd.api.types.is_numeric_dtype(synth_eval_df[col]):
            fig = sdmetrics_get_column_plot(
                real_data=real_eval_df,
                synthetic_data=synth_eval_df,
                column_name=col,
                plot_type='distplot'
            )
            plot_path = os.path.join(RESULTS_DIR, f'sdmetrics_distplot_{col}.png')
        else:
            fig = sdmetrics_get_column_plot(
                real_data=real_eval_df,
                synthetic_data=synth_eval_df,
                column_name=col,
                plot_type='bar'
            )
            plot_path = os.path.join(RESULTS_DIR, f'sdmetrics_barplot_{col}.png')
        if hasattr(fig, 'savefig'):
            fig.savefig(plot_path)
            print(f"[INFO] Saved SDMetrics plot: {plot_path}")
            plt.close(fig)
        else:
            print(f"[WARNING] Could not save SDMetrics plot for column {col}: Not a matplotlib Figure.")
    except Exception as e:
        print(f"[ERROR] SDMetrics plot for column {col} failed: {e}")

# Save summary statistics to text file
with open(os.path.join(RESULTS_DIR, "summary_stats.txt"), "w") as f:
    f.write("=== KS Test & Wasserstein Distance for Numerical Features ===\n")
    for col, stats in results.items():
        f.write(f"{col}: KS={stats['ks_stat']:.4f} (p={stats['ks_p']:.4f}), Wasserstein={stats['wasserstein']:.4f}\n")
    f.write("\n=== SDV Overall Quality Report ===\n")
    f.write(f"Overall Score: {sdv_report.get_score()}\n")
    f.write("Column Shapes:\n" + str(sdv_report.get_details('Column Shapes')) + "\n")
    f.write("Column Pair Trends:\n" + str(sdv_report.get_details('Column Pair Trends')) + "\n")
    f.write("\n=== POI Category Frequency Comparison ===\n")
    f.write(cat_compare.head(20).to_string())

# Print summary statistics
print("\n=== KS Test & Wasserstein Distance for Numerical Features ===")
for col, stats in results.items():
    print(f"{col}: KS={stats['ks_stat']:.4f} (p={stats['ks_p']:.4f}), Wasserstein={stats['wasserstein']:.4f}")

print("\n=== POI Category Frequency Comparison ===")
print(cat_compare.head(20))

def kl_divergence(p, q):
    # p, q: arrays of probabilities (must sum to 1)
    p = np.asarray(p) + 1e-8  # avoid log(0)
    q = np.asarray(q) + 1e-8
    p = p / p.sum()
    q = q / q.sum()
    return entropy(p, q)

# --- Compare more features: lat, lon ---
if 'lat' in df.columns and 'lat' in synth_df.columns:
    plt.figure(figsize=(8,4))
    plt.hist(df['lat'].dropna(), bins=30, alpha=0.5, label='Real')
    plt.hist(synth_df['lat'].dropna(), bins=30, alpha=0.5, label='Synthetic')
    plt.title('Latitude Distribution: Real vs Synthetic')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'lat_dist_comparison.png'))
    plt.close()
    # KL divergence
    real_lat_hist, bins = np.histogram(df['lat'].dropna(), bins=30, density=True)
    synth_lat_hist, _ = np.histogram(synth_df['lat'].dropna(), bins=bins, density=True)
    kl_lat = kl_divergence(real_lat_hist, synth_lat_hist)
    print(f"KL divergence (lat): {kl_lat:.4f}")
else:
    print("[INFO] 'lat' column not found in both datasets.")

if 'lon' in df.columns and 'lon' in synth_df.columns:
    plt.figure(figsize=(8,4))
    plt.hist(df['lon'].dropna(), bins=30, alpha=0.5, label='Real')
    plt.hist(synth_df['lon'].dropna(), bins=30, alpha=0.5, label='Synthetic')
    plt.title('Longitude Distribution: Real vs Synthetic')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'lon_dist_comparison.png'))
    plt.close()
    # KL divergence
    real_lon_hist, bins = np.histogram(df['lon'].dropna(), bins=30, density=True)
    synth_lon_hist, _ = np.histogram(synth_df['lon'].dropna(), bins=bins, density=True)
    kl_lon = kl_divergence(real_lon_hist, synth_lon_hist)
    print(f"KL divergence (lon): {kl_lon:.4f}")
else:
    print("[INFO] 'lon' column not found in both datasets.")

# Save KL divergence results to summary_stats.txt
with open(os.path.join(RESULTS_DIR, "summary_stats.txt"), "a") as f:
    if 'lat' in df.columns and 'lat' in synth_df.columns:
        f.write(f"\nKL divergence (lat): {kl_lat:.4f}\n")
    if 'lon' in df.columns and 'lon' in synth_df.columns:
        f.write(f"KL divergence (lon): {kl_lon:.4f}\n")

# --- Compare histogram: number of check-ins per user per POI ---
# Ensure 'user_id' and 'place_id' exist in synthetic dataset
missing_cols = []
for col in ['user_id', 'place_id']:
    if col not in synth_df.columns:
        missing_cols.append(col)
if missing_cols:
    print(f"[ERROR] Columns missing in synthetic dataset: {missing_cols}. Skipping check-ins per user per POI comparison.")
else:
    real_counts = df.groupby(['user_id', 'place_id']).size()
    synth_counts = synth_df.groupby(['user_id', 'place_id']).size()

    plt.figure(figsize=(8,4))
    plt.hist(real_counts, bins=30, alpha=0.5, label='Real')
    plt.hist(synth_counts, bins=30, alpha=0.5, label='Synthetic')
    plt.title('Check-ins per User per POI: Real vs Synthetic')
    plt.xlabel('Number of Check-ins')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'checkins_per_user_per_poi_hist.png'))
    plt.close()

    # Save summary statistics for check-ins per user per POI
    with open(os.path.join(RESULTS_DIR, "summary_stats.txt"), "a") as f:
        f.write("\n=== Check-ins per User per POI Histogram Stats ===\n")
        f.write(f"Real: mean={real_counts.mean():.2f}, std={real_counts.std():.2f}, min={real_counts.min()}, max={real_counts.max()}\n")
        f.write(f"Synthetic: mean={synth_counts.mean():.2f}, std={synth_counts.std():.2f}, min={synth_counts.min()}, max={synth_counts.max()}\n")

# --- Compare unique user_ids in real and synthetic datasets ---
real_user_ids = set(df['user_id'].astype(str).unique())
synth_user_ids = set(synth_df['user_id'].astype(str).unique())
new_in_synth = synth_user_ids - real_user_ids
missing_in_synth = real_user_ids - synth_user_ids
print(f"\n[USER_ID COMPARISON]")
print(f"Total unique user_id in real: {len(real_user_ids)}")
print(f"Total unique user_id in synthetic: {len(synth_user_ids)}")
print(f"Overlap: {len(real_user_ids & synth_user_ids)}")
print(f"User IDs in synthetic but not in real: {len(new_in_synth)}")
if new_in_synth:
    print(f"Example new user_ids in synthetic: {list(new_in_synth)[:10]}")
print(f"User IDs in real but not in synthetic: {len(missing_in_synth)}")
if missing_in_synth:
    print(f"Example missing user_ids in synthetic: {list(missing_in_synth)[:10]}")

# --- Compare user_id distribution: frequency of check-ins per user ---
real_user_counts = df['user_id'].value_counts().sort_index()
synth_user_counts = synth_df['user_id'].value_counts().sort_index()

# Plot top N users for readability
N = 30
real_top = real_user_counts.sort_values(ascending=False).head(N)
synth_top = synth_user_counts.sort_values(ascending=False).head(N)
plt.figure(figsize=(14,6))
plt.bar(real_top.index.astype(str), real_top.values, alpha=0.6, label='Real', color='tab:blue')
plt.bar(synth_top.index.astype(str), synth_top.values, alpha=0.6, label='Synthetic', color='tab:orange')
plt.title(f'Check-ins per User (Top {N} Users)')
plt.xlabel('user_id')
plt.ylabel('Number of Check-ins')
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f'user_id_checkin_freq_top{N}_1.png'))
plt.close()

# Plot full distribution as histogram (log scale for y)
plt.figure(figsize=(8,5))
plt.hist(real_user_counts, bins=30, alpha=0.5, label='Real')
plt.hist(synth_user_counts, bins=30, alpha=0.5, label='Synthetic')
plt.yscale('log')
plt.title('Distribution of Check-ins per User (Log Scale)')
plt.xlabel('Number of Check-ins')
plt.ylabel('Number of Users (log)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'user_id_checkin_hist_log_2.png'))
plt.close()

# Summary statistics
real_stats = real_user_counts.describe()
synth_stats = synth_user_counts.describe()
with open(os.path.join(RESULTS_DIR, "summary_stats.txt"), "a") as f:
    f.write("\n=== Check-ins per User Statistics ===\n")
    f.write(f"Real: count={real_stats['count']}, mean={real_stats['mean']:.2f}, std={real_stats['std']:.2f}, min={real_stats['min']}, max={real_stats['max']}\n")
    f.write(f"Synthetic: count={synth_stats['count']}, mean={synth_stats['mean']:.2f}, std={synth_stats['std']:.2f}, min={synth_stats['min']}, max={synth_stats['max']}\n")

# KL divergence (align user_id sets)
all_user_ids = sorted(set(real_user_counts.index) | set(synth_user_counts.index))
real_freq = np.array([real_user_counts.get(uid, 0) for uid in all_user_ids])
synth_freq = np.array([synth_user_counts.get(uid, 0) for uid in all_user_ids])
if real_freq.sum() > 0 and synth_freq.sum() > 0:
    real_prob = real_freq / real_freq.sum()
    synth_prob = synth_freq / synth_freq.sum()
    kl_user = kl_divergence(real_prob, synth_prob)
    print(f"KL divergence (user_id distribution): {kl_user:.4f}")
    with open(os.path.join(RESULTS_DIR, "summary_stats.txt"), "a") as f:
        f.write(f"KL divergence (user_id distribution): {kl_user:.4f}\n")
else:
    print("[WARNING] Could not compute KL divergence for user_id distribution (empty frequency array).")
