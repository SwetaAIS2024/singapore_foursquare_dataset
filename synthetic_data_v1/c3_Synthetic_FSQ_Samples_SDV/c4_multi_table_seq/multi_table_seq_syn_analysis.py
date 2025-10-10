import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sdv.multi_table import HMASynthesizer
from sdv.metadata import MultiTableMetadata, SingleTableMetadata
from sdv.sequential import PARSynthesizer

# --- CONFIG ---
from c0_Configuration.s00_config_paths import (
    CHECKINS_PATH,
    CATEGORIES_XLSX,
    PLACE_ID_POI_CAT,
    MULTI_TABLE_SEQ_SYN_ANALYSIS_OUTPUT_DIR,
    OUTPUT_SAMPLED_FSQ_PATH
)

RESULTS_DIR = MULTI_TABLE_SEQ_SYN_ANALYSIS_OUTPUT_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)

topN = 10 # Number of top categories to consider

# --- LOAD DATA ---
# Only check-ins file is available, no header in file
cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon']
checkins_df = pd.read_csv(CHECKINS_PATH, sep='\t', header=None, names=cols)
# checkins_df = pd.read_csv(OUTPUT_SAMPLED_FSQ_PATH, sep='\t', header=None, names=cols)

# Map place_id to POI category
placeid_to_cat = pd.read_csv(PLACE_ID_POI_CAT)
placeid_to_cat_dict = dict(zip(placeid_to_cat['place_id'], placeid_to_cat['category']))
checkins_df['POI category'] = checkins_df['place_id'].map(placeid_to_cat_dict)

# Filter relevant POI categories
relevant_cats_df = pd.read_excel(CATEGORIES_XLSX)
cat_col = 'POI Category in Singapore'
yes_col = 'Relevant to use case '
relevant_categories = [cat.strip().lower() for cat, flag in zip(relevant_cats_df[cat_col], relevant_cats_df[yes_col]) if str(flag).strip().lower() == 'yes' and cat and str(cat).strip()]
relevant_categories = list(dict.fromkeys(relevant_categories))
checkins_df['POI category'] = checkins_df['POI category'].astype(str).str.strip().str.lower()
checkins_df = checkins_df[checkins_df['POI category'].isin(relevant_categories)]

topN_cats = checkins_df['POI category'].value_counts().head(topN).index.tolist()
filtered_df = checkins_df[checkins_df['POI category'].isin(topN_cats)]
checkins_df = checkins_df.sample(n=300, random_state=42)  # Sample for faster processin
# for 1000 samples, the PARsynthesizer is very slow, cannot scale this to larger datasets sizes
# works ok for 100 samples - around 1 minute with topN as 10.

print("[INFO] Loaded check-ins data with shape:", checkins_df.shape)

# --- MULTI-TABLE SYNTHESIS (HMA) ---
# Since only checkins are available, treat as single-table for HMA
data_dict = {
    'checkins': checkins_df
}

metadata = MultiTableMetadata()
metadata.detect_from_dataframes(data_dict)
# remove the primary key if autodetected
metadata.tables['checkins'].primary_key = None

synthesizer = HMASynthesizer(metadata)
print("[INFO] Fitting HMASynthesizer on check-ins data...")
synthesizer.fit(data_dict)
synthetic_data = synthesizer.sample()

# Save synthetic check-ins table
synthetic_data['checkins'].to_csv(os.path.join(RESULTS_DIR, "HMASynthesizer_output.csv"), index=False)
print("[INFO] Synthetic check-ins data saved.")

# --- SEQUENTIAL SYNTHESIS (PAR) FOR TIMESTAMPS ---
# Generate timestamp sequences for each user from checkins
par_metadata = SingleTableMetadata()
# par_metadata.detect_from_dataframe(checkins_df)
par_metadata.detect_from_dataframe(synthetic_data['checkins'])
par_metadata.primary_key = None # No primary key for PAR
par_metadata.sequence_key = 'user_id'  # <-- Set sequence key for PAR
par_metadata.sequence_index = 'datetime'  # <-- Set sequence index directly
par_metadata.update_column('datetime', sdtype='datetime')

# visualization and validation of metadata
print("[INFO] Visualizing and validating PAR metadata...")
# print(par_metadata)
par_metadata.visualize()
# par_metadata.validate()

par = PARSynthesizer(par_metadata)
print("[INFO] Fitting PARSynthesizer for timestamp sequences...")
# par.fit(checkins_df, entity_columns=['user_id'], sequence_index='datetime')
par.fit(checkins_df)
synth_sequences = par.sample(num_sequences=100)
synth_sequences.to_csv(os.path.join(RESULTS_DIR, "PARSynthesizer_output.csv"), index=False)
print("[INFO] Synthetic timestamp sequences saved.")

# --- ANALYSIS: Compare distributions ---
def plot_and_save_hist(real, synth, col, results_dir):
    plt.figure()
    plt.hist(real.dropna(), bins=30, alpha=0.5, label='Real')
    plt.hist(synth.dropna(), bins=30, alpha=0.5, label='Synthetic')
    plt.title(f"Distribution of {col}: Real vs Synthetic")
    plt.legend()
    plt.savefig(os.path.join(results_dir, f"dist_{col}.png"))
    plt.close()

# Compare lat/lon distributions in checkins
for col in ['lat', 'lon']:
    if col in checkins_df.columns and col in synthetic_data['checkins'].columns:
        plot_and_save_hist(checkins_df[col], synthetic_data['checkins'][col], col, RESULTS_DIR)

checkins_df['datetime'] = pd.to_datetime(checkins_df['datetime'], errors='coerce')
checkins_df = checkins_df.sort_values(['user_id', 'datetime'])
# Compare POI category frequencies (top 10)
if 'POI category' in checkins_df.columns and 'POI category' in synthetic_data['checkins'].columns:
    real_cat_freq = checkins_df['POI category'].value_counts(normalize=True)
    synth_cat_freq = synthetic_data['checkins']['POI category'].value_counts(normalize=True)
    cat_compare = pd.DataFrame({'real': real_cat_freq, 'synthetic': synth_cat_freq}).fillna(0)
    top10 = cat_compare.sort_values('real', ascending=False).head(10)
    top10.plot.bar(figsize=(12,6))
    plt.title('POI Category Frequency (Top 10): Real vs Synthetic')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'poi_category_freq_top10_comparison.png'))
    plt.close()

print("[INFO] Multi-table and sequential synthesis analysis complete. See results in:", RESULTS_DIR)