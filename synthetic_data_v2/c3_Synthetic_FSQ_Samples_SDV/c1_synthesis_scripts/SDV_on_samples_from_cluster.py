import os
import pandas as pd
from sdv.single_table import CopulaGANSynthesizer,CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.metadata import Metadata  # Updated import
from sdv.metadata import MultiTableMetadata  # Import for multi-table metadata if needed
import numpy as np

# Paths
from synthetic_data_v2.c0_Configuration.config_paths import (
    CATEGORIES_XLSX,
    PLACE_ID_POI_CAT,
    SDV_OUTPUT_DIR,
    SDV_OUTPUT_FILE,
    OUTPUT_SAMPLED_FSQ_PATH
)


def synth_datetime(row):
    # Example: "Mon Jul 02 11:23:45 +0000 2012"
    # You can set year as needed, e.g., 2012
    import calendar
    dayofweek = int(row['dayofweek'])
    month = int(row['month'])
    hour = int(row['hour'])
    minute = int(row['minute'])
    second = int(row['second'])
    # Find first day in month/year with correct weekday
    year = 2012
    for day in range(1, 8):
        if pd.Timestamp(year=year, month=month, day=day).dayofweek == dayofweek:
            break
    day_name = calendar.day_name[dayofweek]
    month_name = calendar.month_abbr[month]
    return f"{day_name[:3]} {month_name} {day:02d} {hour:02d}:{minute:02d}:{second:02d} +0000 {year}"



# # Load FSQ dataset
# cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon']
# df = pd.read_csv(FSQ_PATH, sep='\t', header=0, names=cols)

df = pd.read_csv(OUTPUT_SAMPLED_FSQ_PATH, sep='\t')
df = df.drop(columns = ['cluster_id', 'sampled_count'])  # Remove cluster_id if present

# Map place_id to POI category
placeid_to_cat = pd.read_csv(PLACE_ID_POI_CAT)
placeid_to_cat_dict = dict(zip(placeid_to_cat['place_id'], placeid_to_cat['category']))
df['POI category'] = df['place_id'].map(placeid_to_cat_dict)

# Load relevant POI categories (marked 'yes')
relevant_cats_df = pd.read_excel(CATEGORIES_XLSX)
cat_col = 'POI Category in Singapore'
yes_col = 'Relevant to use case '
relevant_categories = [cat.strip().lower() for cat, flag in zip(relevant_cats_df[cat_col], relevant_cats_df[yes_col]) if str(flag).strip().lower() == 'yes' and cat and str(cat).strip()]
relevant_categories = list(dict.fromkeys(relevant_categories))
print(f"[INFO] Number of relevant categories: {len(relevant_categories)}")
print(f"[INFO] Example relevant categories: {relevant_categories[:10]}")

# Normalize POI category names for robust filtering
df['POI category'] = df['POI category'].astype(str).str.strip().str.lower()
df = df[df['POI category'].isin(relevant_categories)]
print(f"[INFO] Number of rows after filtering by relevant POI categories: {len(df)}")
print(f"[INFO] Unique POI categories in filtered FSQ: {df['POI category'].unique()}")

# After filtering by relevant POI categories
# Select users with more than 50 check-ins only
user_counts = df['user_id'].value_counts()
active_users = user_counts[user_counts > 50].index
filtered_df = df[df['user_id'].isin(active_users)]

print(f"[INFO] Number of users with >50 check-ins: {len(active_users)}")
print(f"[INFO] Synthetic sample generation will use {len(filtered_df['user_id'].unique())} users and {len(filtered_df)} check-ins.")

# Limit to 5000 check-ins
if len(filtered_df) > 5000:
    filtered_df = filtered_df.sample(n=5000, random_state=42)
print(f"[INFO] Synthetic sample generation will use {len(filtered_df['user_id'].unique())} users and {len(filtered_df)} check-ins.")

# Minimal preprocessing: select columns in the same order as the original dataset (include POI category in SVD data)
sdv_df = filtered_df[['user_id', 'place_id', 'POI category', 'datetime', 'timezone', 'lat', 'lon']].copy()

# --- Encode datetime into hour, minute, second, dayofweek, and month ---
sdv_df['datetime'] = pd.to_datetime(sdv_df['datetime'], errors='coerce')
sdv_df['hour'] = sdv_df['datetime'].dt.hour
sdv_df['minute'] = sdv_df['datetime'].dt.minute
sdv_df['second'] = sdv_df['datetime'].dt.second
sdv_df['dayofweek'] = sdv_df['datetime'].dt.dayofweek
sdv_df['month'] = sdv_df['datetime'].dt.month
sdv_df = sdv_df.drop(columns=['datetime'])

# --- Convert high-cardinality columns to category dtype ---
for col in ['user_id', 'place_id', 'POI category']:
    sdv_df[col] = sdv_df[col].astype('category')
# Ensure timezone is numeric
dtype_timezone = pd.api.types.infer_dtype(sdv_df['timezone'])
if dtype_timezone not in ['integer', 'floating']:
    sdv_df['timezone'] = pd.to_numeric(sdv_df['timezone'], errors='coerce')

# --- Reduce sample size if needed for memory efficiency ---
max_rows = 300000
if len(sdv_df) > max_rows:
    sdv_df = sdv_df.sample(n=max_rows, random_state=42)
    print(f"[INFO] Reduced SVD data sample size to {max_rows} rows for memory efficiency.")

# --- Define num_cols and cat_cols before any dtype or column operations ---
num_cols = ['lat', 'lon', 'hour', 'minute', 'second', 'dayofweek', 'month', 'timezone']
cat_cols = ['user_id', 'place_id', 'POI category']

# --- Drop constant columns (e.g., timezone) ---
if 'timezone' in sdv_df.columns and sdv_df['timezone'].nunique() == 1:
    print("[INFO] Dropping constant column 'timezone'")
    sdv_df = sdv_df.drop(columns=['timezone'])
    num_cols = [col for col in num_cols if col != 'timezone']

# --- Ensure correct dtypes for categorical columns ---
for col in cat_cols:
    sdv_df[col] = sdv_df[col].astype('category')

# --- Ensure correct dtypes for numerical columns and handle NaN/inf ---
for col in num_cols:
    sdv_df[col] = pd.to_numeric(sdv_df[col], errors='coerce')
    if not np.all(np.isfinite(sdv_df[col])):
        print(f"[ERROR] Non-finite values in {col}. Dropping rows.")
        sdv_df = sdv_df[np.isfinite(sdv_df[col])]

# --- SDV requires categorical columns to be object dtype, not category ---
for col in cat_cols:
    sdv_df[col] = sdv_df[col].astype('object')

# --- Ensure place_id is treated as categorical/object everywhere ---
sdv_df['place_id'] = sdv_df['place_id'].astype('object')

# --- Update metadata after all dtype changes ---
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(sdv_df)
for col in num_cols:
    metadata.update_column(col, sdtype="numerical")
for col in cat_cols:
    metadata.update_column(col, sdtype="categorical")

try:
    # synthesizer = CopulaGANSynthesizer(metadata, epochs=100)
    synthesizer = CTGANSynthesizer(metadata, epochs=100)
    print(f"[DEBUG] Using synthesizer: {type(synthesizer)}")
    synthesizer.fit(sdv_df)
except Exception as e:
    print(f"[ERROR] Failed to fit synthesizer on original FSQ dataset: {e}")
    exit(1)


# Ensure synthetic data only picks place_id and POI category pairs from the original data
# Create a mapping of valid place_id to POI category
valid_placeid_to_cat = filtered_df[['place_id', 'POI category']].drop_duplicates()
valid_placeid_cat_set = set(tuple(x) for x in valid_placeid_to_cat.values)

# After generating synthetic data, no need to map POI category, just filter for valid pairs
# n_samples = len(sdv_df)  # Use the same number of samples as the original data
# synth_df = synthesizer.sample(num_rows=n_samples, batch_size=50)


desired_samples = len(sdv_df)
oversample_factor = 10  # Try 10x the desired number
n_samples = desired_samples * oversample_factor
synth_df = synthesizer.sample(num_rows=n_samples, batch_size=50)

# Filter for valid pairs
synth_df = synth_df[synth_df[['place_id', 'POI category']].apply(tuple, axis=1).isin(valid_placeid_cat_set)]

# Downsample to desired number if needed
if len(synth_df) > desired_samples:
    synth_df = synth_df.sample(n=desired_samples, random_state=42)
elif len(synth_df) < desired_samples:
    print(f"[WARNING] Only {len(synth_df)} valid synthetic samples after filtering. Consider increasing oversample_factor or relaxing constraints.")

# After generating synthetic data, convert categories to string/object for saving
for col in cat_cols:
    synth_df[col] = synth_df[col].astype(str)

synth_df['datetime'] = synth_df.apply(synth_datetime, axis=1)

output_cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon']
synth_df[output_cols].to_csv(os.path.join(SDV_OUTPUT_DIR, SDV_OUTPUT_FILE), sep='\t', index=False)
print(f"[INFO] Saved synthetic samples from original FSQ dataset to {os.path.join(SDV_OUTPUT_DIR, SDV_OUTPUT_FILE)}")
