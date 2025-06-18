import pandas as pd
from datetime import datetime

# File paths
org_path = 'analysis_older_dataset/Final_code/syn_org_eval/original_checkins.txt'
syn_path = 'analysis_older_dataset/Final_code/syn_org_eval/synthetic_checkins.txt'

# Read original and synthetic check-ins
def read_checkins(path, is_synthetic=False):
    if is_synthetic:
        df = pd.read_csv(path, sep=',', header=0, dtype=str)
        # Drop the synthetic_user_id column if present
        if 'synthetic_user_id' in df.columns:
            df = df.drop(columns=['synthetic_user_id'])
    else:
        df = pd.read_csv(path, sep='\t', header=None, names=['user_id','place_id','datetime','timezone','lat','lon'], dtype=str)
    # Parse datetime for sorting/analysis
    try:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    except Exception:
        pass
    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    return df

org = read_checkins(org_path)
syn = read_checkins(syn_path, is_synthetic=True)

# Basic stats
print('Original check-ins:', len(org))
print('Synthetic check-ins:', len(syn))
print('Original users:', org.user_id.nunique())
print('Synthetic users:', syn.user_id.nunique())
print('Original POIs:', org.place_id.nunique())
print('Synthetic POIs:', syn.place_id.nunique())

# Check-in count per user
org_user_counts = org.groupby('user_id').size()
syn_user_counts = syn.groupby('user_id').size()

print('\nCheck-in count per user (original):')
print(org_user_counts.describe())
print('\nCheck-in count per user (synthetic):')
print(syn_user_counts.describe())

# Check-in count per POI
org_poi_counts = org.groupby('place_id').size()
syn_poi_counts = syn.groupby('place_id').size()

print('\nCheck-in count per POI (original):')
print(org_poi_counts.describe())
print('\nCheck-in count per POI (synthetic):')
print(syn_poi_counts.describe())

# Temporal distribution (by hour)
org['hour'] = org['datetime'].dt.hour
syn['hour'] = syn['datetime'].dt.hour
print('\nOriginal check-ins by hour:')
print(org['hour'].value_counts().sort_index())
print('\nSynthetic check-ins by hour:')
print(syn['hour'].value_counts().sort_index())

# Save summary CSVs for further analysis
org_user_counts.to_csv('analysis_older_dataset/Final_code/syn_org_eval/org_user_checkin_counts.csv')
syn_user_counts.to_csv('analysis_older_dataset/Final_code/syn_org_eval/syn_user_checkin_counts.csv')
org_poi_counts.to_csv('analysis_older_dataset/Final_code/syn_org_eval/org_poi_checkin_counts.csv')
syn_poi_counts.to_csv('analysis_older_dataset/Final_code/syn_org_eval/syn_poi_checkin_counts.csv')
