import pandas as pd
from geopy.distance import geodesic
from datetime import datetime

# Load check-in data
cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon']
df = pd.read_csv('analysis_older_dataset/singapore_checkins_filtered_with_locations_coord.txt', sep='\t', names=cols)

# Sort by user and time
# Parse datetime
try:
    df['datetime'] = pd.to_datetime(df['datetime'])
except Exception:
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.sort_values(['user_id', 'datetime'])

# Extract travel segments
segments = []
for user, group in df.groupby('user_id'):
    group = group.sort_values('datetime')
    prev = None
    for idx, row in group.iterrows():
        if prev is not None:
            origin = prev
            dest = row
            time_delta = (dest['datetime'] - origin['datetime']).total_seconds() / 60.0  # minutes
            dist = geodesic((origin['lat'], origin['lon']), (dest['lat'], dest['lon'])).km
            segments.append({
                'user_id': user,
                'origin_place_id': origin['place_id'],
                'origin_lat': origin['lat'],
                'origin_lon': origin['lon'],
                'origin_time': origin['datetime'],
                'dest_place_id': dest['place_id'],
                'dest_lat': dest['lat'],
                'dest_lon': dest['lon'],
                'dest_time': dest['datetime'],
                'travel_minutes': time_delta,
                'travel_km': dist
            })
        prev = row

# Save all travel segments
segments_df = pd.DataFrame(segments)
segments_df.to_csv('synthetic_dataset/user_travel_segments.csv', index=False)

# Aggregate: most common OD pairs
od_counts = segments_df.groupby(['origin_place_id', 'dest_place_id']).size().reset_index(name='count')
od_counts = od_counts.sort_values('count', ascending=False)
od_counts.to_csv('synthetic_dataset/most_common_od_pairs.csv', index=False)

print('Travel pattern extraction complete.')
print('Travel segments saved to synthetic_dataset/user_travel_segments.csv')
print('Most common OD pairs saved to synthetic_dataset/most_common_od_pairs.csv')
