import pandas as pd
import os

# Load cluster summaries and timewise category mapping
profile_dir = 'analysis_older_dataset/analysis_set4/codes/Clustering_Profiling/cluster_profiles'
timewise = pd.read_csv('analysis_older_dataset/analysis_set4/codes/Clustering_Profiling/cluster_timewise_category.csv')

final_output = {}
for fname in os.listdir(profile_dir):
    if fname.endswith('_summary.txt'):
        cluster_id = fname.split('_')[1]
        with open(os.path.join(profile_dir, fname)) as f:
            lines = f.readlines()
        top_cats = eval(lines[2].split(':', 1)[1].strip())
        # Time profile (peak hours) - removed erroneous npy read
        # For simplicity, just note if peak is in morning, afternoon, evening, or night
        # Use timewise mapping for a few key hours
        timewise_map = timewise[timewise['cluster_id'] == cluster_id]
        timewise_dict = dict(zip(timewise_map['hour'], timewise_map['top_category']))
        # Heuristic label
        label = 'General'
        if 'Nightlife' in top_cats[0]:
            label = 'Nightlife Lovers'
        elif 'Education' in top_cats[0] or 'Office' in top_cats[0]:
            label = 'Students/Professionals'
        elif 'Mall' in top_cats[0] or 'Shopping' in top_cats[0]:
            label = 'Weekend Mall Visitors'
        final_output[cluster_id] = {
            'POI Preference': top_cats,
            'Time-Wise Behavior': {f'{int(h)}': c for h, c in list(timewise_dict.items())[:3]},
            'Label': label
        }
import json
with open('analysis_older_dataset/analysis_set4/codes/Clustering_Profiling/final_cluster_profiles.json', 'w') as f:
    json.dump(final_output, f, indent=2)
print('Saved final cluster profiles to analysis_older_dataset/analysis_set4/codes/Clustering_Profiling/final_cluster_profiles.json')
