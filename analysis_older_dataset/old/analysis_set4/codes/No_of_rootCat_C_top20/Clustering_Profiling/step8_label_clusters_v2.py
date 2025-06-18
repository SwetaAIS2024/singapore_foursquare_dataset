import numpy as np
import pandas as pd
import os

def get_time_bracket(hour):
    if 8 <= hour < 12:
        return 'Morning (8AM-12PM)'
    elif 12 <= hour < 16:
        return 'Noon (12PM-4PM)'
    elif 16 <= hour < 20:
        return 'Evening (4PM-8PM)'
    elif 20 <= hour < 24:
        return 'Night (8PM-12AM)'
    elif 0 <= hour < 4:
        return 'Late Night (12AM-4AM)'
    else:
        return 'Early Morning (4AM-8AM)'

# Load category labels
cat_labels = []
with open('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/C_time_matrix/user_category_labels.txt') as f:
    for line in f:
        idx, cat = line.strip().split('\t')
        cat_labels.append(cat)

profile_dir = 'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Clustering_Profiling/cluster_profiles'
output = {}
for fname in os.listdir(profile_dir):
    if fname.endswith('_mean_matrix.npy'):
        cluster_id = fname.split('_')[1]
        mean_mat = np.load(os.path.join(profile_dir, fname))
        # POI Preference (most active hour per category)
        cat_pref = mean_mat.max(axis=1)
        top_cats = [cat_labels[i] for i in np.argsort(cat_pref)[::-1][:5]]
        # Time Profile
        time_profile = mean_mat.sum(axis=0)
        # Find the 4-hour bracket with the highest sum
        bracket_sums = {}
        for start in [8, 12, 16, 20, 0, 4]:
            if start == 0:
                hours = list(range(0, 4))
            elif start == 4:
                hours = list(range(4, 8))
            else:
                hours = list(range(start, start+4))
            bracket_sums[get_time_bracket(hours[0])] = time_profile[hours].sum()
        peak_bracket = max(bracket_sums, key=bracket_sums.get)
        # Time-Wise Behavior (for each bracket, get the POI with the highest value summed over all hours in that bracket)
        bracket_map = {}
        bracket_hours = {
            'Morning (8AM-12PM)': range(8, 12),
            'Noon (12PM-4PM)': range(12, 16),
            'Evening (4PM-8PM)': range(16, 20),
            'Night (8PM-12AM)': range(20, 24),
            'Late Night (12AM-4AM)': list(range(0, 4)),
            'Early Morning (4AM-8AM)': range(4, 8)
        }
        for label, hours in bracket_hours.items():
            # Sum over all 7 days for each hour in the bracket
            hour_indices = []
            for d in range(7):
                for h in hours:
                    hour_indices.append(d*24 + h)
            bracket_sum = mean_mat[:, hour_indices].sum(axis=1)
            cat_idx = np.argmax(bracket_sum)
            bracket_map[label] = cat_labels[cat_idx]
        # Label using the global peak in the mean matrix (most frequent POI at its most active time)
        peak_val = np.max(mean_mat)
        peak_cat_idx, peak_hour_global = np.unravel_index(np.argmax(mean_mat), mean_mat.shape)
        peak_cat = cat_labels[peak_cat_idx]
        peak_hour = peak_hour_global % 24  # hour in day
        peak_bracket_for_peak_cat = get_time_bracket(peak_hour)

        # POI-based label
        if 'Nightlife' in peak_cat:
            base_label = 'Nightlife Lovers'
        elif 'Trade School' in peak_cat or 'Professional' in peak_cat:
            base_label = 'Professional'
        elif 'Education' in peak_cat or 'College' in peak_cat:
            base_label = 'Students/Professionals'
        elif 'High School' in peak_cat:
            base_label = 'High School Students'
        elif 'Mall' in peak_cat or 'Shopping' in peak_cat or 'Shop' in peak_cat:
            base_label = 'Mall Visitors'
        elif 'Food' in peak_cat:
            base_label = 'Foodies'
        else:
            base_label = 'General'
        label = f"{base_label} (Most frequent: {peak_bracket_for_peak_cat})"
        output[cluster_id] = {
            'POI Preference': top_cats,
            'Time Profile': f'Peaks during {peak_bracket}',
            'Time-Wise Behavior': bracket_map,
            'Label': label
        }
import json
with open('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Clustering_Profiling/final_cluster_profiles_v2.json', 'w') as f:
    json.dump(output, f, indent=2)
print('Saved improved cluster profiles to analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Clustering_Profiling/final_cluster_profiles_v2.json')
