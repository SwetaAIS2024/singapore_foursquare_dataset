import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
import re

# Load cluster profiles
with open('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Clustering_Profiling/final_cluster_profiles_v2.json') as f:
    profiles = json.load(f)

# 1. Summary Table
summary_rows = []
for cid, prof in profiles.items():
    summary_rows.append({
        'Cluster': cid,
        'Label': prof['Label'],
        'Top POI Preferences': ', '.join(prof['POI Preference']),
        'Peak Time': prof['Time Profile']
    })
sum_df = pd.DataFrame(summary_rows)
sum_df.to_csv('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Visualization/cluster_summary_table.csv', index=False)

# 2. Bar chart for POI Preferences per cluster
for cid, prof in profiles.items():
    plt.figure(figsize=(8, 4))
    sns.barplot(x=prof['POI Preference'], y=list(range(len(prof['POI Preference']))), orient='h', palette='viridis')
    plt.title(f'Cluster {cid} - Top POI Preferences')
    plt.xlabel('POI Category')
    plt.ylabel('Rank')
    plt.tight_layout()
    plt.savefig(f'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Visualization/cluster_{cid}_poi_pref_bar.png')
    plt.close()

# 3. Time-wise POI Category Bar with Text Labels
brackets = [
    'Morning (8AM-12PM)', 'Noon (12PM-4PM)', 'Evening (4PM-8PM)',
    'Night (8PM-12AM)', 'Late Night (12AM-4AM)', 'Early Morning (4AM-8AM)'
]
for cid, prof in profiles.items():
    cats = [prof['Time-Wise Behavior'].get(b, '') for b in brackets]
    plt.figure(figsize=(10, 2))
    plt.bar(range(len(brackets)), [1]*len(brackets), color='lightblue')
    for i, cat in enumerate(cats):
        plt.text(i, 0.5, cat, ha='center', va='center', fontsize=12)
    plt.xticks(range(len(brackets)), brackets, rotation=30, ha='right')
    plt.yticks([])
    plt.title(f'Cluster {cid} - Dominant POI by Time Bracket')
    plt.tight_layout()
    plt.savefig(f'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Visualization/cluster_{cid}_timewise_bar.png')
    plt.close()

# --- 3D Plot: Time Bracket x Dominant POI x Cluster ---
# Load summary table
csv_path = 'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Visualization/cluster_summary_table.csv'
sum_df = pd.read_csv(csv_path)

# Extract time bracket from 'Peak Time' column (fix regex)
def extract_bracket(peak_time):
    match = re.search(r'\((.*?)\)', str(peak_time))
    if match:
        return match.group(1)
    return ''

sum_df['Time Bracket'] = sum_df['Peak Time'].apply(lambda x: re.search(r'\((.*?)\)', str(x)).group(1) if re.search(r'\((.*?)\)', str(x)) else '')
# Use the first POI in 'Top POI Preferences' as dominant
sum_df['Dominant POI'] = sum_df['Top POI Preferences'].apply(lambda x: str(x).split(',')[0].strip())

# Print unique extracted time brackets for debugging
print('Extracted time brackets:', sum_df['Time Bracket'].unique())

# Define the order for time brackets
bracket_order = ['8AM-12PM','12PM-4PM','4PM-8PM','8PM-12AM','12AM-4AM','4AM-8AM']
poi_list = sorted(sum_df['Dominant POI'].unique())
poi_map = {poi: i for i, poi in enumerate(poi_list)}

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

import matplotlib.cm as cm
clusters = sorted(sum_df['Cluster'].astype(int).unique())
colors = cm.get_cmap('tab10', len(clusters))

for idx, row in sum_df.iterrows():
    # Use index in bracket_order for x
    try:
        x = bracket_order.index(row['Time Bracket'])
    except ValueError:
        print(f"Skipping row with unknown time bracket: {row['Time Bracket']}")
        continue
    y = poi_map.get(row['Dominant POI'], -1)
    try:
        z = int(row['Cluster'])
    except Exception:
        continue
    if y == -1:
        print(f"Skipping row with unknown POI: {row['Dominant POI']}")
        continue
    ax.scatter(x, y, z, s=120, color=colors(z), alpha=0.8, edgecolor='k')
    ax.text(x, y, z, f"{row['Label']}", fontsize=9, color=colors(z))

ax.set_xlabel('Time Bracket')
ax.set_ylabel('Dominant POI')
ax.set_zlabel('Cluster')
ax.set_xticks(list(range(len(bracket_order))))
ax.set_xticklabels(bracket_order, rotation=30)
ax.set_yticks(list(range(len(poi_list))))
ax.set_yticklabels(poi_list)
plt.title('3D Plot: Time Bracket x Dominant POI x Cluster')
plt.tight_layout()
plt.savefig('analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Visualization/cluster_3d_time_poi_cluster.png')
plt.show()
# --- End 3D Plot ---

print('Visualization complete. See analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20/Visualization for results.')
