import pandas as pd

# Load check-in and POI category data
checkins = pd.read_csv('analysis_older_dataset/Final_code/analysis_on_rep_users_clusters/cluster_representative_users_checkins.csv')
cat_map = pd.read_csv('analysis_older_dataset/Final_code/analysis_on_rep_users_clusters/sg_place_id_to_category.csv')
cat_dict = dict(zip(cat_map['place_id'], cat_map['category']))

# Map place_id to category
checkins['category'] = checkins['place_id'].map(cat_dict)

# Group by cluster and category, count check-ins
grouped = checkins.groupby(['cluster', 'category']).size().reset_index(name='checkin_count')

# For each cluster, get top POI categories by check-in count
result = []
for cluster, group in grouped.groupby('cluster'):
    top_cats = group.sort_values('checkin_count', ascending=False)
    result.append((cluster, top_cats[['category', 'checkin_count']].values.tolist()))

# Save results to CSV
rows = []
for cluster, cats in sorted(result):
    for cat, count in cats:
        rows.append({'cluster': cluster, 'category': cat, 'checkin_count': count})
output_df = pd.DataFrame(rows)
output_df.to_csv('analysis_older_dataset/Final_code/analysis_on_rep_users_clusters/top_poi_categories_per_cluster.csv', index=False)

# Print results
for cluster, cats in sorted(result):
    print(f'Cluster {cluster}:')
    for cat, count in cats:
        print(f'  {cat}: {count}')
    print()
