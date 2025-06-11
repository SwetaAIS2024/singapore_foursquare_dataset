import pandas as pd

# Path to the cluster profile file (update if needed)
profile_path = "analysis_set3/trial1_graph_based/plots_clustering/user_cluster_profiles_PCA_KMeans.txt"

# Read the file
with open(profile_path, "r") as f:
    lines = f.readlines()

# Helper to parse cluster blocks
def parse_clusters(lines):
    clusters = []
    cluster = {}
    for line in lines:
        line = line.strip()
        if line.startswith("Cluster") and "profile" in line:
            if cluster:
                clusters.append(cluster)
                cluster = {}
            cluster['Cluster'] = int(line.split()[1])
        elif line.startswith("num_edges:"):
            cluster['num_edges'] = float(line.split(":")[1].strip())
        elif line.startswith("num_nodes:"):
            cluster['num_nodes'] = float(line.split(":")[1].strip())
        elif line.startswith("degree_") and ":" in line:
            # e.g. degree_Food: 6.77
            k, v = line.split(":")
            cluster[k.strip()] = float(v.strip())
        elif line.startswith("Full centroid:"):
            # Next lines are centroid values
            cluster['centroid'] = {}
        elif 'centroid' in cluster and line and not line.startswith("Cluster"):
            # Accept both 'degree_Food: 6.77' and 'degree_Food 6.77'
            if ":" in line:
                k, v = line.split(":")
            elif len(line.split()) == 2:
                k, v = line.split()
            else:
                continue
            try:
                cluster['centroid'][k.strip()] = float(v.strip())
            except ValueError:
                pass
    if cluster:
        clusters.append(cluster)
    return clusters

# Main interests: top 3 features in centroid (excluding num_nodes, num_edges, num_self_loops)
def get_main_interests(centroid):
    ignore = {'num_nodes', 'num_edges', 'num_self_loops'}
    # Only consider degree_* features
    filtered = {k: v for k, v in centroid.items() if k.startswith('degree_') and k not in ignore}
    # Sort by value descending, pick top 3
    top = sorted(filtered.items(), key=lambda x: -x[1])[:3]
    return ", ".join([k.replace('degree_', '') for k, v in top])

# Activity level: based on num_edges and num_nodes
def get_activity_level(num_edges, num_nodes):
    if num_edges > 40 and num_nodes > 8.5:
        return "Very High"
    elif num_edges > 30 and num_nodes > 7.5:
        return "High"
    elif num_edges > 20 and num_nodes > 6.5:
        return "Moderate"
    else:
        return "Low"

# Improved lifestyle/type assignment using combinations of top features and activity
LIFESTYLE_MAP = [
    (['College', 'University'], 'High', 'High-activity students/professionals'),
    (['College', 'University'], 'Moderate', 'Students/Professionals, balanced'),
    (['Food', 'Shop', 'Service', 'Professional & Other Places', 'Travel & Transport'], 'Very High', 'Urban explorers, highly diverse users'),
    (['Food', 'Shop', 'Service', 'Arts & Entertainment'], 'Very High', 'Active, cosmopolitan, event-goers'),
    (['Food', 'Shop', 'Service', 'Nightlife'], 'Moderate', 'Social, nightlife and shopping fans'),
    (['Food', 'Nightlife', 'Shop', 'Service'], 'Moderate', 'Social, outgoing, nightlife seekers'),
    (['Shop', 'Service', 'Professional & Other Places', 'Travel & Transport'], 'Moderate', 'Balanced, work/shopping focused'),
    (['Shop', 'Food'], 'Low', 'Routine, low-activity users'),
]

def get_lifestyle(main_interests, activity):
    main_list = [x.strip() for x in main_interests.split(",")]
    for keywords, act_level, desc in LIFESTYLE_MAP:
        if activity == act_level and any(any(key in m for key in keywords) for m in main_list):
            return desc
    # Fallbacks
    if activity == "Very High":
        return "Urban explorers, highly diverse users"
    if activity == "Low":
        return "Routine, low-activity users"
    return "General"

# Instead of parsing the text file, read the centroids CSV directly for main interests
centroids_path = "analysis_set3/trial1_graph_based/plots_clustering/user_cluster_centroids_PCA_KMeans.csv"
centroids_df = pd.read_csv(centroids_path)

# Build summary table from centroids_df
summary_rows = []
for idx, row in centroids_df.iterrows():
    cluster = int(row['Cluster'])
    num_edges = row['num_edges']
    num_nodes = row['num_nodes']
    # Only degree_* columns
    degree_cols = [c for c in centroids_df.columns if c.startswith('degree_')]
    top3 = row[degree_cols].sort_values(ascending=False).head(3)
    main_interests = ', '.join([c.replace('degree_', '') for c in top3.index])
    activity = get_activity_level(num_edges, num_nodes)
    # Distance metrics if present
    total_distance = row['total_distance'] if 'total_distance' in centroids_df.columns else None
    mean_distance = row['mean_distance'] if 'mean_distance' in centroids_df.columns else None
    max_distance = row['max_distance'] if 'max_distance' in centroids_df.columns else None
    min_distance = row['min_distance'] if 'min_distance' in centroids_df.columns else None
    lifestyle = get_lifestyle(main_interests, activity)
    summary_rows.append({
        'Cluster': cluster,
        'Activity Level': activity,
        'Main Interests': main_interests,
        'Lifestyle/Type': lifestyle,
        'Total Distance (km)': total_distance,
        'Mean Distance (km)': mean_distance,
        'Max Distance (km)': max_distance,
        'Min Distance (km)': min_distance
    })

summary_df = pd.DataFrame(summary_rows)
summary_path = "analysis_set3/trial1_graph_based/plots_clustering/user_cluster_summary_table.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Summary table saved to {summary_path}")
print(summary_df)
