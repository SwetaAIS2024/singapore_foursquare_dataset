import pandas as pd
import networkx as nx
import os
from collections import defaultdict
import datetime
import pickle
import csv

# Load POI mapping (place_id -> category)
poi_map = pd.read_csv('analysis_set2/Tensor_factorisation_and_clustering/sg_place_id_to_category.csv')
placeid2cat = dict(zip(poi_map['place_id'], poi_map['category']))

# Load category to root category mapping
cat2root = dict(pd.read_csv('poi_category_to_root_mapping.csv').values)

# Load check-in data
checkins = []
with open('singapore_checkins_filtered_with_locations_coord.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 6:
            continue
        user_id, place_id, timestamp, _, lat, lon = parts[:6]
        if place_id not in placeid2cat:
            continue
        try:
            dt = datetime.datetime.strptime(timestamp, '%a %b %d %H:%M:%S %z %Y')
            hour = dt.hour
        except Exception:
            continue
        poi_cat = placeid2cat[place_id]
        root_cat = cat2root.get(poi_cat, 'Other')
        checkins.append((user_id, dt, root_cat))

# Group check-ins by user and sort by time
user_checkins = defaultdict(list)
for user_id, dt, root_cat in checkins:
    user_checkins[user_id].append((dt, root_cat))
for user_id in user_checkins:
    user_checkins[user_id].sort()

# Create a graph for each user: nodes=root categories, edges=temporal transitions
os.makedirs('analysis_set3/trial1_graph_based/user_graphs', exist_ok=True)
for user_id, visits in user_checkins.items():
    G = nx.DiGraph()
    for i in range(len(visits)-1):
        cat_from = visits[i][1]
        cat_to = visits[i+1][1]
        time_from = visits[i][0]
        time_to = visits[i+1][0]
        hour_from = time_from.hour
        hour_to = time_to.hour
        # Edge: from cat_from to cat_to, with temporal info (hour_from, hour_to)
        if G.has_edge(cat_from, cat_to):
            G[cat_from][cat_to]['count'] += 1
            G[cat_from][cat_to]['hours'].append((hour_from, hour_to))
        else:
            G.add_edge(cat_from, cat_to, count=1, hours=[(hour_from, hour_to)])
    with open(f'analysis_set3/trial1_graph_based/user_graphs/{user_id}.gpickle', 'wb') as f:
        pickle.dump(G, f)
print('Saved user POI root category transition graphs with temporal edge attributes to analysis_set3/trial1_graph_based/user_graphs/')

# Feature extraction for all users (except those with only one check-in)
graph_dir = 'analysis_set3/trial1_graph_based/user_graphs'
feature_csv = 'analysis_set3/trial1_graph_based/user_graph_features_rootcat.csv'

# Collect all root categories from mapping
all_root_categories = sorted(set(cat2root.values()))

# Prepare header: user_id, num_nodes, num_edges, num_self_loops, [cat_* for each root category], [degree_* for each root category]
header = ['user_id', 'num_nodes', 'num_edges', 'num_self_loops']
header += [f'cat_{cat}' for cat in all_root_categories]
header += [f'degree_{cat}' for cat in all_root_categories]

with open(feature_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for fname in os.listdir(graph_dir):
        if not fname.endswith('.gpickle'):
            continue
        user_id = fname.replace('.gpickle', '')
        with open(os.path.join(graph_dir, fname), 'rb') as gf:
            G = pickle.load(gf)
        if G.number_of_nodes() <= 1:
            continue  # skip users with only one check-in
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        num_self_loops = nx.number_of_selfloops(G)
        # Category visit counts (node degree sum for each root category)
        cat_counts = [G.nodes[n]['count'] if n in G and 'count' in G.nodes[n] else 0 for n in all_root_categories]
        # Degree for each root category (out-degree)
        degree_counts = [G.out_degree(cat) if cat in G else 0 for cat in all_root_categories]
        row = [user_id, num_nodes, num_edges, num_self_loops] + cat_counts + degree_counts
        writer.writerow(row)
print(f'Saved user graph features (root categories) for clustering to {feature_csv}')
