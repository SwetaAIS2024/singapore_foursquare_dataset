import pandas as pd
import networkx as nx
import os
from collections import defaultdict
import datetime
import pickle
import csv
from math import radians, sin, cos, sqrt, atan2
import numpy as np

# Load POI mapping (place_id -> category) and also store coordinates
poi_map = pd.read_csv('analysis_set2/Tensor_factorisation_and_clustering/sg_place_id_to_category.csv')
placeid2cat = dict(zip(poi_map['place_id'], poi_map['category']))
# Build a place_id -> (lat, lon) mapping
placeid2coord = {}
with open('singapore_checkins_filtered_with_locations_coord.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 6:
            continue
        place_id = parts[1]
        lat, lon = parts[4], parts[5]
        try:
            placeid2coord[place_id] = (float(lat), float(lon))
        except Exception:
            continue

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
        checkins.append((user_id, dt, root_cat, place_id, float(lat), float(lon)))

# Group check-ins by user and sort by time
user_checkins = defaultdict(list)
for user_id, dt, root_cat, place_id, lat, lon in checkins:
    user_checkins[user_id].append((dt, root_cat, place_id, lat, lon))
for user_id in user_checkins:
    user_checkins[user_id].sort()

# Create a graph for each user: nodes=root categories, edges=temporal transitions with distance
os.makedirs('analysis_set3/trial2_graph_based/user_graphs', exist_ok=True)
def haversine(lat1, lon1, lat2, lon2):
    # Calculate the great-circle distance between two points (in km)
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

for user_id, visits in user_checkins.items():
    G = nx.DiGraph()
    for i in range(len(visits)-1):
        cat_from = visits[i][1]
        cat_to = visits[i+1][1]
        time_from = visits[i][0]
        time_to = visits[i+1][0]
        hour_from = time_from.hour
        hour_to = time_to.hour
        lat1, lon1 = visits[i][3], visits[i][4]
        lat2, lon2 = visits[i+1][3], visits[i+1][4]
        dist = haversine(lat1, lon1, lat2, lon2)
        # Edge: from cat_from to cat_to, with temporal info and distance
        if G.has_edge(cat_from, cat_to):
            G[cat_from][cat_to]['count'] += 1
            G[cat_from][cat_to]['hours'].append((hour_from, hour_to))
            G[cat_from][cat_to]['distances'].append(dist)
            G[cat_from][cat_to]['weight'] += dist
        else:
            G.add_edge(cat_from, cat_to, count=1, hours=[(hour_from, hour_to)], distances=[dist], weight=dist)
    with open(f'analysis_set3/trial2_graph_based/user_graphs/{user_id}.gpickle', 'wb') as f:
        pickle.dump(G, f)
print('Saved user POI root category transition graphs with temporal and distance edge attributes to analysis_set3/trial2_graph_based/user_graphs/')

# Feature extraction for all users (except those with only one check-in)
graph_dir = 'analysis_set3/trial2_graph_based/user_graphs'
feature_csv = 'analysis_set3/trial2_graph_based/user_graph_features_rootcat.csv'

# Collect all root categories from mapping
all_root_categories = sorted(set(cat2root.values()))

# Prepare header: user_id, num_nodes, num_edges, num_self_loops, [cat_* for each root category], [degree_* for each root category], distance features
header = ['user_id', 'num_nodes', 'num_edges', 'num_self_loops']
header += [f'cat_{cat}' for cat in all_root_categories]
header += [f'degree_{cat}' for cat in all_root_categories]
header += ['total_distance', 'mean_distance', 'max_distance', 'min_distance']

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
        # Distance features
        all_distances = []
        for u, v, d in G.edges(data=True):
            if 'distances' in d:
                all_distances.extend(d['distances'])
        if all_distances:
            total_distance = sum(all_distances)
            mean_distance = np.mean(all_distances)
            max_distance = np.max(all_distances)
            min_distance = np.min(all_distances)
        else:
            total_distance = mean_distance = max_distance = min_distance = 0.0
        row = [user_id, num_nodes, num_edges, num_self_loops] + cat_counts + degree_counts + [total_distance, mean_distance, max_distance, min_distance]
        writer.writerow(row)
print(f'Saved user graph features (root categories + distance features) for clustering to {feature_csv}')
