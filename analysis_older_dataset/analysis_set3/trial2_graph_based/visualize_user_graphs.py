import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt

# Directory containing user graphs
graph_dir = 'analysis_set3/trial1_graph_based/user_graphs'
output_dir = 'analysis_set3/trial1_graph_based/user_graph_visualizations'
os.makedirs(output_dir, exist_ok=True)

# Visualize a sample of user graphs (first 10)
graph_files = [f for f in os.listdir(graph_dir) if f.endswith('.gpickle')]
for i, fname in enumerate(graph_files[:10]):
    with open(os.path.join(graph_dir, fname), 'rb') as f:
        G = pickle.load(f)
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    # Show both count and distance (weight) on edges if available
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        label = f"c:{d.get('count', 0)}"
        if 'weight' in d:
            label += f"\nkm:{d['weight']:.1f}"
        edge_labels[(u, v)] = label
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, edge_color='gray', font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title(f'User {fname.replace(".gpickle", "")} POI Category Transition Graph')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{fname.replace(".gpickle", "")}_graph.png'))
    plt.close()
print(f'Saved visualizations for {min(10, len(graph_files))} user graphs to {output_dir}/')

# Directory to save cycle subgraphs
cycle_dir = 'analysis_set3/trial1_graph_based/user_graph_cycles'
os.makedirs(cycle_dir, exist_ok=True)


# Visualize a sample of cycle subgraphs (first 10 found)
cycle_files = [f for f in os.listdir(cycle_dir) if f.endswith('.gpickle')]
for i, fname in enumerate(cycle_files[:10]):
    with open(os.path.join(cycle_dir, fname), 'rb') as f:
        subG = pickle.load(f)
    plt.figure(figsize=(6, 5))
    pos = nx.spring_layout(subG, seed=42)
    edge_labels = {(u, v): d['count'] for u, v, d in subG.edges(data=True) if 'count' in d}
    nx.draw(subG, pos, with_labels=True, node_color='lightgreen', node_size=700, edge_color='orange', font_size=10)
    if edge_labels:
        nx.draw_networkx_edge_labels(subG, pos, edge_labels=edge_labels, font_color='blue')
    plt.title(f'Cycle {i+1}: {fname.replace(".gpickle", "")}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{fname.replace(".gpickle", "")}_cycle.png'))
    plt.close()
print(f'Saved visualizations for {min(10, len(cycle_files))} cycle subgraphs to {output_dir}/')
