
# Dashboard plan 
- Original Dataset
    - Show raw data, summary stats, and upload/download options.
- Clustering Algorithm Selector
    - Sidebar or main page: Dropdown to select which clustering algorithm’s results to view (KMeans, DBSCAN, Agglomerative, etc.).
    - Dynamically load results/plots based on selection.
- User Insights
    - Show user-level cluster assignments, allow filtering by user, cluster, or algorithm.
    - Display user profiles, check-in patterns, etc.
- Cluster Analysis
    - Show cluster profiles, heatmaps, top POIs, time profiles, etc.
    - Compare clusters across algorithms.
- Sampling
    - Show sampled users/data, allow download, and display summary stats.
- Next Pipeline Placeholder
    - Section for JSON dataset generation (show status, allow triggering, or preview generated JSONs).

## Folder structure
<pre>```
project_root/
│
├── c0_config/                      # Config files (YAML/JSON) for pipeline 
├── c1_data/
│   ├── c0_original/                  # Raw/original dataset files
│   ├── c1_processing/                 # Processed/cleaned data, intermediate files
│   └── c2_output/                   # Sampled data for analysis/testing
│
├── c2_clustering/
│   ├── c0_clustering/                # Scripts for different clustering algorithms (KMeans, DBSCAN, etc.)
│   ├── c1_output/                   # Cluster labels, metrics, plots for each algorithm
│   └── c2_post_clustering_analysis/                  # Cluster analysis scripts and outputs (e.g., cluster profiles, heatmaps)
│
├── c3_dashboard/
│   ├── c0_components/                # Custom Streamlit components or helper scripts
│   └── c1_static/                    # Static assets (images, CSS, etc.)
│   ├── c2_streamlit_app.py           # Main Streamlit dashboard app
|
├── c4_sampling/
│   ├── c0_sampling_scripts/          # Scripts for user/data sampling
│   └── c1_sampled_outputs/           # Sampled datasets/results
│
├── c5_json_dataset_generation/
│   ├── c0_scripts/                   # Scripts for generating JSON datasets
│   └── c1_outputs/                   # Generated JSON files
│
parameters
├── README.md
└── requirements.txt
```</pre>

## How to Implement in Streamlit
- Use a sidebar radio or selectbox for navigation:

<pre> ```python page = st.sidebar.radio('Select Page', [
    'Original Dataset',
    'Clustering Algorithm Selector',
    'User Insights',
    'Cluster Analysis',
    'Sampling',
    'JSON Dataset Generation (Next Pipeline)'
]) ``` </pre>

- In each section, load files from the corresponding folders above.
