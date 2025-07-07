
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
├── data/
│   ├── original/                  # Raw/original dataset files
│   ├── processed/                 # Processed/cleaned data, intermediate files
│   └── samples/                   # Sampled data for analysis/testing
│
├── clustering/
│   ├── algorithms/                # Scripts for different clustering algorithms (KMeans, DBSCAN, etc.)
│   ├── results/                   # Cluster labels, metrics, plots for each algorithm
│   └── analysis/                  # Cluster analysis scripts and outputs (e.g., cluster profiles, heatmaps)
│
├── dashboard/
│   ├── streamlit_app.py           # Main Streamlit dashboard app
│   ├── components/                # Custom Streamlit components or helper scripts
│   └── static/                    # Static assets (images, CSS, etc.)
│
├── sampling/
│   ├── sampling_scripts/          # Scripts for user/data sampling
│   └── sampled_outputs/           # Sampled datasets/results
│
├── json_dataset_generation/
│   ├── scripts/                   # Scripts for generating JSON datasets
│   └── outputs/                   # Generated JSON files
│
├── config/                        # Config files (YAML/JSON) for pipeline parameters
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
