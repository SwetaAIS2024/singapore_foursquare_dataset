import streamlit as st
import os
import pandas as pd
import json
from glob import glob
import matplotlib.pyplot as plt
import numpy as np


def list_files(directory, exts=None):
    files = []
    for f in os.listdir(directory):
        if exts is None or any(f.lower().endswith(ext) for ext in exts):
            files.append(f)
    return sorted(files)

def older_clustering_analysis():
    st.title('Past Clustering Results Analysis')
    st.write('This section displays clustering results from older datasets.')

    # --- Sidebar: File Browser ---
    st.sidebar.title('File Browser')

    st.sidebar.subheader('Clustering Results (CSV/Image)')
    clustering_files = list_files(clustering_dir_older, exts=['.csv', '.png', '.jpg', '.jpeg'])
    if clustering_files:
        selected_clustering_file = st.sidebar.selectbox('Select clustering result file', clustering_files)
    else:
        selected_clustering_file = None
        st.sidebar.info('No clustering result files found.')

    st.sidebar.subheader('Cluster Heatmaps')
    heatmap_files = list_files(heatmap_dir_old, exts=['heatmap.png'])
    selected_heatmap_file = st.sidebar.selectbox(
        'Select cluster heatmap file',
        heatmap_files if heatmap_files else ['No heatmap files found'],
        disabled=not bool(heatmap_files)
    )
    if not heatmap_files:
        selected_heatmap_file = None
        st.sidebar.info('No cluster heatmap files found.')

    # --- Main: Display Results ---
    
    if selected_clustering_file:
        if selected_clustering_file.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(clustering_dir_older, selected_clustering_file))
                st.dataframe(df)
            except Exception as e:
                st.warning(f'Could not read CSV: {e}')
        elif selected_clustering_file.endswith(('.png', '.jpg', '.jpeg')):
            try:
                st.image(os.path.join(clustering_dir_older, selected_clustering_file))
            except Exception as e:
                st.warning(f'Could not display image: {e}')
        else:
            st.text('File preview not supported for this file type.')
    
    if selected_heatmap_file:
        heatmap_path = os.path.join(heatmap_dir_old, selected_heatmap_file)
        if os.path.exists(heatmap_path):
            st.subheader(f'Cluster Heatmap: {selected_heatmap_file}')
            st.image(heatmap_path)
        else:
            st.warning(f'Heatmap file {selected_heatmap_file} not found.')
    
    # Cluster scatter plot
    if os.path.exists(cluster_scatter_plot_old):
        st.subheader('Cluster Scatter Plot')
        st.image(cluster_scatter_plot_old, caption='Cluster Scatter Plot of Users')

    # --- Additional: Cluster Heatmap Peak Summary ---
    st.markdown('---')
    st.subheader('Cluster Heatmap Peak Summary (CSV)')
    if os.path.exists(cluster_heatmap_summary_old):
        try:
            df_summary = pd.read_csv(cluster_heatmap_summary_old)
            st.dataframe(df_summary)
        except Exception as e:
            st.warning(f'Could not read heatmap summary CSV: {e}')
    else:
        st.info('Cluster heatmap peak summary CSV not found.')

    # --- Additional: Cluster Centroid-Nearest User Mapping ---
    st.markdown('---')
    st.subheader('Cluster Centroid-Nearest User Mapping (CSV)')
    if os.path.exists(cluster_centroid_nearest_user_old):
        try:
            df_centroid = pd.read_csv(cluster_centroid_nearest_user_old)
            st.dataframe(df_centroid)
        except Exception as e:
            st.warning(f'Could not read centroid-nearest user CSV: {e}')
    else:
        st.info('Cluster centroid-nearest user CSV not found.')

def cluster_analysis_page():
    st.title('Analysis on Representative Users from Clusters')
    st.write('Top POI categories per cluster (by check-in count) for 10 representative users from each cluster:')
    csv_path = 'analysis_older_dataset/Final_code/analysis_on_rep_users_clusters/top_poi_categories_per_cluster.csv'
    if not os.path.exists(csv_path):
        st.warning('Top POI categories per cluster CSV not found.')
        return
    df = pd.read_csv(csv_path)
    clusters = sorted(df['cluster'].unique())
    selected_cluster = st.selectbox('Select cluster', clusters)
    cluster_df = df[df['cluster'] == selected_cluster]
    st.subheader(f'Top POI Categories for Cluster {selected_cluster}')
    st.dataframe(cluster_df[['category', 'checkin_count']].sort_values('checkin_count', ascending=False))
    st.bar_chart(cluster_df.set_index('category')['checkin_count'].sort_values(ascending=False))

def main_page():
    
    st.title('Foursquare Singapore Dataset Clustering Analysis Dashboard')
    
    # --- Sidebar: File Browser ---
    st.sidebar.title('File Browser')

    st.sidebar.subheader('Clustering Results (CSV/Image)')
    if clustering_files:
        selected_clustering_file = st.sidebar.selectbox('Select clustering result file', clustering_files)
    else:
        selected_clustering_file = None
        st.sidebar.info('No clustering result files found.')

    st.sidebar.subheader('User Profile JSONs')
    if json_files:
        selected_json_file = st.sidebar.selectbox('Select user profile JSON', json_files)
    else:
        selected_json_file = None
        st.sidebar.info('No user profile JSON files found.')
    
    # --- Main: Display Results --
    
    st.header('Clustering Results')
    if selected_clustering_file:
        if selected_clustering_file.endswith('.csv'):
            df = pd.read_csv(os.path.join(clustering_dir, selected_clustering_file))
            st.dataframe(df)
        elif selected_clustering_file.endswith(('.png', '.jpg', '.jpeg')):
            st.image(os.path.join(clustering_dir, selected_clustering_file))
        else:
            st.text('File preview not supported for this file type.')
    path = os.path.join(clustering_dir, 'matrix_output_user_cluster_scatter.png')
    if os.path.exists(path):
        st.subheader('Cluster Scatter Plot')
        st.image(path, caption='Cluster Scatter Plot of Users')
    
    st.header('Original vs Synthetic Dataset Comparison')
    org_counts_path = os.path.join(syn_org_eval_dir, 'org_user_checkin_counts.csv')
    syn_counts_path = os.path.join(syn_org_eval_dir, 'syn_user_checkin_counts.csv')
    if os.path.exists(org_counts_path) and os.path.exists(syn_counts_path):
        org_counts = pd.read_csv(org_counts_path, index_col=0, header=None, names=['user_id','org_checkins'])
        syn_counts = pd.read_csv(syn_counts_path, index_col=0, header=None, names=['user_id','syn_checkins'])
        merged = org_counts.join(syn_counts, how='outer')
        st.subheader('Check-in Counts per User')
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(merged.index, merged['org_checkins'].fillna(0), label='Original', color='tab:blue')
        ax.plot(merged.index, merged['syn_checkins'].fillna(0), label='Synthetic', color='tab:orange')
        ax.set_xlabel('User Index')
        ax.set_ylabel('Check-in Count')
        ax.legend()
        st.pyplot(fig)
        st.write('Summary (Original):', org_counts.describe())
        st.write('Summary (Synthetic):', syn_counts.describe())
    else:
        st.info('Comparison CSVs not found. Please generate them in syn_org_eval.')
    st.markdown('---')
    st.caption('Dashboard generated by Streamlit. All results and files are interactive.')
    if selected_json_file:
        st.header('User Profile JSON')
        json_path = os.path.join(json_dir, selected_json_file)
        with open(json_path, 'r') as f:
            user_json = json.load(f)
        st.json(user_json)


if __name__ == '__main__':

    # Clustering results directory - latest results
    clustering_dir = 'analysis_older_dataset/Final_code/clustering'
    json_dir = 'analysis_older_dataset/Final_code/json_dataset_generation'
    syn_org_eval_dir = 'analysis_older_dataset/Final_code/syn_org_eval'

    clustering_files = list_files(clustering_dir, exts=['.csv', '.png', '.jpg', '.jpeg'])
    json_files = list_files(json_dir, exts=['.json'])
    syn_org_eval_files = list_files(syn_org_eval_dir, exts=['.csv', '.txt'])

    
    # Clustering results directory - older results
    clustering_dir_older = 'analysis_older_dataset/old/analysis_set4/codes/No_of_rootCat_C_top20_remove_MALL'
    heatmap_dir_old = 'analysis_older_dataset/old/analysis_set4/codes/No_of_rootCat_C_top20_remove_MALL/Clustering_Profiling/cluster_profiles'
    # each file = '.../cluster_profiles/cluster_0_heatmap.png'
    cluster_scatter_plot_old = 'analysis_older_dataset/old/analysis_set4/codes/No_of_rootCat_C_top20_remove_MALL/Clustering_Profiling/user_clusters_pca123_scatter.png'
    cluster_heatmap_summary_old = 'analysis_older_dataset/old/analysis_set4/codes/No_of_rootCat_C_top20_remove_MALL/Clustering_Profiling/cluster_profiles/cluster_heatmap_peak_summary.csv'
    cluster_centroid_nearest_user_old = 'analysis_older_dataset/old/analysis_set4/codes/No_of_rootCat_C_top20_remove_MALL/Clustering_Profiling/cluster_centroid_nearest_user.csv'

    try:
        # --- Main: Display Results ---
        st.set_page_config(layout="wide", initial_sidebar_state="expanded")
        st.title('FSQ Clustering and Synthetic Dataset Dashboard')
        # Add navigation to Streamlit app
        page = st.sidebar.radio('Select Page', 
                                ['Main Dashboard', 
                                 'Cluster Analysis (Rep Users)',
                                 'Past Clustering Results',])
        
        if page == 'Main Dashboard':
            main_page()
        elif page == 'Cluster Analysis (Rep Users)':
            cluster_analysis_page()
        elif page == 'Past Clustering Results':
            older_clustering_analysis()

    except Exception as e:
        st.error(f'An error occurred: {e}')
        st.info("Reloading the page again.")
        st.rerun()
        # st.info("Rerun completed.") # This line wont work, rerun() restarts the script from the start again