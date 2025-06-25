import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tempfile
import subprocess
import shutil

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("FSQ Check-in Data: End-to-End Clustering Dashboard")

st.write("""
Upload your raw FSQ check-in CSV file (with columns like user_id, place_id, timestamp, etc). The dashboard will process your data stepwise using the analysis scripts from the Final_code folder, showing intermediate and final results interactively.
""")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your check-in CSV or TXT file", type=["csv", "txt"])

# --- Pre-defined mapping/category files (update paths as needed) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLUSTERING_DIR = os.path.join(BASE_DIR, "analysis_older_dataset/Final_code/clustering")
MAPPING_PLACEID_TO_CAT = os.path.join(CLUSTERING_DIR, "sg_place_id_to_category.csv")
RELEVANT_POI_CAT = os.path.join(BASE_DIR, "analysis_older_dataset/Final_code/analysis_on_rep_users_clusters/Relevant_POI_category.xlsx")

# --- Matrix Size Parameters (Sidebar) ---
st.sidebar.header("Matrix Size Parameters")
n_users = st.sidebar.number_input("Number of users", min_value=10, max_value=2000, value=200)
n_spatial_clusters = st.sidebar.number_input("Number of spatial clusters", min_value=2, max_value=50, value=20)
n_categories = st.sidebar.number_input("Number of categories", min_value=5, max_value=180, value=30)

# Helper to run a script and return output file path or error
def run_script(script, args, output_files=None):
    try:
        result = subprocess.run(["python3", script] + args, capture_output=True, text=True)
        if result.returncode == 0:
            if output_files:
                for f in output_files:
                    if not os.path.exists(f):
                        st.error(f"Expected output file not found: {f}")
                        return None
            return True
        else:
            st.error(f"Script failed: {script}\n{result.stderr}")
            return None
    except Exception as e:
        st.error(f"Error running script {script}: {e}")
        return None

if uploaded_file is not None:
    # Save uploaded file to a temp location with correct extension
    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
    if file_ext not in [".csv", ".txt"]:
        st.error("Only CSV or TXT files are supported.")
        st.stop()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    st.success(f"File uploaded: {uploaded_file.name}")

    # --- 1. Matrix Creation ---
    st.header("1. Matrix Creation: User x Spatial Cluster x POI Category x Time Bins")
    matrix_out = os.path.join(tempfile.gettempdir(), "user_spatial_cat_time_matrix.npy")
    user_list_out = os.path.join(tempfile.gettempdir(), "user_list.txt")
    cluster_list_out = os.path.join(tempfile.gettempdir(), "spatial_cluster_list.txt")
    cat_list_out = os.path.join(tempfile.gettempdir(), "poi_cat_list.txt")
    timebin_list_out = os.path.join(tempfile.gettempdir(), "timebin_list.txt")

    # Always run matrix creation, do not skip if files exist
    matrix_script = os.path.join(CLUSTERING_DIR, "s01_build_user_spatial_category_time_matrix.py")
    args = [
        "--input", tmp_path,
        "--placeid_to_cat", MAPPING_PLACEID_TO_CAT,
        "--output_matrix", matrix_out,
        "--output_user_list", user_list_out,
        "--output_cluster_list", cluster_list_out,
        "--output_cat_list", cat_list_out,
        "--output_timebin_list", timebin_list_out,
        "--n_users", str(n_users),
        "--n_spatial_clusters", str(n_spatial_clusters),
        "--n_categories", str(n_categories)
    ]
    # Run script and capture error output
    try:
        result = subprocess.run(["python3", matrix_script] + args, capture_output=True, text=True)
        # Workaround: check for hardcoded output files if expected files are missing
        hardcoded_matrix = os.path.join(CLUSTERING_DIR, "matrix_output.npy")
        hardcoded_user_list = os.path.join(CLUSTERING_DIR, "user_list.txt")
        hardcoded_cluster_list = os.path.join(CLUSTERING_DIR, "spatial_cluster_list.txt")
        hardcoded_cat_list = os.path.join(CLUSTERING_DIR, "poi_cat_list.txt")
        hardcoded_timebin_list = os.path.join(CLUSTERING_DIR, "timebin_list.txt")
        # If expected files are missing but hardcoded exist, copy them
        if not os.path.exists(matrix_out) and os.path.exists(hardcoded_matrix):
            shutil.copy(hardcoded_matrix, matrix_out)
        if not os.path.exists(user_list_out) and os.path.exists(hardcoded_user_list):
            shutil.copy(hardcoded_user_list, user_list_out)
        if not os.path.exists(cluster_list_out) and os.path.exists(hardcoded_cluster_list):
            shutil.copy(hardcoded_cluster_list, cluster_list_out)
        if not os.path.exists(cat_list_out) and os.path.exists(hardcoded_cat_list):
            shutil.copy(hardcoded_cat_list, cat_list_out)
        if not os.path.exists(timebin_list_out) and os.path.exists(hardcoded_timebin_list):
            shutil.copy(hardcoded_timebin_list, timebin_list_out)
        missing = [f for f in [matrix_out, user_list_out, cluster_list_out, cat_list_out, timebin_list_out] if not os.path.exists(f)]
        if result.returncode == 0:
            if missing:
                st.error(f"Expected output file(s) not found: {missing}")
                st.error(f"Script output:\n{result.stdout}\n{result.stderr}")
                st.stop()
            st.success("Matrix created!")
            st.write(f"Matrix file: {matrix_out}")
        else:
            st.error(f"Matrix creation script failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
            st.stop()
    except Exception as e:
        st.error(f"Error running matrix creation script: {e}")
        st.stop()

    # --- 2. Matrix Inspection ---
    st.header("2. Matrix Inspection")
    inspect_script = os.path.join(CLUSTERING_DIR, "s02_inspect_user_spatial_category_time_matrix.py")
    inspect_out = os.path.join(tempfile.gettempdir(), "matrix_inspect.csv")
    args = [
        "--matrix", matrix_out,
        "--output", inspect_out
    ]
    if run_script(inspect_script, args, [inspect_out]):
        df = pd.read_csv(inspect_out)
        st.dataframe(df.head(20))
        st.write(f"Matrix shape: {df.shape}")
    else:
        st.stop()

    # --- 3. Matrix Pre-processing ---
    st.header("3. Matrix Pre-processing")
    # 3.1 Normalization
    st.subheader("a) Normalization")
    norm_matrix_out = os.path.join(tempfile.gettempdir(), "user_spatial_cat_time_matrix_normalized.npy")
    norm_script = os.path.join(CLUSTERING_DIR, "s03_normalize_and_inspect_user_matrix.py")
    args = [
        "--input_matrix", matrix_out,
        "--output_matrix", norm_matrix_out
    ]
    if run_script(norm_script, args, [norm_matrix_out]):
        st.success("Matrix normalized!")
        st.write(f"Normalized matrix file: {norm_matrix_out}")
    else:
        st.stop()

    # 3.2 Flattening
    st.subheader("b) Flattening")
    flat_matrix_out = os.path.join(tempfile.gettempdir(), "user_matrix_flattened.npy")
    flat_script = os.path.join(CLUSTERING_DIR, "s04_flatten_normalized_4D_matrix.py")
    args = [
        "--input_matrix", norm_matrix_out,
        "--output_matrix", flat_matrix_out
    ]
    if run_script(flat_script, args, [flat_matrix_out]):
        st.success("Matrix flattened!")
        st.write(f"Flattened matrix file: {flat_matrix_out}")
    else:
        st.stop()

    # 3.3 Sparse Conversion
    st.subheader("c) Sparse Matrix Conversion")
    sparse_matrix_out = os.path.join(tempfile.gettempdir(), "user_matrix_sparse.npz")
    sparse_script = os.path.join(CLUSTERING_DIR, "s05_sparse_matrix.py")
    args = [
        "--input_matrix", flat_matrix_out,
        "--output_matrix", sparse_matrix_out
    ]
    if run_script(sparse_script, args, [sparse_matrix_out]):
        st.success("Sparse matrix created!")
        st.write(f"Sparse matrix file: {sparse_matrix_out}")
    else:
        st.stop()

    # --- 4. Clustering ---
    st.header("4. Clustering")
    cluster_assign_out = os.path.join(tempfile.gettempdir(), "user_cluster_assignments.csv")
    cluster_plot_out = os.path.join(tempfile.gettempdir(), "user_clusters_plot.png")
    cluster_script = os.path.join(CLUSTERING_DIR, "s06_overall_user_clustering.py")
    args = [
        "--input_matrix", sparse_matrix_out,
        "--user_list", user_list_out,
        "--output_assignments", cluster_assign_out,
        "--output_plot", cluster_plot_out
    ]
    if run_script(cluster_script, args, [cluster_assign_out, cluster_plot_out]):
        st.success("Clustering complete!")
        df = pd.read_csv(cluster_assign_out)
        st.dataframe(df.head(20))
        if os.path.exists(cluster_plot_out):
            st.image(cluster_plot_out, caption="User Clusters Visualization", use_column_width=True)
    else:
        st.stop()

    # --- 5. Post-Clustering Analysis & Sampling ---
    st.header("5. Post-Clustering Analysis & Sampling")
    post_cluster_script = os.path.join(CLUSTERING_DIR, "s07_post_clustering_analysis.py")
    post_cluster_out = os.path.join(tempfile.gettempdir(), "post_clustering_analysis.csv")
    args = [
        "--cluster_assignments", cluster_assign_out,
        "--user_list", user_list_out,
        "--output", post_cluster_out
    ]
    if run_script(post_cluster_script, args, [post_cluster_out]):
        df = pd.read_csv(post_cluster_out)
        st.dataframe(df.head(20))
    else:
        st.warning("Post-clustering analysis script did not produce output.")

    # --- 6. Sampling Representative Users ---
    st.header("6. Sampling Representative Users from Clusters")
    sample_script = os.path.join(CLUSTERING_DIR, "s08_user_profile_from_cluster.py")
    sample_out = os.path.join(tempfile.gettempdir(), "sampled_representative_users.csv")
    args = [
        "--cluster_assignments", cluster_assign_out,
        "--user_list", user_list_out,
        "--output", sample_out
    ]
    if run_script(sample_script, args, [sample_out]):
        df = pd.read_csv(sample_out)
        st.dataframe(df.head(20))
    else:
        st.warning("Sampling script did not produce output.")

    # --- 7. Synthetic Data Generation ---
    st.header("7. Synthetic Data Generation")
    synth_script = os.path.join(CLUSTERING_DIR, "s09_synthetic_data_generation.py")
    synth_out = os.path.join(tempfile.gettempdir(), "synthetic_data.csv")
    args = [
        "--cluster_assignments", cluster_assign_out,
        "--output", synth_out
    ]
    if run_script(synth_script, args, [synth_out]):
        df = pd.read_csv(synth_out)
        st.dataframe(df.head(20))
    else:
        st.warning("Synthetic data generation script did not produce output.")

    # --- Clean up temp file ---
    os.remove(tmp_path)
else:
    st.info("Please upload a CSV file to begin analysis.")

st.markdown("---")
st.caption("Dashboard powered by Streamlit. Analysis scripts from Final_code/clustering folder.")
