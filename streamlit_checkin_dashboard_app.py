import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import tempfile
import subprocess

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("Interactive Check-in Data Dashboard")

st.write("""
Upload your raw check-in CSV file (with columns like user_id, place_id, timestamp, etc). The app will process your data using the analysis scripts from the Final_code folder and generate interactive plots and summaries in real time.
""")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your check-in CSV file", type=["csv"])

if uploaded_file is not None:
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    st.success(f"File uploaded: {uploaded_file.name}")
    st.write("Processing your data...")

    # --- Example: Run the top POI categories per cluster script ---
    # You can add more scripts as needed
    output_csv = os.path.join(tempfile.gettempdir(), "top_poi_categories_per_cluster_user.csv")
    script_path = "analysis_older_dataset/Final_code/analysis_on_rep_users_clusters/top_poi_categories_per_cluster.py"
    # Assume the script can take input and output file arguments (modify as needed)
    try:
        result = subprocess.run([
            "python3", script_path,
            "--input", tmp_path,
            "--output", output_csv
        ], capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_csv):
            st.success("Analysis complete! Displaying results:")
            df = pd.read_csv(output_csv)
            st.dataframe(df)
            st.bar_chart(df.set_index('category')['checkin_count'].sort_values(ascending=False))
        else:
            st.error("Script did not run successfully. Output:\n" + result.stderr)
    except Exception as e:
        st.error(f"Error running analysis script: {e}")

    # --- Clean up temp file ---
    os.remove(tmp_path)
else:
    st.info("Please upload a CSV file to begin analysis.")

st.markdown("---")
st.caption("Dashboard powered by Streamlit. Analysis scripts from Final_code folder.")
