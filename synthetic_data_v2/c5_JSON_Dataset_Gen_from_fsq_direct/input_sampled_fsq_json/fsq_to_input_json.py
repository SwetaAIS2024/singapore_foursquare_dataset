import json
import csv
import time
import pandas as pd
import hashlib
from datetime import datetime
from collections import defaultdict
from statistics import mean
from geopy.geocoders import Nominatim


CLUSTER_SUMMARY = "c4_JSON_Dataset_Generation/utils/u1_cluster_insights/per_cluster_insights/all_clusters_summary.csv"
SAMPLED_FSQ_PLANNING_AREA = "c5_JSON_Dataset_Gen_from_fsq_direct/utils/u3_sampled_fsq_add_planning_area/sampled_FSQ_dataset_with_planning_area.txt"
POI_CAT_MAPPING = "c1_Data_Collection_and_Processing/input_data/sg_place_id_to_category.csv"
JSON_INPUT = "c5_JSON_Dataset_Gen_from_fsq_direct/input_sampled_fsq_json/input.json"
CATEGORIES_XLSX = "./c1_Data_Collection_and_Processing/input_data/Relevant_POI_category.xlsx"


# Load POI category mapping using pandas
def load_poi_mapping(poi_mapping_path):
    df = pd.read_csv(poi_mapping_path)
    poi_mapping = pd.Series(df['category'].values, index=df['place_id']).to_dict()
    return poi_mapping

# Load cluster summary from a CSV file - FIXED VERSION
def load_cluster_summary(cluster_summary_path):
    df = pd.read_csv(cluster_summary_path)
    
    # Check if there's a cluster_id column, if not, use index as cluster_id
    if 'cluster_id' in df.columns:
        cluster_summary = df.set_index('cluster_id').to_dict(orient="index")
    else:
        # Use row index as cluster_id (0, 1, 2, ...)
        cluster_summary = {idx: row for idx, row in enumerate(df.to_dict(orient="records"))}
    
    print(f"Loaded {len(cluster_summary)} clusters")
    print(f"Cluster IDs: {list(cluster_summary.keys())[:10]}...")  # Show first 10
    
    return cluster_summary


def filter_cat_distribution(dataframe_user):
    # Load relevant categories from the Excel file
    relevant_cats_df = pd.read_excel(CATEGORIES_XLSX)
    cat_col = 'POI Category in Singapore'
    yes_col = 'Relevant to use case '

    # Extract relevant categories
    relevant_categories = [
        cat.strip().lower()
        for cat, flag in zip(relevant_cats_df[cat_col], relevant_cats_df[yes_col])
        if str(flag).strip().lower() == 'yes' and cat and str(cat).strip()
    ]
    relevant_categories = list(dict.fromkeys(relevant_categories))  # Remove duplicates

    # Normalize the category column in the user DataFrame
    dataframe_user['category'] = dataframe_user['category'].astype(str).str.strip().str.lower()

    # Filter to only relevant categories
    dataframe_user = dataframe_user[dataframe_user['category'].isin(relevant_categories)]

    return dataframe_user

# Process input file and generate user profiles - FIXED VERSION
def process_input_file(input_file, poi_mapping, cluster_summary):
    user_data = defaultdict(list)
    # Read and group check-ins by user_id
    with open(input_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)  # Read all rows into memory for filtering

    # Convert rows to a DataFrame for filtering
    dataframe_user = pd.DataFrame(rows)

    # Map place_id to POI category
    dataframe_user['category'] = dataframe_user['place_id'].map(poi_mapping).fillna("Unknown Category")

    # Filter out irrelevant categories
    dataframe_user = filter_cat_distribution(dataframe_user)

    for index, row in dataframe_user.iterrows():
        user_id = row['user_id']
        place_id = row['place_id']
        datetime_str = row['datetime']
        planning_area = row['planning_area']

        # Parse datetime and extract metadata
        dt = datetime.strptime(datetime_str, "%a %b %d %H:%M:%S %z %Y")
        day_of_week = dt.strftime("%A")
        time_of_day = dt.strftime("%I:%M %p")
        month_of_year = dt.strftime("%B")
        
        # Convert to ISO 8601 format timestamp
        timestamp_iso = dt.strftime("%Y-%m-%dT%H:%M:%S") + "Z"

        # Map place_id to POI category
        poi_category = poi_mapping.get(place_id, "Unknown Category")
        hash_digest = hashlib.sha256(str(place_id).encode()).hexdigest()[:8]

        # Append check-in metadata
        user_data[user_id].append({
            "poi_id": f"POI-ID-{hash_digest}",
            "poi_category": poi_category,
            "planning_area": planning_area,
            "day_of_week": day_of_week,
            "time_of_day": time_of_day,
            "month_of_year": month_of_year,
            "timestamp": timestamp_iso,
            "lat": row["lat"],
            "lon": row["lon"],
        })

    # Generate user profiles
    user_profiles = []
    
    for user_id, checkins in user_data.items():
        # User Metadata
        user_metadata = checkins
        
        # Append user profile (simplified - only user_id and metadata)
        user_profiles.append({
            "user_id": user_id,
            "user_metadata": user_metadata,
        })
    
    return user_profiles

# Save user profiles to JSON
def save_to_json(output_file, user_profiles):
    # Create output directory if it doesn't exist
    import os
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(user_profiles, f, indent=4)

# Main function
def main():
    # Paths
    
    input_file = SAMPLED_FSQ_PLANNING_AREA
    poi_mapping_path = POI_CAT_MAPPING
    cluster_summary_path = CLUSTER_SUMMARY
    output_file = JSON_INPUT

    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Load data
    poi_mapping = load_poi_mapping(poi_mapping_path)
    print(f"Loaded {len(poi_mapping)} POI mappings")
    
    cluster_summary = load_cluster_summary(cluster_summary_path)

    print("\n" + "="*60)
    print("PROCESSING USER PROFILES")
    print("="*60)
    
    # Process input file (no cluster mapping needed)
    user_profiles = process_input_file(input_file, poi_mapping, cluster_summary)

    print("\n" + "="*60)
    print("SAVING OUTPUT")
    print("="*60)
    
    # Save to JSON
    save_to_json(output_file, user_profiles)
    print(f"✅ {len(user_profiles)} user profiles saved to {output_file}")

if __name__ == "__main__":
    main()