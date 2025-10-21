import json
import csv
import time
import pandas as pd
import hashlib
from datetime import datetime
from collections import defaultdict
from statistics import mean
from geopy.geocoders import Nominatim
from synthetic_data_v2.c0_Configuration.config_paths import (
    CLUSTER_SUMMARY,
    SAMPLED_FSQ_PLANNING_AREA,
    POI_CAT_MAPPING,
    FINAL_CHECKIN_FILE_TO_LLM,
    CATEGORIES_XLSX,
)



# Load POI category mapping using pandas
def load_poi_mapping(poi_mapping_path):
    df = pd.read_csv(poi_mapping_path)
    poi_mapping = pd.Series(df['category'].values, index=df['place_id']).to_dict()
    return poi_mapping

# Load cluster summary from a CSV file
def load_cluster_summary(cluster_summary_path):
    df = pd.read_csv(cluster_summary_path)
    cluster_summary = df.to_dict(orient="records")  # List of dictionaries for each cluster
    return cluster_summary

# Map user_id to cluster_id from the input file
def map_user_to_cluster(input_file):
    user_to_cluster = {}
    with open(input_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            user_id = row['user_id']
            cluster_id = row['cluster_id']
            user_to_cluster[user_id] = cluster_id
    return user_to_cluster

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
    # print("Relevant Categories:", relevant_categories)  # Debugging

    # Normalize the category column in the user DataFrame
    dataframe_user['category'] = dataframe_user['category'].astype(str).str.strip().str.lower()
    # print("Categories in DataFrame before filtering:", dataframe_user['category'].unique())  # Debugging

    # Filter to only relevant categories
    dataframe_user = dataframe_user[dataframe_user['category'].isin(relevant_categories)]
    # print("Categories in DataFrame after filtering:", dataframe_user['category'].unique())  # Debugging

    return dataframe_user


# Process input file and generate user profiles
def process_input_file(input_file, poi_mapping, cluster_summary, user_to_cluster):
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
    # Re-read the filtered DataFrame into a list of dictionaries

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
            "lat": row["lat"],
            "lon": row["lon"],
        })

    # Generate user profiles
    user_profiles = []
    for user_id, checkins in user_data.items():
        # User Metadata
        user_metadata = checkins

        # Cluster Metadata
        cluster_id = user_to_cluster.get(user_id, "Unknown")
        cluster_info = cluster_summary[int(cluster_id)] if cluster_id != "Unknown" else {}
        top_poi_categories = cluster_info.get("top_poi_categories", "Unknown")
        most_active_time = cluster_info.get("hour_peak", "Unknown")
        most_active_day = cluster_info.get("day_peak", "Unknown")
        mean_planning_area = cluster_info.get("planning_area", "Unknown")


        cluster_metadata = {
            "top_poi_categories": top_poi_categories.split(", ")[:10] if isinstance(top_poi_categories, str) else [],
            "most_active_time": most_active_time,
            "most_active_day": most_active_day,
            "mean_planning_area": mean_planning_area,
        }

        # Probable User Profile Tag
        probable_tags = []
        if isinstance(top_poi_categories, str):  # Ensure top_poi_categories is a string
            top_poi_categories_list = top_poi_categories.split(", ")
            if "Office" in top_poi_categories_list:
                probable_tags.append("Office employees")
            if "Flea Market" in top_poi_categories_list:
                probable_tags.append("Old people")
            if "Cosmetics" in top_poi_categories_list:
                probable_tags.append("Ladies")
            if "Gym" in top_poi_categories_list or "Video Games" in top_poi_categories_list:
                probable_tags.append("Teenagers")
        probable_user_profile_tag = ", ".join(probable_tags) if probable_tags else "General"
        # Append user profile
        user_profiles.append({
            "user_id": user_id,
            "user_metadata": user_metadata,
            "cluster_metadata": cluster_metadata,
            "Probable_user_profile_tag": probable_user_profile_tag,
        })

    return user_profiles

# Save user profiles to JSON
def save_to_json(output_file, user_profiles):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(user_profiles, f, indent=4)

# Main function
def main():
    # Paths
    input_file = SAMPLED_FSQ_PLANNING_AREA
    poi_mapping_path = POI_CAT_MAPPING
    cluster_summary_path = CLUSTER_SUMMARY
    output_file = FINAL_CHECKIN_FILE_TO_LLM

    # Load data
    poi_mapping = load_poi_mapping(poi_mapping_path)
    cluster_summary = load_cluster_summary(cluster_summary_path)
    user_to_cluster = map_user_to_cluster(input_file)

    # Process input file
    user_profiles = process_input_file(input_file, poi_mapping, cluster_summary, user_to_cluster)

    # Save to JSON
    save_to_json(output_file, user_profiles)
    print(f"User profiles saved to {output_file}")

if __name__ == "__main__":
    main()