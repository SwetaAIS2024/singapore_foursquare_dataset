import json
import csv
import time
import pandas as pd
import hashlib
from datetime import datetime
from collections import defaultdict
from statistics import mean
from geopy.geocoders import Nominatim


def safe_string(val, default="Unknown"):
    """
    Safely convert value to string, handling NaN, None, and empty values
    """
    import math
    
    if val is None:
        return default
    
    # Handle NaN values (both float('nan') and string 'NaN')
    if isinstance(val, float) and math.isnan(val):
        return default
    
    # Handle string representations of NaN
    if isinstance(val, str) and val.lower() in ['nan', 'null', 'none', '']:
        return default
    
    # Return the string representation of the value
    return str(val).strip() or default


SAMPLED_FSQ_PLANNING_AREA = "json_gen/utils/u3_sampled_fsq_add_planning_area/sampled_FSQ_dataset_with_planning_area.txt"
POI_CAT_MAPPING = "./Input_data/FSQ_SG_2013_POI.csv"
JSON_INPUT_FILTERED = "json_gen/input_sampled_fsq_json/input_filtered.json"
JSON_INPUT_ALL = "json_gen/input_sampled_fsq_json/input_all_categories.json"
CATEGORIES_XLSX = "./Input_data/Relevant_POI_category.xlsx"

# Load POI category and name mapping using pandas
def load_poi_mapping(poi_mapping_path):
    # Read the CSV file assuming columns: place_id, name, lat, lon, category, country
    df = pd.read_csv(poi_mapping_path, header=None, 
                     names=['place_id', 'name', 'lat', 'lon', 'category', 'country'])
    
    # Create mappings for both category and name
    poi_category_mapping = pd.Series(df['category'].values, index=df['place_id']).to_dict()
    poi_name_mapping = pd.Series(df['name'].values, index=df['place_id']).to_dict()
    
    return poi_category_mapping, poi_name_mapping




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
def process_input_file(input_file, poi_category_mapping, poi_name_mapping, apply_category_filter=True):
    user_data = defaultdict(list)
    # Read and group check-ins by user_id
    with open(input_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)  # Read all rows into memory for filtering

    # Convert rows to a DataFrame for filtering
    dataframe_user = pd.DataFrame(rows)

    # Map place_id to POI category and name
    dataframe_user['category'] = dataframe_user['place_id'].map(poi_category_mapping).fillna("Unknown Category")
    dataframe_user['place_name'] = dataframe_user['place_id'].map(poi_name_mapping).fillna("Unknown")
    
    # Clean NaN values in place_name column
    dataframe_user['place_name'] = dataframe_user['place_name'].apply(lambda x: safe_string(x, "Unknown"))

    # Filter out irrelevant categories only if apply_category_filter is True
    if apply_category_filter:
        print("Applying relevant category filter...")
        dataframe_user = filter_cat_distribution(dataframe_user)
    else:
        print("Keeping all categories (no filter applied)...")

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

        # Map place_id to POI category and name
        poi_category = poi_category_mapping.get(place_id, "Unknown Category")
        poi_name = safe_string(poi_name_mapping.get(place_id), "Unknown")
        hash_digest = hashlib.sha256(str(place_id).encode()).hexdigest()[:8]

        # Append check-in metadata
        user_data[user_id].append({
            "poi_id": f"POI-ID-{hash_digest}",
            "poi_name": poi_name,
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

# Process and save both versions of the JSON
def process_and_save_both_versions(input_file, poi_category_mapping, poi_name_mapping):
    print("\n" + "="*60)
    print("PROCESSING FILTERED VERSION (Relevant Categories Only)")
    print("="*60)
    
    # Process with category filter applied
    user_profiles_filtered = process_input_file(input_file, poi_category_mapping, poi_name_mapping, apply_category_filter=True)
    
    print("\n" + "="*60)
    print("PROCESSING ALL CATEGORIES VERSION")
    print("="*60)
    
    # Process without category filter
    user_profiles_all = process_input_file(input_file, poi_category_mapping, poi_name_mapping, apply_category_filter=False)
    
    print("\n" + "="*60)
    print("SAVING BOTH VERSIONS")
    print("="*60)
    
    # Save filtered version
    save_to_json(JSON_INPUT_FILTERED, user_profiles_filtered)
    print(f"✅ Filtered version: {len(user_profiles_filtered)} user profiles saved to {JSON_INPUT_FILTERED}")
    
    # Save all categories version
    save_to_json(JSON_INPUT_ALL, user_profiles_all)
    print(f"✅ All categories version: {len(user_profiles_all)} user profiles saved to {JSON_INPUT_ALL}")
    
    return user_profiles_filtered, user_profiles_all

# Main function
def main():
    # Paths
    input_file = SAMPLED_FSQ_PLANNING_AREA
    poi_mapping_path = POI_CAT_MAPPING

    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Load data
    poi_category_mapping, poi_name_mapping = load_poi_mapping(poi_mapping_path)
    print(f"Loaded {len(poi_category_mapping)} POI category mappings")
    print(f"Loaded {len(poi_name_mapping)} POI name mappings")

    # Process and save both versions
    user_profiles_filtered, user_profiles_all = process_and_save_both_versions(input_file, poi_category_mapping, poi_name_mapping)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"📊 Filtered version (relevant categories): {len(user_profiles_filtered)} users")
    print(f"📊 All categories version: {len(user_profiles_all)} users")
    print(f"📂 Files created:")
    print(f"   - Filtered: {JSON_INPUT_FILTERED}")
    print(f"   - All categories: {JSON_INPUT_ALL}")

if __name__ == "__main__":
    main()