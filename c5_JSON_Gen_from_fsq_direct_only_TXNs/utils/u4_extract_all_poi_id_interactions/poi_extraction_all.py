import json
import os


# Get script directory and construct paths relative to it
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Go up to c5_JSON_Dataset_Gen_from_fsq_direct/

# Updated to read from the final preprocessed synthetic datasets
preprocessing_dir = os.path.join(BASE_DIR, "output_syn_json", "preprocessing")
input_files = [
    os.path.join(preprocessing_dir, "all_categories_5core_filtered.json")
    # os.path.join(preprocessing_dir, "filtered_5core_filtered.json")
]
output_file = os.path.join(SCRIPT_DIR, "all_pois_just5_core_synthetic.json")

# code
unique_pois = {}

def safe_float(val):
    try:
        return float(val)
    except Exception:
        return None

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

# Process all available final synthetic dataset files
for input_file in input_files:
    if not os.path.exists(input_file):
        print(f"Skipping {os.path.basename(input_file)} - file not found")
        continue
    
    print(f"Reading from final synthetic dataset: {input_file}")
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Processing {len(data)} users from {os.path.basename(input_file)}...")

    poi_count = 0
    for i, user_data in enumerate(data):
        # Extract transactions from the new structure
        interaction = user_data.get("interaction", {})
        transactions = interaction.get("transactions", [])
        
        print(f"User {i} has {len(transactions)} transactions")
        
        for transaction in transactions:
            poi_count += 1
            poi_id = transaction.get("poiId")
            if not poi_id or poi_id in unique_pois:
                continue
            
            # Extract POI information from transaction
            poi_name = safe_string(transaction.get("poiName"), "Unknown")
            poi_categories = transaction.get("poiCategories", [])
            planning_area = transaction.get("planning_area")
            user_location = transaction.get("userLocation", {})
            lat = safe_float(user_location.get("latitude"))
            lon = safe_float(user_location.get("longitude"))
            
            if None in (planning_area, lat, lon) or not poi_categories:
                continue
            
            unique_pois[poi_id] = {
                "poiId": poi_id,
                "poiName": poi_name,
                "poiCategories": poi_categories,
                "planningArea": planning_area,
                "userLocation": {
                    "latitude": lat,
                    "longitude": lon
                }
            }
            
            if len(unique_pois) <= 5:  # Show first few POIs
                print(f"Added POI: {poi_id} - {poi_name} in {planning_area}")
    
    print(f"Found {len(unique_pois)} unique POIs so far from {os.path.basename(input_file)}")

unique_pois_list = list(unique_pois.values())


with open(output_file, "w", encoding="utf-8") as f:
    json.dump(unique_pois_list, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(unique_pois_list)} unique POIs from final synthetic dataset to {output_file}")
print(f"Sample POI format:")
if unique_pois_list:
    sample_poi = unique_pois_list[0]
    print(json.dumps(sample_poi, indent=2, ensure_ascii=False))