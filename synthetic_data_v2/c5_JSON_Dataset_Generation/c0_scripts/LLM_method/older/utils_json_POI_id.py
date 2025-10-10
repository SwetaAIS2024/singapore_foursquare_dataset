import pandas as pd

from c0_Configuration.s00_config_paths import PLACE_ID_POI_CAT, CATEGORIES_XLSX

def generate_category_poiid_map(categories_xlsx):
    cat_df = pd.read_excel(categories_xlsx)
    cat_col = 'POI Category in Singapore'
    yes_col = 'Relevant to use case '
    relevant_cats = [
        cat.strip().lower()
        for cat, flag in zip(cat_df[cat_col], cat_df[yes_col])
        if str(flag).strip().lower() == 'yes' and cat and str(cat).strip()
    ]
    relevant_cats = list(dict.fromkeys(relevant_cats))
    # Assign POI-IDs to relevant categories
    cat_to_poiid = {cat: f"POI-{i+1}" for i, cat in enumerate(relevant_cats)}
    return cat_to_poiid, relevant_cats

def generate_placeid_poi_mapping(categories_xlsx, placeid_cat_csv):
    cat_to_poiid, relevant_cats = generate_category_poiid_map(categories_xlsx)
    cat_df = pd.read_excel(categories_xlsx)
    cat_col = 'POI Category in Singapore'
    subcat_col = 'POI Subcategory' if 'POI Subcategory' in cat_df.columns else None
    # Build category -> subcategory mapping (if available)
    if subcat_col:
        cat_map = {row[cat_col].strip().lower(): row[subcat_col] for _, row in cat_df.iterrows()}
    else:
        cat_map = {row[cat_col].strip().lower(): "" for _, row in cat_df.iterrows()}

    # Load place_id to category mapping
    place_cat_df = pd.read_csv(placeid_cat_csv)
    mapping = {}
    for _, row in place_cat_df.iterrows():
        pid = row['place_id']
        cat = str(row['category']).strip().lower()
        if cat in relevant_cats:
            mapping[pid] = {
                'poiId': cat_to_poiid[cat],
                'category': cat.title(),
                'subcategory': cat_map.get(cat, "")
            }
    return mapping

# test this block
if __name__ == "__main__":
    categories_xlsx = CATEGORIES_XLSX
    placeid_cat_csv = PLACE_ID_POI_CAT
    mapping = generate_placeid_poi_mapping(categories_xlsx, placeid_cat_csv)
    
    # Convert mapping to DataFrame and save as CSV
    mapping_df = pd.DataFrame.from_dict(mapping, orient='index')
    mapping_df.index.name = 'place_id'
    mapping_df.reset_index(inplace=True)
    mapping_df.to_csv("c5_JSON_Dataset_Generation/c0_scripts/LLM_based_method_gpt_oss_20b/mapping.csv", index=False)
    print("[INFO] Mapping saved to placeid_poi_mapping.csv")