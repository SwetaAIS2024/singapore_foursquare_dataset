import os

# Get the directory of the folder containing the script
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CHECKINS_PATH_CSV = os.path.join(BASE_DIR, "./Input_data/FSQ_SG_2013_Checkins.csv")
POI_COORDINATES_CSV = os.path.join(BASE_DIR, "./Input_data/FSQ_SG_2013_POI.csv")
CATEGORIES_XLSX = os.path.join(BASE_DIR, "./Input_data/Relevant_POI_category.xlsx")

# json dataset generation
JSON_OUTPUT_DIR = os.path.join(BASE_DIR, "./json_gen/output_syn_json")
# each poi has a mapping id 
POI_UNIQUE_ID_MAPPING = os.path.join(BASE_DIR, "./json_gen/utils/u0_mapping/mapping.csv")


FSQ_WITH_PLANNING_AREA = os.path.join(BASE_DIR,"json_gen/utils/fsq_add_planning_area/FSQ_dataset_with_planning_area.txt")
JSON_INPUT = os.path.join(BASE_DIR, "json_gen/input_sampled_fsq_json/input.json")
JSON_OUTPUT = os.path.join(BASE_DIR, "json_gen/output_syn_json/synthetic_data.json")
ALL_POI_ID_INTERACTIONS = os.path.join(BASE_DIR, "json_gen/utils/extract_all_poi_id_interactions/all_pois.json")
JSON_FUNC = os.path.join(BASE_DIR, "json_gen/func")
JSON_VAL = os.path.join(BASE_DIR, "json_gen/validation_graph")
# Local cache paths
PLANNING_CACHE = os.path.join(BASE_DIR, "json_gen/utils/u1_cluster_insights/planning_area.geojson")
SUBZONE_CACHE = os.path.join(BASE_DIR, "json_gen/utils/u1_cluster_insights/subzone.geojson")