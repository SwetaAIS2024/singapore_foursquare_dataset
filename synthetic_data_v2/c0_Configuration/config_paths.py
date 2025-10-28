import os

# Get the directory of the folder containing the script
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CONFIG_PATH = os.path.join(BASE_DIR, "./c0_Configuration/config_params.json")

MASTER_POI = os.path.join(BASE_DIR, "./c1_Data_Collection_and_Processing/input_data/master_poi_pool.json")

MATRIX_PATH = os.path.join(BASE_DIR, "./c2_Clustering_and_Analysis/clustering_output/matrix_output.npy")
CHECKINS_PATH = os.path.join(BASE_DIR, "./c1_Data_Collection_and_Processing/input_data/singapore_checkins_filtered_with_locations_coord.txt")
CATEGORIES_XLSX = os.path.join(BASE_DIR, "./c1_Data_Collection_and_Processing/input_data/Relevant_POI_category.xlsx")
PLACE_ID_POI_CAT = os.path.join(BASE_DIR, "./c1_Data_Collection_and_Processing/input_data/sg_place_id_to_category.csv")
FINAL_INPUT_DATASET = os.path.join(BASE_DIR, "./c1_Data_Collection_and_Processing/output/")
CLUSTER_OUTPUT_DIR = os.path.join(BASE_DIR, "./c2_Clustering_and_Analysis/clustering_output")
POST_SAMPLING_ANALYSIS_OUTPUT_DIR = os.path.join(BASE_DIR, "./c3_Sampling_After_Clustering/sampled_dataset_analysis/post_sampling_analysis_outputs")
DIMRED_MODEL_PATH = os.path.join(BASE_DIR, "./c2_Clustering_and_Analysis/clustering_output/clustering_kmeans.pkl")

# Individual cluster analysis paths
CLUSTER_DISTRIBUTION_SUMMARY_PATH = os.path.join(BASE_DIR, "./c2_Clustering_and_Analysis/individual_cluster_analysis/cluster_analysis/kmeans/cluster_summary.csv")
CLUSTER_DISTRIBUTION_PARAMS_PATH = os.path.join(BASE_DIR, "./c2_Clustering_and_Analysis/individual_cluster_analysis/cluster_analysis/kmeans")

#post clustering analysis paths
POST_SAMPLING_ANALYSIS_OUTPUT_DIR = os.path.join(BASE_DIR,"./c2_Clustering_and_Analysis/post_clustering_analysis/output/kmeans")

# Paths updated for the sampling script 
ORIGINAL_DATA_PATH = CHECKINS_PATH
CLUSTER_LABELS_PATH = os.path.join(BASE_DIR, "./c2_Clustering_and_Analysis/clustering_output/kmeans/user_cluster_labels_user_to_cluster.txt")
OUTPUT_SAMPLED_FSQ_PATH = os.path.join(BASE_DIR,"./c3_Sampling_After_Clustering/sampled_dataset_after_clustering/sampled_FSQ_dataset_after_clustering.txt")
OUTPUT_SAMPLED_FSQ_PATH_PLANNING_AREA = os.path.join(BASE_DIR,"c4_JSON_Dataset_Generation/c0_scripts/LLM_method/older/sampled_FSQ_dataset_with_planning_area.txt")

# json dataset generation
JSON_OUTPUT_DIR = os.path.join(BASE_DIR, "./c4_JSON_Dataset_Generation/output_syn_json")
# each poi has a mapping id 
POI_UNIQUE_ID_MAPPING = os.path.join(BASE_DIR, "./c4_JSON_Dataset_Generation/utils/u0_mapping/mapping.csv")

# DATA PROCESSING PIPELINE BEFORE GIVING TO MODEL PROMPT
CLUSTER_SUMMARY_DIR = os.path.join(BASE_DIR, "c4_JSON_Dataset_Generation/utils/u1_cluster_insights/per_cluster_insights")
CLUSTER_SUMMARY = os.path.join(BASE_DIR, "c4_JSON_Dataset_Generation/utils/u1_cluster_insights/per_cluster_insights/all_clusters_summary.csv")
CLUSTER_SUMMARY_WITH_PLANNING_AREA = os.path.join(BASE_DIR, "c4_JSON_Dataset_Generation/utils/u1_cluster_insights/per_cluster_insights/all_clusters_summary_with_area.csv")
SAMPLED_FSQ = os.path.join(BASE_DIR,"c3_Sampling_After_Clustering/sampled_dataset_after_clustering/sampled_FSQ_dataset_after_clustering.txt")
SAMPLED_FSQ_PLANNING_AREA = os.path.join(BASE_DIR,"c4_JSON_Dataset_Generation/utils/u3_sampled_fsq_add_planning_area/sampled_FSQ_dataset_with_planning_area.txt")
POI_CAT_MAPPING = os.path.join(BASE_DIR, "c1_Data_Collection_and_Processing/input_data/sg_place_id_to_category.csv")
JSON_INPUT = os.path.join(BASE_DIR, "c4_JSON_Dataset_Generation/input_sampled_fsq_json/input.json")
JSON_OUTPUT = os.path.join(BASE_DIR, "c4_JSON_Dataset_Generation/output_syn_json/synthetic_data.json")
ALL_POI_ID_INTERACTIONS = os.path.join(BASE_DIR, "c4_JSON_Dataset_Generation/utils/u4_extract_all_poi_id_interactions/all_pois.json")
JSON_FUNC = os.path.join(BASE_DIR, "c4_JSON_Dataset_Generation/func")
JSON_VAL = os.path.join(BASE_DIR, "c4_JSON_Dataset_Generation/validation_graph")
# Local cache paths
PLANNING_CACHE = os.path.join(BASE_DIR, "c4_JSON_Dataset_Generation/utils/u1_cluster_insights/planning_area.geojson")
SUBZONE_CACHE = os.path.join(BASE_DIR, "c4_JSON_Dataset_Generation/utils/u1_cluster_insights/subzone.geojson")