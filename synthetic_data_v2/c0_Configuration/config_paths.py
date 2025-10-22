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


# Paths updated for the sampling script 
ORIGINAL_DATA_PATH = CHECKINS_PATH
CLUSTER_LABELS_PATH = os.path.join(BASE_DIR, "./c2_Clustering_and_Analysis/clustering_output/kmeans/user_cluster_labels_user_to_cluster.txt")
OUTPUT_SAMPLED_FSQ_PATH = os.path.join(BASE_DIR,"./c3_Sampling_After_Clustering/sampled_dataset_after_clustering/sampled_FSQ_dataset_after_clustering.txt")
OUTPUT_SAMPLED_FSQ_PATH_PLANNING_AREA = os.path.join(BASE_DIR,"c5_JSON_Dataset_Generation/c0_scripts/LLM_method/older/sampled_FSQ_dataset_with_planning_area.txt")

#sdv 
# SDV_OUTPUT_DIR = os.path.join(BASE_DIR,"./c3_Sampling_After_Clustering/c2_synthesis_outputs")
# SDV_OUTPUT_FILE = "from_cluster_samples_SDV_output.csv"
# SDV_OUTPUT = os.path.join(SDV_OUTPUT_DIR, SDV_OUTPUT_FILE)

#post synthesis analysis
# SDV_POST_SYNTHESIS_ANALYSIS_OUTPUT_DIR = os.path.join(BASE_DIR,"./c3_Sampling_After_Clustering/c3_output_analysis/analysis_SDV_from_cluster_samples")

# multi-table seq synthesis
# MULTI_TABLE_SEQ_SYN_ANALYSIS_OUTPUT_DIR = os.path.join(BASE_DIR,"./c3_Sampling_After_Clustering/c4_multi_table_seq")

# json dataset generation
JSON_OUTPUT_DIR = os.path.join(BASE_DIR, "./c5_JSON_Dataset_Generation/c1_outputs")
# each poi has a mapping id 
POI_UNIQUE_ID_MAPPING = os.path.join(BASE_DIR, "./c5_JSON_Dataset_Generation/c0_scripts/LLM_method/older/mapping.csv")

# DATA PROCESSING PIPELINE BEFORE GIVING TO MODEL PROMPT
# CLUSTER_SUMMARY = os.path.join(BASE_DIR, "c5_JSON_Dataset_Generation/c0_scripts/LLM_method/older/per_cluster_insights/all_clusters_summary.csv")
CLUSTER_SUMMARY_DIR = os.path.join(BASE_DIR, "c5_JSON_Dataset_Generation/older/per_cluster_insights")
CLUSTER_SUMMARY = os.path.join(BASE_DIR, "c5_JSON_Dataset_Generation/c0_scripts/LLM_method/older/per_cluster_insights/all_clusters_summary_with_area.csv")
SAMPLED_FSQ_PLANNING_AREA = os.path.join(BASE_DIR,"c3_Sampling_After_Clustering/sampled_dataset_after_clustering/sampled_FSQ_dataset_after_clustering.txt")
POI_CAT_MAPPING = os.path.join(BASE_DIR, "c1_Data_Collection_and_Processing/input_data/sg_place_id_to_category.csv")
FINAL_CHECKIN_FILE_TO_LLM = os.path.join(BASE_DIR, "c5_JSON_Dataset_Generation/c0_scripts/LLM_method/input_dataset/final_checkin_file.json")

# user summary extraction amd generation 
#PROMPT_TEMPLATE_PATH = os.path.join(BASE_DIR, "c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_summary_extraction/prompt_for_user_summary.json")
SYSTEM_PROMPT_TEMPLATE_PATH = os.path.join(BASE_DIR, "c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_summary_extraction/system_prompt.json")
SUMMARY_OUTPUT_FILE = os.path.join(BASE_DIR, "c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_summary_extraction/user_summaries_hf.csv")
FINAL_JSON_INPUT_LLM = os.path.join(BASE_DIR, "c5_JSON_Dataset_Generation/c0_scripts/LLM_method/input_dataset/final_checkin_file.json")
