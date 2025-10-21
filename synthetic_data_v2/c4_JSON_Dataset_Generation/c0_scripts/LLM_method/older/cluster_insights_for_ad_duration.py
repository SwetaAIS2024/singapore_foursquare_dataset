import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIG ---
from synthetic_data_v2.c0_Configuration.config_paths import PLACE_ID_POI_CAT, CATEGORIES_XLSX
SAMPLED_FILE = "c3_Synthetic_FSQ_Samples_SDV/c0_sampling_scripts/sampled_dataset_after_clustering/sampled_FSQ_dataset_after_clustering.txt"  # Tab-separated
MAPPING_FILE = "c5_JSON_Dataset_Generation/c0_scripts/LLM_based_method_gpt_oss_20b/mapping.csv"  # place_id,poiId,category,subcategory
OUTPUT_DIR = "c5_JSON_Dataset_Generation/c0_scripts/LLM_based_method_gpt_oss_20b/per_cluster_insights"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD DATA ---
df = pd.read_csv(SAMPLED_FILE, sep="\t", parse_dates=["datetime"])
mapping = pd.read_csv(MAPPING_FILE)
df = df.merge(mapping[["place_id", "category"]], on="place_id", how="left")
# df["category"] = df["category"].fillna("Unknown")

# for the unkwown categories, we will use a fallback mapping from PLACE_ID_POI_CAT
# Load the fallback mapping (place_id to category) from PLACE_ID_POI_CAT
place_cat_df = pd.read_csv(PLACE_ID_POI_CAT)
place_cat_df.rename(columns={"category": "fallback_category"}, inplace=True)

# Merge fallback categories for unknowns
df = df.merge(place_cat_df[["place_id", "fallback_category"]], on="place_id", how="left")

# Fill 'Unknown' categories with fallback if available
df["category"] = df.apply(
    lambda row: row["fallback_category"] if row["category"] == "Unknown" and pd.notnull(row["fallback_category"]) else row["category"],
    axis=1
)

# Optionally drop the helper column
df.drop(columns=["fallback_category"], inplace=True)

# --- TIME FEATURES ---
df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
df["dayofweek"] = pd.to_datetime(df["datetime"]).dt.day_name()

# Summary csv
summary_list = []

# --- PER CLUSTER ANALYSIS ---
for cluster in sorted(df["cluster_id"].unique()):
    cluster_df = df[df["cluster_id"] == cluster]

    # Hour-of-day plot
    plt.figure(figsize=(8, 4))
    sns.histplot(cluster_df["hour"], bins=24, kde=False)
    plt.title(f"Cluster {cluster}: Check-in Hour Distribution")
    plt.xlabel("Hour of Day")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cluster_{cluster}_hour_distribution.png")
    plt.close()

    # Day-of-week plot
    plt.figure(figsize=(8, 4))
    sns.countplot(x="dayofweek", data=cluster_df, order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    plt.title(f"Cluster {cluster}: Check-in Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cluster_{cluster}_dayofweek_distribution.png")
    plt.close()

    # POI Category Frequency
    plt.figure(figsize=(10, 4))
    top_cats = cluster_df["category"].value_counts().head(10)
    sns.barplot(x=top_cats.index, y=top_cats.values)
    plt.title(f"Cluster {cluster}: Top 10 POI Categories")
    plt.xlabel("POI Category")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cluster_{cluster}_top_poi_categories.png")
    plt.close()

    # Spatial Distribution
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x="lon", y="lat", data=cluster_df, alpha=0.5)
    plt.title(f"Cluster {cluster}: Spatial Distribution")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cluster_{cluster}_spatial_distribution.png")
    plt.close()

    # Summary Table
    summary = {
        "num_users": cluster_df["user_id"].nunique(),
        "num_checkins": len(cluster_df),
        "top_poi_categories": ", ".join(top_cats.index.tolist()),
        "mean_lat": cluster_df["lat"].mean(),
        "mean_lon": cluster_df["lon"].mean(),
        "hour_peak": cluster_df["hour"].mode().iloc[0] if not cluster_df["hour"].mode().empty else None,
        "day_peak": cluster_df["dayofweek"].mode().iloc[0] if not cluster_df["dayofweek"].mode().empty else None,
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"{OUTPUT_DIR}/cluster_{cluster}_summary.csv", index=False)
    summary_list.append(summary)

all_summary_df = pd.DataFrame(summary_list)
all_summary_df.to_csv(f"{OUTPUT_DIR}/all_clusters_summary.csv", index=False)
print(f"Cluster analysis complete. Plots and tables saved in {OUTPUT_DIR}/")