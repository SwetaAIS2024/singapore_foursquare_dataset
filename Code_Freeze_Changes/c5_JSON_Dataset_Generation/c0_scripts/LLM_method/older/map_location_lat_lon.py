import os
import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from bs4 import BeautifulSoup  # For parsing HTML content

# Local cache paths
PLANNING_CACHE = "C:\\Users\\admin\\Desktop\\sweta\\MPS_syn_data_gen\\singapore_foursquare_dataset\\Code_Freeze_Changes\\c5_JSON_Dataset_Generation\\c0_scripts\\LLM_method\\older\\planning_area.geojson"
SUBZONE_CACHE = "C:\\Users\\admin\\Desktop\\sweta\\MPS_syn_data_gen\\singapore_foursquare_dataset\\Code_Freeze_Changes\\c5_JSON_Dataset_Generation\\c0_scripts\\LLM_method\\older\\subzone.geojson"
INPUT_FILE = "C:\\Users\\admin\\Desktop\\sweta\\MPS_syn_data_gen\\singapore_foursquare_dataset\\Code_Freeze_Changes\\c3_Synthetic_FSQ_Samples_SDV\\c0_sampling_scripts\\sampled_dataset_after_clustering\\sampled_FSQ_dataset_after_clustering.txt"
OUTPUT_FILE = "C:\\Users\\admin\\Desktop\\sweta\\MPS_syn_data_gen\\singapore_foursquare_dataset\\Code_Freeze_Changes\\c5_JSON_Dataset_Generation\\c0_scripts\\LLM_method\\older\\sampled_FSQ_dataset_with_planning_area.txt"

def _load_geojson(cache_path: str) -> gpd.GeoDataFrame:
    """
    Load a GeoJSON file into a GeoDataFrame.
    """
    if os.path.exists(cache_path):
        print(f"Loading GeoJSON from local cache: {cache_path}")
        return gpd.read_file(cache_path)
    else:
        raise RuntimeError(f"GeoJSON file not found at {cache_path}")

try:
    planning_gdf = _load_geojson(PLANNING_CACHE)
    subzone_gdf = _load_geojson(SUBZONE_CACHE)
except RuntimeError as e:
    print(f"Error loading GeoJSON data: {e}")
    planning_gdf = None
    subzone_gdf = None

# Debugging Block
if planning_gdf is not None:
    print("Planning CRS:", planning_gdf.crs)
    print("Planning geometries valid:", planning_gdf.geometry.is_valid.all())
    print("Planning bounds:", planning_gdf.total_bounds)
    if planning_gdf.crs != "EPSG:4326":
        planning_gdf = planning_gdf.to_crs("EPSG:4326")
    if not planning_gdf.geometry.is_valid.all():
        planning_gdf["geometry"] = planning_gdf["geometry"].buffer(0)

if subzone_gdf is not None:
    print("Subzone CRS:", subzone_gdf.crs)
    print("Subzone geometries valid:", subzone_gdf.geometry.is_valid.all())
    print("Subzone bounds:", subzone_gdf.total_bounds)
    if subzone_gdf.crs != "EPSG:4326":
        subzone_gdf = subzone_gdf.to_crs("EPSG:4326")
    if not subzone_gdf.geometry.is_valid.all():
        subzone_gdf["geometry"] = subzone_gdf["geometry"].buffer(0)

def _extract_pln_area_n(description: str) -> str:
    """
    Extract the PLN_AREA_N value from the Description field using BeautifulSoup.
    """
    soup = BeautifulSoup(description, "html.parser")
    rows = soup.find_all("tr")  # Find all rows in the table
    for row in rows:
        th = row.find("th")  # Find the <th> tag in the row
        td = row.find("td")  # Find the <td> tag in the row
        if th and td and th.text.strip() == "PLN_AREA_N":  # Match PLN_AREA_N
            return td.text.strip()  # Return the corresponding <td> value
    return None

def lookup_area(lat: float, lon: float, granularity: str = "planning") -> dict:
    """
    Lookup the planning area or subzone for a given latitude and longitude.
    """
    if granularity == "planning" and planning_gdf is None:
        return {"granularity": granularity, "name": None, "index": None}
    if granularity == "subzone" and subzone_gdf is None:
        return {"granularity": granularity, "name": None, "index": None}
    
    pt = Point(round(lon, 6), round(lat, 6))  # Ensure precision consistency
    gdf = planning_gdf if granularity == "planning" else subzone_gdf

    # Query spatial index
    cand_idx = list(gdf.sindex.query(pt))

    if not cand_idx:
        return {"granularity": granularity, "name": None, "index": None}

    # Check for containment or boundary inclusion using .within()
    hits = gdf.iloc[cand_idx][gdf.iloc[cand_idx].geometry.apply(lambda geom: pt.within(geom))]

    if hits.empty:
        # Check for intersection as a fallback
        hits = gdf.iloc[cand_idx][gdf.iloc[cand_idx].intersects(pt)]

    if hits.empty:
        return {"granularity": granularity, "name": None, "index": None}

    # Extract the PLN_AREA_N value from the Description field
    row = hits.iloc[0]
    description = row.get("Description", "")
    pln_area_n = _extract_pln_area_n(description)

    return {"granularity": granularity, "name": pln_area_n, "index": int(row.name)}

def append_planning_area(input_file: str, output_file: str):
    """
    Read the input file, append the planning area for each row, and save to the output file.
    """
    # Load the input dataset
    df = pd.read_csv(input_file, sep="\t")  # Assuming tab-separated values
    print(f"Loaded {len(df)} rows from {input_file}")

    # Add a new column for the planning area
    planning_areas = []
    for index, row in df.iterrows():
        lat = row["lat"]  # Replace with the actual column name for latitude
        lon = row["lon"]  # Replace with the actual column name for longitude
        planning_area = lookup_area(lat, lon, "planning")["name"]
        planning_areas.append(planning_area)

    df["planning_area"] = planning_areas

    # Save the updated dataset to the output file
    df.to_csv(output_file, sep="\t", index=False)
    print(f"Saved updated dataset with planning areas to {output_file}")

# --- Append Planning Area ---
append_planning_area(INPUT_FILE, OUTPUT_FILE)