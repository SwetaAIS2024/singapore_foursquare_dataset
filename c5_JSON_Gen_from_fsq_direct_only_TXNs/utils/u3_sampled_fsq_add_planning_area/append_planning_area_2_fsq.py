import os
import json
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from bs4 import BeautifulSoup
from c0_Configuration.config_paths import CHECKINS_PATH_CSV, POI_COORDINATES_CSV,  PLANNING_CACHE, SUBZONE_CACHE

# Local cache paths
INPUT_FILE = CHECKINS_PATH_CSV
OUTPUT_FILE = "C:\\Users\\admin\\Desktop\\sweta\\MPS_syn_data_gen\\singapore_foursquare_dataset\\synthetic_data_v2\\c5_JSON_Gen_from_fsq_direct_only_TXNs\\utils\\u3_sampled_fsq_add_planning_area\\sampled_FSQ_dataset_with_planning_area.txt"


def _load_geojson(cache_path: str) -> gpd.GeoDataFrame:
    """
    Load a GeoJSON file into a GeoDataFrame.
    """
    if os.path.exists(cache_path):
        print(f"Loading GeoJSON from local cache: {cache_path}")
        return gpd.read_file(cache_path)
    else:
        raise RuntimeError(f"GeoJSON file not found at {cache_path}")
    
def _extract_pln_area_n(description: str) -> str:
    """
    Extract the PLN_AREA_N value from the Description field using BeautifulSoup.
    """
    if not description or pd.isna(description):
        return None
    
    soup = BeautifulSoup(str(description), "html.parser")
    rows = soup.find_all("tr")
    for row in rows:
        th = row.find("th")
        td = row.find("td")
        if th and td and th.text.strip() == "PLN_AREA_N":
            return td.text.strip()
    return None

def lookup_area(lat: float, lon: float, granularity: str = "planning", gdf=None) -> str:
    """
    Lookup the planning area or subzone for a given latitude and longitude.
    Returns the area name directly (or None).
    """
    if gdf is None or gdf.empty:
        return None
    
    # Create point (lon, lat order for Shapely)
    pt = Point(lon, lat)

    try:
        # Query spatial index - more efficient
        possible_matches_idx = list(gdf.sindex.query(pt, predicate='intersects'))
        
        if not possible_matches_idx:
            return None
        
        # Get candidate polygons
        candidates = gdf.iloc[possible_matches_idx]
        
        # Check which polygon contains the point
        mask = candidates.geometry.contains(pt)
        hits = candidates[mask]
        
        if hits.empty:
            # Fallback: small buffer for boundary cases
            buffered_pt = pt.buffer(0.00001)  # ~1 meter
            mask = candidates.geometry.intersects(buffered_pt)
            hits = candidates[mask]
        
        if hits.empty:
            return None
        
        # Extract planning area name from Description field
        row = hits.iloc[0]
        description = row.get("Description", "")
        pln_area_n = _extract_pln_area_n(description)
        
        return pln_area_n
        
    except Exception as e:
        print(f"Error looking up area for ({lat}, {lon}): {e}")
        return None

def append_planning_area(input_file: str, output_file: str, gdf):
    """
    Read the input file, append the planning area for each row, and save to the output file.
    Rows without planning area will be kept with empty/null planning_area values.
    """
    # Load the checkin dataset - CSV file WITHOUT headers
    # Columns: user_id, place_id, datetime, timezone (NO lat/lon)
    print(f"Loading checkins from: {input_file}")
    df_checkins = pd.read_csv(input_file, sep=",", header=None, 
                              names=['user_id', 'place_id', 'datetime', 'timezone'])
    print(f"Loaded {len(df_checkins)} checkin rows from {input_file}")
    
    # Load POI coordinates from separate file using configured path
    print(f"Loading POI coordinates from: {POI_COORDINATES_CSV}")
    df_poi = pd.read_csv(POI_COORDINATES_CSV, sep=",", header=None,
                         names=['place_id', 'name', 'lat', 'lon', 'category', 'country'])
    print(f"Loaded {len(df_poi)} POI rows from {POI_COORDINATES_CSV}")
    
    # Merge checkins with POI coordinates by place_id
    print("Merging checkins with POI coordinates...")
    df = df_checkins.merge(df_poi[['place_id', 'lat', 'lon']], on='place_id', how='left')
    print(f"After merge: {len(df)} rows")
    
    # Check how many checkins have coordinates
    has_coords = df[df['lat'].notna() & df['lon'].notna()]
    print(f"Checkins with coordinates: {len(has_coords)} ({len(has_coords)/len(df)*100:.1f}%)")
    
    # Use the correct column names
    lat_col = 'lat'
    lon_col = 'lon'
    
    print(f"Using columns: latitude='{lat_col}', longitude='{lon_col}'")
    
    # Validate coordinates
    print(f"Checking for missing coordinates...")
    missing = df[df[lat_col].isna() | df[lon_col].isna()]
    if len(missing) > 0:
        print(f"WARNING: {len(missing)} rows have missing coordinates")
    
    # Show coordinate bounds
    print(f"Coordinate bounds: {lat_col}=[{df[lat_col].min():.6f}, {df[lat_col].max():.6f}], "
          f"{lon_col}=[{df[lon_col].min():.6f}, {df[lon_col].max():.6f}]")

    # Process in chunks with progress
    print(f"\nProcessing {len(df)} rows...")
    planning_areas = []
    none_count = 0
    
    for idx, row in df.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]
        
        planning_area = lookup_area(lat, lon, "planning", gdf=gdf)
        planning_areas.append(planning_area)
        
        if planning_area is None:
            none_count += 1
        
        # Progress every 5000 rows
        if (idx + 1) % 5000 == 0:
            pct_complete = (idx + 1) / len(df) * 100
            pct_matched = ((idx + 1 - none_count) / (idx + 1)) * 100 if idx > 0 else 0
            print(f"Progress: {idx + 1}/{len(df)} ({pct_complete:.1f}%) - "
                  f"Matched: {pct_matched:.1f}%")

    df["planning_area"] = planning_areas
    
    # Summary before cleaning
    print(f"\n{'='*60}")
    print(f"SUMMARY BEFORE CLEANING")
    print(f"{'='*60}")
    print(f"Total rows processed: {len(df):,}")
    print(f"Rows WITH planning area: {len(df) - none_count:,} ({(len(df)-none_count)/len(df)*100:.1f}%)")
    print(f"Rows WITHOUT planning area: {none_count:,} ({none_count/len(df)*100:.1f}%)")
    
    if none_count > 0:
        print(f"\nSample rows without planning area (will be kept with null values):")
        no_area = df[df['planning_area'].isna()].head(5)
        print(no_area[['user_id', 'place_id', 'lat', 'lon', 'planning_area']])
    
    # KEEP ALL ROWS - INCLUDING THOSE WITHOUT PLANNING AREA
    print(f"\n{'='*60}")
    print(f"PREPARING FINAL DATA")
    print(f"{'='*60}")
    initial_count = len(df)
    df_cleaned = df.copy()  # Keep all rows
    
    print(f"Keeping all {len(df_cleaned):,} rows (100.0% of original data)")
    print(f"Rows with planning area: {len(df_cleaned) - none_count:,} ({(len(df_cleaned)-none_count)/len(df_cleaned)*100:.1f}%)")
    print(f"Rows without planning area: {none_count:,} ({none_count/len(df_cleaned)*100:.1f}%)")
    
    # Summary of final dataset
    print(f"\n{'='*60}")
    print(f"FINAL DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total rows in final dataset: {len(df_cleaned):,}")
    print(f"Rows with valid planning areas: {df_cleaned['planning_area'].notna().sum():,}")
    print(f"Rows with null planning areas: {df_cleaned['planning_area'].isna().sum():,}")
    
    # Show distribution (including null values)
    print(f"\nPlanning Area Distribution (Top 15):")
    area_counts = df_cleaned['planning_area'].value_counts(dropna=False)
    for idx, (area, count) in enumerate(area_counts.head(15).items(), 1):
        pct = count / len(df_cleaned) * 100
        area_display = "NULL/UNKNOWN" if pd.isna(area) else str(area)
        print(f"{idx:2d}. {area_display:<20} {count:>6,} ({pct:>5.2f}%)")
    
    # Additional statistics
    print(f"\nDataset Statistics:")
    print(f"Number of unique planning areas (excl. null): {df_cleaned['planning_area'].nunique()}")
    print(f"Number of unique planning areas (incl. null): {df_cleaned['planning_area'].nunique() + (1 if df_cleaned['planning_area'].isna().any() else 0)}")
    print(f"Number of unique users: {df_cleaned['user_id'].nunique()}")
    print(f"Number of unique places: {df_cleaned['place_id'].nunique()}")

    # Save the complete dataset
    print(f"\n{'='*60}")
    print(f"SAVING COMPLETE DATA")
    print(f"{'='*60}")
    print(f"Saving to {output_file}...")
    df_cleaned.to_csv(output_file, sep="\t", index=False)
    print(f"✅ Saved complete dataset with planning areas")
    print(f"   File: {output_file}")
    print(f"   Rows: {len(df_cleaned):,}")
    print(f"   Rows with planning areas: {df_cleaned['planning_area'].notna().sum():,}")
    print(f"   Rows without planning areas: {df_cleaned['planning_area'].isna().sum():,}")


# ==========================================
# MAIN EXECUTION
# ==========================================

print("="*60)
print("LOADING GEOJSON FILES")
print("="*60)

try:
    planning_gdf = _load_geojson(PLANNING_CACHE)
    subzone_gdf = _load_geojson(SUBZONE_CACHE)
except RuntimeError as e:
    print(f"❌ Error loading GeoJSON data: {e}")
    planning_gdf = None
    subzone_gdf = None
    exit(1)

# Validate and fix planning GDF
if planning_gdf is not None:
    print(f"\n{'='*60}")
    print("PLANNING AREA GDF INFO")
    print(f"{'='*60}")
    print(f"CRS: {planning_gdf.crs}")
    print(f"Number of polygons: {len(planning_gdf)}")
    print(f"Bounds: {planning_gdf.total_bounds}")
    print(f"Columns: {list(planning_gdf.columns)}")
    print(f"Geometries valid: {planning_gdf.geometry.is_valid.all()}")
    
    # Convert CRS if needed
    if planning_gdf.crs != "EPSG:4326":
        print("Converting to EPSG:4326...")
        planning_gdf = planning_gdf.to_crs("EPSG:4326")
    
    # Fix invalid geometries
    if not planning_gdf.geometry.is_valid.all():
        print("Fixing invalid geometries...")
        invalid_count = (~planning_gdf.geometry.is_valid).sum()
        print(f"Found {invalid_count} invalid geometries")
        planning_gdf["geometry"] = planning_gdf.geometry.buffer(0)
        print(f"After fix - Geometries valid: {planning_gdf.geometry.is_valid.all()}")
    
    # Show sample description
    if 'Description' in planning_gdf.columns:
        sample_desc = planning_gdf['Description'].iloc[0]
        print(f"\nSample Description field (first 200 chars):")
        print(str(sample_desc)[:200])
        
        # Try to extract area name from sample
        sample_area = _extract_pln_area_n(sample_desc)
        print(f"Extracted area name from sample: {sample_area}")

# Validate subzone GDF
if subzone_gdf is not None:
    print(f"\n{'='*60}")
    print("SUBZONE GDF INFO")
    print(f"{'='*60}")
    print(f"CRS: {subzone_gdf.crs}")
    print(f"Number of polygons: {len(subzone_gdf)}")
    print(f"Bounds: {subzone_gdf.total_bounds}")
    print(f"Geometries valid: {subzone_gdf.geometry.is_valid.all()}")
    
    if subzone_gdf.crs != "EPSG:4326":
        subzone_gdf = subzone_gdf.to_crs("EPSG:4326")
    if not subzone_gdf.geometry.is_valid.all():
        subzone_gdf["geometry"] = subzone_gdf.geometry.buffer(0)

# Process the data
print(f"\n{'='*60}")
print("PROCESSING DATA")
print(f"{'='*60}")

if planning_gdf is not None and not planning_gdf.empty:
    append_planning_area(INPUT_FILE, OUTPUT_FILE, planning_gdf)
else:
    print("❌ Cannot proceed: Planning GeoDataFrame not loaded or empty")