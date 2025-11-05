from pyrosm import OSM
import geopandas as gpd
from shapely.geometry import Point

# Load the .pbf file
osm = OSM("C:\\Users\\admin\\Desktop\\sweta\\MPS_syn_data_gen\\singapore_foursquare_dataset\\Code_Freeze_Changes\\c5_JSON_Dataset_Generation\\c0_scripts\\LLM_method\\input_dataset\\mean_lat_lon_area\\malaysia-singapore-brunei-latest.osm.pbf")

# Extract administrative boundaries
boundaries = osm.get_boundaries()

# Inspect unique admin_level values
print(boundaries["admin_level"].unique())

# Adjust the admin_level filter based on your dataset
planning_areas = boundaries[boundaries["admin_level"] == "8"]  # Adjust as needed

# Save the planning areas to a GeoDataFrame
planning_areas_gdf = gpd.GeoDataFrame(planning_areas)

def get_planning_area_offline(mean_lat, mean_lon, planning_areas_gdf):
    """
    Map latitude and longitude to a planning area using polygons from the .pbf file.
    """
    point = Point(mean_lon, mean_lat)  # Note: Point takes (lon, lat)
    for _, area in planning_areas_gdf.iterrows():
        print(f"Checking area: {area['name']} with geometry: {area['geometry']}")  # Debugging
        if area['geometry'].contains(point):
            return area['name']  # Replace 'name' with the correct column name
    return "Unknown Area"

# Example usage
mean_lat = 1.3141891061838895
mean_lon = 103.86208437103336
planning_area = get_planning_area_offline(mean_lat, mean_lon, planning_areas_gdf)
print(f"Planning Area: {planning_area}")