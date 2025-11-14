# Raw Data Directory

## Contents

- `FSQ_SG_2013_Checkins.csv` - Original Foursquare Singapore check-in data (2013)
- `FSQ_SG_2013_POI.csv` - POI metadata (names, categories, coordinates)
- `Relevant_POI_category.xlsx` - Filtered POI category mappings

## Important

⚠️ **DO NOT MODIFY FILES IN THIS DIRECTORY**

These are the original raw datasets. All transformations should be done in the processing pipeline.

## Data Provenance

- **Source**: Foursquare Singapore Dataset
- **Year**: 2013
- **Format**: CSV (Tab-separated)
- **License**: Research use only

## Processing Pipeline

1. **Raw Data** (this directory) → Never modified
2. **Interim Data** (`data/interim/`) → Temporary transformations
3. **Processed Data** (`data/processed/`) → Clean input for generation
4. **Synthetic Data** (`data/synthetic/`) → Final outputs

## Data Schema

### FSQ_SG_2013_Checkins.csv
- User ID, POI ID, Timestamp, Location coordinates

### FSQ_SG_2013_POI.csv
- POI ID, Name, Category, Latitude, Longitude
