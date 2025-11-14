#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Centralized path configuration for Singapore FSQ synthetic data pipeline.

This module provides absolute paths to all data files, configurations,
and output directories used throughout the project.
"""
import os
from pathlib import Path

# Base directory (project root)
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_INTERIM = BASE_DIR / "data" / "interim"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_SYNTHETIC = BASE_DIR / "data" / "synthetic"

# Input files (raw data)
FSQ_CHECKINS_CSV = DATA_RAW / "FSQ_SG_2013_Checkins.csv"
FSQ_POI_CSV = DATA_RAW / "FSQ_SG_2013_POI.csv"
RELEVANT_CATEGORIES_XLSX = DATA_RAW / "Relevant_POI_category.xlsx"

# Processed input files
INPUT_TO_GENERATOR_INTERIM = DATA_INTERIM / "checkins_interim.csv"

# Processed input files
INPUT_TO_GENERATOR_FILTERED_JSON = DATA_PROCESSED / "input_filtered.json"
INPUT_TO_GENERATOR_ALL_CATEGORIES_JSON = DATA_PROCESSED / "input_all_categories.json"

# Synthetic output files
SYNTHETIC_FILTERED_JSON = DATA_SYNTHETIC / "fsq_to_synthetic_filtered_categories.json"
SYNTHETIC_ALL_CATEGORIES_JSON = (
    DATA_SYNTHETIC / "fsq_to_synthetic_all_categories.json"
)

# GeoJSON files (in project root)
PLANNING_AREA_GEOJSON = BASE_DIR / "planning_area.geojson"
SUBZONE_GEOJSON = BASE_DIR / "subzone.geojson"

# Legacy path aliases for backward compatibility
# TODO: Remove these once all scripts are updated
CHECKINS_PATH_CSV = str(FSQ_CHECKINS_CSV)
POI_COORDINATES_CSV = str(FSQ_POI_CSV)
CATEGORIES_XLSX = str(RELEVANT_CATEGORIES_XLSX)
JSON_OUTPUT_DIR = str(DATA_SYNTHETIC)
PLANNING_CACHE = str(PLANNING_AREA_GEOJSON)
SUBZONE_CACHE = str(SUBZONE_GEOJSON)
