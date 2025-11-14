#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess raw FSQ data to input JSON format.

This script converts raw Foursquare CSV files into the JSON format
required by the synthetic data generation pipeline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.fsq_to_input_json import main as preprocess_main


if __name__ == "__main__":
    preprocess_main()
