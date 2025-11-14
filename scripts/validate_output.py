#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate synthetic dataset quality.

This script validates the quality and correctness of generated
synthetic transaction datasets.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.validate_transactions import main as validate_main


if __name__ == "__main__":
    validate_main()
