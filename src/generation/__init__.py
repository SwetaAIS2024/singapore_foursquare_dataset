"""
Data generation module.

This module contains the core synthetic data generation logic,
converting Foursquare check-ins to transaction-only datasets.
"""
from .transaction_generator import process_input_to_synthetic

__all__ = ['process_input_to_synthetic']
