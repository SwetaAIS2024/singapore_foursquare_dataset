#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
POI Extraction Utility for Final Dataset

Extracts unique POIs from the final dataset (with views, transactions, and reviews).
Creates a comprehensive POI reference with all metadata.

This script is specifically designed for the final_dataset output after views and reviews
have been extrapolated.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Input: Final dataset with views, transactions, and reviews
FINAL_DATASET_DIR = Path(__file__).parent.parent.parent / "data" / "final_dataset"
INPUT_FILE = FINAL_DATASET_DIR / "final_dataset_with_views_reviews.json"
OUTPUT_FILE = Path(__file__).parent / "all_pois_final_dataset.json"
STATS_FILE = FINAL_DATASET_DIR / "poi_extraction_stats.json"

# POI reference file for names
POI_REFERENCE_FILE = Path(__file__).parent.parent.parent / "data" / "processed" / "input_filtered.json"


def safe_float(val):
    """Convert value to float safely, returning None on failure."""
    try:
        return float(val)
    except Exception:
        return None


def safe_string(val, default="Unknown"):
    """
    Safely convert value to string, handling NaN, None, and empty values.
    
    Parameters:
    val: Value to convert
    default: Default value to return if conversion fails
    
    Returns:
    str: Converted string or default value
    """
    import math
    
    if val is None:
        return default
    
    # Handle NaN values (both float('nan') and string 'NaN')
    if isinstance(val, float) and math.isnan(val):
        return default
    
    # Handle string representations of NaN
    if isinstance(val, str) and val.lower() in ['nan', 'null', 'none', '']:
        return default
    
    # Return the string representation of the value
    return str(val).strip() or default


def load_poi_name_mapping():
    """
    Load POI names from the reference input file.
    
    Returns:
    dict: Mapping from poi_id to poi_name
    """
    poi_name_map = {}
    
    if not POI_REFERENCE_FILE.exists():
        print(f"⚠️  POI reference file not found: {POI_REFERENCE_FILE}")
        return poi_name_map
    
    print(f"📖 Loading POI names from: {POI_REFERENCE_FILE}")
    
    with open(POI_REFERENCE_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for user_data in data:
        user_metadata = user_data.get('user_metadata', [])
        for visit in user_metadata:
            poi_id = visit.get('poi_id')
            poi_name = visit.get('poi_name')
            
            if poi_id and poi_name and poi_id not in poi_name_map:
                poi_name_map[poi_id] = safe_string(poi_name, 'Unknown')
    
    print(f"✅ Loaded {len(poi_name_map):,} POI names\n")
    return poi_name_map


def extract_pois_from_final_dataset():
    """
    Extract POIs from final dataset with views, transactions, and reviews.
    
    Returns:
    tuple: (unique_pois dict, statistics dict)
    """
    print("\n" + "="*80)
    print("POI EXTRACTION FROM FINAL DATASET")
    print("="*80 + "\n")
    
    if not INPUT_FILE.exists():
        print(f"❌ Error: Input file not found: {INPUT_FILE}")
        return {}, {}
    
    # Load POI name mapping
    poi_name_map = load_poi_name_mapping()
    
    print(f"📂 Reading from: {INPUT_FILE}")
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"👥 Processing {len(data):,} users from final dataset...")
    
    unique_pois = {}
    poi_statistics = defaultdict(lambda: {
        'view_count': 0,
        'transaction_count': 0,
        'review_count': 0,
        'total_rating': 0.0,
        'rating_count': 0,
        'users': set()
    })
    
    interaction_counts = {
        'views': 0,
        'transactions': 0,
        'reviews': 0
    }
    
    for i, user_data in enumerate(data):
        user_id = user_data.get('user', {}).get('userId')
        interaction = user_data.get("interaction", {})
        
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1:,} users...")
        
        # Process views
        views = interaction.get("views", [])
        interaction_counts['views'] += len(views)
        
        for view in views:
            poi_id = view.get("poiId")
            if poi_id:
                poi_statistics[poi_id]['view_count'] += 1
                poi_statistics[poi_id]['users'].add(user_id)
                
                # Add POI if not seen before
                if poi_id not in unique_pois:
                    poi_categories = view.get("poiCategories", [])
                    user_location = view.get("userLocation", {})
                    
                    unique_pois[poi_id] = {
                        "poiId": poi_id,
                        "poiName": "Unknown",
                        "poiCategories": poi_categories if poi_categories else [],
                        "planningArea": None,
                        "userLocation": {
                            "latitude": safe_float(user_location.get("latitude")),
                            "longitude": safe_float(user_location.get("longitude"))
                        }
                    }
        
        # Process transactions
        transactions = interaction.get("transactions", [])
        interaction_counts['transactions'] += len(transactions)
        
        for transaction in transactions:
            poi_id = transaction.get("poiId")
            if poi_id:
                poi_statistics[poi_id]['transaction_count'] += 1
                poi_statistics[poi_id]['users'].add(user_id)
                
                # Update POI info (transaction has more complete data)
                if poi_id not in unique_pois:
                    unique_pois[poi_id] = {
                        "poiId": poi_id,
                        "poiName": "Unknown",
                        "poiCategories": [],
                        "planningArea": None,
                        "userLocation": {
                            "latitude": None,
                            "longitude": None
                        }
                    }
                
                poi_categories = transaction.get("poiCategories", [])
                planning_area = transaction.get("planning_area")
                user_location = transaction.get("userLocation", {})
                
                # Update with transaction data
                if poi_categories:
                    unique_pois[poi_id]["poiCategories"] = poi_categories
                if planning_area:
                    unique_pois[poi_id]["planningArea"] = planning_area
                
                unique_pois[poi_id]["userLocation"] = {
                    "latitude": safe_float(user_location.get("latitude")),
                    "longitude": safe_float(user_location.get("longitude"))
                }
        
        # Process reviews
        reviews = interaction.get("reviews", [])
        interaction_counts['reviews'] += len(reviews)
        
        for review in reviews:
            poi_id = review.get("poiId")
            if poi_id:
                poi_statistics[poi_id]['review_count'] += 1
                poi_statistics[poi_id]['users'].add(user_id)
                
                rating = review.get("rating")
                if rating is not None:
                    poi_statistics[poi_id]['total_rating'] += rating
                    poi_statistics[poi_id]['rating_count'] += 1
                
                # Update POI if not exists
                if poi_id not in unique_pois:
                    poi_categories = review.get("poiCategories", [])
                    user_location = review.get("userLocation", {})
                    
                    unique_pois[poi_id] = {
                        "poiId": poi_id,
                        "poiName": "Unknown",
                        "poiCategories": poi_categories if poi_categories else [],
                        "planningArea": None,
                        "userLocation": {
                            "latitude": safe_float(user_location.get("latitude")),
                            "longitude": safe_float(user_location.get("longitude"))
                        }
                    }
    
    print(f"\n📊 Interaction counts:")
    print(f"   Views:        {interaction_counts['views']:,}")
    print(f"   Transactions: {interaction_counts['transactions']:,}")
    print(f"   Reviews:      {interaction_counts['reviews']:,}")
    print(f"✅ Found {len(unique_pois):,} unique POIs")
    
    # Add POI names from mapping
    print(f"\n📝 Adding POI names from reference data...")
    
    names_found = 0
    for poi_id, poi_data in unique_pois.items():
        if poi_id in poi_name_map:
            poi_data['poiName'] = poi_name_map[poi_id]
            names_found += 1
        else:
            # Keep "Unknown" if no mapping found
            poi_data['poiName'] = "Unknown"
    
    print(f"✅ Mapped {names_found:,} POI names ({names_found/len(unique_pois)*100:.1f}%)")
    
    # Generate overall statistics
    overall_stats = {
        'total_pois': len(unique_pois),
        'total_users': len(data),
        'total_views': interaction_counts['views'],
        'total_transactions': interaction_counts['transactions'],
        'total_reviews': interaction_counts['reviews'],
        'avg_views_per_poi': interaction_counts['views'] / len(unique_pois) if unique_pois else 0,
        'avg_transactions_per_poi': interaction_counts['transactions'] / len(unique_pois) if unique_pois else 0,
        'avg_reviews_per_poi': interaction_counts['reviews'] / len(unique_pois) if unique_pois else 0,
        'pois_with_reviews': sum(1 for s in poi_statistics.values() if s['review_count'] > 0),
        'review_coverage': sum(1 for s in poi_statistics.values() if s['review_count'] > 0) / len(unique_pois) * 100 if unique_pois else 0
    }
    
    # Top POIs by activity
    sorted_pois = sorted(
        poi_statistics.items(),
        key=lambda x: x[1]['transaction_count'],
        reverse=True
    )
    
    print(f"\n🏆 Top 10 POIs by transaction count:")
    for i, (poi_id, stats) in enumerate(sorted_pois[:10], 1):
        poi = unique_pois.get(poi_id, {})
        category = poi.get('poiCategories', ['Unknown'])[0] if poi.get('poiCategories') else 'Unknown'
        avg_rating = None
        if stats['rating_count'] > 0:
            avg_rating = stats['total_rating'] / stats['rating_count']
        
        rating_str = f"{avg_rating:.1f}★" if avg_rating else "No ratings"
        print(f"   {i:2d}. {poi_id[:20]:20s} | {category:25s} | "
              f"{stats['transaction_count']:4d} txns | {stats['review_count']:3d} reviews | {rating_str}")
    
    return unique_pois, overall_stats


def main():
    """
    Main execution function for POI extraction from final dataset.
    """
    # Extract POIs and statistics
    unique_pois, overall_stats = extract_pois_from_final_dataset()
    
    if not unique_pois:
        print("\n❌ No POIs extracted. Exiting.")
        return 1
    
    # Convert to list for JSON output
    unique_pois_list = list(unique_pois.values())
    
    # Save POI data
    print(f"\n💾 Saving POI data to: {OUTPUT_FILE}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(unique_pois_list, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(unique_pois_list):,} POIs to {OUTPUT_FILE.name}")
    
    # Save statistics
    print(f"💾 Saving statistics to: {STATS_FILE}")
    
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(overall_stats, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved statistics to {STATS_FILE.name}")
    
    # Display sample POI
    if unique_pois_list:
        print(f"\n📋 Sample POI format:")
        sample_poi = unique_pois_list[0]
        print(json.dumps(sample_poi, indent=2, ensure_ascii=False))
    
    # Display overall statistics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"Total POIs:              {overall_stats['total_pois']:,}")
    print(f"Total Users:             {overall_stats['total_users']:,}")
    print(f"Total Views:             {overall_stats['total_views']:,}")
    print(f"Total Transactions:      {overall_stats['total_transactions']:,}")
    print(f"Total Reviews:           {overall_stats['total_reviews']:,}")
    print(f"Avg Views/POI:           {overall_stats['avg_views_per_poi']:.2f}")
    print(f"Avg Transactions/POI:    {overall_stats['avg_transactions_per_poi']:.2f}")
    print(f"Avg Reviews/POI:         {overall_stats['avg_reviews_per_poi']:.2f}")
    print(f"POIs with Reviews:       {overall_stats['pois_with_reviews']:,} ({overall_stats['review_coverage']:.1f}%)")
    
    print(f"\n{'='*80}")
    print("✅ EXTRACTION COMPLETE")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
