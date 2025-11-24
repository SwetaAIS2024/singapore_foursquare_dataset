#!/usr/bin/env python3
"""
Extrapolate Views and Reviews from Synthetic Transaction Data

This script uses the final synthetic transaction output to generate
views and reviews data that were previously empty in the dataset.

Author: Sweta Pattnaik
Date: 25 November 2025
"""

import json
import logging
import argparse
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ViewsReviewsExtrapolator:
    """
    Extrapolates views and reviews from synthetic transaction data.
    """
    
    def __init__(self, transaction_file: Path, output_dir: Path):
        """
        Initialize the extrapolator.
        
        Args:
            transaction_file: Path to the synthetic transaction data file
            output_dir: Directory to save the extrapolated data
        """
        self.transaction_file = transaction_file
        self.output_dir = output_dir
        self.transactions = []
        self.views_data = []
        self.reviews_data = []
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_transaction_data(self) -> None:
        """
        Load synthetic transaction data from file.
        """
        logger.info(f"Loading transaction data from {self.transaction_file}")
        
        try:
            with open(self.transaction_file, 'r') as f:
                self.transactions = json.load(f)
            logger.info(f"Loaded {len(self.transactions)} transactions")
        except FileNotFoundError:
            logger.error(f"Transaction file not found: {self.transaction_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            raise
    
    def extrapolate_views(self) -> None:
        """
        Extrapolate views data from transactions.
        
        Logic:
        - For each transaction, generate a view that happened 20-90 minutes before
        - View happens from home/work location (not at POI)
        - 80% views from home, 20% from other nearby location
        - View duration: 60-180 seconds if converted to transaction, else 1-120 seconds
        """
        logger.info("Extrapolating views data...")
        
        for user_profile in self.transactions:
            user = user_profile.get('user', {})
            interaction = user_profile.get('interaction', {})
            transactions = interaction.get('transactions', [])
            
            if not transactions:
                continue
            
            # Generate user home location (consistent for this user)
            # Use first transaction's location as approximate planning area center
            first_txn = transactions[0]
            first_loc = first_txn.get('userLocation', {})
            planning_lat = first_loc.get('latitude', 1.3521)
            planning_lon = first_loc.get('longitude', 103.8198)
            
            # User home within 1-3km of planning area
            user_home_lat, user_home_lon = self._generate_user_home_location(
                planning_lat, planning_lon
            )
            
            views = []
            
            for txn in transactions:
                # Parse transaction timestamp
                txn_timestamp = datetime.strptime(txn['timestamp'], "%Y-%m-%dT%H:%M:%SZ")
                
                # View happens 20-90 minutes before transaction
                # Weighted distribution: 20% quick (20-30min), 50% moderate (30-60min), 30% slow (60-90min)
                minutes_before = self._weighted_choice(
                    [(20, 30), (30, 60), (60, 90)],
                    [0.2, 0.5, 0.3]
                )
                view_timestamp = txn_timestamp - timedelta(minutes=minutes_before)
                
                # View location: 80% from home, 20% from other location
                view_lat, view_lon = self._get_view_location(
                    user_home_lat, user_home_lon,
                    first_loc.get('latitude', 1.3521),
                    first_loc.get('longitude', 103.8198)
                )
                
                # View duration: 60-180 seconds (since it converted to transaction)
                duration = np.random.randint(60, 181)
                
                # Create view object
                view = {
                    "timestamp": view_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "poiId": txn['poiId'],
                    "poiCategories": txn['poiCategories'],
                    "poiSubcategories": txn.get('poiSubcategories', []),
                    "duration": duration,
                    "referrer": np.random.choice(["map", "search", "ad", "friend"]),
                    "userLocation": {
                        "latitude": view_lat,
                        "longitude": view_lon
                    }
                }
                
                views.append(view)
            
            # Sort views by timestamp
            views.sort(key=lambda x: x['timestamp'])
            
            # Update the user profile with views
            interaction['views'] = views
            
        logger.info(f"Generated views for {len(self.transactions)} users")
    
    def extrapolate_reviews(self) -> None:
        """
        Extrapolate reviews data from transactions.
        
        Logic:
        - Only 40% of transactions get reviews (transaction_to_review_rate)
        - Reviews happen 1-7 days after transaction
        - Review happens from home (90% of time) with small GPS noise
        - Rating: 3.5-5.0 (positive bias)
        """
        logger.info("Extrapolating reviews data...")
        
        transaction_to_review_rate = 0.4
        
        for user_profile in self.transactions:
            user = user_profile.get('user', {})
            interaction = user_profile.get('interaction', {})
            transactions = interaction.get('transactions', [])
            
            if not transactions:
                continue
            
            # Generate user home location (same logic as views)
            first_txn = transactions[0]
            first_loc = first_txn.get('userLocation', {})
            planning_lat = first_loc.get('latitude', 1.3521)
            planning_lon = first_loc.get('longitude', 103.8198)
            
            user_home_lat, user_home_lon = self._generate_user_home_location(
                planning_lat, planning_lon
            )
            
            # Select subset of transactions to get reviews (40%)
            num_reviews = int(len(transactions) * transaction_to_review_rate)
            
            if num_reviews > 0:
                # Randomly select transactions that will get reviews
                review_indices = np.random.choice(
                    len(transactions), 
                    size=min(num_reviews, len(transactions)),
                    replace=False
                )
                
                reviews = []
                
                for idx in review_indices:
                    txn = transactions[idx]
                    
                    # Parse transaction timestamp
                    txn_timestamp = datetime.strptime(txn['timestamp'], "%Y-%m-%dT%H:%M:%SZ")
                    
                    # Review happens 1-7 days after transaction
                    days_after = np.random.randint(1, 8)
                    review_timestamp = txn_timestamp + timedelta(
                        days=days_after,
                        hours=np.random.randint(0, 24),
                        minutes=np.random.randint(0, 60)
                    )
                    
                    # Review location: 90% from home, 10% from other location
                    review_lat, review_lon = self._get_review_location(
                        user_home_lat, user_home_lon
                    )
                    
                    # Rating: 3.5-5.0 (positive bias)
                    rating = round(np.random.uniform(3.5, 5.0), 1)
                    
                    # Create review object
                    review = {
                        "timestamp": review_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "poiId": txn['poiId'],
                        "poiCategories": txn['poiCategories'],
                        "poiSubcategories": txn.get('poiSubcategories', []),
                        "rating": rating,
                        "reviewText": self._generate_review_text(
                            user['userId'], 
                            txn['poiId'], 
                            txn['poiCategories'][0] if txn['poiCategories'] else "Unknown"
                        ),
                        "userLocation": {
                            "latitude": review_lat,
                            "longitude": review_lon
                        }
                    }
                    
                    reviews.append(review)
                
                # Sort reviews by timestamp
                reviews.sort(key=lambda x: x['timestamp'])
                
                # Update the user profile with reviews
                interaction['reviews'] = reviews
        
        logger.info(f"Generated reviews for {len(self.transactions)} users")
    
    def save_final_dataset(self, filename: str = "final_dataset_with_views_reviews.json") -> None:
        """
        Save complete dataset with views, transactions, and reviews to file.
        
        Args:
            filename: Name of the output file
        """
        output_path = self.output_dir / filename
        logger.info(f"Saving complete dataset to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.transactions, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved complete dataset for {len(self.transactions)} users")
    
    def _jitter_location(self, lat: float, lon: float, distance_km: float = 2.0) -> Tuple[float, float]:
        """Add random jitter to location within specified distance."""
        max_deg = distance_km / 111.0
        lat_jitter = np.random.uniform(-max_deg, max_deg)
        lon_jitter = np.random.uniform(-max_deg, max_deg) / math.cos(math.radians(lat))
        return round(lat + lat_jitter, 6), round(lon + lon_jitter, 6)
    
    def _generate_user_home_location(self, planning_area_lat: float, planning_area_lon: float) -> Tuple[float, float]:
        """
        Generate a consistent 'home' location for a user within their planning area.
        This represents where they live and where views/reviews typically happen.
        """
        # Small jitter within 1-3km of planning area center (residential area)
        return self._jitter_location(planning_area_lat, planning_area_lon, 
                                     distance_km=np.random.uniform(1, 3))
    
    def _get_view_location(self, user_home_lat: float, user_home_lon: float, 
                          poi_lat: float, poi_lon: float) -> Tuple[float, float]:
        """
        View happens from home, work, or nearby location (not at the POI).
        80% from home, 20% from other location within 5km of home.
        """
        if np.random.random() < 0.8:
            # View from home (with small jitter ~100m for GPS noise)
            return self._jitter_location(user_home_lat, user_home_lon, distance_km=0.1)
        else:
            # View from somewhere else nearby (work, friend's place, etc.)
            return self._jitter_location(user_home_lat, user_home_lon, 
                                        distance_km=np.random.uniform(2, 5))
    
    def _get_review_location(self, user_home_lat: float, user_home_lon: float) -> Tuple[float, float]:
        """
        Review happens from home (90% of the time) with small GPS noise.
        """
        if np.random.random() < 0.9:
            return self._jitter_location(user_home_lat, user_home_lon, distance_km=0.1)
        else:
            # Occasional review from nearby location
            return self._jitter_location(user_home_lat, user_home_lon, 
                                        distance_km=np.random.uniform(1, 3))
    
    def _weighted_choice(self, ranges: List[Tuple[int, int]], weights: List[float]) -> int:
        """Select a random value from weighted ranges."""
        selected_range = np.random.choice(len(ranges), p=weights)
        min_val, max_val = ranges[selected_range]
        return np.random.randint(min_val, max_val + 1)
    
    def _generate_review_text(self, user_id: str, poi_id: str, category: str) -> str:
        """Generate placeholder review text."""
        templates = [
            f"Great experience at this {category}!",
            f"Good service and quality at this location.",
            f"Enjoyed visiting this {category}.",
            f"Highly recommend this place!",
            f"Nice {category}, will visit again."
        ]
        return np.random.choice(templates)
    
    def generate_statistics(self) -> Dict[str, Any]:
        """
        Generate statistics about the extrapolated data.
        
        Returns:
            Dictionary containing statistics
        """
        total_users = len(self.transactions)
        total_transactions = sum(len(u['interaction']['transactions']) for u in self.transactions)
        total_views = sum(len(u['interaction'].get('views', [])) for u in self.transactions)
        total_reviews = sum(len(u['interaction'].get('reviews', [])) for u in self.transactions)
        
        stats = {
            'total_users': total_users,
            'total_transactions': total_transactions,
            'total_views': total_views,
            'total_reviews': total_reviews,
            'views_per_user': total_views / total_users if total_users > 0 else 0,
            'transactions_per_user': total_transactions / total_users if total_users > 0 else 0,
            'reviews_per_user': total_reviews / total_users if total_users > 0 else 0,
            'view_to_transaction_ratio': total_views / total_transactions if total_transactions > 0 else 0,
            'transaction_to_review_ratio': total_reviews / total_transactions if total_transactions > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("Statistics generated:")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            else:
                logger.info(f"  {key}: {value}")
        
        return stats
    
    def save_statistics(self, stats: Dict[str, Any], 
                       filename: str = "extrapolation_stats.json") -> None:
        """
        Save statistics to file.
        
        Args:
            stats: Statistics dictionary
            filename: Name of the output file
        """
        output_path = self.output_dir / filename
        logger.info(f"Saving statistics to {output_path}")
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def run(self) -> None:
        """
        Execute the complete extrapolation pipeline.
        """
        logger.info("Starting views and reviews extrapolation...")
        
        # Load transaction data
        self.load_transaction_data()
        
        # Extrapolate views and reviews
        self.extrapolate_views()
        self.extrapolate_reviews()
        
        # Save complete dataset
        self.save_final_dataset()
        
        # Generate and save statistics
        stats = self.generate_statistics()
        self.save_statistics(stats)
        
        logger.info("Extrapolation complete!")


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Extrapolate views and reviews from synthetic transaction data'
    )
    
    parser.add_argument(
        '--transaction-file',
        type=Path,
        default=Path('data/synthetic_postprocess/filtered_5core_filtered.json'),
        help='Path to the synthetic transaction data file (default: data/synthetic_postprocess/filtered_5core_filtered.json)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/final_dataset'),
        help='Directory to save final dataset (default: data/final_dataset)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set the logging level (default: INFO)'
    )
    
    return parser.parse_args()


def main():
    """
    Main entry point for the script.
    """
    args = parse_arguments()
    
    # Set logging level
    logger.setLevel(args.log_level)
    
    # Create and run extrapolator
    extrapolator = ViewsReviewsExtrapolator(
        transaction_file=args.transaction_file,
        output_dir=args.output_dir
    )
    
    extrapolator.run()


if __name__ == '__main__':
    main()
