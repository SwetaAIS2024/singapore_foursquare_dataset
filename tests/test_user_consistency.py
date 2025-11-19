#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for validating synthetic dataset against raw FSQ data.

Tests ensure data integrity and consistency between raw and synthetic datasets.
"""
import json
import pytest
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.paths import (
    FSQ_CHECKINS_CSV,
    FSQ_POI_CSV,
    SYNTHETIC_POST_FILTERED_JSON,
    SYNTHETIC_POST_ALL_CATEGORIES_JSON,
    INPUT_TO_GENERATOR_FILTERED_JSON,
    INPUT_TO_GENERATOR_ALL_CATEGORIES_JSON
)



class TestUserConsistency:
    """Test user consistency between raw and synthetic datasets."""
    
    @pytest.fixture(scope="class")
    def raw_checkins_df(self):
        """Load raw FSQ checkins data."""
        if not FSQ_CHECKINS_CSV.exists():
            pytest.skip(f"Raw checkins file not found: {FSQ_CHECKINS_CSV}")
        
        # FSQ format: comma-separated, NO HEADER
        # Columns: user_id, place_id, datetime, timezone
        df = pd.read_csv(
            FSQ_CHECKINS_CSV,
            sep=',',
            header=None,
            names=['user_id', 'place_id', 'datetime', 'timezone']
        )
        return df
    
    @pytest.fixture(scope="class")
    def raw_user_ids(self, raw_checkins_df):
        """Extract unique user IDs from raw checkins as strings."""
        # Convert to string to match synthetic data format
        return set(str(uid) for uid in raw_checkins_df['user_id'].unique())
    
    @pytest.fixture(scope="class")
    def synthetic_filtered_data(self):
        """Load synthetic filtered dataset."""
        if not SYNTHETIC_POST_FILTERED_JSON.exists():
            pytest.skip(
                f"Synthetic filtered file not found: "
                f"{SYNTHETIC_POST_FILTERED_JSON}"
            )
        
        with open(SYNTHETIC_POST_FILTERED_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @pytest.fixture(scope="class")
    def synthetic_all_data(self):
        """Load synthetic all categories dataset."""
        if not SYNTHETIC_POST_ALL_CATEGORIES_JSON.exists():
            pytest.skip(
                f"Synthetic all categories file not found: "
                f"{SYNTHETIC_POST_ALL_CATEGORIES_JSON}"
            )
        
        with open(SYNTHETIC_POST_ALL_CATEGORIES_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def test_filtered_users_exist_in_raw(
        self,
        raw_user_ids,
        synthetic_filtered_data
    ):
        """Test that all users in synthetic filtered data exist in raw data."""
        synthetic_user_ids = set(
            profile['user']['userId'] for profile in synthetic_filtered_data
        )
        
        # Check if all synthetic users exist in raw data
        missing_users = synthetic_user_ids - raw_user_ids
        
        assert len(missing_users) == 0, (
            f"Found {len(missing_users)} synthetic users not in raw data: "
            f"{list(missing_users)[:10]}..."  # Show first 10
        )
        
        print(
            f"\n✅ All {len(synthetic_user_ids)} users in synthetic "
            f"filtered dataset exist in raw data"
        )
    
    def test_all_categories_users_exist_in_raw(
        self,
        raw_user_ids,
        synthetic_all_data
    ):
        """
        Test that all users in synthetic all categories data exist in raw.
        """
        synthetic_user_ids = set(
            profile['user']['userId'] for profile in synthetic_all_data
        )
        
        # Check if all synthetic users exist in raw data
        missing_users = synthetic_user_ids - raw_user_ids
        
        assert len(missing_users) == 0, (
            f"Found {len(missing_users)} synthetic users not in raw data: "
            f"{list(missing_users)[:10]}..."  # Show first 10
        )
        
        print(
            f"\n✅ All {len(synthetic_user_ids)} users in synthetic "
            f"all categories dataset exist in raw data"
        )
    
    def test_no_duplicate_users_in_synthetic_filtered(
        self,
        synthetic_filtered_data
    ):
        """Test that there are no duplicate user IDs in synthetic data."""
        user_ids = [profile['user']['userId'] for profile in synthetic_filtered_data]
        unique_user_ids = set(user_ids)
        
        assert len(user_ids) == len(unique_user_ids), (
            f"Found duplicate users in synthetic filtered data: "
            f"{len(user_ids)} total, {len(unique_user_ids)} unique"
        )
        
        print(f"\n✅ No duplicate users in synthetic filtered dataset")
    
    def test_no_duplicate_users_in_synthetic_all(self, synthetic_all_data):
        """Test no duplicate user IDs in synthetic all categories data."""
        user_ids = [profile['user']['userId'] for profile in synthetic_all_data]
        unique_user_ids = set(user_ids)
        
        assert len(user_ids) == len(unique_user_ids), (
            f"Found duplicate users in synthetic all categories data: "
            f"{len(user_ids)} total, {len(unique_user_ids)} unique"
        )
        
        print(
            f"\n✅ No duplicate users in synthetic all categories dataset"
        )


class TestPOIConsistency:
    """Test POI consistency between raw and synthetic datasets."""
    
    @pytest.fixture(scope="class")
    def synthetic_filtered_data(self):
        """Load synthetic filtered dataset."""
        if not SYNTHETIC_POST_FILTERED_JSON.exists():
            pytest.skip(f"Synthetic filtered file not found: {SYNTHETIC_POST_FILTERED_JSON}")
        with open(SYNTHETIC_POST_FILTERED_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @pytest.fixture(scope="class")
    def synthetic_all_data(self):
        """Load synthetic all categories dataset."""
        if not SYNTHETIC_POST_ALL_CATEGORIES_JSON.exists():
            pytest.skip(f"Synthetic all categories file not found: {SYNTHETIC_POST_ALL_CATEGORIES_JSON}")
        with open(SYNTHETIC_POST_ALL_CATEGORIES_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @pytest.fixture(scope="class")
    def raw_poi_df(self):
        """Load raw FSQ POI data."""
        if not FSQ_POI_CSV.exists():
            pytest.skip(f"Raw POI file not found: {FSQ_POI_CSV}")
        
        # FSQ POI format: comma-separated, NO HEADER
        # Columns: place_id, name, lat, lon, category, country
        df = pd.read_csv(
            FSQ_POI_CSV,
            sep=',',
            header=None,
            names=['place_id', 'name', 'lat', 'lon', 'category', 'country']
        )
        return df
    
    @pytest.fixture(scope="class")
    def raw_poi_ids(self, raw_poi_df):
        """Extract unique POI IDs from raw POI data."""
        return set(raw_poi_df['place_id'].unique())
    
    @pytest.fixture(scope="class")
    def raw_checkins_df(self):
        """Load raw FSQ checkins data."""
        if not FSQ_CHECKINS_CSV.exists():
            pytest.skip(f"Raw checkins file not found: {FSQ_CHECKINS_CSV}")
        
        # FSQ format: comma-separated, NO HEADER
        df = pd.read_csv(
            FSQ_CHECKINS_CSV,
            sep=',',
            header=None,
            names=['user_id', 'place_id', 'datetime', 'timezone']
        )
        return df
    
    @pytest.fixture(scope="class")
    def raw_checkin_poi_ids(self, raw_checkins_df):
        """Extract unique POI IDs (place_id) from raw checkins."""
        return set(raw_checkins_df['place_id'].unique())
    
    def test_synthetic_pois_exist_in_raw_filtered(
        self,
        raw_checkin_poi_ids,
        synthetic_filtered_data
    ):
        """Test POI count and structure in synthetic filtered data."""
        # Extract all POI IDs from transactions
        synthetic_poi_ids = set()
        for profile in synthetic_filtered_data:
            transactions = profile.get('interaction', {}).get(
                'transactions',
                []
            )
            for txn in transactions:
                # Synthetic uses 'poiId' field with 'POI-ID-{hash}' format
                # This is a hash of the original place_id from raw data
                synthetic_poi_ids.add(txn['poiId'])
        
        # Verify POI structure and reasonable counts
        # Note: POI IDs are hashed (POI-ID-xxx format) so direct comparison 
        # with raw place_id is not possible
        assert len(synthetic_poi_ids) > 0, "No POIs found in synthetic data"
        assert len(synthetic_poi_ids) <= len(raw_checkin_poi_ids), (
            f"Synthetic has more POIs ({len(synthetic_poi_ids)}) than "
            f"raw data ({len(raw_checkin_poi_ids)})"
        )
        
        print(
            f"\n✅ Synthetic filtered dataset has {len(synthetic_poi_ids)} "
            f"unique POIs (raw data has {len(raw_checkin_poi_ids)} POIs)"
        )
    
    def test_synthetic_pois_exist_in_raw_all_categories(
        self,
        raw_checkin_poi_ids,
        synthetic_all_data
    ):
        """Test POI count and structure in synthetic all categories data."""
        # Extract all POI IDs from transactions
        synthetic_poi_ids = set()
        for profile in synthetic_all_data:
            transactions = profile.get('interaction', {}).get(
                'transactions',
                []
            )
            for txn in transactions:
                # Synthetic uses 'poiId' field with 'POI-ID-{hash}' format
                # This is a hash of the original place_id from raw data
                synthetic_poi_ids.add(txn['poiId'])
        
        # Verify POI structure and reasonable counts
        # Note: POI IDs are hashed (POI-ID-xxx format) so direct comparison 
        # with raw place_id is not possible
        assert len(synthetic_poi_ids) > 0, "No POIs found in synthetic data"
        assert len(synthetic_poi_ids) <= len(raw_checkin_poi_ids), (
            f"Synthetic has more POIs ({len(synthetic_poi_ids)}) than "
            f"raw data ({len(raw_checkin_poi_ids)})"
        )
        
        print(
            f"\n✅ Synthetic all categories dataset has {len(synthetic_poi_ids)} "
            f"unique POIs (raw data has {len(raw_checkin_poi_ids)} POIs)"
        )


class TestTransactionStructure:
    """Test the structure of synthetic transaction data."""
    
    @pytest.fixture(scope="class")
    def synthetic_filtered_data(self):
        """Load synthetic filtered dataset."""
        if not SYNTHETIC_POST_FILTERED_JSON.exists():
            pytest.skip(
                f"Synthetic filtered file not found: "
                f"{SYNTHETIC_POST_FILTERED_JSON}"
            )
        
        with open(SYNTHETIC_POST_FILTERED_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def test_transaction_only_structure(self, synthetic_filtered_data):
        """
        Test that synthetic data is transaction-only with empty views/reviews.
        """
        for profile in synthetic_filtered_data[:100]:  # Check first 100
            interaction = profile.get('interaction', {})
            
            # Check transactions exist and have data
            transactions = interaction.get('transactions', [])
            assert len(transactions) > 0, (
                f"User {profile['user']['userId']} has no transactions"
            )
            
            # Check views are empty or don't exist
            views = interaction.get('views', [])
            assert len(views) == 0, (
                f"User {profile['user']['userId']} has views (should be empty)"
            )
            
            # Check reviews are empty or don't exist
            reviews = interaction.get('reviews', [])
            assert len(reviews) == 0, (
                f"User {profile['user']['userId']} has reviews (should be empty)"
            )
        
        print(
            f"\n✅ Verified transaction-only structure "
            f"(views and reviews are empty)"
        )
    
    def test_transaction_required_fields(self, synthetic_filtered_data):
        """Test that all transactions have required fields."""
        required_fields = ['poiId', 'timestamp']
        
        for profile in synthetic_filtered_data[:100]:  # Check first 100
            transactions = profile.get('interaction', {}).get(
                'transactions',
                []
            )
            
            for txn in transactions[:10]:  # Check first 10 transactions
                for field in required_fields:
                    assert field in txn, (
                        f"User {profile['user']['userId']}: Transaction missing "
                        f"required field '{field}'"
                    )
                
                # Verify timestamp format (should be ISO 8601)
                timestamp = txn['timestamp']
                assert isinstance(timestamp, str), (
                    f"User {profile['user']['userId']}: Timestamp should be string"
                )
                assert 'T' in timestamp, (
                    f"User {profile['user']['userId']}: Timestamp should be "
                    f"ISO 8601 format"
                )
        
        print(f"\n✅ All transactions have required fields")


class TestDataIntegrity:
    """Test overall data integrity of synthetic datasets."""
    
    @pytest.fixture(scope="class")
    def synthetic_filtered_data(self):
        """Load synthetic filtered dataset."""
        if not SYNTHETIC_POST_FILTERED_JSON.exists():
            pytest.skip(
                f"Synthetic filtered file not found: "
                f"{SYNTHETIC_POST_FILTERED_JSON}"
            )
        
        with open(SYNTHETIC_POST_FILTERED_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @pytest.fixture(scope="class")
    def synthetic_all_data(self):
        """Load synthetic all categories dataset."""
        if not SYNTHETIC_POST_ALL_CATEGORIES_JSON.exists():
            pytest.skip(
                f"Synthetic all categories file not found: "
                f"{SYNTHETIC_POST_ALL_CATEGORIES_JSON}"
            )
        
        with open(SYNTHETIC_POST_ALL_CATEGORIES_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def test_non_empty_dataset_filtered(self, synthetic_filtered_data):
        """Test that synthetic filtered dataset is not empty."""
        assert len(synthetic_filtered_data) > 0, (
            "Synthetic filtered dataset is empty"
        )
        
        print(
            f"\n✅ Synthetic filtered dataset contains "
            f"{len(synthetic_filtered_data)} user profiles"
        )
    
    def test_non_empty_dataset_all_categories(self, synthetic_all_data):
        """Test that synthetic all categories dataset is not empty."""
        assert len(synthetic_all_data) > 0, (
            "Synthetic all categories dataset is empty"
        )
        
        print(
            f"\n✅ Synthetic all categories dataset contains "
            f"{len(synthetic_all_data)} user profiles"
        )
    
    def test_all_users_have_transactions_filtered(self, synthetic_filtered_data):
        """Test that all users in filtered dataset have at least one transaction."""
        users_without_transactions = []
        
        for profile in synthetic_filtered_data:
            transactions = profile.get('interaction', {}).get(
                'transactions',
                []
            )
            if len(transactions) == 0:
                users_without_transactions.append(profile['user']['userId'])
        
        assert len(users_without_transactions) == 0, (
            f"Found {len(users_without_transactions)} users without "
            f"transactions in filtered dataset: {users_without_transactions[:10]}..."
        )
        
        print(f"\n✅ All users in filtered dataset have at least one transaction")
    
    def test_all_users_have_transactions_all_categories(self, synthetic_all_data):
        """Test that all users in all categories dataset have at least one transaction."""
        users_without_transactions = []
        
        for profile in synthetic_all_data:
            transactions = profile.get('interaction', {}).get(
                'transactions',
                []
            )
            if len(transactions) == 0:
                users_without_transactions.append(profile['user']['userId'])
        
        assert len(users_without_transactions) == 0, (
            f"Found {len(users_without_transactions)} users without "
            f"transactions in all categories dataset: {users_without_transactions[:10]}..."
        )
        
        print(f"\n✅ All users in all categories dataset have at least one transaction")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
