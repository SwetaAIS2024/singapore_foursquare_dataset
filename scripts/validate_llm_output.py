#!/usr/bin/env python3
"""
Validate LLM-generated Synthetic Data

Checks funnel consistency, temporal ordering, and data quality.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def validate_temporal_order(user_data):
    """Check that views come before transactions, reviews come after"""
    errors = []
    
    views = user_data['interaction']['views']
    txns = user_data['interaction']['transactions']
    reviews = user_data['interaction']['reviews']
    
    # Build POI -> timestamps map
    poi_views = defaultdict(list)
    poi_txns = defaultdict(list)
    poi_reviews = defaultdict(list)
    
    for view in views:
        poi_id = view['poiId']
        ts = datetime.fromisoformat(view['timestamp'].replace('Z', '+00:00'))
        poi_views[poi_id].append(ts)
    
    for txn in txns:
        poi_id = txn['poiId']
        ts = datetime.fromisoformat(txn['timestamp'].replace('Z', '+00:00'))
        poi_txns[poi_id].append(ts)
    
    for review in reviews:
        poi_id = review['poiId']
        ts = datetime.fromisoformat(review['timestamp'].replace('Z', '+00:00'))
        poi_reviews[poi_id].append(ts)
    
    # Check: Each transaction should have a prior view
    for poi_id, txn_times in poi_txns.items():
        if poi_id not in poi_views:
            errors.append(f"Transaction to {poi_id} has no view")
            continue
        
        for txn_time in txn_times:
            # Find views before this transaction
            prior_views = [v for v in poi_views[poi_id] if v < txn_time]
            if not prior_views:
                errors.append(f"Transaction at {txn_time} has no prior view for {poi_id}")
    
    # Check: Reviews should come after transactions
    for poi_id, review_times in poi_reviews.items():
        if poi_id not in poi_txns:
            errors.append(f"Review for {poi_id} has no transaction")
            continue
        
        for review_time in review_times:
            # Find transactions before this review
            prior_txns = [t for t in poi_txns[poi_id] if t < review_time]
            if not prior_txns:
                errors.append(f"Review at {review_time} has no prior transaction for {poi_id}")
    
    return errors


def validate_funnel_ratios(user_data):
    """Check funnel makes sense"""
    warnings = []
    
    views = user_data['interaction']['views']
    txns = user_data['interaction']['transactions']
    reviews = user_data['interaction']['reviews']
    
    # Should have equal views and transactions (1:1 for this dataset)
    if len(views) != len(txns):
        warnings.append(f"Views ({len(views)}) != Transactions ({len(txns)}). Expected 1:1 ratio.")
    
    # Reviews should be ~40% of transactions
    if len(txns) > 0:
        review_rate = len(reviews) / len(txns)
        if review_rate > 0.6:
            warnings.append(f"Review rate {review_rate:.1%} too high (expected ~40%)")
        elif review_rate < 0.2 and len(txns) > 3:
            warnings.append(f"Review rate {review_rate:.1%} too low (expected ~40%)")
    
    # Can't have more reviews than transactions
    if len(reviews) > len(txns):
        warnings.append(f"More reviews ({len(reviews)}) than transactions ({len(txns)})")
    
    return warnings


def validate_data_structure(user_data):
    """Check required fields exist"""
    errors = []
    
    # Check user structure
    if 'user' not in user_data:
        errors.append("Missing 'user' field")
        return errors
    
    required_user_fields = ['userId', 'age', 'gender', 'location', 'device']
    for field in required_user_fields:
        if field not in user_data['user']:
            errors.append(f"Missing user.{field}")
    
    # Check interaction structure
    if 'interaction' not in user_data:
        errors.append("Missing 'interaction' field")
        return errors
    
    required_interaction_fields = ['views', 'transactions', 'reviews']
    for field in required_interaction_fields:
        if field not in user_data['interaction']:
            errors.append(f"Missing interaction.{field}")
    
    return errors


def validate_file(file_path: Path):
    """Validate entire file"""
    print("\n" + "="*60)
    print("VALIDATING LLM-GENERATED DATA")
    print("="*60 + "\n")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"📂 Loading: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"👥 Total users: {len(data)}\n")
    
    all_errors = []
    all_warnings = []
    
    for i, user_data in enumerate(data, 1):
        user_id = user_data.get('user', {}).get('userId', f'Unknown-{i}')
        
        # Validate structure
        struct_errors = validate_data_structure(user_data)
        if struct_errors:
            all_errors.extend([f"User {user_id}: {e}" for e in struct_errors])
            continue
        
        # Validate temporal order
        temporal_errors = validate_temporal_order(user_data)
        if temporal_errors:
            all_errors.extend([f"User {user_id}: {e}" for e in temporal_errors])
        
        # Validate ratios
        ratio_warnings = validate_funnel_ratios(user_data)
        if ratio_warnings:
            all_warnings.extend([f"User {user_id}: {w}" for w in ratio_warnings])
        
        # Print user summary
        views = len(user_data['interaction']['views'])
        txns = len(user_data['interaction']['transactions'])
        reviews = len(user_data['interaction']['reviews'])
        persona = user_data.get('metadata', {}).get('persona', 'Unknown')
        
        status = "✅" if not temporal_errors else "⚠️"
        print(f"{status} User {user_id} ({persona}): {views}V {txns}T {reviews}R")
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    if all_errors:
        print(f"\n❌ Found {len(all_errors)} ERRORS:")
        for error in all_errors[:10]:  # Show first 10
            print(f"   • {error}")
        if len(all_errors) > 10:
            print(f"   ... and {len(all_errors) - 10} more")
    else:
        print("\n✅ No critical errors found!")
    
    if all_warnings:
        print(f"\n⚠️  Found {len(all_warnings)} warnings:")
        for warning in all_warnings[:10]:  # Show first 10
            print(f"   • {warning}")
        if len(all_warnings) > 10:
            print(f"   ... and {len(all_warnings) - 10} more")
    
    # Overall statistics
    total_views = sum(len(u['interaction']['views']) for u in data)
    total_txns = sum(len(u['interaction']['transactions']) for u in data)
    total_reviews = sum(len(u['interaction']['reviews']) for u in data)
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total views:        {total_views}")
    print(f"   Total transactions: {total_txns}")
    print(f"   Total reviews:      {total_reviews}")
    print(f"   View:Txn ratio:     {total_views/total_txns:.2f}:1" if total_txns > 0 else "   View:Txn ratio:     N/A")
    print(f"   Review rate:        {total_reviews/total_txns*100:.1f}%" if total_txns > 0 else "   Review rate:        N/A")
    
    print("="*60 + "\n")
    
    return len(all_errors) == 0


def main():
    """Main execution"""
    file_path = Path('data/llm_generated/synthetic_data_llm.json')
    
    success = validate_file(file_path)
    
    if success:
        print("✅ Validation passed! Data is ready to use.")
        sys.exit(0)
    else:
        print("❌ Validation failed. Please check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
