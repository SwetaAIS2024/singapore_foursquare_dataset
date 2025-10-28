"""
Debug script to identify why reviews are before transactions
"""
import json
from datetime import datetime

# Load the generated dataset
with open('c4_JSON_Dataset_Generation/output_syn_json/synthetic_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("="*80)
print("DEBUGGING REVIEW TIMING VIOLATIONS")
print("="*80)

violations = []

for user_profile in data[:10]:  # Check first 10 users
    user_id = user_profile['user']['userId']
    interactions = user_profile['interaction']
    
    views = interactions.get('views', [])
    transactions = interactions.get('transactions', [])
    reviews = interactions.get('reviews', [])
    
    # Build POI → events mapping
    poi_events = {}
    
    for view in views:
        poi_id = view['poiId']
        if poi_id not in poi_events:
            poi_events[poi_id] = {'views': [], 'transactions': [], 'reviews': []}
        poi_events[poi_id]['views'].append(view['timestamp'])
    
    for txn in transactions:
        poi_id = txn['poiId']
        if poi_id not in poi_events:
            poi_events[poi_id] = {'views': [], 'transactions': [], 'reviews': []}
        poi_events[poi_id]['transactions'].append(txn['timestamp'])
    
    for review in reviews:
        poi_id = review['poiId']
        if poi_id not in poi_events:
            poi_events[poi_id] = {'views': [], 'transactions': [], 'reviews': []}
        poi_events[poi_id]['reviews'].append(review['timestamp'])
    
    # Check for violations
    for poi_id, events in poi_events.items():
        if events['transactions'] and events['reviews']:
            txn_time = datetime.strptime(events['transactions'][0], "%Y-%m-%dT%H:%M:%SZ")
            review_time = datetime.strptime(events['reviews'][0], "%Y-%m-%dT%H:%M:%SZ")
            
            if review_time <= txn_time:
                violations.append({
                    'user': user_id,
                    'poi': poi_id,
                    'txn_time': txn_time,
                    'review_time': review_time,
                    'delta_hours': (txn_time - review_time).total_seconds() / 3600
                })

print(f"\nFound {len(violations)} violations in first 10 users\n")

for v in violations[:5]:
    print(f"User {v['user']}, POI {v['poi']}:")
    print(f"  Transaction: {v['txn_time']}")
    print(f"  Review:      {v['review_time']}")
    print(f"  Review is {v['delta_hours']:.1f} hours BEFORE transaction")
    print()

# Check if reviews are subset of transactions
print("\n" + "="*80)
print("CHECKING FUNNEL LOGIC")
print("="*80)

for user_profile in data[:5]:
    user_id = user_profile['user']['userId']
    interactions = user_profile['interaction']
    
    view_pois = set(v['poiId'] for v in interactions.get('views', []))
    txn_pois = set(t['poiId'] for t in interactions.get('transactions', []))
    review_pois = set(r['poiId'] for r in interactions.get('reviews', []))
    
    print(f"\nUser {user_id}:")
    print(f"  Views: {len(view_pois)} unique POIs")
    print(f"  Transactions: {len(txn_pois)} unique POIs")
    print(f"  Reviews: {len(review_pois)} unique POIs")
    
    # Check if all transaction POIs have views
    txns_without_views = txn_pois - view_pois
    if txns_without_views:
        print(f"  ❌ {len(txns_without_views)} transactions without views!")
    
    # Check if all review POIs have transactions
    reviews_without_txns = review_pois - txn_pois
    if reviews_without_txns:
        print(f"  ❌ {len(reviews_without_txns)} reviews without transactions!")
        print(f"     Review POIs: {reviews_without_txns}")
