import json
import csv

# Input/output paths
in_path = 'synthetic_dataset/synthetic_user_data.json'
out_path = 'synthetic_dataset/synthetic_checkin_events.csv'

with open(in_path, 'r') as f:
    data = json.load(f)

header = [
    'userId','timestamp','poiId','poiCategories','poiSubcategories','eventType','duration','referrer',
    'transactionId','amount','currency','paymentMethod','rating','reviewText','userLongitude','userLatitude'
]

with open(out_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for user in data:
        userId = user['user']['userId']
        # Views
        for v in user['interactions']['views']:
            writer.writerow([
                userId, v['timestamp'], v['poiId'], v['poiCategories'], v['poiSubcategories'], 'view',
                v.get('duration',''), v.get('referrer',''), '', '', '', '', '', '',
                v['userLocation']['longitude'], v['userLocation']['latitude']
            ])
        # Transactions
        for t in user['interactions']['transactions']:
            writer.writerow([
                userId, t['timestamp'], t['poiId'], t['poiCategories'], t['poiSubcategories'], 'transaction',
                '', '', t.get('transactionId',''), t.get('amount',''), t.get('currency',''), t.get('paymentMethod',''), '', '',
                t['userLocation']['longitude'], t['userLocation']['latitude']
            ])
        # Reviews
        for r in user['interactions']['reviews']:
            writer.writerow([
                userId, r['timestamp'], r['poiId'], r['poiCategories'], r['poiSubcategories'], 'review',
                '', '', '', '', '', '', r.get('rating',''), r.get('reviewText',''),
                r['userLocation']['longitude'], r['userLocation']['latitude']
            ])
print(f'Extracted check-in events to {out_path}')
