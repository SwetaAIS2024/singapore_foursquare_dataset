import json
from datetime import datetime

# Input/output paths
in_path = 'synthetic_dataset/synthetic_user_data.json'
out_path = 'synthetic_dataset/synthetic_checkins_for_comparison.txt'

with open(in_path, 'r') as f:
    data = json.load(f)

with open(out_path, 'w') as f:
    for user in data:
        user_id = user['user']['userId']
        # Views as check-ins
        for v in user['interactions']['views']:
            try:
                dt = datetime.fromisoformat(v['timestamp'])
                dt_str = dt.strftime('%a %b %d %H:%M:%S +0000 %Y')
            except Exception:
                dt_str = v['timestamp']
            timezone = '480'  # Singapore UTC+8
            lat = v['userLocation']['latitude']
            lon = v['userLocation']['longitude']
            place_id = v['poiId']
            f.write(f"{user_id}\t{place_id}\t{dt_str}\t{timezone}\t{lat}\t{lon}\n")
        # Transactions as check-ins
        for t in user['interactions']['transactions']:
            try:
                dt = datetime.fromisoformat(t['timestamp'])
                dt_str = dt.strftime('%a %b %d %H:%M:%S +0000 %Y')
            except Exception:
                dt_str = t['timestamp']
            timezone = '480'
            lat = t['userLocation']['latitude']
            lon = t['userLocation']['longitude']
            place_id = t['poiId']
            f.write(f"{user_id}\t{place_id}\t{dt_str}\t{timezone}\t{lat}\t{lon}\n")
        # Reviews as check-ins
        for r in user['interactions']['reviews']:
            try:
                dt = datetime.fromisoformat(r['timestamp'])
                dt_str = dt.strftime('%a %b %d %H:%M:%S +0000 %Y')
            except Exception:
                dt_str = r['timestamp']
            timezone = '480'
            lat = r['userLocation']['latitude']
            lon = r['userLocation']['longitude']
            place_id = r['poiId']
            f.write(f"{user_id}\t{place_id}\t{dt_str}\t{timezone}\t{lat}\t{lon}\n")
print(f'Synthetic check-ins for comparison saved to {out_path}')
