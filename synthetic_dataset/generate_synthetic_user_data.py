import json
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import uuid

# --- CONFIG ---
N_SYNTH_USERS = 1000  # Number of archetype users to generate
VIEWS_LAMBDA = 30    # Mean for Poisson sampling of view events
TXNS_LAMBDA = 10     # Mean for Poisson sampling of transactions
REVIEWS_LAMBDA = 5   # Mean for Poisson sampling of reviews

# --- Load archetype users (nearest to centroids) ---
centroid_users_path = 'analysis_older_dataset/analysis_set4/codes/No_of_rootCat_C_top20_remove_MALL/Clustering_Profiling/cluster_centroid_nearest_user.csv'
centroid_users = pd.read_csv(centroid_users_path)

# --- Load real POI data from file ---
# Load check-in data with explicit columns
checkin_cols = ['user_id', 'place_id', 'datetime', 'timezone', 'lat', 'lon']
checkins = pd.read_csv('analysis_older_dataset/singapore_checkins_filtered_with_locations_coord.txt', sep='\t', names=checkin_cols)
# Load POI to category mapping
place_cat = pd.read_csv('analysis_older_dataset/sg_place_id_to_category.csv')
# Merge to get category for each POI
poi_df = checkins[['place_id', 'lat', 'lon']].drop_duplicates().merge(place_cat, left_on='place_id', right_on='place_id', how='left')
POI_LIST = [
    {
        'poi_id': row['place_id'],
        'category': row['category'],
        'subcategory': '',  # No subcategory available
        'longitude': row['lon'],
        'latitude': row['lat']
    }
    for _, row in poi_df.iterrows()
]

# Load user home locations (most frequent check-in location per user)
home_locs = {}
with open('synthetic_dataset/user_home_locations.txt', 'r') as f:
    for line in f:
        user, lat, lon = line.strip().split()
        if user not in home_locs:
            home_locs[user] = (float(lat), float(lon))

# --- Helper functions ---
def random_age():
    return random.randint(18, 65)

def random_gender():
    return random.choice(['male', 'female', 'other'])

def random_city_country():
    return 'Singapore', 'Singapore'

def random_platform():
    return random.choice(['Android', 'iOS'])

def random_appversion():
    return f'{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,9)}'

def random_payment():
    return random.choice(['Credit Card', 'Cash', 'Mobile Pay', 'Debit Card'])

def random_currency():
    return 'SGD'

def random_review():
    return random.choice([
        'Great place!', 'Would visit again.', 'Not bad.', 'Could be better.', 'Loved it!', 'Average experience.'
    ])

def random_rating():
    return random.randint(1, 5)

def random_referrer():
    return random.choice(['search', 'recommendation', 'ad', 'friend'])

def random_duration():
    return random.randint(10, 600)  # seconds

def random_amount():
    return round(random.uniform(5, 200), 2)

def random_timestamp(start=None):
    if not start:
        start = datetime(2024, 1, 1)
    delta = timedelta(minutes=random.randint(0, 60*24*180))
    return (start + delta).isoformat()

def pick_poi():
    return random.choice(POI_LIST)

# --- Main synthetic data generation ---
synth_data = []
user_id_counter = 10001  # Start from a reasonably small number
for idx, row in centroid_users.iterrows():
    # Generate a small numeric user ID
    user_id = str(user_id_counter)
    user_id_counter += 1
    age = random_age()
    gender = random_gender()
    city, country = random_city_country()
    # Use a random POI as user's home location (since user_id is synthetic)
    home_poi = pick_poi()
    location = {'longitude': home_poi['longitude'], 'latitude': home_poi['latitude']}
    # Sample event counts from Poisson distribution (minimum 1)
    n_views = max(1, np.random.poisson(VIEWS_LAMBDA))
    n_txns = max(1, np.random.poisson(TXNS_LAMBDA))
    n_reviews = max(1, np.random.poisson(REVIEWS_LAMBDA))
    user_profile = {
        'userId': user_id,
        'age': age,
        'gender': gender,
        'location': location,
        'city': city,
        'country': country
    }
    device = {
        'platform': random_platform(),
        'appVersion': random_appversion()
    }
    # Interactions
    views = []
    for _ in range(n_views):
        poi = pick_poi()
        views.append({
            'timestamp': random_timestamp(),
            'poiId': poi['poi_id'],
            'poiCategories': poi['category'],
            'poiSubcategories': poi['subcategory'],
            'duration': random_duration(),
            'referrer': random_referrer(),
            'userLocation': {'longitude': poi['longitude'], 'latitude': poi['latitude']}
        })
    transactions = []
    for _ in range(n_txns):
        poi = pick_poi()
        transactions.append({
            'timestamp': random_timestamp(),
            'poiId': poi['poi_id'],
            'poiCategories': poi['category'],
            'poiSubcategories': poi['subcategory'],
            'transactionId': f'TXN_{user_id}_{random.randint(1000,9999)}',
            'amount': random_amount(),
            'currency': random_currency(),
            'paymentMethod': random_payment(),
            'userLocation': {'longitude': poi['longitude'], 'latitude': poi['latitude']}
        })
    reviews = []
    for _ in range(n_reviews):
        poi = pick_poi()
        reviews.append({
            'timestamp': random_timestamp(),
            'poiId': poi['poi_id'],
            'poiCategories': poi['category'],
            'poiSubcategories': poi['subcategory'],
            'rating': random_rating(),
            'reviewText': random_review(),
            'userLocation': {'longitude': poi['longitude'], 'latitude': poi['latitude']}
        })
    synth_data.append({
        'user': user_profile,
        'device': device,
        'interactions': {
            'views': views,
            'transactions': transactions,
            'reviews': reviews
        }
    })

# --- Save to JSON ---
out_path = 'synthetic_dataset/synthetic_user_data.json'
with open(out_path, 'w') as f:
    json.dump(synth_data, f, indent=2)
print(f'Synthetic data saved to {out_path}')
