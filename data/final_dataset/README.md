# Final Dataset Directory

## Purpose

This directory contains the **final synthetic dataset** with complete user interaction data including:
- ✅ **Views** - User browsing behavior before visits
- ✅ **Transactions** - Actual check-ins/purchases at POIs
- ✅ **Reviews** - User feedback after visits

## Files

### Generated Dataset
- **`final_dataset_with_views_reviews.json`** - Complete dataset with all interaction types
- **`extrapolation_stats.json`** - Statistical summary of the generation process

## Dataset Structure

Each user profile contains:

```json
{
  "user": {
    "userId": "21418",
    "age": 62,
    "gender": "male",
    "location": {"city": "Singapore", "country": "SG"},
    "device": {"platform": "Android", "appVersion": "2.6.5"}
  },
  "interaction": {
    "views": [
      {
        "timestamp": "2012-04-03T17:30:00Z",
        "poiId": "POI-ID-114d7492",
        "poiCategories": ["Asian Restaurant"],
        "poiSubcategories": [],
        "duration": 120,
        "referrer": "search",
        "userLocation": {"latitude": 1.3521, "longitude": 103.8198}
      }
    ],
    "transactions": [
      {
        "timestamp": "2012-04-03T18:19:55Z",
        "poiId": "POI-ID-114d7492",
        "poiCategories": ["Asian Restaurant"],
        "poiSubcategories": [],
        "transactionId": "687681",
        "amount": 12.88,
        "currency": "SGD",
        "paymentMethod": "mobile_wallet",
        "userLocation": {"latitude": 1.359951, "longitude": 103.884701}
      }
    ],
    "reviews": [
      {
        "timestamp": "2012-04-07T15:22:00Z",
        "poiId": "POI-ID-114d7492",
        "poiCategories": ["Asian Restaurant"],
        "poiSubcategories": [],
        "rating": 4.3,
        "reviewText": "Great experience at this Asian Restaurant!",
        "userLocation": {"latitude": 1.3521, "longitude": 103.8198}
      }
    ]
  }
}
```

## Generation Process

The dataset is generated using `scripts/extrapolate_views_reviews.py` which:

1. **Loads** transaction data from `data/synthetic_postprocess/filtered_5core_filtered.json`
2. **Generates views** for each transaction (20-90 minutes before)
3. **Generates reviews** for 40% of transactions (1-7 days after)
4. **Validates** temporal and spatial consistency
5. **Outputs** complete dataset with statistics

For detailed logic, see: [`docs/VIEWS_REVIEWS_GENERATION_LOGIC.md`](../docs/VIEWS_REVIEWS_GENERATION_LOGIC.md)

## Usage

### Generate Dataset

```bash
# Default paths
python scripts/extrapolate_views_reviews.py

# Custom paths
python scripts/extrapolate_views_reviews.py \
  --transaction-file data/synthetic_postprocess/all_categories_5core_filtered.json \
  --output-dir data/final_dataset \
  --log-level INFO
```

### Load Dataset (Python)

```python
import json

# Load complete dataset
with open('data/final_dataset/final_dataset_with_views_reviews.json', 'r') as f:
    dataset = json.load(f)

# Access user data
for user_profile in dataset:
    user = user_profile['user']
    views = user_profile['interaction']['views']
    transactions = user_profile['interaction']['transactions']
    reviews = user_profile['interaction']['reviews']
    
    print(f"User {user['userId']}: {len(views)} views, {len(transactions)} txns, {len(reviews)} reviews")
```

### Load Statistics

```python
import json

with open('data/final_dataset/extrapolation_stats.json', 'r') as f:
    stats = json.load(f)

print(f"Total users: {stats['total_users']}")
print(f"View→Transaction ratio: {stats['view_to_transaction_ratio']:.2f}")
print(f"Transaction→Review rate: {stats['transaction_to_review_ratio']:.1%}")
```

## Data Quality Guarantees

✅ **Temporal Consistency**: Views → Transactions → Reviews (chronological order)  
✅ **Spatial Realism**: Views/reviews from home, transactions at POI  
✅ **Funnel Integrity**: Every transaction has a view, subset have reviews  
✅ **Realistic Patterns**: Based on real user behavior research  
✅ **GPS Accuracy**: Location jittering simulates real-world GPS noise  

## Expected Statistics

- **View:Transaction ratio**: 1:1 (every transaction has a view)
- **Transaction:Review ratio**: 1:0.4 (40% review rate)
- **Views per user**: Varies by user activity
- **Temporal delays**:
  - View → Transaction: 20-90 minutes (avg ~55 min)
  - Transaction → Review: 1-7 days (avg ~4 days)

## Use Cases

This dataset is suitable for:

1. **Recommendation Systems**
   - POI recommendation
   - Collaborative filtering
   - Content-based filtering

2. **Trajectory Mining**
   - User mobility patterns
   - Visit sequence analysis
   - Spatial-temporal clustering

3. **Review Analysis**
   - Sentiment analysis
   - Rating prediction
   - Review helpfulness

4. **User Behavior Modeling**
   - Conversion funnel analysis
   - Engagement metrics
   - Churn prediction

5. **Machine Learning**
   - Training data for deep learning models
   - Feature engineering experiments
   - Model validation with realistic patterns

## References

- Main Documentation: [`README.md`](../README.md)
- Generation Logic: [`docs/VIEWS_REVIEWS_GENERATION_LOGIC.md`](../docs/VIEWS_REVIEWS_GENERATION_LOGIC.md)
- Category Mapping: [`config/README_CATEGORY_MAPPING.md`](../config/README_CATEGORY_MAPPING.md)

---

**Last Updated**: 25 November 2025  
**Version**: 1.0
