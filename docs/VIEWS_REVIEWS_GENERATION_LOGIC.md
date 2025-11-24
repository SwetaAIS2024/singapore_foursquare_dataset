# Views and Reviews Generation Logic

## Overview

This document explains the logic used to extrapolate **views** and **reviews** from the synthetic transaction data in the Singapore Foursquare dataset. The script `extrapolate_views_reviews.py` implements a realistic user behavior model based on established patterns from location-based service (LBS) research.

## Input Data

The script takes as input the final synthetic transaction dataset:
- **Location**: `data/synthetic_postprocess/filtered_5core_filtered.json`
- **Format**: JSON array of user profiles with transactions already populated
- **Structure**: Each user has transactions with timestamps, POI IDs, locations, and payment information

## Funnel Model

The generation follows a realistic user journey funnel:

```
Views → Transactions → Reviews
100%      100%         40%
```

### Key Principle
- **Every transaction must have a preceding view** (conversion funnel)
- **Not all transactions get reviews** (only ~40% based on research)

---

## Views Generation Logic

### Temporal Relationship
**Views occur BEFORE transactions** (20-90 minutes prior)

The delay distribution follows a weighted pattern:
- **20%** quick decision (20-30 minutes)
- **50%** moderate consideration (30-60 minutes) 
- **30%** extended browsing (60-90 minutes)

**Average delay**: ~55 minutes (realistic consideration time)

### Spatial Behavior

#### User Home Location
Each user has a **consistent home location** generated once:
- Within 1-3 km of their planning area center
- Represents their residential location
- Used consistently across all views and reviews

#### View Location Distribution
Views happen from where users browse on their devices:

- **80%** from home location
  - Small GPS noise (~100m) added for realism
  - Represents users browsing at home before going out
  
- **20%** from other locations
  - Within 2-5 km of home (work, friend's place, etc.)
  - Simulates browsing while already out

**Rationale**: People typically research POIs on their phones before visiting, most commonly while at home or work.

### View Attributes

```json
{
  "timestamp": "2012-04-03T17:30:00Z",     // 20-90 min before transaction
  "poiId": "POI-ID-114d7492",               // Same as transaction
  "poiCategories": ["Asian Restaurant"],     // Same as transaction
  "poiSubcategories": [],
  "duration": 120,                          // 60-180 seconds (converted)
  "referrer": "search",                     // map/search/ad/friend
  "userLocation": {
    "latitude": 1.3521,                     // Home/nearby location
    "longitude": 103.8198
  }
}
```

#### Duration Logic
- **60-180 seconds** for views that converted to transactions
- Longer duration indicates serious consideration
- Aligned with typical app browsing times

#### Referrer Distribution
Randomly assigned from realistic sources:
- `map`: Discovered via map browsing
- `search`: Found through search
- `ad`: Came from advertisement
- `friend`: Recommended by friend

---

## Reviews Generation Logic

### Selection Rate
**40% of transactions generate reviews**

This is based on:
- Industry research showing 30-50% review rates for satisfied customers
- Balanced dataset for ML training
- Realistic user behavior (not everyone reviews)

### Temporal Relationship
**Reviews occur AFTER transactions** (1-7 days later)

The delay distribution:
- Uniformly distributed between 1-7 days
- Additional random hour (0-23) and minute (0-59)
- **Average delay**: ~4 days

**Rationale**: Users need time to reflect on experience, typically reviewing within a week while memory is fresh.

### Spatial Behavior

Reviews are written from comfortable locations:

- **90%** from home location
  - Small GPS noise (~100m) for realism
  - People write reviews at leisure, typically at home
  
- **10%** from other nearby locations
  - Within 1-3 km of home
  - Occasional reviews written elsewhere

### Review Attributes

```json
{
  "timestamp": "2012-04-07T15:22:00Z",      // 1-7 days after transaction
  "poiId": "POI-ID-114d7492",                // Same as transaction
  "poiCategories": ["Asian Restaurant"],      // Same as transaction
  "poiSubcategories": [],
  "rating": 4.3,                             // 3.5-5.0 (positive bias)
  "reviewText": "Great experience at this Asian Restaurant!",
  "userLocation": {
    "latitude": 1.3521,                      // Home location
    "longitude": 103.8198
  }
}
```

#### Rating Distribution
- **Range**: 3.5 to 5.0 stars
- **Positive bias**: Users who had poor experiences (<3.5 stars) are less likely to review
- **Continuous**: Rounded to 1 decimal place (4.3, 4.7, etc.)

#### Review Text
Placeholder templates based on category:
- "Great experience at this {category}!"
- "Good service and quality at this location."
- "Enjoyed visiting this {category}."
- "Highly recommend this place!"
- "Nice {category}, will visit again."

**Note**: In production, these should be replaced with more sophisticated text generation (e.g., using LLMs).

---

## GPS Noise and Realism

### Location Jittering Function
```python
def _jitter_location(lat, lon, distance_km):
    max_deg = distance_km / 111.0  # 1 degree ≈ 111 km
    lat_jitter = random.uniform(-max_deg, max_deg)
    lon_jitter = random.uniform(-max_deg, max_deg) / cos(lat)
    return (lat + lat_jitter, lon + lon_jitter)
```

**Purpose**: Simulates GPS inaccuracy and privacy-preserving location fuzzing.

### Distance Scales
- **100m**: GPS noise (small random variations)
- **1-3km**: Home location offset from planning area
- **2-5km**: Alternative view locations (work, friends)

---

## Implementation Architecture

### Class Structure

```python
class ViewsReviewsExtrapolator:
    - load_transaction_data()      # Load input JSON
    - extrapolate_views()          # Generate views
    - extrapolate_reviews()        # Generate reviews
    - save_final_dataset()         # Save complete dataset
    - generate_statistics()        # Calculate metrics
```

### Key Helper Methods

1. **`_generate_user_home_location()`**
   - Creates consistent home location per user
   - Used for all views and reviews

2. **`_get_view_location()`**
   - 80/20 distribution (home vs. elsewhere)
   - Ensures views don't happen at POI

3. **`_get_review_location()`**
   - 90/10 distribution (home vs. elsewhere)
   - Reviews written from comfortable places

4. **`_weighted_choice()`**
   - Implements weighted temporal delays
   - More realistic than uniform distribution

5. **`_jitter_location()`**
   - Adds GPS noise
   - Privacy-preserving location fuzzing

---

## Statistical Validation

The script generates comprehensive statistics:

```json
{
  "total_users": 5000,
  "total_transactions": 50000,
  "total_views": 50000,
  "total_reviews": 20000,
  "views_per_user": 10.0,
  "transactions_per_user": 10.0,
  "reviews_per_user": 4.0,
  "view_to_transaction_ratio": 1.0,
  "transaction_to_review_ratio": 0.4
}
```

### Expected Ratios
- **Views:Transactions** = 1:1 (strict funnel)
- **Transactions:Reviews** = 1:0.4 (40% review rate)

---

## Usage

### Basic Usage
```bash
python scripts/extrapolate_views_reviews.py
```

### Custom Paths
```bash
python scripts/extrapolate_views_reviews.py \
  --transaction-file data/synthetic_postprocess/all_categories_5core_filtered.json \
  --output-dir data/final_dataset \
  --log-level DEBUG
```

### Output
- **File**: `data/final_dataset/final_dataset_with_views_reviews.json`
- **Statistics**: `data/final_dataset/extrapolation_stats.json`

---

## Design Principles

### 1. Temporal Realism
- Views precede transactions by realistic delays
- Reviews follow transactions after reflection time
- No temporal violations in the funnel

### 2. Spatial Realism
- Views/reviews happen from user's home area
- Transactions happen at POI location
- GPS noise simulates real-world inaccuracy

### 3. Behavioral Realism
- Not all transactions get reviews (40% rate)
- Positive rating bias (3.5-5.0 stars)
- Weighted temporal delays (not uniform)

### 4. Consistency
- Each user has one home location
- Same POI ID across view/transaction/review
- Chronological ordering maintained

### 5. Data Quality
- No funnel violations (view → txn → review order)
- Realistic spatio-temporal patterns
- Suitable for ML training and validation

---

## Research Foundations

This implementation is based on:

1. **Location-Based Services Research**
   - User mobility patterns
   - Check-in behavior studies
   - Review generation patterns

2. **Recommendation Systems Literature**
   - Implicit feedback (views)
   - Explicit feedback (transactions, reviews)
   - Funnel conversion rates

3. **Real-World Observations**
   - App analytics data
   - User behavior studies
   - GPS accuracy patterns

---

## Future Enhancements

Potential improvements:

1. **Advanced Review Text Generation**
   - Use LLMs (GPT, BERT) for realistic reviews
   - Sentiment alignment with ratings
   - Category-specific vocabulary

2. **Category-Specific Patterns**
   - Different review rates by category
   - Temporal patterns by POI type
   - Rating distributions by category

3. **User Personas**
   - Heavy reviewers vs. lurkers
   - Fast vs. slow decision makers
   - Home vs. mobile browsers

4. **Temporal Patterns**
   - Day-of-week effects
   - Time-of-day patterns
   - Seasonal variations

5. **POI Popularity**
   - Popular POIs get more reviews
   - Chain restaurants vs. local spots
   - Tourist destinations vs. local haunts

---

## Contact & Maintenance

**Author**: Sweta Pattnaik  
**Date**: 25 November 2025  
**Version**: 1.0  

For questions or improvements, please refer to the main project documentation.
