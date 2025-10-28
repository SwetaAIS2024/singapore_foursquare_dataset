"""
Diagnostic Script: Analyze Synthetic Data Quality for Model Training
Checks for common issues that cause poor model performance
"""
import json
import os
from collections import defaultdict, Counter

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
SYNTHETIC_DATA = os.path.join(BASE_DIR, "output_syn_json", "fsq_to_synthetic.json")

print("="*80)
print("SYNTHETIC DATA QUALITY DIAGNOSTIC")
print("="*80)

# Load data
print(f"\nLoading data from: {SYNTHETIC_DATA}")
with open(SYNTHETIC_DATA, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total users: {len(data)}")

# Analysis containers
all_pois = []
user_poi_counts = []
poi_visit_counts = defaultdict(int)
category_counts = defaultdict(int)
user_interaction_counts = []

for user_data in data:
    user_pois = []
    user_interactions = 0
    
    # Count views
    if 'views' in user_data['interaction']:
        for view in user_data['interaction']['views']:
            poi_id = view['poiId']
            all_pois.append(poi_id)
            user_pois.append(poi_id)
            poi_visit_counts[poi_id] += 1
            user_interactions += 1
            
            if view['poiCategories']:
                category_counts[view['poiCategories'][0]] += 1
    
    # Count transactions
    if 'transactions' in user_data['interaction']:
        user_interactions += len(user_data['interaction']['transactions'])
    
    # Count reviews
    if 'reviews' in user_data['interaction']:
        user_interactions += len(user_data['interaction']['reviews'])
    
    user_poi_counts.append(len(set(user_pois)))
    user_interaction_counts.append(user_interactions)

# Calculate statistics
unique_pois = len(poi_visit_counts)
total_visits = len(all_pois)
avg_pois_per_user = sum(user_poi_counts) / len(user_poi_counts) if user_poi_counts else 0
avg_interactions_per_user = sum(user_interaction_counts) / len(user_interaction_counts) if user_interaction_counts else 0

print("\n" + "="*80)
print("1. POI COVERAGE & OVERLAP")
print("="*80)
print(f"Total POI visits: {total_visits:,}")
print(f"Unique POIs: {unique_pois:,}")
print(f"Avg visits per POI: {total_visits/unique_pois:.2f}")
print(f"Avg unique POIs per user: {avg_pois_per_user:.1f}")
print(f"Avg total interactions per user: {avg_interactions_per_user:.1f}")

# Check POI distribution
visit_distribution = Counter(poi_visit_counts.values())
print(f"\nPOI Visit Distribution:")
print(f"  POIs with 1 visit: {visit_distribution[1]:,} ({visit_distribution[1]/unique_pois*100:.1f}%)")
print(f"  POIs with 2-5 visits: {sum(visit_distribution[i] for i in range(2,6)):,}")
print(f"  POIs with 6-10 visits: {sum(visit_distribution[i] for i in range(6,11)):,}")
print(f"  POIs with 11-50 visits: {sum(visit_distribution[i] for i in range(11,51)):,}")
print(f"  POIs with 50+ visits: {sum(visit_distribution[i] for i in range(51, max(visit_distribution.keys())+1) if i in visit_distribution):,}")

# Most popular POIs
top_pois = sorted(poi_visit_counts.items(), key=lambda x: x[1], reverse=True)[:20]
print(f"\nTop 20 Most Visited POIs:")
for poi_id, count in top_pois[:10]:
    print(f"  {poi_id}: {count} visits")

print("\n" + "="*80)
print("2. TRAIN/TEST OVERLAP POTENTIAL")
print("="*80)
# Simulate 80/20 split
num_train = int(len(data) * 0.8)
train_data = data[:num_train]
test_data = data[num_train:]

train_pois = set()
test_pois = set()

for user_data in train_data:
    if 'views' in user_data['interaction']:
        for view in user_data['interaction']['views']:
            train_pois.add(view['poiId'])

for user_data in test_data:
    if 'views' in user_data['interaction']:
        for view in user_data['interaction']['views']:
            test_pois.add(view['poiId'])

overlap_pois = train_pois & test_pois
overlap_rate = len(overlap_pois) / len(test_pois) if test_pois else 0

print(f"Train users: {len(train_data)}")
print(f"Test users: {len(test_data)}")
print(f"Train POIs: {len(train_pois):,}")
print(f"Test POIs: {len(test_pois):,}")
print(f"Overlapping POIs: {len(overlap_pois):,}")
print(f"Test POI Overlap Rate: {overlap_rate*100:.1f}%")

if overlap_rate < 0.5:
    print("\n⚠️  WARNING: Less than 50% of test POIs are in training set!")
    print("   Model cannot predict POIs it has never seen.")
    print("   Recommendation: Add more popular POI revisits to increase overlap.")

print("\n" + "="*80)
print("3. USER BEHAVIOR PATTERNS")
print("="*80)

# Check for revisits
user_revisit_rates = []
for user_data in data:
    user_pois = []
    if 'views' in user_data['interaction']:
        for view in user_data['interaction']['views']:
            user_pois.append(view['poiId'])
    
    if len(user_pois) > 0:
        unique = len(set(user_pois))
        revisit_rate = 1 - (unique / len(user_pois))
        user_revisit_rates.append(revisit_rate)

avg_revisit_rate = sum(user_revisit_rates) / len(user_revisit_rates) if user_revisit_rates else 0

print(f"Average user revisit rate: {avg_revisit_rate*100:.1f}%")

if avg_revisit_rate < 0.1:
    print("\n⚠️  WARNING: Very low revisit rate (<10%)!")
    print("   Users don't revisit POIs - no loyalty patterns for model to learn.")
    print("   Recommendation: Add favorite POI revisits (10-20% of visits).")

print("\n" + "="*80)
print("4. CATEGORY DISTRIBUTION")
print("="*80)
top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:15]
total_cat_visits = sum(category_counts.values())
print(f"Top 15 Categories:")
for cat, count in top_categories:
    pct = count / total_cat_visits * 100
    print(f"  {cat:<35} {count:>6} ({pct:>5.1f}%)")

print("\n" + "="*80)
print("5. RECOMMENDATIONS")
print("="*80)

issues = []

if overlap_rate < 0.5:
    issues.append("❌ CRITICAL: Low train/test POI overlap (<50%)")
    issues.append("   Fix: Re-enable popular POI injection to ensure common POIs appear in both sets")

if avg_revisit_rate < 0.15:
    issues.append("❌ CRITICAL: Low revisit rate (<15%)")
    issues.append("   Fix: Add 15-20% favorite POI revisits per user")

if unique_pois / len(data) > 50:
    issues.append("⚠️  Too many unique POIs per user (>50)")
    issues.append("   Fix: May need to reduce POI diversity or increase user overlap")

cold_start_rate = visit_distribution[1] / unique_pois
if cold_start_rate > 0.7:
    issues.append("❌ CRITICAL: 70%+ of POIs have only 1 visit (cold start problem)")
    issues.append("   Fix: Need more POIs with 5-10+ visits to establish patterns")

if issues:
    print("\n🔴 ISSUES FOUND:")
    for issue in issues:
        print(issue)
    print("\n💡 SOLUTION: Re-enable POI injection and revisit logic in direct_conversion_fsq_to_json.py")
    print("   Lines to uncomment:")
    print("   - Popular POI injection (3-5 popular POIs per user)")
    print("   - Favorite POI revisits (10-20% additional visits)")
else:
    print("\n✅ Data quality looks good! Issues may be in model architecture or training.")

print("\n" + "="*80)
