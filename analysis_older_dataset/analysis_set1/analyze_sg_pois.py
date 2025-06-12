# Script to analyze POIs in Singapore from the TIST2015 POI dataset
# - Counts number of POIs in Singapore (country code = SG)
# - Counts number of unique categories in Singapore
# - Prints basic stats

import os
import matplotlib.pyplot as plt
from collections import Counter

# Path to the large POI file
POI_FILE = 'dataset_TIST2015_POIs.txt'

num_pois_sg = 0
categories_sg = set()

with open(POI_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 5:
            continue
        country_code = parts[4]
        if country_code == 'SG':
            num_pois_sg += 1
            categories_sg.add(parts[3])

# --- Additional analysis: Category counts ---
category_counts = Counter()
with open(POI_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 5:
            continue
        if parts[4] == 'SG':
            category_counts[parts[3]] += 1

# Write stats to a file in descending order of count
with open('preliminary_analysis.txt', 'w', encoding='utf-8') as out:
    out.write(f"Number of POIs in Singapore (SG): {num_pois_sg}\n")
    out.write(f"Number of unique categories in Singapore: {len(categories_sg)}\n")
    out.write("Categories and their counts (descending order):\n")
    for cat, count in category_counts.most_common():
        out.write(f"- {cat}: {count}\n")

# Plot top 20 categories by count
most_common = category_counts.most_common(20)
categories, counts = zip(*most_common)
plt.figure(figsize=(10, 6))
plt.barh(categories[::-1], counts[::-1], color='skyblue')
plt.xlabel('Number of POIs')
plt.title('Top 20 POI Categories in Singapore')
plt.tight_layout()
plt.savefig('sg_poi_top20_categories.png')
plt.show()

# Plot all 399 categories by count
all_categories, all_counts = zip(*category_counts.most_common())
plt.figure(figsize=(12, 40))
plt.barh(all_categories[::-1], all_counts[::-1], color='skyblue')
plt.xlabel('Number of POIs')
plt.title('All POI Categories in Singapore (Descending Order)')
plt.tight_layout()
plt.savefig('sg_poi_all_categories.png')
plt.close()

# Print top 5 and bottom 5 categories for extra insight
print("Top 5 categories:")
for cat, cnt in most_common[:5]:
    print(f"{cat}: {cnt}")
print("\nBottom 5 categories:")
for cat, cnt in category_counts.most_common()[-5:]:
    print(f"{cat}: {cnt}")

print(f"Number of POIs in Singapore (SG): {num_pois_sg}")
print(f"Number of unique categories in Singapore: {len(categories_sg)}")
print("Categories:")
for cat in sorted(categories_sg):
    print(f"- {cat}")

# --- Extract Singapore POI IDs ---
sg_poi_ids = set()
with open(POI_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 5:
            continue
        if parts[4] == 'SG':
            sg_poi_ids.add(parts[0])

# --- Filter checkins for Singapore POIs ---
checkins_file = 'dataset_TIST2015_Checkins.txt'
sg_checkins_file = 'singapore_checkins.txt'
with open(checkins_file, 'r', encoding='utf-8') as fin, open(sg_checkins_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        venue_id = parts[1]
        if venue_id in sg_poi_ids:
            fout.write(line)
print(f"Singapore check-ins written to {sg_checkins_file}")
