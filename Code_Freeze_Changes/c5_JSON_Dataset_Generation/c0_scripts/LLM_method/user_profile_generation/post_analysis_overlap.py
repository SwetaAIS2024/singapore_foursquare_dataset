import json
import csv

with open("c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_profile_generation/app_profiles_all_users_version_4.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare CSV output
csv_rows = []
header = [
    "userId",
    "total_interactions",
    "views",
    "transactions",
    "reviews",
    "views_percent",
    "transactions_percent",
    "reviews_percent",
    "views∩transactions",
    "views∩reviews",
    "transactions∩reviews",
    "views∩transactions∩reviews",
    "ratio_close_to_70_20_10"
]

# Totals
total_users = len(data)
total_views = 0
total_txns = 0
total_reviews = 0
total_vt = 0
total_vr = 0
total_tr = 0
total_vtr = 0

for user in data:
    user_id = user["user"]["userId"]
    views = user["interaction"]["views"]
    txns = user["interaction"]["transactions"]
    reviews = user["interaction"]["reviews"]

    # Use poiId for overlap calculation
    views_set = set(v["poiId"] for v in views)
    txns_set = set(t["poiId"] for t in txns)
    reviews_set = set(r["poiId"] for r in reviews)

    vt = len(views_set & txns_set)
    vr = len(views_set & reviews_set)
    tr = len(txns_set & reviews_set)
    vtr = len(views_set & txns_set & reviews_set)

    n_views = len(views)
    n_txns = len(txns)
    n_reviews = len(reviews)
    total = n_views + n_txns + n_reviews
    views_pct = (n_views / total * 100) if total else 0
    txns_pct = (n_txns / total * 100) if total else 0
    reviews_pct = (n_reviews / total * 100) if total else 0
    close_to_ratio = (
        (65 <= views_pct <= 75) and
        (15 <= txns_pct <= 25) and
        (5 <= reviews_pct <= 15)
    )

    csv_rows.append([
        user_id,
        total,
        n_views,
        n_txns,
        n_reviews,
        f"{views_pct:.1f}",
        f"{txns_pct:.1f}",
        f"{reviews_pct:.1f}",
        vt,
        vr,
        tr,
        vtr,
        "YES" if close_to_ratio else "NO"
    ])
    total_views += n_views
    total_txns += n_txns
    total_reviews += n_reviews
    total_vt += vt
    total_vr += vr
    total_tr += tr
    total_vtr += vtr

grand_total = total_views + total_txns + total_reviews
views_pct_total = (total_views / grand_total * 100) if grand_total else 0
txns_pct_total = (total_txns / grand_total * 100) if grand_total else 0
reviews_pct_total = (total_reviews / grand_total * 100) if grand_total else 0

# Add TOTAL row
csv_rows.append([
    "TOTAL",
    grand_total,
    total_views,
    total_txns,
    total_reviews,
    f"{views_pct_total:.1f}",
    f"{txns_pct_total:.1f}",
    f"{reviews_pct_total:.1f}",
    total_vt,
    total_vr,
    total_tr,
    total_vtr,
    ""
])

# Write to CSV
with open("c5_JSON_Dataset_Generation/c0_scripts/LLM_method/user_profile_generation/user_interaction_stats.csv", "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(csv_rows)

print("CSV file 'user_interaction_stats.csv' generated.\n")
print(f"Number of users: {total_users}")
print(f"Total interactions: {grand_total}")
print(f"  - Total views: {total_views}")
print(f"  - Total transactions: {total_txns}")
print(f"  - Total reviews: {total_reviews}")
print(f"  - Views %: {views_pct_total:.1f}%")
print(f"  - Transactions %: {txns_pct_total:.1f}%")
print(f"  - Reviews %: {reviews_pct_total:.1f}%")
print(f"  - Views∩Transactions: {total_vt} ({(total_vt/grand_total*100):.2f}%)")
print(f"  - Views∩Reviews: {total_vr} ({(total_vr/grand_total*100):.2f}%)")
print(f"  - Transactions∩Reviews: {total_tr} ({(total_tr/grand_total*100):.2f}%)")
print(f"  - Views∩Transactions∩Reviews: {total_vtr} ({(total_vtr/grand_total*100):.2f}%)")