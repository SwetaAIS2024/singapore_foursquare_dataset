# summarize_histograms.py
"""
Script to summarize the findings in the histogram for each user.
For each user in the histograms folder, this script will:
- Load the day, noon, and evening histogram images (if present)
- Use matplotlib to extract the bar heights and labels from each histogram
- For each time bin, print and save a summary of the top 3 POI categories by average check-ins per day
- Save the summary as a text file in the user's histogram folder
"""
import os
import numpy as np
from PIL import Image
import pytesseract
import csv

HIST_BASE = 'histograms'
TIME_BINS = ['day', 'noon', 'evening']

for user in os.listdir(HIST_BASE):
    user_dir = os.path.join(HIST_BASE, user)
    if not os.path.isdir(user_dir):
        continue
    summary_lines = []
    summary_lines.append(f"User: {user}\n")
    for bin_name in TIME_BINS:
        csv_path = os.path.join(user_dir, f"{bin_name}_hist.csv")
        if not os.path.exists(csv_path):
            continue
        cats = []
        vals = []
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                cats.append(row['POI Category'])
                vals.append(float(row['Avg Check-ins per Day']))
        if cats and vals:
            arr = np.array(vals)
            idx = np.argsort(arr)[::-1][:3]  # top 3
            summary_lines.append(f"Top 3 POI categories for {bin_name} (by avg check-ins per day):")
            for rank, i in enumerate(idx, 1):
                summary_lines.append(f"  {rank}. POI Category: {cats[i]} | Avg Check-ins/Day: {vals[i]:.2f}")
            summary_lines.append("")
        else:
            summary_lines.append(f"Could not extract data for {bin_name} histogram.")
            summary_lines.append("")
    # Save summary
    with open(os.path.join(user_dir, 'histogram_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    print(f"Summary written for user {user}")
