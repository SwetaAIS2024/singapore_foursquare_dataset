#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract categories filtered with 'yes' from Relevant_POI_category_original.xlsx
and create category_mapping.csv template.
"""
import pandas as pd
from pathlib import Path

# Read the Excel file
excel_path = Path("data/raw/Relevant_POI_category_original.xlsx")
print(f"Reading Excel file: {excel_path}")

df = pd.read_excel(excel_path)

print(f"\nColumns in Excel file: {df.columns.tolist()}")
print(f"Total rows: {len(df)}")
print(f"\nFirst few rows:")
print(df.head(10))

# Find the column that contains 'yes' filter
# Common column names: 'Relevant', 'Filter', 'Include', 'Yes/No', etc.
filter_column = None
category_column = None

for col in df.columns:
    col_lower = str(col).lower()
    if 'relevant' in col_lower or 'filter' in col_lower or 'include' in col_lower:
        filter_column = col
    if 'category' in col_lower or 'name' in col_lower or 'type' in col_lower:
        if category_column is None:  # Get the first one
            category_column = col

print(f"\nDetected columns:")
print(f"  Category column: {category_column}")
print(f"  Filter column: {filter_column}")

# Filter rows with 'yes'
if filter_column:
    # Try different variations of 'yes'
    filtered_df = df[df[filter_column].astype(str).str.lower().str.strip() == 'yes']
    print(f"\nFiltered rows with 'yes': {len(filtered_df)}")
else:
    print("\nNo filter column found, using all rows")
    filtered_df = df

# Extract unique categories
if category_column:
    categories = filtered_df[category_column].dropna().unique()
    categories = sorted([str(cat).strip() for cat in categories if str(cat).strip() != 'nan'])
else:
    # Try first column
    categories = filtered_df.iloc[:, 0].dropna().unique()
    categories = sorted([str(cat).strip() for cat in categories if str(cat).strip() != 'nan'])

print(f"\nExtracted {len(categories)} unique categories:")
for i, cat in enumerate(categories[:20], 1):
    print(f"  {i}. {cat}")
if len(categories) > 20:
    print(f"  ... and {len(categories) - 20} more")

# Create CSV file
output_path = Path("config/category_mapping.csv")
output_df = pd.DataFrame({
    'original_category': categories,
    'group_category': [''] * len(categories)  # Empty for user to fill
})

output_df.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n✅ Created category mapping template: {output_path}")
print(f"   Please fill in the 'group_category' column with your 50 group categories")
