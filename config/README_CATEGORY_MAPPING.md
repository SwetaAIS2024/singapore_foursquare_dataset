# Category Mapping Configuration

## Overview

This directory contains the category mapping file used to group detailed POI categories into broader group categories during postprocessing.

## Purpose

The synthetic data generator creates transactions with detailed POI categories (e.g., "Indian Restaurant", "Chinese Restaurant", "Italian Restaurant"). The category grouping feature consolidates these into higher-level groups (e.g., all become "Restaurant").

This is useful for:
- Simplifying analysis with fewer categories
- Creating consistent category labels across datasets
- Reducing sparsity in category-based models
- Standardizing category naming conventions

## File Formats

### JSON Format (Recommended)
```json
{
  "Indian Restaurant": "Restaurant",
  "Chinese Restaurant": "Restaurant",
  "Italian Restaurant": "Restaurant",
  "Coffee Shop": "Cafe",
  "Café": "Cafe",
  "Bar": "Bar",
  "Cocktail Bar": "Bar"
}
```

### CSV Format
```csv
original_category,group_category
Indian Restaurant,Restaurant
Chinese Restaurant,Restaurant
Italian Restaurant,Restaurant
Coffee Shop,Cafe
Café,Cafe
Bar,Bar
Cocktail Bar,Bar
```

## Setup Instructions

### Step 1: Create Your Mapping File

You have two options:

**Option A: Rename the template**
```bash
# For JSON format
cp category_mapping_template.json category_mapping.json

# For CSV format
cp category_mapping_template.csv category_mapping.csv
```

**Option B: Create from scratch**
Create a new file named `category_mapping.json` or `category_mapping.csv` in this directory.

### Step 2: Add Your Mappings

Replace the template content with your actual 180 → 50 category mappings.

**Important:**
- Use exact category names as they appear in your data
- Category names are case-sensitive
- Unmapped categories will remain unchanged
- One original category can only map to one group category

### Step 3: Run Postprocessing

The postprocessing script will automatically detect and use the mapping file:

```bash
python scripts/postprocess_synthetic_data.py
```

To disable category grouping:
```python
# In the script or when called programmatically
main(apply_grouping=False)
```

## Example Mapping Strategy

### Consolidating Restaurants (180 → 50 categories)

**Original (detailed):**
- Indian Restaurant
- Chinese Restaurant
- Italian Restaurant
- Thai Restaurant
- Japanese Restaurant
- French Restaurant
- Mexican Restaurant
- ... (many more)

**Grouped (consolidated):**
- Restaurant (for all sit-down restaurants)
- Fast Food (for quick service)
- Cafe (for coffee shops and cafes)

### Recommended Group Categories (50 groups)

Here's a suggested structure for 50 group categories:

1. **Food & Drink (15 groups)**
   - Restaurant
   - Fast Food
   - Cafe
   - Bar
   - Nightlife
   - Bakery
   - Dessert Shop
   - Food Court
   - Food Truck
   - Brewery
   - Winery
   - Ice Cream Shop
   - Juice Bar
   - Tea Room
   - Diner

2. **Shopping (10 groups)**
   - Shopping Mall
   - Clothing Store
   - Electronics Store
   - Grocery Store
   - Bookstore
   - Pharmacy
   - Convenience Store
   - Market
   - Department Store
   - Specialty Shop

3. **Services (8 groups)**
   - Bank
   - Post Office
   - Government Building
   - Medical Center
   - Salon & Spa
   - Automotive
   - Professional Services
   - Repair Shop

4. **Entertainment (7 groups)**
   - Movie Theater
   - Museum
   - Art Gallery
   - Music Venue
   - Theater
   - Arcade
   - Bowling Alley

5. **Outdoors & Recreation (5 groups)**
   - Park
   - Beach
   - Gym
   - Sports Facility
   - Outdoor Recreation

6. **Travel & Transport (3 groups)**
   - Hotel
   - Transportation Hub
   - Travel Service

7. **Other (2 groups)**
   - Education
   - Miscellaneous

## Verification

After running postprocessing with category mapping, check the output:

1. **Console Output:** Shows mapping statistics
   ```
   Category Grouping Results:
     Total transactions processed: 56,789
     Categories mapped: 55,000 (96.85%)
     Categories unmapped: 1,789 (3.15%)
   
   Group Category Distribution (Top 20):
     Restaurant                    : 15,234 (26.82%)
     Cafe                          :  8,901 (15.67%)
     Shopping                      :  7,456 (13.13%)
     ...
   ```

2. **Output Files:**
   - `filtered_grouped.json` - Data with grouped categories (before 5-core)
   - `filtered_5core_filtered.json` - Final filtered data with grouped categories

3. **Unmapped Categories Warning:**
   If categories are found in your data but not in the mapping file, they'll be listed:
   ```
   ⚠️  Warning: 5 categories not found in mapping:
      - 'Burger Restaurant' (234 transactions)
      - 'Ramen Shop' (156 transactions)
      - ...
   ```
   Add these to your mapping file to ensure complete coverage.

## Troubleshooting

### Problem: Categories not being mapped

**Solution:** Check for exact name matches (case-sensitive)
```python
# Your data has: "Coffee Shop"
# Your mapping has: "coffee shop"  ❌ Won't match
# Should be: "Coffee Shop"  ✅ Will match
```

### Problem: Too many unmapped categories

**Solution:** Extract actual category names from your data first:
```python
import json

with open('data/synthetic/fsq_to_synthetic_all_categories.json', 'r') as f:
    data = json.load(f)

categories = set()
for user in data:
    for txn in user['interaction']['transactions']:
        if txn['poiCategories']:
            categories.add(txn['poiCategories'][0])

print(f"Found {len(categories)} unique categories:")
for cat in sorted(categories):
    print(f"  - {cat}")
```

### Problem: Mapping file not found

**Solution:** Ensure the file is named exactly:
- `category_mapping.json` OR
- `category_mapping.csv`

And located in the `config/` directory.

## Advanced Usage

### Custom Mapping File Location

```python
from scripts.postprocess_synthetic_data import main

# Use custom mapping file path
main(category_mapping_file='path/to/my_mapping.json')
```

### Programmatic Access

```python
from scripts.postprocess_synthetic_data import (
    load_category_mapping,
    apply_category_grouping,
    load_synthetic_data
)

# Load your data
data = load_synthetic_data('data/synthetic/output.json')

# Load mapping
mapping = load_category_mapping('config/category_mapping.json')

# Apply grouping
grouped_data, stats = apply_category_grouping(data, mapping)

# Save result
import json
with open('output_grouped.json', 'w') as f:
    json.dump(grouped_data, f, indent=2)
```

## Notes

- Category grouping is applied **before** 5-core filtering
- The original detailed categories are replaced, not added to
- If you need both versions, save the output before and after grouping
- Grouping affects the category distribution statistics in the report

## Support

For questions or issues with category mapping:
1. Check the console output for unmapped categories
2. Verify your mapping file format
3. Ensure category names match exactly (including case and spaces)
4. See GitHub Issues for common problems and solutions
