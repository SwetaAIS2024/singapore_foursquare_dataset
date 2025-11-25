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
