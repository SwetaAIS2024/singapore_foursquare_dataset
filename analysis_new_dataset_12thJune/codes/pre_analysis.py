import pandas as pd

# Path to your check-in CSV file
csv_path = 'analysis_new_dataset_12thJune/latest_dataset/places_sg.csv'  # Update if needed

df = pd.read_csv(csv_path)

with open('analysis_new_dataset_12thJune/codes/pre_analysis_places_sg.txt', 'w') as f:
    f.write('--- Dataset Pre-Analysis ---\n')
    f.write(f'Number of rows: {len(df)}\n')
    f.write(f'Columns: {list(df.columns)}\n')
    f.write(f'First 5 rows:\n{df.head()}\n')
    f.write(f'Column types:\n{df.dtypes}\n')
    f.write(f'Null values per column:\n{df.isnull().sum()}\n')
    f.write(f'Unique primary categories: {df["primary_category"].nunique()}\n')
    f.write(f'Top 10 primary categories:\n{df["primary_category"].value_counts().head(10)}\n')
    f.write(f'Unique secondary categories: {df["secondary_category"].nunique()}\n')
    f.write(f'Unique tertiary categories: {df["tertiary_category"].nunique()}\n')
print('Pre-analysis complete. Results saved to pre_analysis.txt')


# Path to your check-in CSV file
csv_path = 'analysis_new_dataset_12thJune/latest_dataset/categories_sg.csv'  # Update if needed

df2 = pd.read_csv(csv_path)
# category_id,category_level,category_name,category_label,level1_category_id,level1_category_name,
# level2_category_id,level2_category_name,level3_category_id,level3_category_name,level4_category_id,
# level4_category_name,level5_category_id,level5_category_name,level6_category_id,level6_category_name


with open('analysis_new_dataset_12thJune/codes/pre_analysis_categories_sg.txt', 'w') as f:
    f.write('--- Dataset Pre-Analysis ---\n')
    f.write(f'Number of rows: {len(df2)}\n')
    f.write(f'Columns: {list(df2.columns)}\n')
    f.write(f'First 5 rows:\n{df2.head()}\n')
    f.write(f'Column types:\n{df2.dtypes}\n')
    f.write(f'Null values per column:\n{df2.isnull().sum()}\n')
    # Category level analysis
    for level in range(1, 7):
        col_name = f'level{level}_category_name'
        if col_name in df2.columns:
            f.write(f'Unique {col_name}: {df2[col_name].nunique()}\n')
            f.write(f'Top 10 {col_name}:\n{df2[col_name].value_counts().head(10)}\n')
    # General category_name and category_label
    if 'category_name' in df2.columns:
        f.write(f'Unique category_name: {df2["category_name"].nunique()}\n')
        f.write(f'Top 10 category_name:\n{df2["category_name"].value_counts().head(10)}\n')
    if 'category_label' in df2.columns:
        f.write(f'Unique category_label: {df2["category_label"].nunique()}\n')
        f.write(f'Top 10 category_label:\n{df2["category_label"].value_counts().head(10)}\n')
print('Pre-analysis for categories_sg.csv complete. Results saved to pre_analysis_categories_sg.txt')