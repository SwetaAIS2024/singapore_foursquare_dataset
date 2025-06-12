import csv

# Path to the large POI file
input_path = 'original_dataset_truncated/dataset_TIST2015_POIs.txt'
output_path = 'analysis_set2/Tensor_factorisation_and_clustering/sg_place_id_to_category.csv'

with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['place_id', 'category'])
    for line in infile:
        parts = line.strip().split('\t')
        if len(parts) < 5:
            continue
        place_id, lat, lon, category, country = parts[:5]
        if country == 'SG':
            writer.writerow([place_id, category])
print(f'Saved Singapore POI mapping to {output_path}')
