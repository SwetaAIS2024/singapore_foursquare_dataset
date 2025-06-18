# Print the first 10 lines of the large POI file to inspect its structure
input_path = 'original_dataset_truncated/dataset_TIST2015_POIs.txt'
with open(input_path, 'r', encoding='utf-8') as infile:
    for i, line in enumerate(infile):
        print(line.strip())
        if i >= 9:
            break
