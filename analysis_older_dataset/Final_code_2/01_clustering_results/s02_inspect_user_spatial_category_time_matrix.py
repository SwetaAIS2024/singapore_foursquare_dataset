import pandas as pd
import numpy as np
from s00_config_paths import MATRIX_PATH

if __name__ == "__main__":
    # Use config_paths.py for paths
    matrix = np.load(MATRIX_PATH)
    summary = {
        "n_users": [matrix.shape[0]],
        "n_spatial_clusters": [matrix.shape[1]],
        "n_categories": [matrix.shape[2]],
        "n_timebins": [matrix.shape[3]],
        "total_nonzero": [int(np.count_nonzero(matrix))],
        "max_value": [float(matrix.max())],
        "min_value": [float(matrix.min())]
    }
    df = pd.DataFrame(summary)
    output = MATRIX_PATH.replace('.npy', '_summary.csv')
    df.to_csv(output, index=False)
    print(f"Matrix summary written to {output}")
