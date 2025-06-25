import argparse
import pandas as pd
import numpy as np
from s00_config_paths import MATRIX_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', type=str, default=MATRIX_PATH)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    matrix = np.load(args.matrix)
    summary = {
        "n_users": [matrix.shape[0]],
        "n_clusters": [matrix.shape[1]],
        "n_categories": [matrix.shape[2]],
        "n_timebins": [matrix.shape[3]],
        "total_nonzero": [int(np.count_nonzero(matrix))],
        "max_value": [float(matrix.max())],
        "min_value": [float(matrix.min())]
    }
    df = pd.DataFrame(summary)
    df.to_csv(args.output, index=False)
    print(f"Matrix summary written to {args.output}")
