import numpy as np
import sys
import os
import traceback

if __name__ == "__main__":
    # Use config_paths.py for paths
    from s00_config_paths import MATRIX_PATH
    input_matrix = MATRIX_PATH
    output_matrix = MATRIX_PATH.replace('.npy', '_normalized.npy')
    try:
        if not os.path.exists(input_matrix):
            print(f"[ERROR] Input matrix file does not exist: {input_matrix}", file=sys.stderr)
            sys.exit(1)
        file_size = os.path.getsize(input_matrix)
        if file_size == 0:
            print(f"[ERROR] Input matrix file is empty: {input_matrix}", file=sys.stderr)
            sys.exit(1)
        # Load shape only
        matrix = np.load(input_matrix, mmap_mode='r')
        shape = matrix.shape
        # Prepare memmap for output
        matrix_norm = np.memmap(output_matrix, dtype=np.float32, mode='w+', shape=shape)
        # Per-user normalization in chunks
        for user_idx in range(shape[0]):
            user_matrix = matrix[user_idx]
            user_sum = user_matrix.sum()
            if user_sum == 0:
                matrix_norm[user_idx] = 0
            else:
                matrix_norm[user_idx] = user_matrix / user_sum
        # Flush memmap to disk
        matrix_norm.flush()
        del matrix_norm
        print(f"Saved normalized matrix to {output_matrix}")
    except Exception as e:
        print(f"[ERROR] Normalization failed: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        print(f"[ERROR] Arguments: input_matrix={input_matrix}, output_matrix={output_matrix}", file=sys.stderr)
        if not os.path.exists(input_matrix):
            print(f"[ERROR] Input matrix file does not exist: {input_matrix}", file=sys.stderr)
        sys.exit(1)
