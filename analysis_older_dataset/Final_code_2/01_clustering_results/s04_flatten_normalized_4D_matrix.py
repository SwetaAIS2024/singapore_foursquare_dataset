import numpy as np
import os

if __name__ == "__main__":
    from s00_config_paths import MATRIX_PATH
    input_matrix = MATRIX_PATH.replace('.npy', '_normalized.npy')
    output_matrix = MATRIX_PATH.replace('.npy', '_flattened.npy')
    if not os.path.exists(input_matrix):
        print(f"[ERROR] Input matrix file does not exist: {input_matrix}")
        exit(1)
    matrix = np.load(input_matrix, mmap_mode='r')
    n_users = matrix.shape[0]
    flat_dim = np.prod(matrix.shape[1:])
    user_vectors = np.memmap(output_matrix, dtype=np.float32, mode='w+', shape=(n_users, flat_dim))
    for user_idx in range(n_users):
        user_vectors[user_idx] = matrix[user_idx].reshape(-1)
    del user_vectors
    print(f"Saved flattened user vectors to {output_matrix}")
