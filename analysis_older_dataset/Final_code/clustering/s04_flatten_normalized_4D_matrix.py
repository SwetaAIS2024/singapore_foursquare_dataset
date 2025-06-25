import argparse
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_matrix', type=str, required=True, help='Path to normalized 4D matrix (npy)')
    parser.add_argument('--output_matrix', type=str, required=True, help='Path to save flattened matrix (npy)')
    args = parser.parse_args()
    if not os.path.exists(args.input_matrix):
        print(f"[ERROR] Input matrix file does not exist: {args.input_matrix}")
        exit(1)
    matrix = np.load(args.input_matrix)
    print(f"Loaded normalized matrix with shape: {matrix.shape}")
    n_users = matrix.shape[0]
    user_vectors = matrix.reshape(n_users, -1)
    print(f"Flattened user vectors shape: {user_vectors.shape}")
    np.save(args.output_matrix, user_vectors)
    print(f"Saved flattened user vectors to {args.output_matrix}")
    num_total = user_vectors.size
    num_nonzero = np.count_nonzero(user_vectors)
    sparsity = 1.0 - (num_nonzero / num_total)
    print(f"Sparsity of flattened user vectors: {sparsity:.6f} (fraction of zero entries)")
    user_idx = 0
    nonzero = np.nonzero(user_vectors[user_idx])[0]
    print(f"Nonzero entries for user {user_idx} in flattened vector:")
    for idx in nonzero:
        print(f"  Index {idx}, Value: {user_vectors[user_idx, idx]:.4f}")
