import argparse
import numpy as np
import sys
import os
import traceback

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_matrix', type=str, required=True)
    parser.add_argument('--output_matrix', type=str, required=True)
    args = parser.parse_args()
    try:
        print(f"[DEBUG] Current working directory: {os.getcwd()}")
        print(f"[DEBUG] Input matrix path: {args.input_matrix}")
        print(f"[DEBUG] Output matrix path: {args.output_matrix}")
        print(f"[DEBUG] Input matrix exists: {os.path.exists(args.input_matrix)}")
        # Enhanced: Check file existence and size before loading
        if not os.path.exists(args.input_matrix):
            print(f"[ERROR] Input matrix file does not exist: {args.input_matrix}", file=sys.stderr)
            sys.exit(1)
        file_size = os.path.getsize(args.input_matrix)
        print(f"[DEBUG] Input matrix file size: {file_size} bytes")
        if file_size == 0:
            print(f"[ERROR] Input matrix file is empty: {args.input_matrix}", file=sys.stderr)
            sys.exit(1)
        # Try loading the matrix with detailed error reporting
        try:
            matrix = np.load(args.input_matrix)
        except Exception as e:
            print(f"[ERROR] Failed to load input matrix file '{args.input_matrix}': {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
        print(f"[INFO] Loaded matrix from {args.input_matrix}")
        print(f"[INFO] Matrix shape: {matrix.shape}, dtype: {matrix.dtype}")
        print(f"[INFO] Matrix sample (first 5 elements): {matrix.flat[:5]}")

        # Per-user normalization: divide each user's matrix by their total check-ins (across all spatial clusters, categories, and time)
        user_sums = matrix.sum(axis=(1,2,3), keepdims=True)
        print("Calculated per-user sums for normalization.")
        print(f"User sums shape: {user_sums.shape}, dtype: {user_sums.dtype}, min: {user_sums.min()}, max: {user_sums.max()}")
        user_sums[user_sums == 0] = 1  # Avoid division by zero
        matrix_norm = matrix / user_sums
        print("Per-user normalization complete.")

        # Inspection logic: print nonzero entries for the first user (user index 0)
        user_idx = 0
        nonzero = np.nonzero(matrix_norm[user_idx])
        print(f"Nonzero entries for user {user_idx} (normalized):")
        for spatial_cluster, cat, t in zip(*nonzero):
            val = matrix_norm[user_idx, spatial_cluster, cat, t]
            if val > 0:
                print(f"  Spatial Cluster {spatial_cluster}, Category {cat}, Hour {t}, Value: {val:.4f}")

        # Optionally, save the normalized matrix
        print(f"[DEBUG] About to save normalized matrix. Shape: {matrix_norm.shape}, dtype: {matrix_norm.dtype}")
        try:
            np.save(args.output_matrix, matrix_norm)
            print(f"[DEBUG] np.save completed for {args.output_matrix}")
        except Exception as save_e:
            print(f"[ERROR] Failed to save normalized matrix: {save_e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
        print(f"Saved normalized matrix to {args.output_matrix}")
    except Exception as e:
        print(f"[ERROR] Normalization failed: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        print(f"[ERROR] Arguments: input_matrix={args.input_matrix}, output_matrix={args.output_matrix}", file=sys.stderr)
        if not os.path.exists(args.input_matrix):
            print(f"[ERROR] Input matrix file does not exist: {args.input_matrix}", file=sys.stderr)
        sys.exit(1)
