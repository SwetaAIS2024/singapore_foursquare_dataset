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
        # Input validation
        if not os.path.exists(input_matrix):
            print(f"[ERROR] Input matrix file does not exist: {input_matrix}", file=sys.stderr)
            sys.exit(1)
        file_size = os.path.getsize(input_matrix)
        if file_size == 0:
            print(f"[ERROR] Input matrix file is empty: {input_matrix}", file=sys.stderr)
            sys.exit(1)

        print("[INFO] Loading input matrix...")
        # Load the matrix in read-only mode to get shape and dtype
        orig_matrix = np.load(input_matrix, mmap_mode='r')
        shape = orig_matrix.shape
        dtype = orig_matrix.dtype
        print(f"[INFO] Matrix shape: {shape}, dtype: {dtype}")

        # Create numpy array for output
        print("[INFO] Creating output array...")
        result_matrix = np.zeros(shape, dtype=np.float32)

        # Process the matrix in chunks to save memory
        CHUNK_SIZE = 100  # Process 100 users at a time
        for start_idx in range(0, shape[0], CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, shape[0])
            print(f"[INFO] Processing users {start_idx} to {end_idx}...")
            
            # Load chunk of users
            user_chunk = orig_matrix[start_idx:end_idx]
            
            # Calculate sums for this chunk
            user_sums = user_chunk.sum(axis=(1,2,3), keepdims=True)
            user_sums[user_sums == 0] = 1  # Avoid division by zero
            
            # Normalize chunk
            result_matrix[start_idx:end_idx] = user_chunk / user_sums
        
        # Save the normalized matrix
        print("[INFO] Saving normalized matrix...")
        np.save(output_matrix, result_matrix)
        print(f"[SUCCESS] Saved normalized matrix to {output_matrix}")
        
        # Optional: Verify the saved file
        print("[INFO] Verifying saved file...")
        verify_matrix = np.load(output_matrix, mmap_mode='r')
        print(f"[INFO] Verification - Normalized matrix shape: {verify_matrix.shape}, dtype: {verify_matrix.dtype}")
        nonzero_count = np.count_nonzero(verify_matrix[0])
        print(f"[INFO] First user has {nonzero_count} nonzero entries")
        del verify_matrix
        
    except Exception as e:
        print(f"[ERROR] Normalization failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
