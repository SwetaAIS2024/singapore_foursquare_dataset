import numpy as np
import os

import numpy as np
import os
import sys
import traceback

if __name__ == "__main__":
    from s00_config_paths import MATRIX_PATH
    input_matrix = MATRIX_PATH.replace('.npy', '_normalized.npy')
    output_matrix = MATRIX_PATH.replace('.npy', '_normalized_flattened.npy')
    
    if not os.path.exists(input_matrix):
        print(f"[ERROR] Input matrix file does not exist: {input_matrix}")
        exit(1)
    
    try:
        print("[INFO] Loading normalized matrix...")
        matrix = np.load(input_matrix, mmap_mode='r')
        print(f"[INFO] Input matrix shape: {matrix.shape}, dtype: {matrix.dtype}")
        
        n_users = matrix.shape[0]
        flat_dim = np.prod(matrix.shape[1:])
        print(f"[INFO] Will flatten to shape: ({n_users}, {flat_dim})")
        
        # Create output array
        print("[INFO] Creating output array...")
        result_matrix = np.zeros((n_users, flat_dim), dtype=np.float32)
        
        # Process in chunks to save memory
        CHUNK_SIZE = 100
        for start_idx in range(0, n_users, CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, n_users)
            print(f"[INFO] Processing users {start_idx} to {end_idx}...")
            
            # Load and flatten chunk
            chunk = matrix[start_idx:end_idx]
            result_matrix[start_idx:end_idx] = chunk.reshape((end_idx - start_idx, -1))
            
            # Free memory
            del chunk
        
        # Save the flattened matrix
        print("[INFO] Saving flattened matrix...")
        np.save(output_matrix, result_matrix)
        print(f"[SUCCESS] Saved flattened user vectors to {output_matrix}")
        
        # Verify the saved file
        print("[INFO] Verifying saved file...")
        verify_matrix = np.load(output_matrix, mmap_mode='r')
        print(f"[INFO] Verification - Matrix shape: {verify_matrix.shape}, dtype: {verify_matrix.dtype}")
        nonzero = np.count_nonzero(verify_matrix[0])
        print(f"[INFO] First user has {nonzero} nonzero entries")
        print(f"[INFO] Sample values - First user, first 5 elements: {verify_matrix[0, :5]}")
        del verify_matrix
        
    except Exception as e:
        print(f"[ERROR] Flattening failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
