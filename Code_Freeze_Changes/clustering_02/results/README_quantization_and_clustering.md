# README: Quantization and Memory-Efficient Matrix Processing for Large-Scale Clustering

## Overview
This project addresses the challenge of efficiently building, quantizing, normalizing, flattening, and clustering a large user × spatial_cluster × category × time matrix for clustering analysis. The matrix size (e.g., 2000 users × 50 spatial clusters × 168 weekly time bins × 180 POIs) can reach 11GB or more, so all steps are designed to be batch-wise and memory-efficient, suitable for cloud or distributed environments.

## Key Methods and Approaches

### 1. Quantization Methods
- **KBinsDiscretizer (scikit-learn):**
  - Supports 'uniform', 'quantile', and 'kmeans' binning strategies.
  - Used with 'ordinal' encoding to convert continuous values to integer bin indices.
- **Alternative Quantization:**
  - Manual binning with `numpy.digitize` or `pandas.cut`/`qcut`.
  - Log-scaling or rounding for highly skewed data.
  - Clustering-based binning (e.g., 1D KMeans).

### 2. Data Type Optimization
- After quantization, values are cast to the lowest-precision integer type possible (e.g., `np.uint8` for 0–255 bins, `np.uint16` for more bins).
- This reduces memory usage by up to 4x compared to float32.
- Sparse storage (`scipy.sparse`) is used if the matrix is mostly zeros.

### 3. Batch-wise Processing Pipeline
- **Matrix Construction:**
  - User × spatial_cluster × category × time matrix is built in batches.
  - Each batch is quantized and saved independently (as `.npz` or sparse format).
- **Normalization:**
  - Each batch is normalized independently (row-wise or as appropriate).
- **Flattening:**
  - Each normalized batch is flattened and saved as a dense `.npy` file.
  - Feature variance is logged for diagnostics.
- **Clustering:**
  - MiniBatchKMeans is used in true batch-wise mode with `.partial_fit()`.
  - Cluster labels are assigned batch-wise, never loading the full matrix.
- **Post-Clustering Analysis:**
  - Cluster size distributions and PCA scatter plots are generated batch-wise.
  - All-zero user vectors can be filtered out before clustering.

### 4. Diagnostics and Quality Checks
- Feature variance and nonzero counts per user are logged after quantization and normalization.
- If most user vectors are all zeros or have low variance, clustering quality will be poor (most users in one cluster).
- Recommendations include adjusting quantization bins, filtering all-zero users, or revisiting normalization.

## Example Code Snippet: Quantization and Type Casting
```python
import numpy as np

# Quantize to 8 bins, then cast to uint8
n_bins = 8
matrix_quantized = np.digitize(matrix, np.linspace(matrix.min(), matrix.max(), n_bins+1)) - 1
matrix_uint8 = matrix_quantized.astype(np.uint8)
```

## Output Files
- Batch-wise `.npz` (sparse), `_normalized.npz`, `_flattened.npy`
- `matrix_metadata.json`
- Cluster labels `.npy`, user-cluster mapping `.csv`
- Diagnostic logs and PCA plots

## Recommendations
- Use the lowest-precision integer type that fits your quantized values.
- Use sparse storage for highly sparse matrices.
- Always process and save data in batches to avoid memory issues.
- Monitor feature variance and nonzero counts to ensure clustering quality.

---
**For more details, see the code in the `analysis_older_dataset/Final_code_2/01_clustering_results/` directory.**
