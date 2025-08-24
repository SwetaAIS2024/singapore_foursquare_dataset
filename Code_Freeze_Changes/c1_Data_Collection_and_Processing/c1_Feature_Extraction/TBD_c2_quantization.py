import numpy as np
from scipy import sparse
from sklearn.preprocessing import KBinsDiscretizer

def quantize_sparse_matrix(matrix, n_bins=256, strategy='quantile'):
    """
    Quantize a matrix to uint8 format using KBinsDiscretizer or ordinal encoding if unique values are few.
    Returns the quantized matrix and quantization parameters for reconstruction.
    """
    # Checking if the input matrix is in csr format, if not, convert it 
    if not sparse.isspmatrix_csr(matrix):
        matrix = matrix.tocsr()
    # Extracting the non zeroes data from the sparse matrix    
    data = matrix.data
    # Returning a trivial meta data if the data is empty, that is matrix has no non-zero values 
    if len(data) == 0:
        return matrix, {'bin_edges': None, 'min_val': 0, 'max_val': 0}
    
    data = data.reshape(-1, 1) #reshaping the data to column vector for the scikit learn discretizer 
    unique_values = np.unique(data)
    n_unique = len(unique_values)
    
    # if all the values are same, just one similar value, the values are casted to uint8 and returning the meta data
    if n_unique == 1: 
        quantized_data = data.astype(np.uint8)
        quantized_matrix = sparse.csr_matrix((quantized_data.ravel(), matrix.indices, matrix.indptr),
                                             shape=matrix.shape, dtype=np.uint8)
        metadata = {
            'bin_edges': [unique_values[0]],
            'min_val': float(data.min()),
            'max_val': float(data.max())
        }
        return quantized_matrix, metadata
    
    # if the no of unique values is lesssthan or equal to no of bins 
    # and the values are consecutive integers, mapping the values to bins using the ordinal encoding
    if n_unique <= n_bins and np.all(np.diff(unique_values) == 1):
        value_to_bin = {v: i for i, v in enumerate(unique_values)}
        quantized_data = np.vectorize(value_to_bin.get)(data.ravel()).astype(np.uint8)
        quantized_matrix = sparse.csr_matrix((quantized_data, matrix.indices, matrix.indptr),
                                             shape=matrix.shape, dtype=np.uint8)
        metadata = {
            'bin_edges': unique_values.tolist(),
            'min_val': float(data.min()),
            'max_val': float(data.max()),
            'encoding': 'ordinal'
        }
        return quantized_matrix, metadata
    
    # general case - many values and non consecutive values 
    actual_bins = max(2, min(n_bins, n_unique))
    discretizer = KBinsDiscretizer(n_bins=actual_bins, encode='ordinal', strategy=strategy)
    quantized_data = discretizer.fit_transform(data)
    quantized_data = quantized_data.astype(np.uint8)
    quantized_matrix = sparse.csr_matrix((quantized_data.ravel(), matrix.indices, matrix.indptr),
                                       shape=matrix.shape, dtype=np.uint8)
    metadata = {
        'bin_edges': discretizer.bin_edges_[0].tolist(),
        'min_val': float(data.min()),
        'max_val': float(data.max()),
        'encoding': 'binned'
    }
    return quantized_matrix, metadata
