import numpy as np
from scipy import sparse
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_selection import VarianceThreshold

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

import json
import os
import sys
import traceback
from c0_Configuration.config_paths import FINAL_INPUT_DATASET, CLUSTER_OUTPUT_DIR, DIMRED_MODEL_PATH
import joblib
import gc
from c1_Data_Collection_and_Processing.util import load_config


def dimension_reduction(algo, batch_files, feature_indices, svd_components, encoding_dim=32, epochs=20, batch_size=250):
    # --- Second pass: Fit SVD, scaler, selector --- 
    # DIMENSIONALITY REDUCTION AND FEATURE SELECTION BEFORE CLUSTERING 
    feature_autoencoded = None
    all_X = []
    for batch_file in batch_files:
        X_sparse = sparse.load_npz(batch_file)
        X_sparse = X_sparse[:, feature_indices]
        all_X.append(X_sparse)
    X_all = vstack(all_X)
    # Normalizing the data 
    X_all = normalize(X_all, norm='l2', axis=1)
    # why l2 - kmeans and svd based on euclidean distance,
    # so l2 norm 

    if algo == 'truncatedsvd':
        svd = None
        scaler = None
        selector = None
        n_features = None

        # Fit SVD, scaler, selector globally
        # PCA needs dense data, but TruncatedSVD can handle sparse data
        svd = TruncatedSVD(n_components=svd_components, random_state=42)
        X_reduced = svd.fit_transform(X_all)
        print(f"[INFO] SVD explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reduced)
        selector = VarianceThreshold(threshold=0.0)
        X_high_var = selector.fit_transform(X_scaled)
        feature_autoencoded = X_high_var
        # Now X_high_var is ready for clustering
        #X_all = X_high_var
        # model dictionary 
        model_dict = {'svd': svd, 'scaler': scaler, 'selector': selector}

    
    elif algo == 'dnn':
        # why dnn here - 
        # AEs can learn non-linear representations, revealing pat-
        # terns that cannot be captured by linear methods, like Prin-
        # cipal Component Analysis (PCA) [60].
        """Reduce dimensionality using a DNN autoencoder and return embeddings."""
        # Convert to dense if sparse
        if sparse.issparse(X_all):
            X_all = X_all.toarray()
        input_dim = X_all.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(256, activation='relu')(input_layer)
        encoded = Dense(128, activation='relu')(encoded)
        encoded = Dropout(0.1)(encoded) # for regularization
        encoded = Dense(64, activation='relu')(encoded)
        # Latent space - this is the compressed representation
        latent = Dense(encoding_dim, activation='relu')(encoded) 
        decoded = Dense(64, activation='relu')(latent)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(256, activation='relu')(decoded)
        output_layer = Dense(input_dim, activation='sigmoid')(decoded)
        autoencoder = Model(input_layer, output_layer)
        encoder = Model(input_layer, latent)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(X_all, X_all, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.1, verbose=2)
        X_embedded = encoder.predict(X_all)
        feature_autoencoded = X_embedded
        print(f"[INFO] DNN autoencoder reduced feature shape: {feature_autoencoded.shape}")
        # model dictionary for saving the pkl file
        model_dict = {'autoencoder': autoencoder, 'encoder': encoder}
        # need to save the model file as dimred_model_path
        

    else:
        raise ValueError(f"Unsupported algorithm for dimension reduction: {algo}")
    
    # save the model dictionary to a common PKL file 
    dimred_model_path = DIMRED_MODEL_PATH
    joblib.dump(model_dict, dimred_model_path)
    
    return feature_autoencoded

def Clustering_Main():
    """Main function to handle configuration and execute the clustering"""
    try:
        # Load configuration
        config = load_config()
        svd_components = int(config.get('clustering_svd_components'))
        n_clusters = int(config.get('n_clusters'))
        clustering_batch_size = int(config.get('clustering_batch_size'))
        algo = config.get('clustering_algo', 'kmeans').lower()  # 'kmeans', 'dbscan', 'agg', 'gmm', 'all' are the options , default is "kmeans"

        # Load metadata to find all batch files and list is created to store the batch files path
        meta_path = os.path.join(FINAL_INPUT_DATASET, "matrix_metadata.json")
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        # this list below 
        batch_files = [os.path.join(FINAL_INPUT_DATASET, os.path.basename(f)) for f in metadata.get("batch_files", [])]
        print(f"[INFO] Found {len(batch_files)} batch files for clustering.")

        
        

        # --- First pass: Find union of nonzero columns across all batches ---
        #Scans all batches to find which columns (features) have any nonzero value.
        #Computes the union of all nonzero columns across batches.
        #Stores the indices of these columns and prints the total count.
        all_nonzero_cols = None
        user_ids = []  # List to store user IDs from all batches
        
        for batch_file in batch_files:

            # extract the user id here itslef 
            user_id_file = batch_file.replace('.npz', '_user_ids.npy')
            batch_user_ids = np.load(user_id_file)
            user_ids.extend(batch_user_ids.tolist())
            
            # Load the sparse matrix from the batch file
            X_sparse = sparse.load_npz(batch_file)
            nonzero_cols = X_sparse.getnnz(axis=0) > 0
            if all_nonzero_cols is None:
                all_nonzero_cols = nonzero_cols
            else:
                all_nonzero_cols = all_nonzero_cols | nonzero_cols  # union
        feature_indices = np.where(all_nonzero_cols)[0]
        n_features = len(feature_indices)
        print(f"[INFO] Total features with nonzero values across all batches: {n_features}")
        print(f"[INFO] Total user_ids loaded: {len(user_ids)}")


        X_high_var = None
        dimension_reduction_algo = config.get('dimension_reduction_algo')
        X_high_var = dimension_reduction(dimension_reduction_algo, batch_files, feature_indices, svd_components, encoding_dim=32, epochs=20, batch_size=250)

        # --- CLUSTERING ---
        algos_to_run = [algo] if algo != 'all' else ['kmeans', 'dbscan', 'agg', 'gmm']
        for algo_name in algos_to_run:
            print(f"[INFO] Running clustering algorithm: {algo_name}")
            algo_output_dir = os.path.join(CLUSTER_OUTPUT_DIR, algo_name)
            os.makedirs(algo_output_dir, exist_ok=True)
            cluster_labels_path = os.path.join(algo_output_dir, "user_cluster_labels.npy")
            if algo_name == 'kmeans':
                # print('dfdfgdsfgdf')
                kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=clustering_batch_size, n_init=10, random_state=42)
                #kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42) -> this one is not giving good results
                kmeans.fit(X_high_var)
                all_labels = kmeans.labels_
            elif algo_name == 'dbscan':
                dbscan = DBSCAN(eps=float(config.get('dbscan_eps', 0.5)), min_samples=int(config.get('dbscan_min_samples', 5)), n_jobs=-1)
                all_labels = dbscan.fit_predict(X_high_var)
            elif algo_name == 'agg':
                agg = AgglomerativeClustering(n_clusters=n_clusters)
                all_labels = agg.fit_predict(X_high_var)
            elif algo_name == 'gmm':
                gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
                all_labels = gmm.fit_predict(X_high_var)
            else:
                raise ValueError(f"Unknown clustering algorithm: {algo_name}")
            # Saving the cluster labels as .npy file
            print(f"[INFO] Saving cluster labels to {cluster_labels_path}...")
            np.save(cluster_labels_path, all_labels)
            # Save user-to-cluster mapping as JSON and TXT
            user_cluster_json = cluster_labels_path.replace('.npy', '_user_to_cluster.json')
            user_cluster_txt = cluster_labels_path.replace('.npy', '_user_to_cluster.txt')
            # user_to_cluster = {str(i): int(label) for i, label in enumerate(all_labels)} # this is wrong, need to have actual user_id 
            # Assuming you have a list of user_ids in the same order as X_high_var
            
            user_to_cluster = {str(user_id): int(label) for user_id, label in zip(user_ids, all_labels)}
            with open(user_cluster_json, 'w') as f:
                json.dump(user_to_cluster, f, indent=2)
            with open(user_cluster_txt, 'w') as f:
                for user, label in user_to_cluster.items():
                    f.write(f"{user}\t{label}\n")
            print(f"[INFO] User-to-cluster mapping saved to {user_cluster_json} and {user_cluster_txt}")
        print("[INFO] Clustering complete!")
        return 0
    except Exception as e:
        print(f"[ERROR] Clustering failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(Clustering_Main())
    gc.collect()  # Cleanup memory after execution
