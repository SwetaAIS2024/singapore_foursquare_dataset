import gc
from .util import main_matrix_build
from .util import load_config

if __name__ == "__main__":
    print("[INFO] Starting matrix building process...")
    
    config = load_config()   
    eps_km= float(config.get('eps_km'))
    min_samples= int(config.get('min_samples'))
    n_time_bins= int(config.get('n_time_bins'))
    n_users= int(config.get('n_users'))
    n_spatial_clusters= int(config.get('n_spatial_clusters'))
    n_categories= int(config.get('n_categories'))
    n_quantization_bins= int(config.get('n_quantization_bins'))
    batch_size= int(config.get('batch_size'))
    
    result = main_matrix_build(eps_km, min_samples, n_time_bins, n_users, n_spatial_clusters, n_categories, n_quantization_bins, batch_size)
    if result != 0:
        print("[ERROR] Matrix building failed")
    else:
        print("[INFO] Matrix building completed successfully")
    gc.collect()