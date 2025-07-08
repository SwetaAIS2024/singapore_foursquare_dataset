from .c1_build_matrix import main

if __name__ == "__main__":
    print("[INFO] Starting matrix building process...")
    result = main(eps_km=0.3, min_samples=5)
    if result != 0:
        print("[ERROR] Matrix building failed")
    else:
        print("[INFO] Matrix building completed successfully")