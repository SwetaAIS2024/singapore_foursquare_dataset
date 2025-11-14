from setuptools import setup, find_packages

setup(
    name="singapore-fsq-synthetic",
    version="1.0.0",
    description="Singapore Foursquare Synthetic Dataset Generation",
    author="Your Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "geopandas>=0.12.0",
        "shapely>=2.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.64.0",
    ],
    entry_points={
        "console_scripts": [
            "fsq-preprocess=scripts.preprocess_fsq_data:main",
            "fsq-generate=scripts.generate_synthetic_data:main",
            "fsq-validate=scripts.validate_output:main",
        ],
    },
)
