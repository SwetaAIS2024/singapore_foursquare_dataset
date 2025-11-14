.PHONY: install clean generate validate preprocess postprocess test help

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make clean        - Remove generated files"
	@echo "  make preprocess   - Preprocess raw FSQ data"
	@echo "  make generate     - Generate synthetic datasets"
	@echo "  make postprocess  - Apply 5-core filtering to synthetic data"
	@echo "  make validate     - Validate synthetic outputs"
	@echo "  make test         - Run tests"

install:
	pip install -r requirements.txt
	pip install -e .

clean:
	rm -rf data/synthetic/*
	rm -rf data/processed/*
	rm -rf logs/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

preprocess:
	python scripts/preprocess_fsq_data.py

generate:
	python scripts/generate_synthetic_data.py

postprocess:
	python scripts/postprocess_synthetic_data.py

validate:
	python scripts/validate_output.py

test:
	pytest tests/ -v
