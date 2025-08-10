.PHONY: install format lint fix test synthetic-data e2e
install:
	pip install -r requirements.txt
	pip install -e .
	pre-commit install || true
format:
	black . && isort --profile=black --line-length=100 . && ruff format .
lint:
	ruff check . && black --check . && isort --profile=black --line-length=100 --check-only .
fix:
	ruff check --fix . || true; black . || true; isort --profile=black --line-length=100 . || true
test:
	pytest -q
synthetic-data:
	python scripts/generate_synthetic_data.py --bids_root data --subs 01 02 --sessions 01 02
e2e:
	python scripts/run_end_to_end.py
