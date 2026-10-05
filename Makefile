.PHONY: install lint format typecheck test etl train train-quick train-full predict promote publish-report present mlflow docker docker-run clean

# Every command runs inside the project virtualenv (.venv/) managed by Poetry.
RUN ?= poetry run

# Create .venv/ inside the project, install the locked dependencies + dev tools
# and the git hooks (pre-commit, pre-push).
install:
	poetry install
	$(RUN) pre-commit install

# All static checks from .pre-commit-config.yaml (ruff, mypy, file hygiene, ...).
# no-commit-to-branch is skipped so the target also works on main.
lint:
	SKIP=no-commit-to-branch $(RUN) pre-commit run --all-files

format:
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

typecheck:
	$(RUN) mypy

# Full test suite with the coverage report (fails below the threshold in pyproject.toml).
test:
	$(RUN) pytest -q --cov

etl:
	$(RUN) kt etl --data-source sample

# Full pipeline on the committed sample (offline).
train:
	$(RUN) kt train --data-source sample

# Same with minimal budgets (config/quick.yaml) — a smoke run in about a minute.
train-quick:
	$(RUN) kt train --data-source sample --quick

# Full ASSISTments dataset (downloaded on first run; 20–45 min on a laptop CPU).
train-full:
	$(RUN) kt train --data-source full

# Next-attempt P(correct) per skill for the example students, with the local model.
predict:
	$(RUN) kt predict --model artifacts/model --input examples/history.csv

# Approval step: the registered "challenger" becomes the "champion".
promote:
	$(RUN) kt promote

# Publish the latest run (artifacts/) as the versioned results referenced from README.
publish-report:
	cp artifacts/metrics.json reports/metrics.json
	cp artifacts/figures/*.png reports/figures/

present:
	poetry install --with presentation
	$(RUN) python scripts/build_deck.py

mlflow:
	$(RUN) mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

docker:
	docker build -t kt-pipeline:latest .

docker-run:
	docker run --rm -v $$(pwd)/artifacts:/app/artifacts -v $$(pwd)/mlruns:/app/mlruns kt-pipeline:latest

clean:
	rm -rf data/raw/* data/processed/* mlruns/* artifacts/*
