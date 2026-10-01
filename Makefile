.PHONY: install lint format typecheck test etl train train-quick train-full present mlflow docker docker-run clean

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

test:
	$(RUN) pytest -q

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

present:
	poetry install --with presentation
	$(RUN) python scripts/build_deck.py

mlflow:
	MLFLOW_ALLOW_FILE_STORE=true $(RUN) mlflow ui --backend-store-uri file:./mlruns --port 5000

docker:
	docker build -t kt-pipeline:latest .

docker-run:
	docker run --rm -v $$(pwd)/reports:/app/reports -v $$(pwd)/mlruns:/app/mlruns kt-pipeline:latest

clean:
	rm -rf data/raw/* data/processed/* mlruns/* reports/figures/* reports/metrics.json
