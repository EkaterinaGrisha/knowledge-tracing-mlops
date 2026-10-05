.PHONY: install etl all all-full test lint format typecheck present mlflow docker docker-run clean

# Every command runs inside the project virtualenv (.venv/) managed by Poetry.
RUN ?= poetry run

# OpenMP guards prevent duplicate-libomp segfaults seen on macOS arm64
# when PyTorch and lightgbm/xgboost share one process. MLflow 3.x refuses the
# ./mlruns file store unless explicitly allowed (same flag as in the Dockerfile).
OMP_GUARDS = KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MLFLOW_ALLOW_FILE_STORE=true

# Create .venv/ inside the project, install the locked dependencies + dev tools
# and the git hooks (pre-commit, pre-push).
install:
	poetry install
	$(RUN) pre-commit install

# Run the full pipeline on the committed sample (fast, offline, used by CI).
all:
	$(OMP_GUARDS) $(RUN) python -m src.pipeline --config config/config.yaml --data-source sample

# Run the full pipeline on the full ASSISTments dataset (downloads on first run).
all-full:
	$(OMP_GUARDS) $(RUN) python -m src.pipeline --config config/config.yaml --data-source full

etl:
	$(RUN) python -m src.etl.run --config config/config.yaml --data-source sample

test:
	$(OMP_GUARDS) $(RUN) pytest -q

# All static checks from .pre-commit-config.yaml (ruff, mypy, file hygiene, ...).
# no-commit-to-branch is skipped so the target also works on main.
lint:
	SKIP=no-commit-to-branch $(RUN) pre-commit run --all-files

format:
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

typecheck:
	$(RUN) mypy

present:
	poetry install --with presentation
	$(RUN) python -m src.presentation.build_deck

mlflow:
	MLFLOW_ALLOW_FILE_STORE=true $(RUN) mlflow ui --backend-store-uri file:./mlruns --port 5000

docker:
	docker build -t kt-pipeline:latest .

docker-run:
	docker run --rm -v $$(pwd)/reports:/app/reports -v $$(pwd)/mlruns:/app/mlruns kt-pipeline:latest

clean:
	rm -rf data/raw/* data/processed/* mlruns/* reports/figures/* reports/metrics.json
