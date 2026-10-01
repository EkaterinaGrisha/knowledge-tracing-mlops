#!/usr/bin/env bash
# Run the ML pipeline on the committed sample with quick budgets.
# Sets OpenMP env vars to avoid duplicate-libomp segfaults on macOS arm64.
set -euo pipefail
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
# MLflow 3.x refuses the ./mlruns file store unless explicitly allowed.
export MLFLOW_ALLOW_FILE_STORE=true
poetry run python -m src.pipeline --data-source sample --quick
