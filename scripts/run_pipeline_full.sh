#!/usr/bin/env bash
# Final/production run: full ASSISTments dataset, normal training budgets from config.yaml.
# OpenMP env vars prevent the macOS arm64 duplicate-libomp segfault.
# Expected wall-clock on a CPU MacBook: 20-45 minutes (mostly DKT+Optuna and FLAML).
set -euo pipefail
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
# MLflow 3.x refuses the ./mlruns file store unless explicitly allowed.
export MLFLOW_ALLOW_FILE_STORE=true
poetry run python -m src.pipeline --data-source full
