# syntax=docker/dockerfile:1
# ----------------------------------------------------------------------------
# Knowledge Tracing ML pipeline — reproducible container image.
# Two stages: the builder installs the dependencies pinned in poetry.lock into
# an in-project virtualenv; the runtime image receives only that .venv and the
# code (no Poetry, no build caches). The dependency layer is cached until
# pyproject.toml / poetry.lock change, so code edits rebuild in seconds.
# ----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.5.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_CACHE_DIR=/tmp/poetry-cache

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app
COPY pyproject.toml poetry.lock poetry.toml README.md ./
# Runtime dependencies only (no dev tools). On Linux the lock file resolves
# torch to the CPU-only build, so there is no CUDA payload.
RUN poetry install --only main --no-root && rm -rf "${POETRY_CACHE_DIR}"

# The knowledge_tracing package itself, in a separate layer so code edits do
# not reinstall dependencies. It is an editable install pointing at /app/src,
# which the runtime stage provides at the same path.
COPY src ./src
RUN poetry install --only-root


FROM python:3.11-slim AS runtime

# - PYTHONDONTWRITEBYTECODE: no .pyc clutter
# - PYTHONUNBUFFERED: stream logs straight to docker logs
# - VIRTUAL_ENV / PATH: use the virtualenv built in the previous stage
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MLFLOW_ALLOW_FILE_STORE=true \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:${PATH}"

# System lib required by lightgbm (libgomp); apt lists cleaned up after.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user (security: limits blast radius if the container is
# compromised). The user owns /app and the code so the pipeline can write
# reports/ and mlruns/; the virtualenv stays root-owned and read-only.
RUN useradd --create-home --uid 1000 mluser \
    && mkdir /app && chown mluser:mluser /app
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
# Application code + committed sample dataset.
COPY --chown=mluser:mluser . .
USER mluser

# Default: run the pipeline on the committed sample (fully offline, reproducible).
# Override CMD to run on the full dataset:  docker run kt-pipeline ... --data-source full
ENTRYPOINT ["python", "-m", "knowledge_tracing.pipeline"]
CMD ["--data-source", "sample"]
