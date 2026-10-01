"""Knowledge tracing on ASSISTments 2009: ETL, BKT / DKT / AutoML models, monitoring."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("knowledge-tracing")
except PackageNotFoundError:  # code bundled with an MLflow model, not installed
    __version__ = "0+unknown"
