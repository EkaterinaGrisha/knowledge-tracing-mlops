"""Exceptions of the package (kept free of heavy imports for the CLI)."""


class DataQualityError(RuntimeError):
    """The data-quality gate failed; training must not proceed."""
