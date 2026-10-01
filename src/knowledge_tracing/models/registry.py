"""The set of models the pipeline trains."""

from __future__ import annotations

from ..config import ModelsConfig
from .automl_flaml import AutoMLModel
from .base import KnowledgeTracingModel
from .bkt import BKTModel
from .dkt import DKTModel
from .dkt_optuna import DKTOptunaModel


def build_models(cfg: ModelsConfig, seed: int) -> list[KnowledgeTracingModel]:
    """All models in training order.

    The order is part of the reproducibility contract: DKT training seeds
    NumPy's global RNG, which later models may draw from.
    """
    return [
        BKTModel(cfg.bkt),
        DKTModel(cfg.dkt, seed),
        DKTOptunaModel(cfg.dkt_optuna, cfg.dkt, seed),
        AutoMLModel(cfg.automl_flaml, seed),
    ]
