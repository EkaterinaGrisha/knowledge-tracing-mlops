"""Bayesian Knowledge Tracing (single-skill, EM-fit) — baseline model.

Core forward/backward + EM are adapted from the MindFlow adaptive-ml service.
Latent state L_t in {0=not known, 1=known}; no forgetting; emission governed by
slip/guess. Here we fit one parameter set per skill on the train split and
evaluate one-step-ahead correctness prediction on held-out students.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import override

from ..config import BKTConfig
from ..etl.datasets import SplitData, StudentSequence, TrainingData
from .base import KnowledgeTracingModel, NotFittedError, Predictions


@dataclass
class BktParams:
    """BKT parameters of one skill.

    Attributes:
        p_init: Probability the skill is known before the first attempt.
        p_learn: Probability of learning the skill after an attempt.
        p_slip: Probability of a wrong answer although the skill is known.
        p_guess: Probability of a right answer although the skill is unknown.
    """

    p_init: float
    p_learn: float
    p_slip: float
    p_guess: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Parameters in the order (p_init, p_learn, p_slip, p_guess)."""
        return (self.p_init, self.p_learn, self.p_slip, self.p_guess)


DEFAULT_PARAMS = BktParams(0.25, 0.15, 0.10, 0.20)

# EM keeps every parameter inside a plausible range (no degenerate solutions
# such as "always guessing" or "never learning").
P_INIT_BOUNDS = (0.01, 0.99)
P_LEARN_BOUNDS = (0.01, 0.50)
P_SLIP_BOUNDS = (0.01, 0.30)
P_GUESS_BOUNDS = (0.01, 0.40)
# EM stops once the log-likelihood improves by less than this.
EM_TOLERANCE = 1e-4
# Keeps normalisations away from division by zero.
_EPS = 1e-12


def _emission(y: int, p: BktParams) -> tuple[float, float]:
    if y == 1:
        return p.p_guess, 1.0 - p.p_slip
    return 1.0 - p.p_guess, p.p_slip


def forward_alpha(seq: list[int], p: BktParams) -> tuple[np.ndarray, float]:
    """Scaled forward pass of the BKT hidden Markov model.

    Args:
        seq: Correctness (0/1) of consecutive attempts at one skill.
        p: Parameters of the skill.

    Returns:
        Normalised forward probabilities of shape (len(seq), 2) — columns are
        "unknown" and "known" — and the log-likelihood of the sequence.
    """
    T = len(seq)
    alpha = np.zeros((T, 2))
    log_lik = 0.0
    e0, e1 = _emission(seq[0], p)
    alpha[0, 0] = (1.0 - p.p_init) * e0
    alpha[0, 1] = p.p_init * e1
    c = alpha[0].sum()
    if c > 0:
        alpha[0] /= c
        log_lik += np.log(c)
    for t in range(1, T):
        e0, e1 = _emission(seq[t], p)
        prior_unknown = alpha[t - 1, 0] * (1.0 - p.p_learn)
        prior_known = alpha[t - 1, 0] * p.p_learn + alpha[t - 1, 1] * 1.0
        alpha[t, 0] = prior_unknown * e0
        alpha[t, 1] = prior_known * e1
        c = alpha[t].sum()
        if c > 0:
            alpha[t] /= c
            log_lik += np.log(c)
    return alpha, log_lik


def _backward_beta(seq: list[int], p: BktParams) -> np.ndarray:
    T = len(seq)
    beta = np.ones((T, 2))
    for t in range(T - 2, -1, -1):
        e0, e1 = _emission(seq[t + 1], p)
        beta[t, 1] = beta[t + 1, 1] * 1.0 * e1
        beta[t, 0] = beta[t + 1, 0] * (1.0 - p.p_learn) * e0 + beta[t + 1, 1] * p.p_learn * e1
        s = beta[t].sum()
        if s > 0:
            beta[t] /= s
    return beta


def fit_em(seqs: list[list[int]], n_iter: int = 30, init: BktParams = DEFAULT_PARAMS) -> BktParams:
    """Fit the parameters of one skill with Expectation-Maximisation.

    Args:
        seqs: Correctness sequences of all students for this skill.
        n_iter: Maximum number of EM iterations.
        init: Starting parameters.

    Returns:
        Parameters clipped to the plausible ranges defined in this module.
    """
    p = BktParams(*init.as_tuple())
    last_ll = -np.inf
    for _ in range(n_iter):
        num_init = den_init = 0.0
        num_learn = den_learn = 0.0
        num_slip = den_slip = 0.0
        num_guess = den_guess = 0.0
        total_ll = 0.0
        for seq in seqs:
            T = len(seq)
            if T == 0:
                continue
            alpha, ll = forward_alpha(seq, p)
            beta = _backward_beta(seq, p)
            total_ll += ll
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + _EPS
            num_init += gamma[0, 1]
            den_init += 1.0
            for t in range(T - 1):
                e0n, e1n = _emission(seq[t + 1], p)
                xi_00 = alpha[t, 0] * (1.0 - p.p_learn) * e0n * beta[t + 1, 0]
                xi_01 = alpha[t, 0] * p.p_learn * e1n * beta[t + 1, 1]
                xi_11 = alpha[t, 1] * 1.0 * e1n * beta[t + 1, 1]
                norm = xi_00 + xi_01 + xi_11 + _EPS
                xi_00 /= norm
                xi_01 /= norm
                num_learn += xi_01
                den_learn += xi_00 + xi_01
            for t in range(T):
                den_slip += gamma[t, 1]
                if seq[t] == 0:
                    num_slip += gamma[t, 1]
                den_guess += gamma[t, 0]
                if seq[t] == 1:
                    num_guess += gamma[t, 0]
        if den_init > 0:
            p.p_init = float(np.clip(num_init / den_init, *P_INIT_BOUNDS))
        if den_learn > 0:
            p.p_learn = float(np.clip(num_learn / den_learn, *P_LEARN_BOUNDS))
        if den_slip > 0:
            p.p_slip = float(np.clip(num_slip / den_slip, *P_SLIP_BOUNDS))
        if den_guess > 0:
            p.p_guess = float(np.clip(num_guess / den_guess, *P_GUESS_BOUNDS))
        if abs(total_ll - last_ll) < EM_TOLERANCE:
            break
        last_ll = total_ll
    return p


def predict_next_correct(seq: list[int], p: BktParams) -> list[float]:
    """P(correct) of every attempt given only the attempts before it.

    Args:
        seq: Correctness (0/1) of consecutive attempts at one skill.
        p: Parameters of the skill.

    Returns:
        One prediction per attempt; the first one uses ``p_init`` only.
    """
    T = len(seq)
    if T == 0:
        return []
    preds: list[float] = []
    p_known_init = p.p_init
    preds.append(p_known_init * (1.0 - p.p_slip) + (1.0 - p_known_init) * p.p_guess)
    p_unknown = 1.0 - p.p_init
    p_known = p.p_init
    for t in range(T - 1):
        e0, e1 = _emission(seq[t], p)
        post0 = p_unknown * e0
        post1 = p_known * e1
        s = post0 + post1 + _EPS
        post0 /= s
        post1 /= s
        p_unknown = post0 * (1.0 - p.p_learn)
        p_known = post0 * p.p_learn + post1 * 1.0
        preds.append(p_known * (1.0 - p.p_slip) + p_unknown * p.p_guess)
    return preds


# ── ASSISTments adapter: per-skill fit + evaluation ────────────────────────


def _per_skill_subsequences(
    sequences: list[StudentSequence], n_skills: int
) -> list[list[list[int]]]:
    """Group each student's interactions by skill, preserving within-skill order."""
    by_skill: list[list[list[int]]] = [[] for _ in range(n_skills)]
    for s in sequences:
        buckets: dict[int, list[int]] = {}
        for k, y in zip(s.skills, s.correct, strict=True):
            buckets.setdefault(int(k), []).append(int(y))
        for k, seq in buckets.items():
            by_skill[k].append(seq)
    return by_skill


def fit_bkt_per_skill(
    train_sequences: list[StudentSequence], n_skills: int, em_iters: int
) -> dict[int, BktParams]:
    """Fit one parameter set per skill; skills without data keep the defaults."""
    by_skill = _per_skill_subsequences(train_sequences, n_skills)
    params: dict[int, BktParams] = {}
    for k in range(n_skills):
        seqs = [s for s in by_skill[k] if len(s) > 0]
        params[k] = fit_em(seqs, n_iter=em_iters) if seqs else BktParams(*DEFAULT_PARAMS.as_tuple())
    return params


def predict_bkt(
    params_by_skill: dict[int, BktParams], sequences: list[StudentSequence], n_skills: int
) -> tuple[np.ndarray, np.ndarray]:
    """One-step-ahead predictions: (observed correctness, P(correct)) per interaction."""
    by_skill = _per_skill_subsequences(sequences, n_skills)
    y_true: list[int] = []
    y_pred: list[float] = []
    for k in range(n_skills):
        p = params_by_skill.get(k, BktParams(*DEFAULT_PARAMS.as_tuple()))
        for seq in by_skill[k]:
            preds = predict_next_correct(seq, p)
            y_true.extend(seq)
            y_pred.extend(preds)
    return np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=float)


class BKTModel(KnowledgeTracingModel):
    """Per-skill BKT fitted with EM — the interpretable baseline."""

    name = "BKT"

    def __init__(self, cfg: BKTConfig) -> None:
        self.cfg = cfg
        self.params_: dict[int, BktParams] | None = None
        self.n_skills_ = 0

    @override
    def fit(self, data: TrainingData) -> None:
        self.n_skills_ = data.n_skills
        self.params_ = fit_bkt_per_skill(data.train.sequences, data.n_skills, self.cfg.em_iters)

    @override
    def predict(self, split: SplitData) -> Predictions:
        if self.params_ is None:
            raise NotFittedError(f"{self.name} is not fitted")
        return predict_bkt(self.params_, split.sequences, self.n_skills_)
