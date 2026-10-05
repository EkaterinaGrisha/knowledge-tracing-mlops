"""Deep Knowledge Tracing (Piech-style LSTM) — sequence model.

Adapted from the MindFlow adaptive-ml service. Each interaction (skill, correct)
is encoded as a single integer 2*skill + correct, embedded, run through an LSTM,
and projected to a per-skill correctness logit. At step t the model predicts the
correctness of step t+1; loss is taken only on the skill actually seen next
(one-step-ahead, identical protocol to the BKT baseline).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing_extensions import override

from ..config import DKTConfig
from ..etl.datasets import SplitData, StudentSequence, TrainingData
from .base import KnowledgeTracingModel, NotFittedError, Predictions

LOG = logging.getLogger(__name__)

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

# A sequence needs at least one transition (two attempts) to be supervised.
MIN_SUPERVISED_LENGTH = 2
# Gradients are clipped to this global norm to keep LSTM training stable.
GRAD_CLIP_NORM = 5.0


@dataclass(frozen=True)
class DKTHyperparameters:
    """Architecture and optimiser settings of one DKT network."""

    embed_dim: int
    hidden_dim: int
    dropout: float
    lr: float
    batch_size: int


class DKTNetwork(nn.Module):
    """Embedding -> LSTM -> per-skill logit of answering the next exercise correctly."""

    def __init__(self, n_concepts: int, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.n_concepts = n_concepts
        self.embed = nn.Embedding(2 * n_concepts, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, n_concepts)

    @override
    def forward(self, x_idx: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Per-skill logits after every step of the padded input sequences."""
        emb = self.embed(x_idx)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=x_idx.size(1))
        return self.head(self.drop(out))


def _supervised(sequences: list[StudentSequence]) -> list[StudentSequence]:
    """Sequences with at least one transition to supervise."""
    return [s for s in sequences if len(s) >= MIN_SUPERVISED_LENGTH]


def _input_indices(seq: StudentSequence) -> np.ndarray:
    return 2 * seq.skills + seq.correct


def _collate(batch: list[StudentSequence], device: torch.device) -> Batch:
    max_t = max(len(s) for s in batch)
    size = len(batch)
    x = np.zeros((size, max_t), dtype=np.int64)
    lengths = np.zeros((size,), dtype=np.int64)
    next_skill = np.zeros((size, max_t), dtype=np.int64)
    next_correct = np.zeros((size, max_t), dtype=np.float32)
    for b, seq in enumerate(batch):
        length = len(seq)
        x[b, :length] = _input_indices(seq)
        lengths[b] = length
        if length > 1:
            next_skill[b, : length - 1] = seq.skills[1:length]
            next_correct[b, : length - 1] = seq.correct[1:length].astype(np.float32)
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(lengths).to(device),
        torch.from_numpy(next_skill).to(device),
        torch.from_numpy(next_correct).to(device),
    )


def _supervision_mask(lengths: torch.Tensor, max_t: int) -> torch.Tensor:
    idx = torch.arange(max_t, device=lengths.device).unsqueeze(0)
    return (idx < (lengths - 1).unsqueeze(1)).float()


def train_dkt(
    sequences: list[StudentSequence],
    n_concepts: int,
    hp: DKTHyperparameters,
    *,
    epochs: int,
    device: torch.device,
    seed: int,
    verbose: bool = False,
) -> tuple[DKTNetwork, list[float]]:
    """Train a DKT network with masked one-step-ahead binary cross-entropy.

    Args:
        sequences: Training sequences; those without a transition are skipped.
        n_concepts: Number of skills (size of the output layer).
        hp: Architecture and optimiser settings.
        epochs: Number of passes over the training sequences.
        device: Torch device to train on.
        seed: Seed for weight initialisation, dropout and batch order.
        verbose: Log the mean loss every five epochs.

    Returns:
        The trained network and the mean training loss of every epoch.
    """
    train = _supervised(sequences)
    torch.manual_seed(seed)
    # Also seed NumPy's global RNG for third-party code that relies on it.
    np.random.seed(seed)  # noqa: NPY002
    model = DKTNetwork(n_concepts, hp.embed_dim, hp.hidden_dim, hp.dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=hp.lr)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(order), hp.batch_size):
            batch = [train[j] for j in order[i : i + hp.batch_size]]
            x, lengths, next_skill, next_correct = _collate(batch, device)
            logits = model(x, lengths)
            logit_at_target = logits.gather(2, next_skill.unsqueeze(-1)).squeeze(-1)
            raw = loss_fn(logit_at_target, next_correct)
            mask = _supervision_mask(lengths, x.size(1))
            loss = (raw * mask).sum() / mask.sum().clamp_min(1.0)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        if verbose and (epoch + 1) % 5 == 0:
            LOG.info("[DKT] epoch %2d/%d loss=%.4f", epoch + 1, epochs, avg)
    return model, losses


@torch.no_grad()
def predict_dkt(
    model: DKTNetwork,
    sequences: list[StudentSequence],
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """One-step-ahead predictions: (observed correctness, P(correct)) per transition."""
    supervised = _supervised(sequences)
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    for i in range(0, len(supervised), batch_size):
        batch = supervised[i : i + batch_size]
        x, lengths, next_skill, next_correct = _collate(batch, device)
        logits = model(x, lengths)
        probs = torch.sigmoid(logits.gather(2, next_skill.unsqueeze(-1)).squeeze(-1))
        mask = _supervision_mask(lengths, x.size(1)).bool().cpu().numpy()
        y_true.append(next_correct.cpu().numpy()[mask])
        y_pred.append(probs.cpu().numpy()[mask])
    return np.concatenate(y_true), np.concatenate(y_pred)


def pick_device() -> torch.device:
    """CUDA when available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DKTModel(KnowledgeTracingModel):
    """DKT with the hand-picked architecture from the config."""

    name = "DKT"

    def __init__(self, cfg: DKTConfig, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self.device = pick_device()
        self.hyperparameters_: DKTHyperparameters | None = None
        self.network_: DKTNetwork | None = None
        self.losses_: list[float] = []
        self.n_skills_ = 0

    @override
    def fit(self, data: TrainingData) -> None:
        cfg = self.cfg
        self._fit_with(
            data,
            DKTHyperparameters(cfg.embed_dim, cfg.hidden_dim, cfg.dropout, cfg.lr, cfg.batch_size),
        )

    def _fit_with(self, data: TrainingData, hp: DKTHyperparameters) -> None:
        self.hyperparameters_ = hp
        self.n_skills_ = data.n_skills
        self.network_, self.losses_ = train_dkt(
            data.train.sequences,
            data.n_skills,
            hp,
            epochs=self.cfg.epochs,
            device=self.device,
            seed=self.seed,
        )

    @override
    def predict(self, split: SplitData) -> Predictions:
        if self.network_ is None:
            raise NotFittedError(f"{self.name} is not fitted")
        return predict_dkt(self.network_, split.sequences, self.device)

    @override
    def training_curve(self) -> list[float]:
        return list(self.losses_)
