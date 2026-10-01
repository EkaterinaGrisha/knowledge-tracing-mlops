"""Deep Knowledge Tracing (Piech-style LSTM) — sequence model.

Adapted from the MindFlow adaptive-ml service. Each interaction (skill, correct)
is encoded as a single integer 2*skill + correct, embedded, run through an LSTM,
and projected to a per-skill correctness logit. At step t the model predicts the
correctness of step t+1; loss is taken only on the skill actually seen next
(one-step-ahead, identical protocol to the BKT baseline).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..etl.datasets import StudentSequence
from ..utils import get_logger

LOG = get_logger()

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class DKTNetwork(nn.Module):
    """Embedding -> LSTM -> per-skill logit of answering the next exercise correctly."""

    def __init__(self, n_concepts: int, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.n_concepts = n_concepts
        self.embed = nn.Embedding(2 * n_concepts, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, n_concepts)

    def forward(self, x_idx: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x_idx)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=x_idx.size(1))
        return self.head(self.drop(out))


def _supervised(sequences: list[StudentSequence]) -> list[StudentSequence]:
    """Sequences with at least one transition to supervise (length >= 2)."""
    return [s for s in sequences if len(s) >= 2]


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
    *,
    device: torch.device,
    embed_dim: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    verbose: bool = False,
) -> tuple[DKTNetwork, list[float]]:
    train = _supervised(sequences)
    torch.manual_seed(seed)
    # Also seed NumPy's global RNG for third-party code that relies on it.
    np.random.seed(seed)  # noqa: NPY002
    model = DKTNetwork(n_concepts, embed_dim, hidden_dim, dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(order), batch_size):
            batch = [train[j] for j in order[i : i + batch_size]]
            x, lengths, next_skill, next_correct = _collate(batch, device)
            logits = model(x, lengths)
            logit_at_target = logits.gather(2, next_skill.unsqueeze(-1)).squeeze(-1)
            raw = loss_fn(logit_at_target, next_correct)
            mask = _supervision_mask(lengths, x.size(1))
            loss = (raw * mask).sum() / mask.sum().clamp_min(1.0)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
