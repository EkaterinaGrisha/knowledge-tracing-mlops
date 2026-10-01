"""Deep Knowledge Tracing (Piech-style LSTM) — sequence model.

Adapted from the MindFlow adaptive-ml service. Each interaction (skill, correct)
is encoded as a single integer 2*skill + correct, embedded, run through an LSTM,
and projected to a per-skill correctness logit. At step t the model predicts the
correctness of step t+1; loss is taken only on the skill actually seen next
(one-step-ahead, identical protocol to the BKT baseline).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..utils import get_logger

LOG = get_logger()


class DKTModel(nn.Module):
    def __init__(
        self, n_concepts: int, embed_dim: int = 64, hidden_dim: int = 64, dropout: float = 0.2
    ):
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


@dataclass
class StudentTrace:
    concept_idx: np.ndarray
    correct: np.ndarray
    length: int


def traces_from_sequences(sequences: list[dict]) -> list[StudentTrace]:
    """Build DKT traces directly from ordered ASSISTments sequences (no synthetic interleave)."""
    traces: list[StudentTrace] = []
    for s in sequences:
        skills = np.asarray(s["skills"], dtype=np.int64)
        correct = np.asarray(s["correct"], dtype=np.int64)
        if len(skills) >= 2:  # need at least one transition to supervise
            traces.append(StudentTrace(skills, correct, len(skills)))
    return traces


def make_input_indices(trace: StudentTrace) -> np.ndarray:
    return 2 * trace.concept_idx + trace.correct


def collate_traces(traces: list[StudentTrace], device: torch.device):
    max_t = max(tr.length for tr in traces)
    B = len(traces)
    x = np.zeros((B, max_t), dtype=np.int64)
    lengths = np.zeros((B,), dtype=np.int64)
    next_c = np.zeros((B, max_t), dtype=np.int64)
    next_y = np.zeros((B, max_t), dtype=np.float32)
    for b, tr in enumerate(traces):
        L = tr.length
        x[b, :L] = make_input_indices(tr)
        lengths[b] = L
        if L > 1:
            next_c[b, : L - 1] = tr.concept_idx[1:L]
            next_y[b, : L - 1] = tr.correct[1:L].astype(np.float32)
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(lengths).to(device),
        torch.from_numpy(next_c).to(device),
        torch.from_numpy(next_y).to(device),
    )


def _supervision_mask(lengths: torch.Tensor, max_t: int) -> torch.Tensor:
    idx = torch.arange(max_t, device=lengths.device).unsqueeze(0)
    return (idx < (lengths - 1).unsqueeze(1)).float()


def train_dkt(
    train_traces: list[StudentTrace],
    n_concepts: int,
    *,
    device: torch.device,
    embed_dim: int = 64,
    hidden_dim: int = 64,
    dropout: float = 0.2,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 5e-3,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[DKTModel, list[float]]:
    torch.manual_seed(seed)
    # Also seed NumPy's global RNG for third-party code that relies on it.
    np.random.seed(seed)  # noqa: NPY002
    model = DKTModel(n_concepts, embed_dim, hidden_dim, dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_traces))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(order), batch_size):
            batch = [train_traces[j] for j in order[i : i + batch_size]]
            x, lengths, next_c, next_y = collate_traces(batch, device)
            logits = model(x, lengths)
            logit_at_target = logits.gather(2, next_c.unsqueeze(-1)).squeeze(-1)
            raw = loss_fn(logit_at_target, next_y)
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
def evaluate_dkt(
    model: DKTModel, traces: list[StudentTrace], device: torch.device, batch_size: int = 64
):
    from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    for i in range(0, len(traces), batch_size):
        batch = traces[i : i + batch_size]
        x, lengths, next_c, next_y = collate_traces(batch, device)
        logits = model(x, lengths)
        probs = torch.sigmoid(logits.gather(2, next_c.unsqueeze(-1)).squeeze(-1))
        mask = _supervision_mask(lengths, x.size(1)).bool().cpu().numpy()
        y_true.append(next_y.cpu().numpy()[mask])
        y_pred.append(probs.cpu().numpy()[mask])
    yt = np.concatenate(y_true)
    yp = np.concatenate(y_pred)
    auc = float(roc_auc_score(yt, yp))
    acc = float(accuracy_score(yt, (yp > 0.5).astype(int)))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    return auc, acc, rmse, yt, yp


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
