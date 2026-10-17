import torch

# Single source of truth for the optimal-threshold search (used by Trainer.eval_epoch
# and scripts/eval.py). The floor goes well below 0.5 because losses that lower the
# probability scale (focal loss) have their optimal operating point below 0.30 — a
# higher floor would systematically understate them.
THRESHOLD_GRID: list[float] = [
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
]


def f1_score(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Macro-averaged F1 for multi-label classification."""
    tp = (preds & labels.bool()).sum(dim=0).float()
    fp = (preds & ~labels.bool()).sum(dim=0).float()
    fn = (~preds & labels.bool()).sum(dim=0).float()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.mean().item()


def precision(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Macro-averaged precision for multi-label classification."""
    tp = (preds & labels.bool()).sum(dim=0).float()
    fp = (preds & ~labels.bool()).sum(dim=0).float()
    return (tp / (tp + fp + 1e-8)).mean().item()


def recall(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Macro-averaged recall for multi-label classification."""
    tp = (preds & labels.bool()).sum(dim=0).float()
    fn = (~preds & labels.bool()).sum(dim=0).float()
    return (tp / (tp + fn + 1e-8)).mean().item()


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """Sample-averaged accuracy for multi-label classification."""
    correct = (preds == labels.bool()).float().mean(dim=1)
    return correct.mean().item()


def roc_auc(scores: torch.Tensor, positives: torch.Tensor) -> float | None:
    """ROC AUC of one column (Mann-Whitney U on average ranks, ties count ½).

    Returns None when the column has no positives or no negatives (undefined).
    """
    pos = positives.bool()
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(scores.double())
    return ((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def roc_auc_macro(probs: torch.Tensor, hard: torch.Tensor) -> float:
    """Mean ROC AUC over the class columns where it is defined — per label for
    multi-label, one-vs-rest for multi-class, the positive column for binary (the
    MedMNIST evaluator's definition). NaN if no column is defined."""
    aucs = [a for c in range(hard.shape[1])
            if (a := roc_auc(probs[:, c], hard[:, c])) is not None]
    return sum(aucs) / len(aucs) if aucs else float("nan")


def _average_ranks(x: torch.Tensor) -> torch.Tensor:
    """1-based ranks of ``x``; tied values share the mean of their ranks (vectorized)."""
    sorted_x, order = torch.sort(x)
    _, counts = torch.unique_consecutive(sorted_x, return_counts=True, dim=0)
    ends = torch.cumsum(counts, dim=0)                  # 1-based last rank of each tie group
    group_rank = (ends - counts + 1 + ends).double() / 2
    ranks = torch.empty_like(x, dtype=torch.float64)
    ranks[order] = torch.repeat_interleave(group_rank, counts)
    return ranks


def eta_str(epoch_times: list[float], epochs_done: int, epochs_total: int) -> str:
    """Human-readable ETA string given a list of past epoch durations."""
    if not epoch_times:
        return "?"
    remaining_s = (epochs_total - epochs_done) * (sum(epoch_times) / len(epoch_times))
    h, m = int(remaining_s // 3600), int((remaining_s % 3600) // 60)
    return f"{h}h {m:02d}m"
