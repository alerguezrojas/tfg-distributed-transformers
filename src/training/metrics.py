import torch


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


def eta_str(epoch_times: list[float], epochs_done: int, epochs_total: int) -> str:
    """Human-readable ETA string given a list of past epoch durations."""
    if not epoch_times:
        return "?"
    remaining_s = (epochs_total - epochs_done) * (sum(epoch_times) / len(epoch_times))
    h, m = int(remaining_s // 3600), int((remaining_s % 3600) // 60)
    return f"{h}h {m:02d}m"
