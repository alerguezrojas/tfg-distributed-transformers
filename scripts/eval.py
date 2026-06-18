"""Evaluation script — measures model performance on the test split.

Loads a saved checkpoint and evaluates it on BigEarthNet-S2 test set,
reporting per-class and aggregate metrics with optimal threshold search.

Usage:
    # Evaluate best checkpoint with default config
    uv run python scripts/eval.py --checkpoint checkpoints/verode/checkpoint_epoch_007.pt

    # Override config (e.g. to use cluster dataset paths)
    uv run python scripts/eval.py \\
        --checkpoint checkpoints/verode/checkpoint_epoch_007.pt \\
        --config configs/train_cluster_v3.yaml

    # Save per-class results to CSV
    uv run python scripts/eval.py \\
        --checkpoint checkpoints/verode/checkpoint_epoch_007.pt \\
        --output logs/verode/eval_results.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import BigEarthNetDataset, get_transforms
from src.models.vit import build_model
from src.training import metrics as m

_THRESHOLD_GRID = m.THRESHOLD_GRID

CLASS_NAMES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the test set")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                        help="Config YAML (used for dataset paths, model name, batch size)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model name from config")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (default: from config)")
    parser.add_argument("--output", type=str, default=None,
                        help="CSV path to save per-class results")
    parser.add_argument("--split", choices=["test", "val"], default="test",
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Override metadata.parquet path (e.g. a demo subset)")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Stop after N batches (quick sanity check)")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """Run inference and return probabilities + labels."""
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    n_batches = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        all_probs.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels.cpu())
        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    return probs, labels, total_loss / n_batches


def threshold_search(probs, labels):
    """Find threshold that maximises macro F1."""
    best_t, best_f1 = 0.5, 0.0
    for t in _THRESHOLD_GRID:
        preds = (probs > t).long()
        f1 = m.f1_score(preds, labels)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


def per_class_metrics(probs, labels, threshold):
    """Compute F1, precision, recall per class."""
    preds = (probs > threshold).long()
    n_classes = labels.shape[1]
    rows = []
    for i in range(n_classes):
        p = preds[:, i]
        l = labels[:, i].long()
        tp = (p & l).sum().item()
        fp = (p & ~l.bool()).sum().item()
        fn = (~p.bool() & l.bool()).sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        rows.append({
            "class_idx": i,
            "class_name": CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"class_{i}",
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
        })
    return rows


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model or cfg["model"]["name"]
    batch_size = args.batch_size or cfg["training"].get("batch_size", 32)
    num_classes = cfg["model"].get("num_classes", 19)

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Modelo     : {model_name}")
    print(f"Split      : {args.split}")
    print(f"Dispositivo: {device}")

    # ── Build model ───────────────────────────────────────────────────────────
    # dropout is a no-op at eval time (model.eval() disables it) and does not
    # affect the state_dict shape, so build_model need not take it.
    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    resumed_epoch = ckpt.get("epoch", "?")
    print(f"Epoch      : {resumed_epoch}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    metadata_path = args.metadata or cfg["data"]["metadata"]
    ds = BigEarthNetDataset(
        cfg["data"]["root"], metadata_path, split=args.split,
        transform=get_transforms("val"),
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4), pin_memory=True,
    )
    print(f"Patches    : {len(ds)}")
    if len(ds) == 0:
        print(f"\nERROR: the '{args.split}' split is empty in {metadata_path}. "
              "This metadata may not contain that split (e.g. the demo subset has "
              "only train/val). Use --split val or a full-dataset metadata.")
        sys.exit(1)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    probs, labels, avg_loss = evaluate(model, loader, device, max_batches=args.max_batches)
    preds_05 = (probs > 0.5).long()

    f1_05     = m.f1_score(preds_05, labels)
    acc_05    = m.accuracy(preds_05, labels)
    prec_05   = m.precision(preds_05, labels)
    rec_05    = m.recall(preds_05, labels)

    best_t, best_f1 = threshold_search(probs, labels)
    preds_bt = (probs > best_t).long()

    print("\n── Aggregate metrics ─────────────────────────────────────────────")
    print(f"  BCE Loss          : {avg_loss:.6f}")
    print(f"  F1  (t=0.50)      : {f1_05:.4f}")
    print(f"  Accuracy (t=0.50) : {acc_05:.4f}")
    print(f"  Precision(t=0.50) : {prec_05:.4f}")
    print(f"  Recall   (t=0.50) : {rec_05:.4f}")
    print(f"  Optimal threshold : {best_t:.2f}")
    print(f"  F1  (t={best_t:.2f})     : {best_f1:.4f}")

    pc = per_class_metrics(probs, labels, best_t)
    pc_sorted = sorted(pc, key=lambda r: r["f1"], reverse=True)

    print("\n── Per-class F1 (threshold={:.2f}, sorted desc) ─────────────────".format(best_t))
    for row in pc_sorted:
        bar = "█" * int(row["f1"] * 20)
        print(f"  {row['class_name'][:40]:<40} F1={row['f1']:.3f}  {bar}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["class_idx", "class_name", "f1", "precision", "recall"])
            writer.writeheader()
            writer.writerows(pc)
        # Append aggregate row
        with open(out_path, "a") as f:
            f.write(f"\n# aggregate,loss={avg_loss:.6f},f1_t05={f1_05:.4f},"
                    f"f1_opt={best_f1:.4f},threshold={best_t:.2f},"
                    f"accuracy={acc_05:.4f},precision={prec_05:.4f},recall={rec_05:.4f}\n")
        print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()
