"""Evaluation script — measures model performance on the test split.

Loads a saved checkpoint and evaluates it on the test split of the dataset named
in the config (``data.name``, default BigEarthNet-S2), reporting per-class and
aggregate metrics — with optimal threshold search for sigmoid tasks.

Usage:
    # Evaluate the best checkpoint of a run (a folder resolves to its best.pt)
    uv run python scripts/eval.py --checkpoint checkpoints/verode/vit_small/05102026_101500

    # Override config (e.g. to use cluster dataset paths)
    uv run python scripts/eval.py \\
        --checkpoint checkpoints/verode/checkpoint_epoch_007.pt \\
        --config configs/train_cluster_v3.yaml

    # Dataset in another location (only data.root moves)
    uv run python scripts/eval.py --checkpoint checkpoints/verode/vit_small/05102026_101500 \\
        --data-root /data/bigearthnet

    # Ensemble: average the probabilities of several checkpoints (e.g. three seeds)
    uv run python scripts/eval.py --config configs/train_panda.yaml \\
        --checkpoint runA/best.pt --checkpoint runB/best.pt --checkpoint runC/best.pt \\
        --save-predictions logs/local/panda/ensemble_preds.npz

    # Save per-class results to CSV
    uv run python scripts/eval.py \\
        --checkpoint checkpoints/verode/checkpoint_epoch_007.pt \\
        --output logs/verode/eval_results.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.clinical import clinical_cutoffs, group_summary
from src.analysis.io import default_predictions_path, save_predictions
from src.data.registry import dataset_spec_from_config
from src.models.vit import build_model
from src.training import metrics as m
from src.training.checkpoints import resolve_checkpoint, strip_wrapper_prefix
from src.training.config_overrides import add_data_root_override, apply_common_overrides
from src.training.tasks import MultiLabelTask, Task


AUTO_PATH = "@auto"     # value of --save-predictions given without a path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the test set")
    parser.add_argument("--checkpoint", type=str, action="append", required=True,
                        help="Checkpoint .pt file (best.pt, last.pt or an old "
                             "checkpoint_epoch_NNN.pt); a run folder resolves to its best.pt. "
                             "Repeat it to evaluate an ensemble: the members' probabilities are "
                             "averaged and every metric is computed on that average")
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                        help="Config YAML (used for dataset paths, model name, batch size)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model name from config")
    add_data_root_override(parser)
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
    parser.add_argument("--save-predictions", type=str, nargs="?", const=AUTO_PATH,
                        default=None, metavar="PATH",
                        help=".npz path to save the per-sample probabilities, targets, "
                             "threshold, classes, task and group variables (format in "
                             "docs/inspeccion.md). Without a path it goes to the run's log "
                             "folder as test_preds_<ts>.npz, which the web Inspector reads")
    parser.add_argument("--threshold-from", choices=["checkpoint", "val", "test"],
                        default="checkpoint",
                        help="Where the decision threshold is tuned: 'checkpoint' (the one "
                             "chosen on validation during training; falls back to 'val' "
                             "for older checkpoints), 'val' (tune now on the validation "
                             "split) or 'test' (tuned on the evaluated split itself — "
                             "optimistic, kept only to reproduce older numbers)")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None, task: Task | None = None):
    """Run inference and return probabilities, targets and mean loss."""
    task = task or MultiLabelTask()
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    n_batches = 0
    criterion = task.build_criterion()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        all_probs.append(task.probabilities(logits).cpu())
        all_labels.append(labels.cpu())
        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    return probs, labels, total_loss / n_batches


def average_probabilities(probs_list, labels_list):
    """Mean of the members' probabilities (an ensemble) and their shared targets.

    Pure: every member must have produced the same output shape and evaluated the same
    targets (same split, same order); otherwise it raises ``ValueError``. A single member
    is returned untouched."""
    if not probs_list or len(probs_list) != len(labels_list):
        raise ValueError("Se necesita al menos un miembro, con sus probabilidades y objetivos")
    for i, (p, y) in enumerate(zip(probs_list, labels_list)):
        if p.shape != probs_list[0].shape:
            raise ValueError(f"El miembro {i + 1} produce probabilidades de forma "
                             f"{tuple(p.shape)}, distinta de la del primero "
                             f"{tuple(probs_list[0].shape)}")
        if y.shape != labels_list[0].shape or not torch.equal(y, labels_list[0]):
            raise ValueError(f"El miembro {i + 1} no evalúa los mismos objetivos que el primero")
    if len(probs_list) == 1:
        return probs_list[0], labels_list[0]
    return torch.stack(probs_list).mean(dim=0), labels_list[0]


def threshold_search(probs, labels, task: Task | None = None):
    """(threshold, macro F1) maximising F1 — same search as during training.

    Tasks without a tunable threshold (multi-class) return the default 0.5 and
    its F1."""
    task = task or MultiLabelTask()
    hard = task.hard_targets(labels.float())
    found = task.search_threshold(probs, hard)
    if found is None:
        return 0.5, m.f1_score(task.predict(probs), hard)
    return found


def choose_threshold(source: str, ckpt_metrics: dict, task: Task, probs, labels,
                     val_outputs=None) -> tuple[float, str]:
    """Decision threshold and where it came from.

    'checkpoint' → the validation-tuned threshold stored at training time (older
    checkpoints lack it and fall back to 'val'); 'val' → tuned now on the validation
    split (``val_outputs()`` returns its probabilities and targets); 'test' → tuned on
    the evaluated split itself, which overstates the result.
    """
    if source == "checkpoint":
        stored = ckpt_metrics.get("optimal_threshold")
        if stored is not None:
            return float(stored), "checkpoint"
        source = "val"
    if source == "val":
        val_probs, val_labels = val_outputs()
        return threshold_search(val_probs, val_labels, task)[0], "val"
    return threshold_search(probs, labels, task)[0], "test"


def per_class_metrics(probs, labels, threshold, class_names=(), task: Task | None = None):
    """Compute F1, precision, recall per class."""
    task = task or MultiLabelTask()
    preds = task.predict(probs, threshold)
    hard = task.hard_targets(labels.float())
    n_classes = hard.shape[1]
    rows = []
    for i in range(n_classes):
        p = preds[:, i]
        l = hard[:, i]
        tp = (p & l).sum().item()
        fp = (p & ~l.bool()).sum().item()
        fn = (~p.bool() & l.bool()).sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        rows.append({
            "class_idx": i,
            "class_name": class_names[i] if i < len(class_names) else f"class_{i}",
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
        })
    return rows


def dataset_groups(ds, n: int) -> dict:
    """Grouping variables of the dataset (duck typing: ``ds.groups()``), or {} if it has none
    or they do not cover the ``n`` evaluated samples (e.g. with ``--max-batches``)."""
    getter = getattr(ds, "groups", None)
    if not callable(getter):
        return {}
    groups = {k: np.asarray(v) for k, v in getter().items()}
    if any(v.shape[0] < n for v in groups.values()):
        return {}
    return {k: v[:n] for k, v in groups.items()}      # the loader keeps the dataset order


def _fmt(x) -> str:
    return "  n/a " if x is None or np.isnan(x) else f"{x:.3f}"


def print_group_metrics(probs, hard, task_name, groups, threshold, ordinal) -> None:
    """Per-variable, per-value n and main metrics (nothing if there are no groups)."""
    if not groups:
        return
    rows = group_summary(probs, hard, task_name, groups, threshold, ordinal)
    qwk_head = "  QWK  " if ordinal else ""
    for var in groups:
        print(f"\n── Metrics by group: {var} ───────────────────────────────────────")
        print(f"  {'group':<20} {'n':>6}  {'Acc':>5}  {'F1':>5}  {'AUC':>5}{qwk_head}")
        for r in (r for r in rows if r["variable"] == var):
            qwk = f"  {_fmt(r['qwk'])}" if ordinal else ""
            print(f"  {str(r['group'])[:20]:<20} {r['n']:>6}  {_fmt(r['accuracy'])}  "
                  f"{_fmt(r['f1'])}  {_fmt(r['auc'])}{qwk}")


def print_clinical_cutoffs(probs, hard, class_names) -> None:
    """Table of the binary 'grade >= k' cut-offs of an ordinal task."""
    rows = clinical_cutoffs(probs, hard, class_names)
    print("\n── Clinical cut-offs (grade >= k; score = P(grade >= k)) ─────────")
    print(f"  {'cut':<10} {'n+':>6} {'n-':>6}  {'Sens':>5}  {'Spec':>5}  {'PPV':>5}  "
          f"{'NPV':>5}  {'AUC':>5}")
    for r in rows:
        print(f"  {r['label'][:10]:<10} {r['n_pos']:>6} {r['n_neg']:>6}  "
              f"{_fmt(r['sensitivity'])}  {_fmt(r['specificity'])}  {_fmt(r['ppv'])}  "
              f"{_fmt(r['npv'])}  {_fmt(r['auc'])}")


def build_eval_model(cfg: dict, model_name: str, spec):
    """Architecture of the trained model, with random weights (the checkpoint fills them).

    Everything that changes the architecture or the readout of the features must match the
    training config: ``img_size`` and ``pooling`` (the latter is not in the state_dict, so a
    mismatch would evaluate silently with the wrong pooling).
    """
    # dropout is a no-op at eval time (model.eval() disables it) and does not
    # affect the state_dict shape, so build_model need not take it.
    return build_model(
        model_name=model_name,
        num_classes=spec.num_classes,
        pretrained=False,
        in_chans=spec.in_channels,
        img_size=cfg["model"].get("img_size"),
        pooling=cfg["model"].get("pooling"),
    )


def checkpoint_model_name(path) -> str | None:
    """Model a checkpoint was trained with (``model_name`` key), or None for old ones.
    Read with mmap so peeking at an ensemble does not load every set of weights."""
    ckpt = torch.load(resolve_checkpoint(path), map_location="cpu", weights_only=True, mmap=True)
    return ckpt.get("model_name")


def check_model_matches(path, ckpt_name: str | None, model_name: str) -> None:
    """Fail clearly if the checkpoint says it holds another model than the one requested."""
    if ckpt_name is not None and ckpt_name != model_name:
        sys.exit(f"ERROR: {path} es un checkpoint de '{ckpt_name}', pero se pide '{model_name}' "
                 f"(--model o la config). Quita --model para usar el del checkpoint.")


def resolve_model_name(cli_model: str | None, cfg: dict, paths) -> str:
    """--model, else the model recorded in the checkpoint(s), else the config's.

    Checkpoints without ``model_name`` (older runs) fall back to the config, as always. An
    ensemble must not mix models unless --model forces one (a mismatch then fails on load)."""
    names = {p: checkpoint_model_name(p) for p in paths}
    if cli_model:
        for p, n in names.items():
            check_model_matches(p, n, cli_model)
        return cli_model
    distinct = {n for n in names.values() if n is not None}
    if len(distinct) > 1:
        sys.exit("ERROR: los checkpoints son de modelos distintos ("
                 + ", ".join(f"{p}: {n}" for p, n in names.items())
                 + "); un conjunto necesita el mismo modelo.")
    return distinct.pop() if distinct else cfg["model"]["name"]


def load_member(path, cfg: dict, model_name: str, spec, device):
    """Model with the checkpoint's weights (DDP / model-parallel prefixes stripped), and the
    checkpoint dict (epoch, metrics)."""
    ckpt = torch.load(resolve_checkpoint(path), map_location=device, weights_only=True)
    check_model_matches(path, ckpt.get("model_name"), model_name)
    model = build_eval_model(cfg, model_name, spec)
    model.load_state_dict(strip_wrapper_prefix(ckpt["model_state_dict"]))
    return model.to(device), ckpt


def evaluate_members(paths, cfg: dict, model_name: str, spec, loader, device, task,
                     max_batches=None, preloaded=None):
    """Probabilities, targets, mean loss and per-member checkpoint info of one checkpoint or
    of an ensemble (the average of the members' probabilities).

    Members are loaded one at a time and freed before the next, so only one model lives in
    memory. ``preloaded`` is the already-loaded first member ``(model, ckpt)``, if any."""
    all_probs, all_labels, losses, infos = [], [], [], []
    for i, path in enumerate(paths):
        model, ckpt = preloaded if (i == 0 and preloaded) else load_member(
            path, cfg, model_name, spec, device)
        infos.append({"epoch": ckpt.get("epoch", "?"), "metrics": ckpt.get("metrics", {})})
        probs, labels, loss = evaluate(model, loader, device, max_batches=max_batches, task=task)
        all_probs.append(probs)
        all_labels.append(labels)
        losses.append(loss)
        del model, ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    probs, labels = average_probabilities(all_probs, all_labels)
    return probs, labels, sum(losses) / len(losses), infos


def stored_threshold_metrics(infos: list[dict]) -> dict:
    """Metrics dict for ``choose_threshold``: a single checkpoint keeps its own; an ensemble
    uses the mean of the members' validation-tuned thresholds (if any lacks it → {} and the
    threshold is tuned on validation)."""
    if len(infos) == 1:
        return infos[0]["metrics"]
    stored = [i["metrics"].get("optimal_threshold") for i in infos]
    if any(t is None for t in stored):
        return {}
    return {"optimal_threshold": float(np.mean(stored))}


def main():
    args = parse_args()
    paths = [resolve_checkpoint(c) for c in args.checkpoint]
    is_ensemble = len(paths) > 1
    if is_ensemble and args.save_predictions == AUTO_PATH:
        sys.exit("ERROR: con varios checkpoints (conjunto), --save-predictions necesita una "
                 "ruta explícita: no hay un único run del que deducir el sello.")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    apply_common_overrides(cfg, args)       # --data-root (eval has no --seed/--epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = resolve_model_name(args.model, cfg, paths)
    batch_size = args.batch_size or cfg["training"].get("batch_size", 32)
    spec = dataset_spec_from_config(cfg)
    task = spec.make_task()

    if is_ensemble:
        print(f"Conjunto   : {len(paths)} checkpoints (probabilidades promediadas): "
              + ", ".join(str(p) for p in paths))
    else:
        print(f"Checkpoint : {paths[0]}")
    print(f"Dataset    : {spec.name} ({task.name}, {spec.num_classes} clases)")
    print(f"Modelo     : {model_name}")
    print(f"Split      : {args.split}")
    print(f"Dispositivo: {device}")

    # ── Build model ───────────────────────────────────────────────────────────
    # A single checkpoint is loaded here (fails before the dataset is built); an ensemble
    # loads its members one at a time inside evaluate_members, so only one lives in memory.
    first = None if is_ensemble else load_member(paths[0], cfg, model_name, spec, device)
    if first is not None:
        print(f"Epoch      : {first[1].get('epoch', '?')}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_cfg = dict(cfg["data"])
    if args.metadata:
        data_cfg["metadata"] = args.metadata
    metadata_path = data_cfg.get("metadata", data_cfg.get("root"))

    def _loader(split: str):
        split_ds = spec.dataset(data_cfg, split)
        return split_ds, DataLoader(
            split_ds, batch_size=batch_size, shuffle=False,
            num_workers=cfg["data"].get("num_workers", 4), pin_memory=True,
        )

    ds, loader = _loader(args.split)
    print(f"Patches    : {len(ds)}")
    if len(ds) == 0:
        print(f"\nERROR: the '{args.split}' split is empty in {metadata_path}. "
              "This metadata may not contain that split (e.g. the demo subset has "
              "only train/val). Use --split val or a full-dataset metadata.")
        sys.exit(1)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    probs, labels, avg_loss, infos = evaluate_members(
        paths, cfg, model_name, spec, loader, device, task,
        max_batches=args.max_batches, preloaded=first)
    del first
    if is_ensemble:
        print("Epochs     : " + ", ".join(str(i["epoch"]) for i in infos))
    hard = task.hard_targets(labels)
    agg = task.metrics(task.predict(probs), hard)
    f1_05, acc_05 = agg["f1"], agg["accuracy"]
    prec_05, rec_05 = agg["precision"], agg["recall"]
    auc = m.roc_auc_macro(probs, hard)       # threshold-free; MedMNIST's main metric

    best_t, best_f1, source = 0.5, f1_05, None
    if task.searches_threshold:
        def _val_outputs():
            print("  (ajustando el umbral sobre el split de validación...)")
            val_probs, val_labels, _, _ = evaluate_members(
                paths, cfg, model_name, spec, _loader("val")[1], device, task,
                max_batches=args.max_batches)
            return val_probs, val_labels
        best_t, source = choose_threshold(args.threshold_from, stored_threshold_metrics(infos),
                                          task, probs, labels, _val_outputs)
        best_f1 = m.f1_score(task.predict(probs, best_t), hard)

    print("\n── Aggregate metrics ─────────────────────────────────────────────")
    print(f"  Loss              : {avg_loss:.6f}")
    print(f"  F1  (t=0.50)      : {f1_05:.4f}")
    print(f"  Accuracy (t=0.50) : {acc_05:.4f}")
    print(f"  Precision(t=0.50) : {prec_05:.4f}")
    print(f"  Recall   (t=0.50) : {rec_05:.4f}")
    print(f"  AUC (macro)       : {auc:.4f}")
    if "qwk" in agg:                         # ordinal datasets (a grade): PANDA's metric
        print(f"  QWK (cuadrático)  : {agg['qwk']:.4f}")
    if task.searches_threshold:
        note = "  (aviso: ajustado sobre el propio test, optimista)" if source == "test" else ""
        print(f"  Threshold         : {best_t:.2f} (de {source}){note}")
        print(f"  F1  (t={best_t:.2f})     : {best_f1:.4f}")
    else:
        print(f"  Optimal threshold : n/a ({task.name})")

    pc = per_class_metrics(probs, labels, best_t, spec.classes, task)
    pc_sorted = sorted(pc, key=lambda r: r["f1"], reverse=True)

    print("\n── Per-class F1 (threshold={:.2f}, sorted desc) ─────────────────".format(best_t))
    for row in pc_sorted:
        bar = "█" * int(row["f1"] * 20)
        print(f"  {row['class_name'][:40]:<40} F1={row['f1']:.3f}  {bar}")

    # ── Per-group metrics / clinical cut-offs / saved predictions ─────────────
    groups = dataset_groups(ds, len(probs))
    ordinal = bool(getattr(task, "ordinal", False))
    probs_np, hard_np = probs.numpy(), hard.numpy()
    print_group_metrics(probs_np, hard_np, task.name, groups, best_t, ordinal)
    if ordinal:
        print_clinical_cutoffs(probs_np, hard_np, spec.classes)
    if args.save_predictions:
        target = (default_predictions_path(
                      paths[0], checkpoint_dir=cfg.get("checkpoint", {}).get("dir"),
                      env=cfg.get("output", {}).get("env", "local"),
                      warn=lambda msg: print(f"  aviso: {msg}"))
                  if args.save_predictions == AUTO_PATH else args.save_predictions)
        save_predictions(target, probs_np, hard_np, best_t, spec.classes, task.name, ordinal,
                         groups, threshold_source=source)
        print(f"\nPredicciones guardadas en: {target}")

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
            thresh = (f"f1_opt={best_f1:.4f},threshold={best_t:.2f},threshold_source={source},"
                      if task.searches_threshold else "")
            f.write(f"\n# aggregate,loss={avg_loss:.6f},f1_t05={f1_05:.4f},{thresh}"
                    f"accuracy={acc_05:.4f},precision={prec_05:.4f},recall={rec_05:.4f},"
                    f"auc={auc:.4f}\n")
        print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()
