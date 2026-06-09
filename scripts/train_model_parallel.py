"""Model-parallel (pipeline) training for the BigEarthNet ViT.

Splits the ViT across the visible GPUs (stage 0 + stage 1) with
:class:`ModelParallelViT` and runs a self-contained training loop — model
parallelism needs custom device handling (input on stage 0, logits/loss on
stage 1), so it does not reuse the single-device Trainer stack.

It writes the SAME artefacts as the other trainers so the dashboard discovers
the run automatically:
  logs/{env}/model_parallel/{model}/train_{ts}.log
  logs/{env}/model_parallel/{model}/epoch_metrics_{ts}.csv

Device assignment:
  - >=2 CUDA GPUs -> stage 0 on cuda:0, stage 1 on cuda:1 (true model parallel)
  - 1 CUDA GPU    -> both stages on cuda:0 (runs; demonstrates the path)
  - no GPU        -> both stages on cpu (smoke testing)

Usage (Kaggle 2xT4 — see docs/model_parallel_runbook.md):
  python scripts/train_model_parallel.py --config configs/train_model_parallel_kaggle.yaml --trace simple
"""
import argparse
import csv
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import BigEarthNetDataset, get_transforms
from src.models.model_parallel import ModelParallelViT
from src.models.vit import BigEarthModel, build_llrd_optimizer
from src.training import metrics as M

_CSV_COLS = ["epoch", "train_loss", "val_loss", "train_f1", "val_f1",
             "train_acc", "val_acc", "val_prec", "val_rec", "epoch_time_s"]


def parse_args():
    p = argparse.ArgumentParser(description="Model-parallel training for BigEarthNet-S2")
    p.add_argument("--config", type=str, default="configs/train_model_parallel_kaggle.yaml")
    p.add_argument("--model", type=str, default=None, help="Override model name (timm ViT)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--split-block", type=int, default=None,
                   help="Block index where stage 0 ends (default: half the blocks)")
    p.add_argument("--devices", type=str, default=None,
                   help="Comma-separated device list, e.g. 'cuda:0,cuda:1'. Auto-detected if omitted.")
    p.add_argument("--trace", choices=["off", "simple"], default="simple")
    return p.parse_args()


def _pick_devices(arg: str | None) -> list[str]:
    if arg:
        return [d.strip() for d in arg.split(",")]
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n >= 2:
        return ["cuda:0", "cuda:1"]
    if n == 1:
        return ["cuda:0", "cuda:0"]
    return ["cpu", "cpu"]


def _make_logger(trace: str, log_path: Path) -> logging.Logger:
    logger = logging.getLogger("model_parallel")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if trace == "simple":
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


@torch.no_grad()
def _evaluate(model, loader, criterion, out_dev) -> dict:
    model.eval()
    total_loss, n = 0.0, 0
    preds_all, labels_all = [], []
    for images, labels in loader:
        labels = labels.to(out_dev)
        logits = model(images)
        loss = criterion(logits, labels)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        n += bs
        preds_all.append((logits > 0).int().cpu())
        labels_all.append(labels.int().cpu())
    preds = torch.cat(preds_all)
    labels = torch.cat(labels_all)
    return {
        "loss": total_loss / max(n, 1),
        "f1": M.f1_score(preds, labels),
        "acc": M.accuracy(preds, labels),
        "prec": M.precision(preds, labels),
        "rec": M.recall(preds, labels),
    }


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.model:
        cfg["model"]["name"] = args.model
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    model_name = cfg["model"]["name"]
    env = cfg.get("output", {}).get("env", "local")
    epochs = int(cfg["training"]["epochs"])
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["lr"])
    weight_decay = float(cfg["training"].get("weight_decay", 0.05))
    warmup = int(cfg["training"].get("warmup_epochs", 0))
    llrd_decay = float(cfg["training"].get("llrd_decay", 0.75))
    num_workers = int(cfg["data"].get("num_workers", 4))

    ts = datetime.now().strftime("%d%m%Y_%H%M%S")
    out_dir = Path(f"logs/{env}/model_parallel/{model_name}")
    log_path = out_dir / f"train_{ts}.log"
    csv_path = out_dir / f"epoch_metrics_{ts}.csv"
    logger = _make_logger(args.trace, log_path)

    devices = _pick_devices(args.devices)
    distinct = len(set(devices)) > 1
    logger.info(f"Model-parallel ViT on devices={devices} "
                f"({'true 2-device split' if distinct else 'single device — demo path'})")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = BigEarthNetDataset(cfg["data"]["root"], cfg["data"]["metadata"],
                                  split="train", transform=get_transforms("train"))
    val_ds = BigEarthNetDataset(cfg["data"]["root"], cfg["data"]["metadata"],
                                split="val", transform=get_transforms("val"))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # ── Model (split across devices) ────────────────────────────────────────────
    base = BigEarthModel(model_name=model_name, num_classes=cfg["model"].get("num_classes", 19),
                         pretrained=cfg["model"].get("pretrained", True),
                         dropout=cfg["model"].get("dropout", 0.1))
    model = ModelParallelViT(base, devices=devices, split_block=args.split_block)
    out_dev = model.output_device
    n_blocks = len(base.backbone.blocks)

    optimizer = build_llrd_optimizer(base, lr_base=lr, weight_decay=weight_decay,
                                     llrd_decay=llrd_decay)
    criterion = nn.BCEWithLogitsLoss()

    def lr_scale(ep: int) -> float:
        if warmup and ep < warmup:
            return (ep + 1) / warmup
        progress = (ep - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    # Config line (parsed by the dashboard's Run -> Info panel)
    logger.info(
        f"Configuración: modelo={model_name} | paralelismo=modelo(pipeline) | "
        f"split_block={model.split_block}/{n_blocks} | devices={'+'.join(devices)} | "
        f"batch={batch_size} | epochs={epochs} | lr={lr} | "
        f"train={len(train_ds)} | val={len(val_ds)}"
    )

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(_CSV_COLS)

    grad_clip = float(cfg["training"].get("grad_clip", 0) or 0)
    best_f1, epoch_times = 0.0, []
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        tot_loss, n = 0.0, 0
        tr_preds, tr_labels = [], []
        for images, labels in train_loader:
            labels = labels.to(out_dev)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(base.parameters(), grad_clip)
            optimizer.step()
            bs = labels.size(0)
            tot_loss += loss.item() * bs
            n += bs
            tr_preds.append((logits.detach() > 0).int().cpu())
            tr_labels.append(labels.int().cpu())
        scheduler.step()

        tp, tl = torch.cat(tr_preds), torch.cat(tr_labels)
        train = {"loss": tot_loss / max(n, 1), "f1": M.f1_score(tp, tl), "acc": M.accuracy(tp, tl)}
        val = _evaluate(model, val_loader, criterion, out_dev)
        dt = time.perf_counter() - t0
        epoch_times.append(dt)
        best_f1 = max(best_f1, val["f1"])

        logger.info(
            f"── Epoch {epoch}/{epochs}  ({dt:.1f}s)\n"
            f"    loss      train={train['loss']:.4f}  val={val['loss']:.4f}\n"
            f"    f1        train={train['f1']:.4f}  val={val['f1']:.4f}  (best={best_f1:.4f})\n"
            f"    accuracy  train={train['acc']:.4f}  val={val['acc']:.4f}\n"
            f"    precision val={val['prec']:.4f}    recall val={val['rec']:.4f}\n"
            f"    ETA: {M.eta_str(epoch_times, epoch, epochs)} ({dt:.0f}s/epoch)"
        )
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, round(train["loss"], 5), round(val["loss"], 5),
                round(train["f1"], 5), round(val["f1"], 5),
                round(train["acc"], 5), round(val["acc"], 5),
                round(val["prec"], 5), round(val["rec"], 5), round(dt, 2),
            ])

    logger.info(f"Done. Best Val F1={best_f1:.4f}. Artefacts in {out_dir}")


if __name__ == "__main__":
    main()
