"""Single-GPU training script for BigEarthNet-S2."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import BigEarthNetDataset, get_transforms
from src.models.vit import build_model
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT on BigEarthNet-S2 (single GPU)")
    parser.add_argument("--data-root", type=str, required=True, help="Path to BigEarthNet-S2 root")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/single_gpu")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Datasets
    train_dataset = BigEarthNetDataset(args.data_root, split="train", transform=get_transforms("train"))
    val_dataset = BigEarthNetDataset(args.data_root, split="val", transform=get_transforms("val"))
    print(f"Train: {len(train_dataset)} patches | Val: {len(val_dataset)} patches")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    model = build_model(
        model_name=args.model_name,
        pretrained=not args.no_pretrained,
    )
    print(f"Modelo: {args.model_name} | Parámetros: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer y scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Entrenamiento
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer.fit(train_loader, val_loader, epochs=args.epochs)


if __name__ == "__main__":
    main()
