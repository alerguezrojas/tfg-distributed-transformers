"""Predict — classify loose images (or folders of them) with a trained checkpoint.

Runs outside the training loop: loads one checkpoint (or an ensemble: the members' probabilities
are averaged, as in ``eval.py``) and classifies the given image files. The dataset named in the
config (``data.name``) decides how a file is read and how it is transformed — the SAME evaluation
transform used at test time (``spec.transforms("test")``) — and the task decides how the output is
read: softmax datasets give the top class (ordinal ones also the expected grade, sum k·p_k),
sigmoid datasets the classes above the threshold stored in the checkpoint.

Datasets whose samples are not a single image file (BigEarthNet: a multiband patch) have no image
reader and fail with a clear error.

Usage:
    # One image, with the model recorded in the checkpoint
    uv run python scripts/predict.py --checkpoint checkpoints/local/single/vit_small/RUN \\
        --config configs/train_panda.yaml mosaic.jpg

    # A folder (recursive, sorted) with an ensemble, saving every class probability
    uv run python scripts/predict.py --config configs/train_panda.yaml \\
        --checkpoint runA/best.pt --checkpoint runB/best.pt mosaics/ --out preds.csv

Security: checkpoints are loaded with ``weights_only=True``. Input paths are local and read as
given; when a folder is walked, symbolic links to directories are not followed and links to files
are skipped if they point outside the folder. Every file must be a regular file with an image
extension; PIL (through the dataset's reader) opens it.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.data.registry import dataset_spec_from_config  # noqa: E402
from src.training.checkpoints import resolve_checkpoint  # noqa: E402
from eval import (load_member, resolve_model_name,  # noqa: E402
                  stored_threshold_metrics)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
DEFAULT_THRESHOLD = 0.5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Classify images with a trained checkpoint")
    parser.add_argument("inputs", nargs="+", metavar="INPUT",
                        help="Image files or folders (searched recursively, sorted)")
    parser.add_argument("--checkpoint", type=str, action="append", required=True,
                        help="Checkpoint .pt file or run folder (resolves to its best.pt). "
                             "Repeat it for an ensemble: the probabilities are averaged")
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                        help="Config YAML (dataset, model options such as img_size)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override the model (default: the one stored in the checkpoint)")
    parser.add_argument("--out", type=str, default=None,
                        help="CSV path with the probability of every class")
    parser.add_argument("--batch-size", type=int, default=16, help="Images per batch")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None,
                        help="Device (default: cuda if available, else cpu)")
    return parser.parse_args(argv)


# ── Inputs ──────────────────────────────────────────────────────────────────────

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _walk_images(folder: Path) -> list[Path]:
    """Images under ``folder`` in a deterministic order. Directory links are not followed and
    file links that resolve outside ``folder`` are skipped."""
    root = folder.resolve()
    found = []
    for dirpath, dirnames, filenames in os.walk(folder, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if not is_image_file(path):
                continue
            if path.is_symlink() and not path.resolve().is_relative_to(root):
                continue
            found.append(path)
    return found


def collect_inputs(inputs) -> list[Path]:
    """Expand files and folders into the list of image files (order kept, no duplicates).
    Raises ``ValueError`` for a missing path, a file that is not an image or an empty folder."""
    files: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_dir():
            found = _walk_images(path)
            if not found:
                raise ValueError(f"{path}: la carpeta no contiene imágenes "
                                 f"({', '.join(IMAGE_EXTENSIONS)})")
            files += found
        elif path.is_file():
            if not is_image_file(path):
                raise ValueError(f"{path}: no es un fichero de imagen "
                                 f"({', '.join(IMAGE_EXTENSIONS)})")
            files.append(path)
        else:
            raise ValueError(f"{path}: no existe o no es un fichero regular")
    return list(dict.fromkeys(files))


class ImageFiles(Dataset):
    """Image files read by the dataset's reader and transformed like its evaluation split."""

    def __init__(self, files, reader, transform):
        self.files, self.reader, self.transform = list(files), reader, transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        try:
            return self.transform(self.reader(path))
        except (OSError, ValueError) as err:     # PIL: not an image / truncated file
            raise ValueError(f"{path}: no se puede leer como imagen ({err})") from err


# ── Inference ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(model, loader, device, task) -> torch.Tensor:
    """Probabilities (N, C) of every image of the loader, in order."""
    model.eval()
    return torch.cat([task.probabilities(model(images.to(device))).cpu() for images in loader])


def predict_members(paths, cfg: dict, model_name: str, spec, loader, device, task):
    """Mean probabilities of one checkpoint or of an ensemble, and the members' checkpoint
    info. Members are loaded one at a time, so only one model lives in memory."""
    probs, infos = [], []
    for path in paths:
        model, ckpt = load_member(path, cfg, model_name, spec, device)
        infos.append({"epoch": ckpt.get("epoch", "?"), "metrics": ckpt.get("metrics", {})})
        probs.append(infer(model, loader, device, task))
        del model, ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.stack(probs).mean(dim=0), infos


def decision_threshold(task, infos) -> tuple[float, bool]:
    """Threshold stored in the checkpoint(s) (sigmoid tasks) and whether it was found; softmax
    tasks have none and return the default."""
    if not task.searches_threshold:
        return DEFAULT_THRESHOLD, True
    stored = stored_threshold_metrics(infos).get("optimal_threshold")
    return (DEFAULT_THRESHOLD, False) if stored is None else (float(stored), True)


def build_table(files, probs, classes, task, threshold: float) -> pd.DataFrame:
    """One row per image: file, predicted, prob (the highest class probability), the expected
    grade for ordinal tasks, and one ``p_<class>`` column per class. Sigmoid tasks list every
    class above ``threshold`` in ``predicted`` (``-`` if none)."""
    p = probs.numpy() if torch.is_tensor(probs) else np.asarray(probs)
    hard = task.predict(torch.as_tensor(p), threshold).numpy().astype(bool)
    if task.sigmoid_outputs:
        predicted = [";".join(c for c, on in zip(classes, row) if on) or "-" for row in hard]
    else:
        predicted = [classes[i] for i in p.argmax(axis=1)]
    table = {"file": [str(f) for f in files], "predicted": predicted,
             "prob": p.max(axis=1).round(4)}
    if getattr(task, "ordinal", False):
        table["expected_grade"] = (p * np.arange(p.shape[1])).sum(axis=1).round(3)
    for i, name in enumerate(classes):
        table[f"p_{name}"] = p[:, i].round(5)
    return pd.DataFrame(table)


def print_table(table: pd.DataFrame) -> None:
    cols = [c for c in ("file", "predicted", "prob", "expected_grade") if c in table]
    print(table[cols].to_string(index=False))


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out) if args.out else None
    if out is not None and out.suffix.lower() != ".csv":
        sys.exit(f"ERROR: --out debe ser un fichero .csv, no '{out}'")
    try:
        files = collect_inputs(args.inputs)
    except ValueError as err:
        sys.exit(f"ERROR: {err}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    spec = dataset_spec_from_config(cfg)
    if spec.image_reader is None:
        sys.exit(f"ERROR: el dataset '{spec.name}' no se puede leer desde imágenes sueltas "
                 f"(sus muestras no son un único fichero de imagen); `predict` no está "
                 f"disponible para él.")
    task = spec.make_task()
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("ERROR: --device cuda pedido, pero CUDA no está disponible")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    paths = [resolve_checkpoint(c) for c in args.checkpoint]
    model_name = resolve_model_name(args.model, cfg, paths)
    print(f"Checkpoint : {', '.join(str(p) for p in paths)}"
          + (f" (conjunto de {len(paths)}, probabilidades promediadas)" if len(paths) > 1 else ""))
    print(f"Dataset    : {spec.name} ({task.name}, {spec.num_classes} clases)")
    print(f"Modelo     : {model_name}")
    print(f"Imágenes   : {len(files)}   Dispositivo: {device}\n")

    dataset = ImageFiles(files, spec.image_reader(cfg["data"]), spec.transforms("test"))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    try:
        probs, infos = predict_members(paths, cfg, model_name, spec, loader, device, task)
    except ValueError as err:                       # unreadable image
        sys.exit(f"ERROR: {err}")

    threshold, found = decision_threshold(task, infos)
    if not found:
        print(f"[aviso] el checkpoint no guarda el umbral; se usa {DEFAULT_THRESHOLD}")
    table = build_table(files, probs, spec.classes, task, threshold)
    if task.searches_threshold:
        print(f"Umbral     : {threshold:.2f}\n")
    print_table(table)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print(f"\nProbabilidades guardadas en: {out}")


if __name__ == "__main__":
    main()
