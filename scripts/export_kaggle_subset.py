"""Exporta un subset autocontenido de BigEarthNet-S2 listo para subir a Kaggle.

Selecciona N train + M val (aleatorio con seed fijo, seed=42, así el subset
coincide con el demo de Verode) y copia
SOLO las 3 bandas RGB (B04/B03/B02) de cada patch, preservando la estructura
    <salida>/BigEarthNet-S2/<scene_id>/<patch_id>/<patch_id>_<band>.tif
y escribe <salida>/metadata_demo.parquet.

Las rutas salen del perfil de la máquina (``configs/machines/<perfil>.yaml``,
``data_roots.bigearthnet``: ``root`` con los .tif y ``metadata_dir`` con
``metadata.parquet``), no de argumentos: la salida es
``<metadata_dir>/kaggle_subset_<n_train>_<n_val>/``.

El resultado (~1 GB para 5000+1500 patches) se comprime y se sube como dataset
de Kaggle para medir el speedup 1×T4 vs 2×T4 con datos reales.

Uso:
    uv run python scripts/export_kaggle_subset.py --n-train 5000 --n-val 1500
    uv run python scripts/export_kaggle_subset.py --machine local   # otro perfil
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.machine import select_machine  # noqa: E402

RGB_BANDS = ["B04", "B03", "B02"]


def _patch_dir(root: Path, patch_id: str) -> Path:
    scene_id = "_".join(patch_id.rsplit("_", 2)[:-2])
    return root / scene_id / patch_id


def bigearthnet_paths(profile) -> tuple[Path, Path]:
    """``(root con los .tif, metadata.parquet)`` del perfil de máquina."""
    entry = profile.data_roots.get("bigearthnet")
    if isinstance(entry, str):
        entry = {"root": entry, "metadata_dir": str(Path(entry).parent)}
    if not isinstance(entry, dict) or not entry.get("root") or not entry.get("metadata_dir"):
        raise SystemExit(f"ERROR: el perfil '{profile.name}' no define data_roots.bigearthnet "
                         "con root y metadata_dir (ver configs/machines/example.yaml)")
    return Path(entry["root"]), Path(entry["metadata_dir"]) / "metadata.parquet"


def main():
    ap = argparse.ArgumentParser(description="Exporta subset BigEarthNet para Kaggle")
    ap.add_argument("--machine", default=None,
                    help="Perfil de máquina (por defecto $PARAVIT_MACHINE o el del host)")
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-val", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    selection = select_machine(args.machine, os.environ)
    if selection is None:
        raise SystemExit("ERROR: no hay perfil de máquina activo (use --machine o PARAVIT_MACHINE)")
    root, metadata = bigearthnet_paths(selection.profile)
    out = metadata.parent / f"kaggle_subset_{args.n_train}_{args.n_val}"
    out_data = out / "BigEarthNet-S2"
    out_data.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(metadata)
    train = df[df["split"] == "train"].sample(n=min(args.n_train, (df["split"] == "train").sum()),
                                              random_state=args.seed)
    val = df[df["split"] == "validation"].sample(n=min(args.n_val, (df["split"] == "validation").sum()),
                                                 random_state=args.seed)
    subset = pd.concat([train, val]).reset_index(drop=True)
    print(f"Subset: train={len(train)}  val={len(val)}  total={len(subset)}")

    copied, missing = 0, 0
    for i, patch_id in enumerate(subset["patch_id"].tolist(), 1):
        src_dir = _patch_dir(root, patch_id)
        dst_dir = _patch_dir(out_data, patch_id)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for band in RGB_BANDS:
            src_f = src_dir / f"{patch_id}_{band}.tif"
            if not src_f.exists():
                missing += 1
                continue
            shutil.copy2(src_f, dst_dir / src_f.name)
            copied += 1
        if i % 500 == 0:
            print(f"  {i}/{len(subset)} patches...")

    subset.to_parquet(out / "metadata_demo.parquet")
    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"\nHecho. {copied} ficheros copiados, {missing} ausentes.")
    print(f"Salida: {out}  ({size_mb:.0f} MB)")
    print(f"Comprimir:  cd {out.parent} && zip -r -q {out.name}.zip {out.name}")


if __name__ == "__main__":
    main()
