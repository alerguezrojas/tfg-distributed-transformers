"""Genera un metadata.parquet reducido para smoke tests del DDP heterogéneo.

Toma N patches de train y M de validation del metadata real (aleatorio con
seed fijo) para que un test multi-nodo complete en pocos minutos en vez de
horas. Los patches del subset existen en el disco (son filas del metadata real).

Uso:
    uv run python scripts/make_smoke_subset.py \
        --metadata ~/datasets/bigearthnet/metadata.parquet \
        --out ~/datasets/bigearthnet/metadata_smoke.parquet \
        --n-train 1600 --n-val 400
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Crea un metadata reducido para smoke tests")
    ap.add_argument("--metadata", required=True, help="Ruta al metadata.parquet real")
    ap.add_argument("--out", required=True, help="Ruta de salida del metadata reducido")
    ap.add_argument("--n-train", type=int, default=1600)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.metadata)
    if not src.exists():
        raise SystemExit(f"No existe el metadata: {src}")

    df = pd.read_parquet(src)
    print(f"Metadata original: {len(df):,} filas | splits: {df['split'].value_counts().to_dict()}")

    train = df[df["split"] == "train"]
    val = df[df["split"] == "validation"]

    n_train = min(args.n_train, len(train))
    n_val = min(args.n_val, len(val))

    sub_train = train.sample(n=n_train, random_state=args.seed)
    sub_val = val.sample(n=n_val, random_state=args.seed)
    subset = pd.concat([sub_train, sub_val]).reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(out)
    print(f"Subset escrito en {out}")
    print(f"  train={n_train}  validation={n_val}  total={len(subset)}")


if __name__ == "__main__":
    main()
