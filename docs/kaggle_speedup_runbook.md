# Kaggle 2×T4 — speedup real single-GPU vs DDP multi-GPU

Plan B del estudio distribuido: como Verode solo tiene **1 GPU usable**, el
speedup **positivo** (distribuido que *gana*) se mide en **Kaggle**, que ofrece
**2× Tesla T4 gratis** (30 h/semana). Mismo modelo (vit_tiny), mismo subset
(5000/1500), mismos epochs (3) que el estudio de Verode → comparable.

Resultado esperado: **~1.5-1.9×** de speedup con 2 GPUs, que se contrasta con:
- la predicción del `DDPOptimizer` del feasibility, y
- el resultado *negativo* del heterogéneo V100+CPU de Verode (distribuido más lento).

---

## 0. Preparar el subset (en local, una vez)

```bash
uv run python scripts/export_kaggle_subset.py \
  --metadata /media/alejandro/SSD/datasets/bigearthnet/metadata.parquet \
  --root /media/alejandro/SSD/datasets/bigearthnet/BigEarthNet-S2 \
  --out ~/kaggle_bigearthnet_demo --n-train 5000 --n-val 1500

cd ~ && zip -r -q kaggle_bigearthnet_demo.zip kaggle_bigearthnet_demo
```

Sube `kaggle_bigearthnet_demo.zip` como **Kaggle Dataset** (kaggle.com → Datasets
→ New Dataset → arrastra el zip; Kaggle lo descomprime solo).

## 1. Crear el notebook

- kaggle.com → Code → New Notebook.
- Panel derecho → **Accelerator: GPU T4 ×2**.
- **Internet: ON** (para `git clone` + `pip install`).
- Add Input → tu dataset `kaggle_bigearthnet_demo`.

## 2. Celdas (pega cada bloque en una celda y ejecuta en orden)

### Celda 1 — repo + dependencias
```python
!git clone https://github.com/alerguezrojas/tfg-distributed-transformers.git
%cd tfg-distributed-transformers
!pip -q install timm rasterio pyarrow torchinfo nvidia-ml-py
import torch
print("GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
```

### Celda 2 — autodescubrir el dataset y generar los configs
```python
import glob, os, yaml
meta = glob.glob('/kaggle/input/**/metadata_demo.parquet', recursive=True)[0]
root = os.path.join(os.path.dirname(meta), 'BigEarthNet-S2')
print("metadata:", meta); print("root:", root, "existe:", os.path.isdir(root))

base = dict(
    data=dict(root=root, metadata=meta, num_workers=4),
    model=dict(name="vit_tiny_patch16_224", pretrained=True, num_classes=19, dropout=0.1),
    training=dict(epochs=3, lr=0.0001, lr_min=0.000001, weight_decay=0.05,
                  warmup_epochs=1, llrd_decay=0.75, grad_clip=1.0,
                  label_smoothing=0.0, mixup_alpha=0.0,
                  early_stopping_patience=10, log_batch_every=5),
    checkpoint=dict(dir="checkpoints/kaggle"),
    output=dict(env="kaggle"),
)

single = {**base, "training": {**base["training"], "batch_size": 96}}
ddp = {**base, "training": {**base["training"], "batch_size": 48},
       "distributed": {"backend": "nccl"}}   # 48×2 = 96 global

yaml.safe_dump(single, open("configs/_kaggle_single.yaml", "w"))
yaml.safe_dump(ddp, open("configs/_kaggle_ddp.yaml", "w"))
print("configs escritos")
```

### Celda 3 — baseline SINGLE-GPU (1 T4)
```python
# Forzamos 1 sola GPU visible para la baseline
!CUDA_VISIBLE_DEVICES=0 python scripts/train_single_gpu.py \
    --config configs/_kaggle_single.yaml \
    --trace simple --layers confusion batch-monitor --fn energy timing
```

### Celda 4 — DDP en 2×T4
```python
!torchrun --nproc_per_node=2 scripts/train_ddp.py \
    --config configs/_kaggle_ddp.yaml \
    --trace simple --layers confusion batch-monitor --fn energy timing
```

### Celda 5 — speedup rápido + empaquetar resultados
```python
import pandas as pd, glob, shutil
s = pd.read_csv(sorted(glob.glob("logs/kaggle/single/**/epoch_metrics_*.csv", recursive=True))[-1])
d = pd.read_csv(sorted(glob.glob("logs/kaggle/ddp/**/epoch_metrics_*.csv", recursive=True))[-1])
ts, td = s["epoch_time_s"].mean(), d["epoch_time_s"].mean()
print(f"Single  epoch medio: {ts:6.1f}s  | Val F1 {s['val_f1'].max():.4f}")
print(f"DDP 2GPU epoch medio: {td:6.1f}s  | Val F1 {d['val_f1'].max():.4f}")
print(f"SPEEDUP: {ts/td:.2f}×   (eficiencia {ts/td/2*100:.0f}%)")

shutil.make_archive("/kaggle/working/kaggle_results", "zip", "logs/kaggle")
print("Descarga /kaggle/working/kaggle_results.zip desde el panel Output")
```

## 3. Traer los resultados al repo

Descarga `kaggle_results.zip` (panel **Output** del notebook), y en local:

```bash
unzip -o ~/Descargas/kaggle_results.zip -d logs/kaggle/
git add logs/kaggle/
git commit -m "data: Kaggle 2xT4 speedup vit_tiny (single vs DDP)"
git push origin main
```

En la web → pestaña **Análisis DDP** aparecerán los runs `kaggle/single` y
`kaggle/ddp` → speedup positivo, y junto a ellos el `verode/ddp_hetero`
(negativo). La comparación completa del TFG en una sola pantalla.

## Notas

- Si una T4 se queda sin memoria, baja `batch_size` (p.ej. single 64 / ddp 32).
- T4 no tiene NVLink → la comunicación va por PCIe; por eso la eficiencia no
  llega al 100% (esperable ~75-90% con vit_tiny). El feasibility lo predice.
- El pretrained de timm requiere internet (ya está ON).
