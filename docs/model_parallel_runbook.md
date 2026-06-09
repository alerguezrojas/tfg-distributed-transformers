# Kaggle 2×T4 — paralelismo de MODELO (pipeline) vs paralelismo de datos

Tercera estrategia de distribución del estudio. Las dos primeras reparten los
**datos** (DDP homogéneo NCCL y heterogéneo GPU+CPU). Aquí se reparte el
**modelo**: los 12 bloques del ViT-base se parten **6/6** entre las 2 Tesla T4
(stage 0 en `cuda:0`, stage 1 en `cuda:1`), en **un solo proceso**.

Es paralelismo de modelo *naive* (sin pipelining de micro-batches): mientras la
stage 1 calcula, la stage 0 está ociosa. En un modelo que **cabe** en una sola
GPU (el ViT-base entra en 8 GB) esto es **más lento** que el de datos y sin
ahorro de memoria. El valor es **didáctico**: demuestra el mecanismo y por qué
el paralelismo de modelo se reserva para modelos que no caben en una GPU (con
pipelining/GPipe se recuperaría la utilización). Es el contrapunto esperado al
DDP, que sí gana cuando es compute-bound (vit_base 1.90× en estas mismas T4).

Validación local antes de Kaggle: el forward model-parallel es **numéricamente
equivalente** al modelo normal (`tests/unit/test_model_parallel.py`, en CPU), y
el bucle de entrenamiento se ha probado cruzando una frontera de dispositivo
real (`cuda:0`→`cpu`). En Kaggle el único cambio es `cpu`→`cuda:1`.

> Requiere **timm 1.x** (el forward usa `patch_embed` / `_pos_embed` /
> `forward_head` de la `VisionTransformer` de timm). Kaggle suele traer timm 1.x;
> la celda 1 lo fija por si acaso.

---

## 0. Dataset

Reutiliza el mismo subset que el estudio de speedup
(`docs/kaggle_speedup_runbook.md`, paso 0): el Kaggle Dataset
`kaggle_bigearthnet_demo` con `metadata_demo.parquet` (5000/1500) +
`BigEarthNet-S2/`. Si ya lo subiste, no hay que repetirlo.

## 1. Notebook

- kaggle.com → Code → New Notebook.
- **Accelerator: GPU T4 ×2**  ·  **Internet: ON**.
- Add Input → `kaggle_bigearthnet_demo`.

## 2. Celdas

### Celda 1 — repo + dependencias
```python
# Repo privado → token de GitHub (fine-grained, Contents:Read) en Kaggle Secrets como GH_TOKEN.
from kaggle_secrets import UserSecretsClient
tok = UserSecretsClient().get_secret("GH_TOKEN")
!git clone https://x-access-token:{tok}@github.com/alerguezrojas/tfg-distributed-transformers.git
%cd tfg-distributed-transformers
!git checkout develop          # o main, según dónde esté integrado
!pip -q install "timm>=1.0" rasterio pyarrow torchinfo nvidia-ml-py
import torch
print("GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
```

### Celda 2 — generar el config apuntando al dataset de Kaggle
```python
import glob, os, yaml
meta = glob.glob('/kaggle/input/**/metadata_demo.parquet', recursive=True)[0]
root = os.path.join(os.path.dirname(meta), 'BigEarthNet-S2')
print("metadata:", meta, "| root existe:", os.path.isdir(root))

cfg = dict(
    data=dict(root=root, metadata=meta, num_workers=4),
    model=dict(name="vit_base_patch16_224", pretrained=True, num_classes=19, dropout=0.1),
    training=dict(epochs=3, batch_size=96, lr=0.0001, weight_decay=0.05,
                  warmup_epochs=1, llrd_decay=0.75, grad_clip=1.0),
    output=dict(env="kaggle"),
)
os.makedirs("configs", exist_ok=True)
with open("configs/_mp_kaggle.yaml", "w") as f:
    yaml.safe_dump(cfg, f)
```

### Celda 3 — entrenamiento model-parallel (1 proceso, 2 GPUs)
```python
# NO se usa torchrun: es un único proceso que parte el modelo entre cuda:0 y cuda:1.
!python scripts/train_model_parallel.py \
    --config configs/_mp_kaggle.yaml --devices cuda:0,cuda:1 --trace simple
```

### Celda 4 — (opcional) baseline single-GPU comparable
```python
# Mismo modelo/subset/epochs en 1 sola T4 para la comparación de tiempos.
cfg_single = dict(cfg); cfg_single["model"] = dict(cfg["model"])
with open("configs/_single_kaggle.yaml","w") as f: yaml.safe_dump(cfg_single, f)
!CUDA_VISIBLE_DEVICES=0 python scripts/train_single_gpu.py \
    --config configs/_single_kaggle.yaml --trace simple --layers confusion --fn timing
```

### Celda 5 — empaquetar los artefactos para bajarlos
```python
!cd logs && zip -r -q /kaggle/working/logs_model_parallel.zip kaggle/model_parallel kaggle/single 2>/dev/null
print("Descarga logs_model_parallel.zip desde el panel Output y descomprímelo en logs/ del repo local.")
```

## 3. En local: integrar y visualizar

```bash
unzip -o ~/Downloads/logs_model_parallel.zip -d logs/
uv run streamlit run src/web/app.py
```

El run aparece como `model_parallel` y se puede **superponer** con el single y el
DDP en **Comparison → Overlay runs** (curvas F1/loss + tiempo por epoch). Lo
esperado: F1 casi idéntico a los demás (la sincronización es matemáticamente
exacta) y **tiempo/epoch mayor** que el single → confirma que el paralelismo de
modelo *naive* no acelera un modelo que ya cabe en una GPU.

## 4. Para la memoria

Tabla de estrategias de distribución sobre el mismo subset (5000/1500, 3 epochs):

| Estrategia | Qué reparte | Procesos | Resultado esperado |
|---|---|---|---|
| Single-GPU | — | 1 | baseline |
| DDP (datos, NCCL) | datos | 2 | **gana** si compute-bound (vit_base 1.90×) |
| Heterogéneo (datos, gloo) | datos (GPU+CPU) | 2 | penaliza (0.12×, nodo lento manda) |
| **Model-parallel (pipeline)** | **modelo** | **1** | **más lento** aquí; útil solo si el modelo no cabe en 1 GPU |
