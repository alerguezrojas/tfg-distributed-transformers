# CLAUDE.md — tfg-distributed-transformers

Contexto completo del proyecto para continuar el trabajo en cualquier máquina.

---

## Sobre el proyecto

**TFG:** "Entrenamiento distribuido de modelos abiertos de aprendizaje automático basados en Transformers"
**Tutor:** Paco Almeida (Universidad de La Laguna)
**Alumno:** Alejandro Rodríguez Rojas
**Entrega:** junio/julio 2026
**Repo:** https://github.com/alerguezrojas/tfg-distributed-transformers

El objetivo es demostrar la aplicación de principios SOLID y patrones de diseño (especialmente Decorator) al ciclo de entrenamiento de un ViT sobre BigEarthNet-S2, y escalar a entrenamiento distribuido con PyTorch DDP.

---

## Hardware

### Local (desarrollo)
- **GPU:** NVIDIA RTX 3060 Ti (8 GB VRAM)
- **Driver:** nvidia-driver-580-open, kernel 6.8
- **Dataset:** SSD externo montado en `/media/alejandro/SSD/` (ext4, ~120 GB)
- **Gestión de paquetes:** `uv` (ver sección de dependencias)

### Clúster VERODE (ULL) — entrenamiento
- **Login:** `ssh alu0101317038@verode00.pcg.ull.es`
- **Nodos de cómputo:** verode[16-21] (5 nodos)
- **GPU:** Tesla V100-PCIE, **32 GB VRAM** por nodo
- **CUDA:** 12.0, Driver 525.147.05
- **CPUs por nodo:** 16
- **RAM por nodo:** ~112 GB
- **Sistema de colas:** Slurm 20.11.04
- **Almacenamiento:** `/home/bejeque/alu0101317038/` (NFS, ~453 GB libres)

#### Configuración del clúster (hacer en cada sesión SSH)
```bash
module add slurm/client/20.11.04   # o añadir al ~/.bashrc
```

#### Entorno en el clúster
- **Miniconda:** instalado en `~/miniconda3`, `auto_activate_base=false`
- **zstd:** instalado via `conda install -c conda-forge zstd`
- **Python/PyTorch:** pendiente de instalar (ver sección siguiente)

#### Problemas conocidos del clúster
- `sbatch` falla con "I/O error writing script/environment to file" — bug de configuración del clúster
- Alternativa: usar `nohup comando &` para jobs largos en el login node
- `srun` funciona para jobs interactivos cortos

---

## Dataset: BigEarthNet-S2 v2.0

### En local (SSD)
- **Dataset:** `/media/alejandro/SSD/datasets/bigearthnet/BigEarthNet-S2/`
- **Metadata:** `/media/alejandro/SSD/datasets/bigearthnet/metadata.parquet`

### En el clúster VERODE
- **Dataset:** `~/datasets/bigearthnet/BigEarthNet-S2/` ✓ completo (549 488 patches verificados)
- **Metadata:** `~/datasets/bigearthnet/metadata.parquet` ✓ descargado
- **Archivo comprimido:** `~/datasets/bigearthnet/BigEarthNet-S2.tar.zst` ✓ guardado (63 GB)

#### Descarga (Zenodo record 10891137)
```bash
# Dataset principal (~59 GB comprimido, ~120 GB extraído)
nohup wget -c "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1" \
     -O ~/datasets/bigearthnet/BigEarthNet-S2.tar.zst >> ~/logs/download.log 2>&1 &

# Metadata (3.4 MB, segundos)
wget -c "https://zenodo.org/records/10891137/files/metadata.parquet?download=1" \
     -O ~/datasets/bigearthnet/metadata.parquet
```

#### Extracción
```bash
# OJO: usar ruta absoluta a zstd — el PATH del login node no incluye conda
nohup tar --use-compress-program=/home/bejeque/alu0101317038/miniconda3/bin/zstd \
    -xf ~/datasets/bigearthnet/BigEarthNet-S2.tar.zst \
    -C ~/datasets/bigearthnet/ >> ~/logs/extract.log 2>&1 &
# Resultado: ~/datasets/bigearthnet/BigEarthNet-S2/scene_id/patch_id/*.tif
```

#### Verificar dataset
```bash
# Contar patches en disco (debe coincidir con local: 549 488)
find ~/datasets/bigearthnet/BigEarthNet-S2/ -mindepth 2 -maxdepth 2 -type d | wc -l
```

### Estructura y descripción
- **Estructura de directorios:** `root/scene_id/patch_id/*.tif`
  - `scene_id` = `patch_id` sin los dos últimos segmentos (`_row_col`)
- **Tamaño:** 480 038 patches — Train: 237 871 | Val: 122 342 | Test: 119 825
- **Bandas usadas:** B04, B03, B02 (proxy RGB de Sentinel-2)
- **Escala:** reflectancia / 10 000, clipped [0, 1]
- **Tarea:** clasificación multi-label, 19 clases CORINE Land Cover
- **Pérdida:** `BCEWithLogitsLoss` (sin sigmoid en el modelo)
- **Métricas:** macro F1 + sample-averaged accuracy + precision + recall

---

## Modelo

- `vit_base_patch16_224` de **timm**, pretrained ImageNet
- 85 813 267 parámetros, embed_dim = 768
- Cabeza personalizada: `Dropout(0.1) → Linear(768, 19)`
- Forward devuelve logits crudos (sin sigmoid)
- Fichero: `src/models/vit.py` — clase `BigEarthViT`, función `build_model()`

---

## Stack de entrenamiento (arquitectura Decorator)

### Principio de diseño

El patrón **Decorator OOP** se aplica al ciclo de entrenamiento. Todos los decoradores extienden `TrainerDecorator`, que usa `__getattr__` para delegar transparentemente cualquier atributo al trainer envuelto (model, optimizer, device…). Esto evita duplicar propiedades en cada decorador.

### Jerarquía de clases

```
BaseTrainer (ABC)
└── Trainer                      # lógica pura, sin prints
└── TrainerDecorator             # base de todos los decoradores
    ├── MetricsLoggerDecorator   # nivel 1: epoch-level, prints simples + ETA
    ├── BatchMetricsDecorator    # nivel 2: tqdm por batch (solo didáctico)
    ├── LayerHooksDecorator      # nivel 3: forward hooks en Linear layers (solo didáctico)
    ├── TracingDecorator         # nivel 4: logging estructurado a fichero + ETA
    └── DeepTracingDecorator     # nivel 5: trazado máximo (fichero aparte)
```

**Nota:** `TensorBoardDecorator` fue eliminado. `BatchMetricsDecorator` y `LayerHooksDecorator` se mantienen solo por valor didáctico del TFG, no se usan en producción.

Ficheros:
- `src/training/base_trainer.py` — contrato abstracto
- `src/training/trainer.py` — implementación pura
- `src/training/trainer_decorators.py` — decoradores niveles 1–4
- `src/training/deep_tracing.py` — decorador nivel 5 (máxima profundidad)
- `src/training/logger_setup.py` — `setup_logger()` con formato timestamp
- `src/training/python_decorators.py` — decoradores Python `@` (contraste didáctico)

### Métricas disponibles

`Trainer.train_epoch` devuelve: `loss`, `f1`, `accuracy`, `time`
`Trainer.eval_epoch` devuelve: `loss`, `f1`, `accuracy`, `precision`, `recall`

### DeepTracingDecorator (nivel más profundo)

Registra:
- **Forward hooks** en todos los módulos hoja → `act_mean`, `act_std`, `act_max`, `dead_ratio`
- **Backward hooks** (`register_full_backward_hook`) → `grad_norm`, `grad_max`, `vanishing`, `exploding`
- **Parameter hooks** (`param.register_hook`) → `weight_norm`, `grad_norm`, `update_ratio`
- **GPU memory** (`torch.cuda.memory_allocated`) por step
- **Learning rate** por grupo del optimizer
- **torchinfo** summary al inicio (ejecutado en CPU para no fragmentar VRAM)
- **Tabla por bloque**: patch_embed + `attn.proj` de cada uno de los 12 bloques + head (14 puntos)
- **Alertas de anomalías**: neuronas muertas, gradiente explosivo/evanescente, update ratio anómalo

Todos los tensores se calculan en GPU con `.detach().float()` y solo se transfiere el escalar final con `.item()`.

### Script de entrenamiento

`scripts/train_single_gpu.py` — flag `--trace` con tres modos:

```bash
# --trace off   → MetricsLoggerDecorator  (sin hooks, máxima velocidad)
# --trace simple → TracingDecorator        (timestamps + log a fichero)
# --trace deep   → DeepTracingDecorator    (trazado completo por capa)
```

Log con timestamp: `logs/train_YYYYMMDD_HHMMSS.log`

---

## Feasibility Checker

`scripts/check_feasibility.py` — análisis de viabilidad previo al entrenamiento.

Usa datos sintéticos (sin tocar el dataset) para medir throughput real y estimar tiempos.

Arquitectura (patrón Facade + SRP):
- `ModelAnalyzer` — FLOPs, parámetros, memoria estática
- `HardwareProbe` — VRAM disponible
- `Benchmarker` — mide throughput real por (batch_size, trace_mode)
- `TimeEstimator` — convierte tiempos en estimaciones
- `ReportFormatter` — imprime el informe
- `FeasibilityChecker` — Facade que coordina todo

```bash
# Config por defecto
uv run python scripts/check_feasibility.py

# Comparar batch sizes y epochs
uv run python scripts/check_feasibility.py --batch-sizes 16 32 64 128 --epochs 30

# Solo algunos modos
uv run python scripts/check_feasibility.py --batch-sizes 32 64 --trace-modes off deep
```

**Resultado conocido en RTX 3060 Ti (local):**
- batch_size=32 óptimo: ~65 imgs/s, 4.95 GB VRAM
- batch_size=64 OOM (necesita ~11.5 GB)
- `--trace deep` añade ~22% overhead vs off

**En V100 32 GB (clúster):** pendiente de ejecutar — con 32 GB VRAM batch_size=64/128 deberían funcionar.

---

## Configuración (`configs/train.yaml`)

```yaml
data:
  root: "/media/alejandro/SSD/datasets/bigearthnet/BigEarthNet-S2"
  metadata: "/media/alejandro/SSD/datasets/bigearthnet/metadata.parquet"
  num_workers: 4

model:
  name: "vit_base_patch16_224"
  pretrained: true
  num_classes: 19

training:
  epochs: 30
  batch_size: 32          # 64 OOM en RTX 3060 Ti (8 GB); en V100 puede subirse
  lr: 0.0001              # OJO: no usar 1e-4, se parsea como string en YAML
  weight_decay: 0.0001
  log_batch_every: 50     # DeepTracingDecorator: tabla de capas cada N batches

checkpoint:
  dir: "checkpoints/single_gpu"
```

**Para el clúster**, sobreescribir las rutas de datos:
```bash
python scripts/train_single_gpu.py \
  --data-root ~/datasets/bigearthnet/BigEarthNet-S2 \
  --config configs/train.yaml \
  --trace simple
```
*(o crear `configs/train_cluster.yaml` con las rutas del clúster)*

**Notas:**
- `1e-4` en YAML se parsea como string. Usar siempre `0.0001`.
- En el clúster con V100 32 GB, ejecutar el feasibility checker para calibrar batch_size óptimo.

---

## Gestión de dependencias

### Local (uv)
```bash
uv sync                                          # instalar entorno
uv run python scripts/train_single_gpu.py ...   # ejecutar
uv add <paquete>                                 # añadir dependencia
```

### Clúster (conda — pendiente de configurar)
```bash
# Crear entorno con Python 3.12 y PyTorch CUDA 12.0
conda create -n tfg python=3.12 -y
conda activate tfg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm torchinfo tqdm pyyaml rasterio pandas pyarrow tensorboard

# Ejecutar
conda activate tfg
python scripts/train_single_gpu.py --data-root ~/datasets/bigearthnet/BigEarthNet-S2 --trace simple
```

Dependencias principales: `torch`, `timm`, `torchvision`, `torchinfo`, `tqdm`, `tensorboard`, `rasterio`, `pandas`, `pyarrow`, `pyyaml`

---

## Git workflow

```
main ← develop ← feature/xxx
```

- Siempre crear feature branch desde `develop`
- PRs: feature → develop → main
- Rama actual: `main` (todo mergeado)
- **No añadir Co-Authored-By en los commits**

---

## Estado actual del proyecto

### Completado
- [x] Pipeline de datos: `BigEarthNetDataset` con metadata.parquet
- [x] Modelo: `BigEarthViT` (ViT + cabeza multi-label)
- [x] Entrenamiento single-GPU: `Trainer` + `Scheduler` + checkpoints
- [x] Patrón Decorator completo (niveles 1–4 en `trainer_decorators.py`)
- [x] Decoradores Python `@` (`@timed`, `@log_call`, `@retry_on_cuda_oom`)
- [x] `DeepTracingDecorator` con trazado a nivel neurona/capa (14 puntos: patch_embed + 12 bloques + head)
- [x] Métricas completas: train F1/acc, val F1/acc/precision/recall, best F1, ETA
- [x] Flag `--trace off/simple/deep` en script de entrenamiento
- [x] Log con timestamp a fichero
- [x] `check_feasibility.py` con benchmark, estimaciones y análisis de memoria
- [x] Acceso al clúster VERODE (ULL) con V100 32 GB
- [x] Miniconda + zstd instalados en el clúster
- [x] Dataset descargado y extraído en el clúster (549 488 patches, verificado contra local)
- [x] metadata.parquet descargado en el clúster

### Pendiente inmediato (clúster)
- [ ] Crear entorno conda con PyTorch + CUDA 12.0 en el clúster
- [ ] Ejecutar `check_feasibility.py` en el clúster para calibrar batch_size en V100
- [ ] Adaptar `configs/train.yaml` o crear `configs/train_cluster.yaml` con rutas del clúster
- [ ] Lanzar entrenamiento completo 30 epochs en el clúster

### Pendiente futuro
- [ ] Implementar entrenamiento distribuido (PyTorch DDP) con múltiples V100
- [ ] Proyección multi-GPU en feasibility checker
- [ ] Visualización de attention maps

---

## Errores conocidos y soluciones

| Error | Causa | Solución |
|-------|-------|----------|
| `lr '<=' not supported between float and str` | `1e-4` en YAML se parsea como string | Usar `0.0001` en el YAML |
| `property 'model' has no setter` | Intentar poner `model` como `@property` abstracta en BaseTrainer | No declarar propiedades en BaseTrainer; usar `__getattr__` en TrainerDecorator |
| `CUDA out of memory` con batch_size=64 (local) | ViT-B necesita ~11.5 GB para batch 64 | Usar batch_size=32 en local (4.95 GB) |
| `CUDA out of memory` en hooks | `.float()` en GPU de tensores grandes | Calcular en GPU con `.detach().float()`, transferir solo el escalar con `.item()` |
| Hooks muy lentos (6 GB/batch transferidos) | `.detach().cpu()` copia tensor entero a RAM | Usar `.detach().float().mean().item()` — solo transfiere 4 bytes |
| `nvidia driver` no funciona con kernel 6.8 | Driver 470 incompatible | Actualizar a `nvidia-driver-580-open` |
| `sbatch` falla en clúster VERODE | Bug de configuración de Slurm (I/O error spool) | Usar `nohup comando &` para background, `srun` para interactivo |

---

## Comandos útiles

### Local
```bash
# Análisis de viabilidad previo
uv run python scripts/check_feasibility.py --batch-sizes 16 32 64 --epochs 30

# Entrenamiento rápido (test, 1 epoch)
uv run python scripts/train_single_gpu.py --epochs 1 --batch-size 32 --trace deep

# Entrenamiento completo
uv run python scripts/train_single_gpu.py --config configs/train.yaml --batch-size 32 --trace simple

# Ver logs en tiempo real
tail -f logs/train_*.log
```

### Clúster VERODE
```bash
# Conectar
ssh alu0101317038@verode00.pcg.ull.es

# Cargar Slurm (o añadir al ~/.bashrc)
module add slurm/client/20.11.04

# Ver estado del clúster
sinfo -s
squeue -u $USER

# Verificar descarga en curso
ps aux | grep wget
tail ~/logs/download.log

# Extracción del dataset (cuando termine la descarga)
tar --use-compress-program=zstd -xf ~/datasets/bigearthnet/BigEarthNet-S2.tar.zst \
    -C ~/datasets/bigearthnet/

# Verificar estructura del dataset
ls ~/datasets/bigearthnet/
ls ~/datasets/bigearthnet/BigEarthNet-S2/ | head -5

# Activar entorno conda (cuando esté creado)
conda activate tfg

# Lanzar entrenamiento en el clúster
nohup python scripts/train_single_gpu.py \
  --data-root ~/datasets/bigearthnet/BigEarthNet-S2 \
  --config configs/train.yaml \
  --trace simple >> ~/logs/train_cluster.log 2>&1 &
```
