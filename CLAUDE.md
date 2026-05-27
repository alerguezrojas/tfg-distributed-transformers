# CLAUDE.md — tfg-distributed-transformers

Contexto completo del proyecto para continuar el trabajo en cualquier máquina.

---

## Sobre el proyecto

**TFG:** "Entrenamiento distribuido de modelos abiertos de aprendizaje automático basados en Transformers"
**Tutor:** Paco Almeida (Universidad de La Laguna)
**Alumno:** Alejandro Rodríguez Rojas
**Entrega:** junio/julio 2026
**Repo:** https://github.com/alerguezrojas/tfg-distributed-transformers

El objetivo es demostrar la aplicación de principios SOLID y patrones de diseño (Decorator + Template Method) al ciclo de entrenamiento de un ViT sobre BigEarthNet-S2, y escalar a entrenamiento distribuido con PyTorch DDP.

---

## Hardware

### Local (desarrollo)
- **GPU:** NVIDIA RTX 3060 Ti (8 GB VRAM)
- **Driver:** nvidia-driver-580-open, kernel 6.8
- **Dataset:** SSD externo montado en `/media/alejandro/SSD/` (ext4, ~120 GB)
- **Gestión de paquetes:** `uv`

### Clúster VERODE (ULL) — entrenamiento
- **Login:** `ssh alu0101317038@verode00.pcg.ull.es`
- **Nodos de cómputo:** verode[16-21] — hardware heterogéneo, solo verode21 es compatible con PyTorch 2.x
  - **verode16:** Tesla M2090 (2011, CC 2.0, 6 GB) — driver no activo + CC < 3.7 → incompatible
  - **verode18:** Tesla K40m (2013, CC 3.5, 11 GB) — driver 460/CUDA 11.2 + CC < 3.7 → incompatible
  - **verode21:** Tesla V100-PCIE (2017, CC 7.0, 32 GB) — operativo ✓
- **CUDA:** 12.0, Driver 525.147.05 (verode21)
- **CPUs por nodo:** 16
- **RAM por nodo:** ~112 GB
- **Sistema de colas:** Slurm 20.11.04
- **Almacenamiento:** `/home/bejeque/alu0101317038/` (NFS)

#### Configuración del clúster (hacer en cada sesión SSH)
```bash
module add slurm/client/20.11.04   # o añadir al ~/.bashrc
```

#### Entorno en el clúster
- **Miniconda:** instalado en `~/miniconda3`, `auto_activate_base=false` — solo para zstd
- **zstd:** instalado via conda (`~/miniconda3/bin/zstd`) — necesario para extraer el dataset
- **uv:** instalado en `~/.local/bin/uv`
- **Entorno Python:** `~/tfg-distributed-transformers/.venv` creado con `uv sync`
- **PyTorch:** `2.7.1+cu118` — instalado con cu118 para compatibilidad con driver 525 (CUDA 12.0 máx.)
  - ⚠️ `uv sync` instala cu13 por defecto → incompatible. Después de sync, ejecutar:
  ```bash
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
  ```
  - ⚠️ `python -m pip` no está disponible en el venv del clúster → usar siempre `uv pip` en su lugar

#### Problemas conocidos del clúster
- `sbatch` falla con "I/O error writing script/environment to file" — bug de configuración de Slurm, no reparable por el usuario
- Alternativa: usar `tmux` + `srun` para jobs que sobrevivan a desconexiones SSH
- El login node (`verode00`) no soporta instrucciones AVX2 → "Illegal instruction" al ejecutar Python con numpy/torch. Usar siempre nodo de cómputo (`srun`) para ejecutar código
- `srun` es el único mecanismo que funciona; ver sección de comandos

---

## Dataset: BigEarthNet-S2 v2.0

### En local (SSD)
- **Dataset:** `/media/alejandro/SSD/datasets/bigearthnet/BigEarthNet-S2/`
- **Metadata:** `/media/alejandro/SSD/datasets/bigearthnet/metadata.parquet`

### En el clúster VERODE
- **Dataset:** `~/datasets/bigearthnet/BigEarthNet-S2/` ✓ completo (549 488 patches verificados)
- **Metadata:** `~/datasets/bigearthnet/metadata.parquet` ✓
- **Archivo comprimido:** `~/datasets/bigearthnet/BigEarthNet-S2.tar.zst` ✓ guardado (63 GB)

#### Descarga (Zenodo record 10891137)
```bash
nohup wget -c "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1" \
     -O ~/datasets/bigearthnet/BigEarthNet-S2.tar.zst >> ~/logs/download.log 2>&1 &

wget -c "https://zenodo.org/records/10891137/files/metadata.parquet?download=1" \
     -O ~/datasets/bigearthnet/metadata.parquet
```

#### Extracción
```bash
# OJO: usar ruta absoluta a zstd — el PATH del login node no incluye conda
nohup tar --use-compress-program=/home/bejeque/alu0101317038/miniconda3/bin/zstd \
    -xf ~/datasets/bigearthnet/BigEarthNet-S2.tar.zst \
    -C ~/datasets/bigearthnet/ >> ~/logs/extract.log 2>&1 &
```

#### Verificar dataset
```bash
find ~/datasets/bigearthnet/BigEarthNet-S2/ -mindepth 2 -maxdepth 2 -type d | wc -l
# Debe dar 549 488
```

### Estructura y descripción
- **Directorios:** `root/scene_id/patch_id/*.tif` — `scene_id` = `patch_id` sin los dos últimos segmentos
- **Splits:** Train 237 871 | Val 122 342 | Test 119 825
- **Bandas:** B04, B03, B02 (proxy RGB), reflectancia / 10 000, clipped [0, 1]
- **Tarea:** clasificación multi-label, 19 clases CORINE Land Cover
- **Pérdida:** `BCEWithLogitsLoss` (sin sigmoid en el modelo)
- **Métricas:** macro F1 + sample-averaged accuracy + precision + recall

---

## Modelo

- `vit_base_patch16_224` de **timm**, pretrained ImageNet
- 85 813 267 parámetros, embed_dim = 768
- Cabeza personalizada: `Dropout(p) → Linear(768, 19)` — p=0.1 (configs v1/v2), p=0.3 (config v3)
- Forward devuelve logits crudos (sin sigmoid)
- Fichero: `src/models/vit.py` — clase `BigEarthViT`, función `build_model()`

---

## Arquitectura de decoradores

### Principio de diseño

Se combinan dos patrones de diseño:
- **Decorator (GoF):** capas que envuelven al trainer añadiendo comportamiento sin modificarlo
- **Template Method:** el bucle de entrenamiento se define UNA sola vez en `EpochController`; las subclases solo sobreescriben los hooks `_on_*`

Esto elimina la duplicación del bucle que tendría el Decorator puro.

### Jerarquía de clases

```
BaseTrainer (ABC)
├── Trainer                        # lógica pura: label smoothing, mixup, threshold search
└── TrainerDecorator               # base OOP: delega todos los métodos al trainer envuelto
    ├── LossReporter               # metric reporter: train_loss / val_loss
    ├── F1Reporter                 # metric reporter: train_f1 / val_f1
    ├── AccuracyReporter           # metric reporter: train_acc / val_acc
    ├── PrecisionRecallReporter    # metric reporter: val_precision / val_recall
    ├── PlottingDecorator          # aspecto: guarda curvas PNG tras cada epoch
    ├── LayerHooksDecorator        # aspecto: forward hooks en capas Linear
    ├── ConfusionMatrixDecorator   # aspecto: PNG de barras por clase + heatmap 19×19 normalizado tras cada eval
    ├── BatchMonitorDecorator      # aspecto: CSV con running loss por batch
    └── EpochController            # Template Method: define fit() con hooks _on_*
        └── TracingDecorator       # controlador: logging a consola y/o fichero
            └── DeepTracingDecorator  # controlador: hereda TracingDecorator + trazado profundo

TrainingSessionBuilder             # Builder fluent API: monta el stack completo
augmentations.mixup_batch()        # mezcla pares de batch con coef. Beta(α,α)
```

### Ficheros

```
src/training/
  base_trainer.py          # ABC con train_epoch, eval_epoch, save_checkpoint, fit
  trainer.py               # implementación pura, usa metrics.py; devuelve _preds/_labels en eval_epoch
  builder.py               # TrainingSessionBuilder — fluent API para montar el stack de decoradores
  metrics.py               # f1_score, precision, recall, accuracy, eta_str
  logger_setup.py          # setup_logger() con formato timestamp
  fn_decorators.py         # decoradores @ de Python: timed, log_call, measure_energy, retry_on_cuda_oom
  decorators/
    base.py                # TrainerDecorator + EpochController (con early stopping: patience)
    tracing.py             # TracingDecorator (consola o fichero según logger=)
    deep_tracing.py        # DeepTracingDecorator — features: set[str] para inspección modular
    plotting.py            # PlottingDecorator (aspecto, guarda PNG)
    layer_hooks.py         # LayerHooksDecorator (aspecto, forward hooks)
    confusion.py           # ConfusionMatrixDecorator — PNG barras por clase + heatmap 19×19 normalizado
    batch_monitor.py       # BatchMonitorDecorator — CSV con running loss por batch
    metric_reporters.py    # LossReporter, F1Reporter, AccuracyReporter, PrecisionRecallReporter
    __init__.py
```

### Tres tipos de decoradores

**Decoradores OOP (Patrón Decorator GoF)** — `decorators/`

Envuelven el objeto trainer completo. Hay tres subtipos:
- **Controladores** (`EpochController`): controlan el bucle; solo uno activo por ejecución
  - `TracingDecorator` — logging a consola (`logger=None`) o a fichero (`logger=Logger`); imprime marcador de epoch y ETA
  - `DeepTracingDecorator` — extiende `TracingDecorator`; añade hooks en cada capa y tabla por bloque del ViT
- **Aspecto** (`TrainerDecorator`): envuelven métodos concretos; combinables libremente
  - `PlottingDecorator` — acumula métricas y guarda PNG tras cada eval epoch; expone `_record_train_result()` para recibir métricas de train cuando `DeepTracingDecorator` gestiona el bucle directamente
  - `LayerHooksDecorator` — captura activaciones de capas Linear cada N epochs
  - `ConfusionMatrixDecorator` — dos PNGs por eval epoch: barras F1/prec/rec por clase + heatmap 19×19 normalizado (celda (i,j) = P(predice j | verdadero es i), diagonal = recall)
  - `BatchMonitorDecorator` — CSV con running loss cada N batches dentro del epoch
- **Metric reporters** (`TrainerDecorator`): cada uno imprime una métrica independiente; activables con `--metrics`
  - `LossReporter` — cachea train_loss en train_epoch, imprime train+val loss tras eval_epoch
  - `F1Reporter` — ídem para F1 macro
  - `AccuracyReporter` — ídem para accuracy
  - `PrecisionRecallReporter` — imprime val_precision y val_recall (sin equivalente en train)

**Decoradores `@` de Python** — `fn_decorators.py`

Envuelven funciones individuales, no objetos. Se aplican a métodos del trainer en tiempo de ejecución:
- `@timed` — tiempo de ejecución
- `@log_call` — traza de entrada/salida
- `@measure_energy` — muestrea potencia GPU en hilo de fondo, informa Julios/Wh
- `@retry_on_cuda_oom` — reintenta una vez tras liberar caché CUDA en OOM

Todos rutean a `logging.getLogger("trainer")` cuando hay fichero de log activo; caen a `print()` en modo `--trace off`.

**Técnicas de regularización en `Trainer`** (v3)
- **Label smoothing** (`label_smoothing: float`) — suaviza targets: 0→ls/2, 1→1-ls/2
- **Mixup** (`mixup_alpha: float`) — mezcla pares del batch con λ ~ Beta(α,α); 50% prob por batch
- **Threshold search** — tras cada `eval_epoch`, busca en [0.30…0.60] el threshold que maximiza F1 macro; reportado en log como `threshold óptimo`; no afecta al criterio de checkpoint (siempre threshold=0.5 para consistencia)

### Stack resultante

```
TracingDecorator / DeepTracingDecorator   ← controlador (--trace)
  └── PrecisionRecallReporter             ← metric reporter (--metrics, solo off/simple)
        └── AccuracyReporter
              └── F1Reporter
                    └── LossReporter
                          └── PlottingDecorator       ← aspecto (--layers plot)
                                └── LayerHooksDecorator  ← aspecto (--layers hooks)
                                      └── Trainer
                                            train_epoch = measure_energy(timed(fn))  ← --fn
```

**Nota sobre `--trace deep`:** `DeepTracingDecorator.train_epoch` gestiona el bucle de entrenamiento directamente (necesario para las tablas por batch). Esto significa:
- Los metric reporters y `--metrics` se ignoran (deep gestiona sus propias métricas en `_on_epoch_end`)
- `LayerHooksDecorator` no activa (deep registra sus propios hooks más completos)
- `@fn` en `train_epoch` no dispara; sí dispara en `eval_epoch`
- `PlottingDecorator` recibe métricas de train vía `_propagate_train_result()` al final de cada epoch

### DeepTracingDecorator — detalle

Registra en cada epoch:
- **Forward hooks** en todos los módulos hoja → `act_mean`, `act_std`, `act_max`, `dead_ratio`
- **Backward hooks** → `grad_norm`, `grad_max`, `vanishing`, `exploding`
- **Param hooks** → `weight_norm`, `grad_norm`, `update_ratio` (healthy: 1e-4 – 1.0)
- **GPU memory** por batch
- **Learning rate** por grupo del optimizer
- **torchinfo** summary al inicio
- **Tabla por bloque**: patch_embed + `attn.proj` de 12 bloques + head = 14 puntos
- **Alertas**: neuronas muertas, gradiente explosivo/evanescente, update ratio anómalo

Todos los tensores se calculan en GPU con `.detach().float()`, solo se transfiere el escalar con `.item()`.

---

## Script de entrenamiento

`scripts/train_single_gpu.py`

### Flags

```
--trace off|simple|deep    Controlador OOP:
                             off    → TracingDecorator sin fichero (solo consola)
                             simple → TracingDecorator con log a logs/train_FECHA.log
                             deep   → DeepTracingDecorator con log a logs/train_deep_FECHA.log

--model NAME               Override del modelo timm (default: cfg model.name)
                             Ejemplos: vit_tiny_patch16_224, resnet50, efficientnet_b0
                             Si el modelo no es ViT/DeiT/Swin → AdamW estándar (sin LLRD)

--layers [plot] [hooks] [confusion] [batch-monitor]
                           Decoradores de aspecto (combinables):
                             plot         → PlottingDecorator, PNG en plots/training_FECHA.png
                             hooks        → LayerHooksDecorator, activaciones cada 5 epochs
                             confusion    → ConfusionMatrixDecorator, PNG barras + heatmap 19×19 por clase tras cada eval
                             batch-monitor → BatchMonitorDecorator, CSV con loss por batch

--fn [timing] [energy]     Decoradores @ de Python (combinables):
                             timing → @timed en train_epoch y eval_epoch
                             energy → @measure_energy en train_epoch y eval_epoch
                             (con --trace deep solo aplica a eval_epoch)

--metrics [loss] [f1] [accuracy] [precision_recall]
                           Metric reporters individuales (solo para --trace off/simple):
                             sin args (--metrics) → desactiva todos
                             por defecto → todos activos

--inspect [model-summary] [batch-table] [grad-monitor] [anomalies]
                           Inspección modular con DeepTracingDecorator (combinable con --trace simple):
                             model-summary → torchinfo al inicio
                             batch-table  → tabla de capas cada N batches
                             grad-monitor → backward hooks en todos los módulos hoja
                             anomalies    → alertas de neuronas muertas / gradientes
                           Si se pasa --inspect, activa DeepTracingDecorator automáticamente.
                           --trace deep equivale a --inspect model-summary batch-table grad-monitor anomalies
```

### Ejemplos

```bash
# Solo consola
uv run python scripts/train_single_gpu.py --trace off

# Log a fichero + gráficas + confusion matrix
uv run python scripts/train_single_gpu.py --trace simple --layers plot confusion

# Solo F1 y loss en pantalla
uv run python scripts/train_single_gpu.py --trace simple --metrics loss f1

# Modelo pequeño para test rápido (~10x más rápido que vit_base)
uv run python scripts/train_single_gpu.py --model vit_tiny_patch16_224 --epochs 1 --trace simple

# Inspección modular: solo summary del modelo + monitor de gradientes
uv run python scripts/train_single_gpu.py --trace simple --inspect model-summary grad-monitor anomalies

# Batch monitor + gráficas
uv run python scripts/train_single_gpu.py --trace simple --layers plot batch-monitor

# Test rápido 1 epoch con todo activo (modelo pequeño)
uv run python scripts/train_single_gpu.py --model vit_tiny_patch16_224 --epochs 1 \
  --trace simple --layers plot confusion batch-monitor \
  --inspect model-summary grad-monitor anomalies

# Trazado profundo (equivalente al --inspect completo)
uv run python scripts/train_single_gpu.py --trace deep --layers plot --fn energy

# Con config del clúster
uv run python scripts/train_single_gpu.py --config configs/train_cluster.yaml --trace simple
```

---

## Entrenamiento distribuido (DDP)

`scripts/train_ddp.py` — punto de entrada para `torchrun` (PyTorch DistributedDataParallel).

### Arquitectura DDP

- **`src/training/ddp_trainer.py`**: `DDPTrainer(Trainer)` — subclase mínima que sobreescribe tres métodos:
  - `train_epoch`: llama `sampler.set_epoch(epoch)` para shuffle correcto
  - `eval_epoch`: reúne predicciones de todos los procesos con `dist.all_gather`; promedia loss con `dist.all_reduce`; recalcula métricas globales
  - `save_checkpoint`: solo el proceso con `rank=0` guarda el checkpoint
- **`src/training/builder.py`**: acepta `rank`, `world_size` y `distributed`; si `distributed=True` crea `DDPTrainer` (independientemente de `world_size`), si no crea `Trainer`
- **`src/training/decorators/base.py`**: `EpochController` añade `dist.barrier()` entre epochs; early stopping se decide en rank 0 y se broadcast a todos los procesos
- **`src/training/decorators/tracing.py`**: `_emit()` comprueba `rank == 0` antes de escribir logs (solo el proceso principal escribe)

### Lanzamiento

```bash
# Smoke test local (1 GPU, usa DDPTrainer real — distributed=True siempre activo en train_ddp.py):
torchrun --nproc_per_node=1 scripts/train_ddp.py \
  --model vit_tiny_patch16_224 --epochs 1 \
  --config configs/train.yaml --trace simple
# → Val F1=0.4353, completado sin errores (verificado 20/05/26)
```

**En Verode — GPU real (NCCL, cuando haya 2 nodos operativos):**
Usar el sistema de colas (Slurm) como recomienda el cotutor. Sin `--gres` si la especificación gres falla.
```bash
# Desde login node en tmux:
/opt/soft/slurm/20.11.04/bin/srun --partition=batch \
  --nodes=2 --nodelist=verode16,verode21 \
  --ntasks-per-node=1 --cpus-per-task=8 --time=72:00:00 \
  bash -c '
    cd ~/tfg-distributed-transformers
    .venv/bin/torchrun --nnodes=2 --nproc_per_node=1 \
      --node_rank=$SLURM_NODEID \
      --master_addr=verode16 --master_port=29500 \
      scripts/train_ddp.py \
      --config configs/train_ddp_verode.yaml --trace simple
  '
```

**En Verode — test funcional CPU (gloo, nodos down en Slurm):**
verode16 y verode18 aparecen como `down*` en Slurm y no son asignables por el sistema de colas,
pero sí son accesibles via SSH directo. Para el test CPU se usa SSH en dos terminales tmux:
```bash
# Terminal 1 — SSH a verode16 (nodo 0, master):
ssh verode16
cd ~/tfg-distributed-transformers && git pull origin main
.venv/bin/torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr=verode16 --master_port=29500 \
  scripts/train_ddp.py --config configs/train_ddp_cpu_test.yaml --trace simple

# Terminal 2 — SSH a verode21 (nodo 1):
ssh verode21
cd ~/tfg-distributed-transformers
.venv/bin/torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
  --master_addr=verode16 --master_port=29500 \
  scripts/train_ddp.py --config configs/train_ddp_cpu_test.yaml --trace simple
```
Extremadamente lento (CPU vs V100 ~100x). Solo para confirmar que la comunicación
multi-nodo y la sincronización de gradientes funcionan antes de tener hardware homogéneo.

### Configs DDP

- `configs/train_ddp_verode.yaml` — batch_size=64 **por GPU** (global batch = 128 con 2 GPUs), backend NCCL, para V100.
- `configs/train_ddp_cpu_test.yaml` — batch_size=4, backend **gloo**, `pretrained=false`, 1 epoch. Valida infraestructura multi-nodo sin GPU compatible.

---

## Feasibility Checker

`scripts/check_feasibility.py` — análisis de viabilidad previo al entrenamiento. Usa datos sintéticos (sin tocar el dataset) para medir throughput real y estimar tiempos.

Arquitectura (patrón Facade + SRP):
- `ModelAnalyzer` — FLOPs, parámetros, memoria estática
- `HardwareProbe` — VRAM disponible
- `Benchmarker` — throughput real por (batch_size, trace_mode)
- `TimeEstimator` — convierte throughput en estimaciones de tiempo
- `ReportFormatter` — imprime el informe + escribe CSV estructurado
- `FeasibilityChecker` — Facade que coordina todo

```bash
uv run python scripts/check_feasibility.py
uv run python scripts/check_feasibility.py --batch-sizes 16 32 64 128 --epochs 30
uv run python scripts/check_feasibility.py --batch-sizes 32 64 --trace-modes off deep
# Factor NFS para corregir estimación en Verode (NFS añade ~30% de latencia I/O)
uv run python scripts/check_feasibility.py --batch-sizes 64 --nfs-factor 1.3
# Override de modelo (uno o varios separados por espacio)
uv run python scripts/check_feasibility.py --model resnet50 --batch-sizes 32 64
uv run python scripts/check_feasibility.py --model vit_tiny_patch16_224 vit_small_patch16_224 vit_base_patch16_224 resnet50 --batch-sizes 16 32 64 128
```

Genera dos artefactos en `logs/{env}/`:
- `feasibility_DDMMYYYY_HHMMSS.log` — informe de texto legible
- `feasibility_DDMMYYYY_HHMMSS.csv` — CSV estructurado con filas `#meta` (modelo/hardware) y filas de benchmark; consumido por la pestaña Feasibility del dashboard

La tabla de estimaciones muestra **train/epoch**, **eval/epoch**, **total/epoch** y **total N epochs** por separado.

**Resultados conocidos en RTX 3060 Ti:**
- batch_size=32: ~65 imgs/s, 4.95 GB VRAM ← óptimo local
- batch_size=64: OOM (necesita ~11.5 GB)
- `--trace deep` añade ~22% overhead vs off

**En V100 32 GB (clúster):** ejecutado el 2026-05-07 en verode21.
- Batch óptimo: **64** (100.6 imgs/s) — batch=128 también cabe (16.55 GB, 100.5 imgs/s, sin ganancia real)
- batch=64 OOM solo en local (8 GB); el V100 tiene 34 GB, caben hasta batch=128
- `--trace deep` añade **18% overhead** a batch 64 (vs 22% en local a batch 32)
- Estimación 30 epochs batch 64: ~19h 42m (off) / ~23h 10m (deep)
- ⚠ La estimación subestima el tiempo real: no cuenta eval (~22 min/epoch) ni latencia NFS (ver resultados)

---

## Estructura de artefactos

Los artefactos se organizan por entorno (`local`/`verode`), modo (`single`/`ddp`) y modelo:

```
logs/
  local/
    single/{model}/   # train_*.log, epoch_metrics_*.csv, perclass_metrics_*.csv,
                      # batch_metrics_*.csv, confusion_matrix_*.csv
    ddp/{model}/      # ídem para runs distribuidos
    feasibility/      # feasibility_*.log + feasibility_*.csv
  verode/
    single/{model}/
    ddp/{model}/
    feasibility/
plots/
  local/
    single/{model}/   # training_*.png, perclass_*.png, confusion_matrix_*.png
    ddp/{model}/
  verode/
    single/{model}/
    ddp/{model}/
checkpoints/
  local/
    single/{model}/   # excluidos de git (*.pt)
    ddp/{model}/
  verode/
    single/{model}/
    ddp/{model}/
```

- El builder lee `output.env` del config y deduce `mode`/`model` automáticamente.
- Los runs anteriores a mayo 2026 usan la estructura plana (`logs/{env}/`) — el dashboard los descubre igual vía `rglob`.
- Los ficheros se nombran con formato **DDMMYYYY_HHMMSS**.
- **git:** todos los logs y CSVs bajo `logs/` se commitean (los `*.pt` de checkpoints, no).

---

## Configuración

### `configs/train.yaml` — local (baseline, sin regularización v3)
```yaml
data:
  root: "/media/alejandro/SSD/datasets/bigearthnet/BigEarthNet-S2"
  metadata: "/media/alejandro/SSD/datasets/bigearthnet/metadata.parquet"
  num_workers: 4
model:
  name: "vit_base_patch16_224"
  pretrained: true
  num_classes: 19
  dropout: 0.1
training:
  epochs: 30
  batch_size: 32          # 64 OOM en RTX 3060 Ti (8 GB)
  lr: 0.0001              # OJO: no usar 1e-4, se parsea como string en YAML
  weight_decay: 0.05
  warmup_epochs: 5
  llrd_decay: 0.75
  grad_clip: 1.0
  early_stopping_patience: 10
  log_batch_every: 50
checkpoint:
  dir: "checkpoints/local"
output:
  env: "local"
```

### `configs/train_v3.yaml` — local con regularización v3
Igual que `train.yaml` más:
```yaml
model:
  dropout: 0.3
training:
  weight_decay: 0.1
  label_smoothing: 0.1
  mixup_alpha: 0.2
```

### `configs/train_cluster.yaml` — clúster VERODE (baseline)
Igual que `train.yaml` pero con rutas del clúster y batch=64:
```yaml
data:
  root: "/home/bejeque/alu0101317038/datasets/bigearthnet/BigEarthNet-S2"
  metadata: "/home/bejeque/alu0101317038/datasets/bigearthnet/metadata.parquet"
training:
  batch_size: 64
checkpoint:
  dir: "checkpoints/verode"
output:
  env: "verode"
```

### `configs/train_cluster_v3.yaml` — clúster VERODE con regularización v3
Igual que `train_cluster.yaml` más `label_smoothing: 0.1`, `mixup_alpha: 0.2`, `dropout: 0.3`, `weight_decay: 0.1`.

---

## Resultados de entrenamiento

### Local — RTX 3060 Ti, batch_size=32, 30 epochs (completado 2026-05-01/02)

| Epoch | Train F1 | Val F1 | Val Loss |
|-------|----------|--------|----------|
| 1  | 0.587 | 0.593 | 0.168 |
| 4  | 0.718 | 0.657 | 0.155 |
| 9  | 0.825 | **0.659** ← mejor | 0.183 |
| 30 | 0.947 | 0.654 | 0.674 |

- **Mejor Val F1: 0.6586** (epoch 9) — guardado en `checkpoints/local/checkpoint_epoch_009.pt`
- Duración: ~32.5 horas (~65 min/epoch)
- Sobreajuste claro a partir del epoch 9: train loss → 0.0001, val loss sigue subiendo
- Log completo: `logs/local/train_legacy.log`

### Clúster — V100 32 GB, batch_size=64, 30 epochs (completado 2026-05-07/09)

Ejecutado con la versión previa a la refactorización (sin metric reporters), con `--trace deep`.

| Epoch | Train F1 | Val F1 | Val Loss |
|-------|----------|--------|----------|
| 1  | 0.5865 | 0.6121 | 0.1628 |
| 2  | 0.6832 | 0.6388 | 0.1557 |
| 4  | 0.7415 | 0.6578 | 0.1567 |
| 9  | 0.8535 | 0.6540 | 0.2125 |
| 18 | 0.9383 | 0.6565 | 0.4688 |
| 26 | 0.9471 | 0.6587 | 0.6401 |
| 28 | 0.9473 | **0.6588** ← mejor | 0.6526 |
| 30 | 0.9473 | 0.6588 | 0.6554 |

- **Mejor Val F1: 0.6588** (epoch 28) — checkpoints en `~/tfg-distributed-transformers/checkpoints/verode/` en verode21
- Duración real: **~45h 50m** (08:41 May 7 → 06:29 May 9)
  - Train: ~67 min/epoch | Eval: ~22 min/epoch | Total: ~89 min/epoch
  - Epoch 1: ~135 min (torchinfo + hook registration + warmup GPU)
  - Epoch 7: ~103 min (contención de recursos en verode21)
- Overfitting severo: val F1 se estabiliza en 0.65-0.66 desde epoch 4, train F1 sigue subiendo hasta 0.947
- Val loss diverge monotónicamente (0.16 → 0.66) mientras train loss cae a 0.0001
- Sin anomalías de gradiente detectadas por DeepTracingDecorator en ningún epoch
- Log completo: `logs/verode/train_deep_20260507_084113.log`

### Clúster v2 — V100 32 GB, batch_size=64, early stopping (completado 2026-05-11/12)

Ejecutado con `feature/training-improvements`: LLRD (decay=0.75, 30 grupos), warmup lineal (5 epochs),
cosine scheduler, grad_clip=1.0, early stopping patience=10. Flags: `--trace simple --layers plot hooks --fn energy`.

| Epoch | Train F1 | Val F1 | Val Loss |
|-------|----------|--------|----------|
| 1  | 0.4708 | 0.5442 | 0.1816 |
| 2  | 0.6159 | 0.6239 | 0.1562 |
| 4  | 0.7127 | 0.6696 | 0.1464 |
| 7  | 0.7828 | **0.6707** ← mejor | 0.1637 |
| 10 | 0.8461 | 0.6671 | 0.2053 |
| 17 | 0.9186 | 0.6652 | 0.3365 ← early stop |

- **Mejor Val F1: 0.6707** (epoch 7) — guardado en `checkpoints/verode/` en verode21
- **Early stopping** paró en epoch 17 (sin mejora desde epoch 7 — patience=10)
- Duración: **~19h** (15:08 May 11 → 10:19 May 12) — 17 epochs × ~67 min/epoch
- Tiempo ahorrado vs v1: ~27h (se habrían necesitado 30 epochs × 89 min = 45h)
- Overfitting idéntico: val F1 plana en 0.67 desde epoch 4, train F1 → 0.92
- `[energy]` reportó "GPU no disponible" en verode — pynvml no accede al driver en ese entorno; no afecta al entrenamiento
- Activaciones de hooks estables en epochs 5/10/15: mlp.fc1 ~1.83, attn.qkv ~0.71, sin neuronas muertas
- Log completo: `logs/verode/train_20260511_150808.log` | Plot: `plots/verode/training_20260511_150808.png`

### Clúster v3 — V100 32 GB, batch_size=64, label smoothing + mixup (completado 2026-05-13/14)

Ejecutado con `configs/train_cluster_v3.yaml`: label smoothing=0.1, mixup α=0.2, dropout=0.3, weight decay=0.1, LLRD decay=0.75, warmup=5, early stopping patience=10. Flags: `--trace simple --layers plot confusion batch-monitor --fn energy`.

| Epoch | Train F1 | Val F1 | Val Loss | Threshold óptimo | F1@threshold |
|-------|----------|--------|----------|-------------------|--------------|
| 1  | 0.4813 | 0.5357 | 0.1813 | 0.30 | 0.5778 |
| 2  | 0.6122 | 0.6237 | 0.1596 | 0.30 | 0.6394 |
| 3  | 0.6813 | 0.6548 | 0.1501 | 0.35 | 0.6668 |
| 4  | 0.7122 | 0.6523 | 0.1478 | 0.30 | 0.6683 |
| 5  | 0.7351 | 0.6687 | 0.1488 | 0.35 | 0.6763 |
| 6  | 0.7535 | **0.6738** ← mejor | 0.1545 | 0.35 | 0.6799 |
| 7  | 0.7816 | 0.6651 | 0.1641 | 0.35 | 0.6711 |
| 10 | 0.8454 | 0.6624 | 0.1992 | 0.35 | 0.6665 |
| 16 | 0.9120 | 0.6630 | 0.3121 ← early stop | 0.35 | 0.6655 |

- **Mejor Val F1: 0.6738** (epoch 6, threshold=0.5) — checkpoint en `checkpoints/verode/` en verode21
- **Con threshold óptimo 0.35:** F1=0.6799 en el mejor epoch (mejora práctica para inferencia)
- **Early stopping** paró en epoch 16 (sin mejora desde epoch 6 — patience=10)
- Duración: **~18h** (13 May 16:15 → 14 May 10:13) — ~67 min/epoch (train+eval)
- **Gap train-val en mejor epoch (6):** 0.7535 - 0.6738 = 0.08 (vs 0.11 en v2 — reducción clara)
- Overfitting reducido pero no eliminado: val F1 plana en 0.66-0.67 desde epoch 6, train F1 → 0.91
- Val loss empieza a divergir desde epoch 6 (mínima fue epoch 4: 0.1478)
- `[energy]` reportó "GPU no disponible" en verode — pynvml no accede al driver; no afecta al entrenamiento
- Log completo: `logs/verode/train_13052026_161533.log` | Plot: `plots/verode/training_13052026_161533.png`

### Local — ResNet50, batch_size=32, 2 epochs (smoke test 2026-05-14)

Prueba de soporte genérico timm con modelo convolucional. Config: `train_v3.yaml` (label smoothing + mixup).

| Epoch | Train F1 | Val F1 | Val Loss | Threshold óptimo |
|-------|----------|--------|----------|-----------------|
| 1 | 0.2311 | 0.3588 | 0.2463 | 0.30 |
| 2 | 0.4046 | **0.4725** | 0.2174 | 0.30 |

- **~17 min/epoch** — 4× más rápido que ViT-Base (~65 min/epoch en RTX 3060 Ti)
- Val F1 > Train F1 en ambos epochs — modelo aún en fase de aprendizaje rápido, sin overfitting
- Val loss bajando fuerte — con entrenamiento completo llegaría claramente más lejos
- Threshold óptimo 0.30 (más bajo que el 0.35 del ViT) — ResNet más conservador en sus predicciones
- A epoch 2, ViT-Base v3 tenía Val F1=0.6237 — ventaja clara del transformer con preentrenamiento ImageNet
- **Valida que el soporte genérico timm funciona correctamente** para modelos no-ViT (sin LLRD, AdamW estándar)
- Log: `logs/local/train_14052026_170438.log` | Plot: `plots/local/training_14052026_170438.png`

### Clúster v3b — V100 32 GB, batch_size=64, stack completo con energía (completado 2026-05-14/15)

Misma config que v3 (`configs/train_cluster_v3.yaml`). Objetivo: verificar pynvml funcionando y obtener datos de consumo energético. Flags: `--trace simple --layers plot confusion batch-monitor hooks --fn energy timing`.

| Epoch | Train F1 | Val F1 | Val Loss | Threshold óptimo | F1@threshold |
|-------|----------|--------|----------|------------------|--------------|
| 1  | 0.4711 | 0.5593 | 0.1777 | 0.35 | 0.5865 |
| 2  | 0.6186 | 0.6170 | 0.1567 | 0.30 | 0.6370 |
| 3  | 0.6829 | 0.6295 | 0.1502 | 0.30 | 0.6631 |
| 4  | 0.7150 | 0.6685 | 0.1487 | 0.35 | 0.6757 |
| 5  | 0.7377 | **0.6708** ← mejor | 0.1502 | 0.35 | 0.6788 |
| 7  | 0.7838 | 0.6700 | 0.1613 | 0.35 | 0.6766 |
| 10 | 0.8465 | 0.6655 | 0.2039 | 0.40 | 0.6673 |
| 15 | 0.9044 | 0.6538 | 0.2909 ← early stop | 0.30 | 0.6579 |

- **Mejor Val F1: 0.6708** (epoch 5) — prácticamente igual a v3 (0.6738); diferencia de 0.003 es variación aleatoria
- **Early stopping** en epoch 15 (sin mejora desde epoch 5 — patience=10)
- Duración: **~17h 17m** (14 May 14:57 → 15 May 08:14) — ~69 min/época
- **Energía (primera medición real):** eval_epoch consume ~35 Wh/época a ~100-104 W de potencia media en V100; total estimado 15 evals ≈ 530 Wh solo en evaluación
- El patrón de overfitting es idéntico a v3: val F1 plana en 0.67 desde epoch 5, train F1 → 0.90
- Val loss mínima en epoch 4 (0.1487), empieza a divergir desde epoch 5 — igual que v3
- Log: `logs/verode/train_14052026_145711.log` | Plot: `plots/verode/training_14052026_145711.png`

### Comparativa de todas las ejecuciones en clúster

| | v1 (sin mejoras) | v2 (LLRD + warmup + early stop) | v3 (label smoothing + mixup) | v3b (stack completo + energía) |
|---|---|---|---|---|
| **Config** | `train_cluster.yaml` | `train_cluster.yaml` + flags | `train_cluster_v3.yaml` | `train_cluster_v3.yaml` |
| **Trace mode** | `--trace deep` | `--trace simple` | `--trace simple` | `--trace simple` |
| **LLRD** | No | Sí (decay=0.75) | Sí (decay=0.75) | Sí (decay=0.75) |
| **Warmup** | No | Sí (5 epochs) | Sí (5 epochs) | Sí (5 epochs) |
| **Early stopping** | No | Sí (patience=10) | Sí (patience=10) | Sí (patience=10) |
| **Label smoothing** | No | No | Sí (0.1) | Sí (0.1) |
| **Mixup** | No | No | Sí (α=0.2) | Sí (α=0.2) |
| **Dropout** | 0.1 | 0.1 | 0.3 | 0.3 |
| **Weight decay** | 0.05 | 0.05 | 0.1 | 0.1 |
| **Energía medida** | No | No | No | **Sí (~35 Wh/eval, ~100 W)** |
| **Epochs ejecutados** | 30 | 17 | 16 | 15 |
| **Duración** | ~45.8h | ~19h | **~18h** | ~17.3h |
| **Mejor Val F1** | 0.6588 (epoch 28) | 0.6707 (epoch 7) | **0.6738 (epoch 6)** | 0.6708 (epoch 5) |
| **Gap train-val en mejor epoch** | ~0.34 | ~0.11 | **~0.08** | ~0.07 |

**Conclusiones v1 → v2:**
- LLRD + warmup mejoraron Val F1 en +0.012 y aceleraron la convergencia (mejor epoch: 28 → 7)
- Early stopping ahorró ~27h eliminando epochs innecesarios
- El techo de generalización (~0.67 Val F1) es una limitación del dataset/regularización, no del hardware
- El cuello de botella NFS persiste: añadir GPUs (DDP) no escala linealmente si el I/O es el límite

**Conclusiones v2 → v3:**
- Label smoothing + mixup mejoraron Val F1 en +0.003 (modesto pero consistente)
- La reducción del gap train-val (0.11 → 0.08) confirma que la regularización adicional funciona
- La convergencia al mejor epoch fue más rápida (epoch 7 → epoch 6)
- El techo del dataset (~0.67-0.68 Val F1 a threshold=0.5) parece real — las clases raras limitan F1 macro
- **Para inferencia usar threshold=0.35:** consistentemente mejora F1 en ~0.005-0.006 sobre threshold=0.5
- El siguiente paso para mejorar resultados es DDP (más datos efectivos por epoch) o cambio de dataset split

---

## Gestión de dependencias

### Local (uv)
```bash
uv sync                   # instalar entorno
uv run python ...         # ejecutar
uv add <paquete>          # añadir dependencia
```

### Clúster
El entorno `.venv` ya está creado. Si se reinstala desde cero:
```bash
cd ~/tfg-distributed-transformers
uv sync
# Después, reinstalar PyTorch con cu118 (cu13 por defecto no es compatible con driver 525):
# OJO: usar uv pip, NO python -m pip (el módulo pip no está disponible en el venv del clúster)
uv pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
```

Dependencias principales: `torch`, `timm`, `torchvision`, `torchinfo`, `tqdm`, `rasterio`, `pandas`, `pyarrow`, `pyyaml`, `matplotlib`, `nvidia-ml-py`, `streamlit`, `plotly`

---

## Dashboard web

`src/web/` — interfaz Streamlit profesional para gestionar y analizar el proyecto de principio a fin.

```
src/web/
  __init__.py
  app.py                    # Streamlit entrypoint — 9 tabs
  run_registry.py           # descubre runs con rglob (estructura plana y profunda);
                            # RunInfo con env, mode, model, epoch/perclass/batch CSV paths
  log_parser.py             # parsea logs --trace simple y --trace deep → DataFrame (fallback)
  batch_parser.py           # lee batch_metrics_*.csv → DataFrame por batch
  perclass_parser.py        # lee perclass_metrics_*.csv → DataFrame por clase
  feasibility_parser.py     # lee feasibility_*.csv → (metadata dict, benchmark DataFrame)
  confusion_matrix_parser.py # lee confusion_matrix_*.csv → matriz numpy por epoch
```

### Arranque

```bash
uv run streamlit run src/web/app.py
# Abre http://localhost:8501
```

### Tabs

| Tab | Contenido |
|-----|-----------|
| Curves | Curvas de F1/loss/accuracy; metric cards; epoch time chart; descarga CSV |
| Per-class | Tabla ranking de clases + tendencia multi-clase + confusion matrix (normalizada/absoluta) |
| Batch | Running loss por batch con moving average y detección de picos |
| Compare | Superpone hasta 4 runs; tabla resumen comparativa |
| Feasibility | Benchmark VRAM/throughput; estimaciones; lanzar feasibility check desde la web |
| Time | Tiempo real por epoch vs estimación; tendencia lineal; warmup detection |
| Info | Config YAML, anomaly log, dataset info, log completo con buscador |
| Launcher | Lanzar entrenamientos single-GPU o feasibility check con output en tiempo real |
| Live | Monitor en vivo: epoch progress, GPU usage, últimas líneas del log, auto-refresh |

Descubre runs recursivamente en toda la estructura `logs/` (tanto flat legacy como profunda env/mode/model).
Compatible con `--trace simple`, `--trace deep` y logs legacy.

---

## Git workflow

```
main ← develop ← feature/xxx
```

- Las feature branches salen de `develop` y hacen PR a `develop`
- Cuando `develop` está validado, se mergea a `main`
- **No añadir Co-Authored-By en los commits ni "Generated with Claude" en los PRs**

### Configuración SSH en Verode (hecho una sola vez)
```bash
ssh-keygen -t ed25519 -C "alu0101317038@verode" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub   # añadir en github.com → Settings → SSH keys
git remote set-url origin git@github.com:alerguezrojas/tfg-distributed-transformers.git
```

---

## Estado actual del proyecto

### Completado
- [x] Pipeline de datos: `BigEarthNetDataset` con metadata.parquet
- [x] Modelo: `BigEarthViT` (ViT + cabeza multi-label, soporte genérico timm)
- [x] Entrenamiento single-GPU: `Trainer` + LLRD + warmup + cosine scheduler + checkpoints
- [x] Arquitectura de decoradores: Decorator (GoF) + Template Method
  - `decorators/`: `TracingDecorator`, `DeepTracingDecorator`, `PlottingDecorator`, `LayerHooksDecorator`
  - `decorators/confusion.py`: `ConfusionMatrixDecorator` — barras F1/prec/rec por clase + heatmap 19×19 normalizado
  - `decorators/batch_monitor.py`: `BatchMonitorDecorator` — CSV con running loss por batch
  - `decorators/metric_reporters.py`: `LossReporter`, `F1Reporter`, `AccuracyReporter`, `PrecisionRecallReporter`
  - `fn_decorators.py`: `@timed`, `@log_call`, `@measure_energy`, `@retry_on_cuda_oom` — rutean a logger
  - `builder.py`: `TrainingSessionBuilder` — fluent API para montar el stack completo
  - `augmentations.py`: `mixup_batch()` — mezcla de batch compatible con multi-label
- [x] Técnicas anti-overfitting v3: label smoothing, mixup, threshold search, dropout 0.3, weight decay 0.1
- [x] `metrics.py`: métricas extraídas en módulo propio (sin duplicación)
- [x] Flags `--trace / --layers / --fn / --metrics / --inspect / --model` en script de entrenamiento
- [x] Inspección modular: `--inspect model-summary batch-table grad-monitor anomalies`
- [x] Early stopping: `patience` configurable en `EpochController`
- [x] Log con timestamp (DDMMYYYY) a fichero en `logs/{env}/{mode}/{model}/` + gráficas PNG en `plots/{env}/{mode}/{model}/`
- [x] `check_feasibility.py`: benchmark train+eval por separado, `--nfs-factor`, auto-save log + CSV en `logs/{env}/feasibility/`
- [x] Entrenamiento local: 30 epochs, Val F1=0.6586 (01-02/05/26) → `logs/local/train_legacy.log`
- [x] Test local stack completo: 1 epoch vit_tiny, Val F1=0.4457 (11/05/26)
- [x] Smoke test v3 local: 1 epoch vit_tiny, Val F1=0.4019, threshold óptimo=0.30 (13/05/26)
- [x] Clúster VERODE: V100 32 GB, dataset 549 488 patches, PyTorch cu118, SSH key configurada
- [x] Entrenamiento clúster v1: 30 epochs, batch=64, Val F1=0.6588 (07-09/05/26) → `logs/verode/train_deep_20260507_084113.log`
- [x] Entrenamiento clúster v2: 17 epochs, Val F1=0.6707, early stop epoch 17 (11-12/05/26) → `logs/verode/train_20260511_150808.log`
- [x] Entrenamiento clúster v3: 16 epochs, Val F1=0.6738, early stop epoch 16 (13-14/05/26) → `logs/verode/train_13052026_161533.log`
- [x] Configs v3 listos: `configs/train_v3.yaml` (local) y `configs/train_cluster_v3.yaml` (Verode)
- [x] Diagrama de clases actualizado: `docs/class_diagram.puml` + `docs/class_diagram.png` — incluye src.web (RunInfo, run_registry, log_parser, batch_parser, app), metrics.py y logger_setup.py; eliminado `docs/class_diagram_pre_v3.png`
- [x] Smoke test ResNet50 local: 2 epochs, Val F1=0.4725, threshold óptimo=0.30 (14/05/26) → `logs/local/train_14052026_170438.log`
- [x] Fix pynvml en Verode: `nvidia-ml-py` no estaba instalado → `uv sync` + `uv pip install torch cu118`; confirmado funcionando
- [x] Entrenamiento v3b en Verode: 15 epochs, Val F1=0.6708 (epoch 5), early stop epoch 15 (14-15/05/26) → `logs/verode/train_14052026_145711.log` — confirma energía funcional (~35 Wh/eval, ~100 W media V100)
- [x] **Entrenamiento distribuido DDP (20/05/26):** `DDPTrainer`, `scripts/train_ddp.py`, `configs/train_ddp_verode.yaml`; smoke test local 1 proceso completado sin errores (Val F1=0.4353) → `logs/local/train_20260520_221708.log`
- [x] **Fix DDPTrainer con 1 GPU (21/05/26):** `TrainingSessionBuilder` usa `distributed=True` en vez de `world_size>1`; `torchrun --nproc_per_node=1` ahora usa `DDPTrainer` real
- [x] **Web dashboard v2 (20/05/26):** 7 tabs, CSV-driven (epoch_metrics, perclass_metrics, feasibility), Plotly interactivo por clase, pestaña Feasibility, pestaña Time Analysis; `perclass_parser.py`, `feasibility_parser.py`; `check_feasibility.py` añade `--model` y escribe CSV
- [x] Diagrama de clases v2: DDPTrainer, TracingDecorator con epoch_csv, ConfusionMatrixDecorator con write_csv, ReportFormatter con write_csv, RunInfo con epoch/perclass csv paths, web con 7 tabs (20/05/26)
- [x] **Heatmap 19×19 de confusión — CSV + Plotly interactivo (26/05/26):** `ConfusionMatrixDecorator` genera `confusion_matrix_TIMESTAMP.csv`; `confusion_matrix_parser.py` lee el CSV; sub-tab muestra heatmap Plotly interactivo con hover y selector de epoch; fallback a PNG
- [x] **Web dashboard v3 (27/05/26):** 9 tabs, interfaz profesional sin emojis; Launcher (lanzar entrenamientos con output en tiempo real); Live Monitor (auto-refresh, GPU via nvidia-smi); mejoras en todas las pestañas (moving average, comparativa multi-run, anomaly detection, etc.)
- [x] **Gestión de carpetas y gitignore (27/05/26):** estructura `{env}/{mode}/{model}/` para logs, plots y checkpoints; feasibility en `{env}/feasibility/`; `run_registry.py` con rglob; `RunInfo` añade `mode` y `model`; `.gitignore` corregido — todos los CSVs y logs bajo `logs/` se commitean
- [x] Diagrama de clases v3: RunInfo con mode/model, web con 9 tabs, confusion_matrix_parser (27/05/26)
- [x] **Multi-model feasibility (27/05/26):** `check_feasibility.py --model` acepta N modelos separados por espacio (`nargs="+"`) — cada modelo genera su propio par log/CSV con timestamp independiente
- [x] **DDP CPU/gloo support (27/05/26):** `train_ddp.py` lee `backend` del config; `DDPTrainer` omite `device_ids` en CPU; `configs/train_ddp_cpu_test.yaml` con backend gloo, vit_tiny, pretrained=false — permite validar infraestructura multi-nodo sin GPU compatible
- [x] **Fix ZeroDivisionError scheduler (27/05/26):** `T_max = max(1, epochs - warmup_epochs)` en `builder.py` — evita división por cero cuando `epochs ≤ warmup_epochs`
- [x] **Feasibility multi-modelo local (27/05/26):** vit_tiny, vit_small, vit_base, resnet50 con batch-sizes 16 y 32; trace-modes off y simple → 4 pares log/CSV en `logs/local/feasibility/`
- [x] **Entrenamiento local vit_tiny 5 epochs (27/05/26):** Val F1=0.590 (epoch 5, mejorando en todos los epochs), ~11 min/epoch, stack completo (plot, hooks, confusion, batch-monitor, energy, timing) → `logs/local/single/vit_tiny_patch16_224/train_27052026_221827.log`

### Pendiente
- [ ] DDP real en Verode con 2 GPUs: `torchrun --nproc_per_node=2` y medir speedup vs single-GPU
- [ ] Proyección multi-GPU en feasibility checker (estimación de throughput con N GPUs)
- [ ] Comparar throughput single-GPU vs multi-GPU para cuantificar speedup DDP

---

## Errores conocidos y soluciones

| Error | Causa | Solución |
|-------|-------|----------|
| `lr '<=' not supported between float and str` | `1e-4` en YAML se parsea como string | Usar `0.0001` en el YAML |
| `CUDA not available` en el clúster | `uv sync` instala torch cu13, incompatible con driver 525 | Reinstalar con `--index-url .../cu118` tras el sync |
| `Illegal instruction` en login node | Login node sin AVX2; numpy/torch usan AVX2 | Ejecutar siempre en nodo de cómputo via `srun` |
| `sbatch` I/O error | Bug de configuración de Slurm en VERODE | Usar `tmux` + `srun` |
| `CUDA out of memory` batch_size=64 (local) | ViT-B necesita ~11.5 GB para batch 64 | Usar batch_size=32 en local (4.95 GB) |
| `CUDA out of memory` en hooks | Tensores grandes copiados a RAM | Calcular en GPU con `.detach().float()`, solo `.item()` para el escalar |
| `FileNotFoundError` metadata.parquet | `configs/train.yaml` tiene rutas del SSD local | Usar `configs/train_cluster.yaml` en el clúster |
| `nvidia driver` no funciona con kernel 6.8 | Driver 470 incompatible | Actualizar a `nvidia-driver-580-open` |
| `[energy] GPU no disponible (pynvml no instalado)` en Verode | `uv sync` antes de añadir `nvidia-ml-py` al pyproject.toml | `uv sync` + reinstall torch cu118 |
| `srun --gres=gpu:1` falla en Verode | El recurso GPU se llama `gpu:tesla` en este clúster | Usar `--gres=gpu:tesla:1` |
| `libcudnn.so.9` en srun no-interactivo | `LD_LIBRARY_PATH` vacío en shells no-interactivos; cu13 instalado por `uv sync` | Reinstalar cu118: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --force-reinstall` |
| `ZeroDivisionError` en cosine scheduler | `epochs ≤ warmup_epochs` → `T_max = 0` | `builder.py` usa `max(1, epochs - warmup_epochs)` |

---

## Comandos útiles

### Local
```bash
# Feasibility checker (genera .log + .csv)
uv run python scripts/check_feasibility.py --batch-sizes 16 32 --epochs 5
uv run python scripts/check_feasibility.py --model resnet50 --batch-sizes 32 64

# Test rápido single-GPU (1 epoch)
uv run python scripts/train_single_gpu.py --model vit_tiny_patch16_224 --epochs 1 --trace simple

# Entrenamiento completo con config v3
uv run python scripts/train_single_gpu.py --config configs/train_v3.yaml --trace simple --layers plot

# Smoke test DDP local (1 proceso, valida sin 2 GPUs)
torchrun --nproc_per_node=1 scripts/train_ddp.py \
  --model vit_tiny_patch16_224 --epochs 1 --config configs/train.yaml --trace simple

# Dashboard web (9 tabs)
uv run streamlit run src/web/app.py

# Ver log en tiempo real (ajusta la ruta al modelo concreto)
tail -f logs/local/single/vit_base_patch16_224/train_*.log
```

### Clúster VERODE
```bash
# Conectar
ssh alu0101317038@verode00.pcg.ull.es

# Cargar Slurm
module add slurm/client/20.11.04

# Ver estado
sinfo -N
squeue -a

# Abrir sesión tmux (para que el job sobreviva a desconexiones)
tmux new-session -s training   # nueva sesión
tmux attach -t training        # reconectar a sesión existente
# Ctrl+B, D → desconectarse sin matar la sesión

# Entrar al nodo de cómputo (interactivo)
/opt/soft/slurm/20.11.04/bin/srun --partition=batch --nodelist=verode21 --gres=gpu:tesla:1 --time=72:00:00 --pty bash

cd ~/tfg-distributed-transformers

# Verificar CUDA
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# Feasibility previo al entrenamiento (usar --config para que guarde en logs/verode/)
.venv/bin/python scripts/check_feasibility.py \
  --config configs/train_cluster_v3.yaml \
  --batch-sizes 32 64 128 --epochs 30 --nfs-factor 1.3

# Entrenamiento completo
.venv/bin/python scripts/train_single_gpu.py \
  --config configs/train_cluster_v3.yaml \
  --trace simple \
  --layers plot confusion batch-monitor \
  --fn energy

# Ver log en tiempo real
tail -f logs/verode/train_*.log

# Al terminar: subir resultados (gitignore ya configurado, no necesita -f)
git add logs/verode/ plots/verode/
git commit -m "feat: add Verode training results"
git push origin main
```
