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
- **Nodos de cómputo:** verode[16-21] — solo verode21 operativo actualmente
- **GPU:** Tesla V100-PCIE, **32 GB VRAM** por nodo
- **CUDA:** 12.0, Driver 525.147.05
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
    ├── ConfusionMatrixDecorator   # aspecto: PNG de F1/prec/rec por clase tras cada eval
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
    confusion.py           # ConfusionMatrixDecorator — PNG con F1/prec/rec por clase (multi-label)
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
  - `ConfusionMatrixDecorator` — PNG con F1/prec/rec por clase (multi-label) tras cada eval epoch
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
                             confusion    → ConfusionMatrixDecorator, PNG por clase tras cada eval
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

## Feasibility Checker

`scripts/check_feasibility.py` — análisis de viabilidad previo al entrenamiento. Usa datos sintéticos (sin tocar el dataset) para medir throughput real y estimar tiempos.

Arquitectura (patrón Facade + SRP):
- `ModelAnalyzer` — FLOPs, parámetros, memoria estática
- `HardwareProbe` — VRAM disponible
- `Benchmarker` — throughput real por (batch_size, trace_mode)
- `TimeEstimator` — convierte throughput en estimaciones de tiempo
- `ReportFormatter` — imprime el informe
- `FeasibilityChecker` — Facade que coordina todo

```bash
uv run python scripts/check_feasibility.py
uv run python scripts/check_feasibility.py --batch-sizes 16 32 64 128 --epochs 30
uv run python scripts/check_feasibility.py --batch-sizes 32 64 --trace-modes off deep
# Nuevo: factor NFS para corregir estimación en Verode (NFS añade ~30% de latencia I/O)
uv run python scripts/check_feasibility.py --batch-sizes 64 --nfs-factor 1.3
```

El informe se guarda automáticamente en `logs/{env}/feasibility_DDMMYYYY.log`.
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

Logs, plots y checkpoints se separan por entorno para no mezclar ejecuciones locales y de clúster:

```
logs/
  local/        # RTX 3060 Ti — commiteados en git (excluye *.csv grandes)
  verode/       # V100 Verode — commiteados en git
plots/
  local/
  verode/
checkpoints/
  local/        # excluidos de git (*.pt)
  verode/       # excluidos de git (*.pt)
```

El builder lee `output.env` del config (`"local"` o `"verode"`) y escribe en el subdirectorio correcto.
Los ficheros se nombran con formato **DDMMYYYY_HHMMSS** (desde housekeeping, mayo 2026).

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

### Comparativa de todas las ejecuciones en clúster

| | v1 (sin mejoras) | v2 (LLRD + warmup + early stop) | v3 (label smoothing + mixup) |
|---|---|---|---|
| **Config** | `train_cluster.yaml` | `train_cluster.yaml` + flags | `train_cluster_v3.yaml` |
| **Trace mode** | `--trace deep` | `--trace simple` | `--trace simple` |
| **LLRD** | No | Sí (decay=0.75) | Sí (decay=0.75) |
| **Warmup** | No | Sí (5 epochs) | Sí (5 epochs) |
| **Early stopping** | No | Sí (patience=10) | Sí (patience=10) |
| **Label smoothing** | No | No | Sí (0.1) |
| **Mixup** | No | No | Sí (α=0.2) |
| **Dropout** | 0.1 | 0.1 | 0.3 |
| **Weight decay** | 0.05 | 0.05 | 0.1 |
| **Epochs ejecutados** | 30 | 17 | 16 |
| **Duración** | ~45.8h | ~19h | **~18h** |
| **Mejor Val F1** | 0.6588 (epoch 28) | 0.6707 (epoch 7) | **0.6738 (epoch 6)** |
| **Gap train-val en mejor epoch** | ~0.34 | ~0.11 | **~0.08** |

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

`src/web/` — interfaz Streamlit para visualizar resultados de entrenamiento (no lanza el training).

```
src/web/
  __init__.py
  app.py            # Streamlit entrypoint — 5 tabs
  run_registry.py   # descubre runs en logs/ y plots/ por timestamp
  log_parser.py     # parsea logs --trace simple y --trace deep → DataFrame por epoch
  batch_parser.py   # lee batch_metrics_*.csv → DataFrame por batch
```

### Arranque

```bash
uv run streamlit run src/web/app.py
# Abre http://localhost:8501
```

### Tabs

| Tab | Contenido |
|-----|-----------|
| Training Curves | Plotly interactivo: loss, F1, accuracy, precision/recall (train + val) |
| Per-class Metrics | PNGs de ConfusionMatrixDecorator por epoch (requiere `--layers confusion`) |
| Batch Monitor | Running loss intra-epoch por batch (requiere `--layers batch-monitor`) |
| Compare Runs | Superpone hasta 4 runs en el mismo gráfico |
| Run Info | Metadatos, tiempos, log crudo (200 primeras líneas) |

El dashboard detecta automáticamente los runs existentes en `logs/`. Compatible con ambos formatos de log (`--trace simple` y `--trace deep`).

---

## Git workflow

```
main ← feature/xxx   (PR directo a main)
```

- Las feature branches salen de `main` y hacen PR directo a `main`
- `develop` existía en versiones anteriores pero ya no se usa — `main` es la rama de integración
- **No añadir Co-Authored-By en los commits**

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
  - `decorators/confusion.py`: `ConfusionMatrixDecorator` — F1/prec/rec por clase (multi-label)
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
- [x] Log con timestamp (DDMMYYYY) a fichero en `logs/{env}/` + gráficas PNG en `plots/{env}/`
- [x] `check_feasibility.py`: benchmark train+eval por separado, `--nfs-factor`, auto-save log
- [x] Dashboard web Streamlit: `src/web/` con 5 tabs (curvas, por clase, batch, comparar, info)
  - Escanea `logs/local/` y `logs/verode/`, soporta formato simple y deep, y logs legacy
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
- [x] Entrenamiento v3b en Verode en curso: stack completo (`--trace simple --layers plot confusion batch-monitor hooks --fn energy timing`) con pynvml funcional; resultados pendientes de commitear

### Pendiente
- [ ] Implementar entrenamiento distribuido (PyTorch DDP) con múltiples V100
- [ ] Proyección multi-GPU en feasibility checker
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

---

## Comandos útiles

### Local
```bash
# Feasibility checker
uv run python scripts/check_feasibility.py --batch-sizes 16 32 64 --epochs 30

# Test rápido (1 epoch)
uv run python scripts/train_single_gpu.py --model vit_tiny_patch16_224 --epochs 1 --trace simple

# Entrenamiento completo con config v3
uv run python scripts/train_single_gpu.py --config configs/train_v3.yaml --trace simple --layers plot

# Ver log en tiempo real
tail -f logs/local/train_*.log
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
