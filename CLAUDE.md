# CLAUDE.md — tfg-distributed-transformers

Contexto completo del proyecto para continuar el trabajo en cualquier máquina.

---

## Sobre el proyecto

**TFG:** "Entrenamiento distribuido de modelos abiertos de aprendizaje automático basados en Transformers"
**Tutor:** Francisco Carmelo Almeida Rodríguez (Universidad de La Laguna)
**Cotutor:** Daniel Suárez Labena
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
- **Pérdida:** `BCEWithLogitsLoss` por defecto (sin sigmoid en el modelo); opcionalmente `FocalLoss` o `pos_weight` vía `training.loss`/`pos_weight` (ver `src/training/losses.py`)
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
  losses.py                # FocalLoss multi-label + pos_weight ('auto' del metadata) + build_criterion
  config_validator.py      # validate_config() — valida el YAML antes de entrenar
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

## CLI unificado (`tfg`)

`src/cli.py` (Typer) + lanzador `tfg.py` en la raíz → **un único punto de entrada en terminal** para todo lo que usa la máquina. Separación tipo W&B/MLflow/TensorBoard: **el terminal hace** (toca la GPU), **la web mira** (visualización de solo lectura).

```bash
uv run tfg.py --help
uv run tfg.py estimate --model vit_base_patch16_224 --gpu "Tesla T4" --strategy ddp --n-gpus 2 --precision amp
uv run tfg.py train --strategy {single|ddp|model-parallel|heterogeneous} --config ... [--dry-run]
uv run tfg.py benchmark --model vit_base_patch16_224 --batch-sizes 32,64 --epochs 30
uv run tfg.py eval --checkpoint checkpoints/local/checkpoint_epoch_009.pt --split test
uv run tfg.py dashboard
```

- Subcomandos: `train` (elige estrategia → lanza el script correcto, con `torchrun` para ddp/heterogéneo), `estimate` (antes `predict`; predictor **analítico** en terminal, sin GPU — tabla rich), `benchmark` (antes `feasibility`; **medición empírica** real en la máquina), `eval` (test split), `runs` (lista los entrenamientos de `logs/` con Best Val F1 / Test F1), `dashboard` (streamlit), `menu` (interactivo guiado — pregunta los parámetros, ideal para la defensa). **Renombrados el 24/06 (analítico/empírico)**; los builders internos (`build_feasibility_cmd`) y el paquete `src/web/tabs/feasibility/` conservan el nombre antiguo.
- Builders puros `build_train_cmd` / `build_feasibility_cmd` / `build_eval_cmd` + helper `_run_row` (testeados: `tests/unit/test_cli.py`, 15 tests). `--dry-run` imprime el comando sin ejecutarlo (para copiarlo en Verode/Kaggle). Opciones de lista por coma (`--batch-sizes 32,64`, `--layers plot,confusion`).
- Los scripts de `scripts/` siguen funcionando por separado; `tfg` solo los unifica. Dependencia nueva: `typer`. Rama: `feature/unified-cli`.

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
- `configs/train_heterogeneous_ddp.yaml` — DDP heterogéneo verode21 (V100, batch=192, weight=48) + verode16/18 (CPU, batch=4, weight=1). Backend **gloo**. Label smoothing + mixup v3.
- `configs/train_heterogeneous_ddp_demo.yaml` — demo heterogéneo COMPLETO vit_tiny, subset (metadata_demo.parquet 5000/1500), 3 epochs, batch GPU 96 / CPU 4. Termina en ~13 min y genera todas las métricas para la web.

#### Configs del estudio apples-to-apples (single vs distribuido, mismo modelo/subset/epochs)
- `configs/train_demo_single.yaml` — baseline **single-GPU** (vit_tiny, subset 5000, 3 epochs, batch 96). Pareja del demo heterogéneo y del DDP.
- `configs/train_demo_ddp.yaml` — DDP **NCCL multi-GPU real** (batch 48/GPU = 96 global). Para 2 GPUs físicas (Kaggle 2×T4). ⚠️ NCCL **no permite 2 ranks en la misma GPU** ("Duplicate GPU detected"), así que con 1 sola V100 en Verode la comparación distribuida posible es el heterogéneo V100+CPU (gloo).

#### Speedup positivo en GPUs reales (Kaggle 2×T4)
Como Verode solo tiene 1 GPU usable, el speedup positivo se mide en **Kaggle** (2× Tesla T4 gratis). Ver `docs/kaggle_speedup_runbook.md`.
- `scripts/export_kaggle_subset.py` — exporta un subset autocontenido (5000/1500, bandas B04/B03/B02, mismo seed=42) listo para subir como dataset de Kaggle (~570 MB / zip 391 MB).
- El notebook clona el repo (privado → token de GitHub vía Kaggle Secrets), genera configs al vuelo, corre single (1 T4) + DDP (2 T4) y emite CSVs en `logs/kaggle/{single,ddp}/{model}/`.

### DDP heterogéneo — verode21 (GPU) + verode16 (CPU)

```bash
# Primero hacer git pull en ambos nodos:
# ssh verode21 && cd ~/tfg-distributed-transformers && git pull origin main
# ssh verode16 && cd ~/tfg-distributed-transformers && git pull origin main

# Terminal 1 — verode21 (V100, rank 0):
ssh verode21
cd ~/tfg-distributed-transformers
.venv/bin/torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr=verode21 --master_port=29500 \
  scripts/train_heterogeneous_ddp.py \
  --config configs/train_heterogeneous_ddp.yaml \
  --trace simple --layers confusion batch-monitor --fn energy

# Terminal 2 — verode16 (CPU, rank 1):
ssh verode16
cd ~/tfg-distributed-transformers
.venv/bin/torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
  --master_addr=verode21 --master_port=29500 \
  scripts/train_heterogeneous_ddp.py \
  --config configs/train_heterogeneous_ddp.yaml \
  --trace simple --layers confusion batch-monitor --fn energy
```

**Cómo funciona:**
- `HeterogeneousDistributedSampler` da a rank 0 (GPU) ≈94% del dataset y a rank 1 (CPU) ≈6%, proporcionalmente a sus compute_weights (16:1)
- `HeterogeneousDDPTrainer` normaliza los gradientes por el batch global real: `loss = criterion_sum / global_batch_size` → gradiente matemáticamente correcto aunque los batch sizes sean distintos
- Solo rank 0 (GPU) escribe logs, CSVs y checkpoints
- Los artefactos van a `logs/verode/ddp_hetero/vit_base_patch16_224/`

---

## Benchmark — medición empírica (antes "Feasibility Checker")

`scripts/benchmark.py` (antes `check_feasibility.py`) — **medición empírica** previa al entrenamiento. Usa datos sintéticos (sin tocar el dataset) para medir throughput real y estimar tiempos. Comando: `tfg benchmark`.

Arquitectura (patrón Facade + SRP):
- `ModelAnalyzer` — FLOPs, parámetros, memoria estática
- `HardwareProbe` — GPU (VRAM, compute capability) + CPU (cores, RAM)
- `DiskProbe` / `DatasetProfiler` — tipo de disco, NFS, I/O real, `io_bottleneck_ratio`
- `Benchmarker` — throughput real por (batch_size, trace_mode)
- `TimeEstimator` — convierte throughput en estimaciones de tiempo
- `DDPOptimizer` — escenarios 1/2/4/8 GPUs, speedup real, cuello de botella
- `PerformancePredictor` — predicción F1 empírica (datos históricos)
- `ConvergenceStudy` (`src/training/convergence_study.py`) — **estudio empírico real**: LR range test + mini-training de convergencia con datos reales + gradient noise scale
- `ReportFormatter` — imprime el informe + escribe CSV estructurado (bloques `#meta`, `#cpu`, `#disk`, `#dataset`, `#prediction`, `#curve_*`, `#ddp`, `#study_*`)
- `FeasibilityChecker` — Facade que coordina todo

```bash
uv run python scripts/benchmark.py
uv run python scripts/benchmark.py --batch-sizes 16 32 64 128 --epochs 30
uv run python scripts/benchmark.py --batch-sizes 32 64 --trace-modes off deep
# Factor NFS para corregir estimación en Verode (NFS añade ~30% de latencia I/O)
uv run python scripts/benchmark.py --batch-sizes 64 --nfs-factor 1.3
# Override de modelo (uno o varios separados por espacio)
uv run python scripts/benchmark.py --model resnet50 --batch-sizes 32 64
# Elegir GPU en máquina multi-GPU (registra sus núcleos CUDA/Tensor en el bloque #gpu del CSV)
uv run python scripts/benchmark.py --model vit_base_patch16_224 --batch-sizes 64 --device 1
# Estudio empírico REAL (mini-training + LR range test + gradient noise) — más lento (~3-8 min)
uv run python scripts/benchmark.py --model vit_base_patch16_224 --batch-sizes 64 \
  --dataset-path ~/datasets/bigearthnet/BigEarthNet-S2 --convergence-study --study-steps 80
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

Los artefactos se organizan por entorno (`local`/`verode`/`kaggle`), modo (`single`/`ddp`/`ddp_hetero`) y modelo:

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
    ddp_hetero/{model}/   # DDP heterogéneo GPU+CPU (gloo)
    feasibility/
  kaggle/
    single/{model}/       # baseline 1×T4
    ddp/{model}/          # DDP 2×T4 (NCCL) — speedup positivo
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

### Clúster v4 — V100 32 GB, batch_size=64, versión actual del proyecto (completado 2026-05-27/28)

Mismo config que v3/v3b (`configs/train_cluster_v3.yaml`). Primera ejecución con la nueva estructura de carpetas `{env}/{mode}/{model}/`. Flags: `--trace simple --layers plot confusion batch-monitor --fn energy timing`.

| Epoch | Train F1 | Val F1 | Val Loss | Threshold óptimo | F1@threshold |
|-------|----------|--------|----------|------------------|--------------|
| 1  | 0.4254 | 0.5259 | 0.2108 | 0.30 | 0.5758 |
| 2  | 0.5707 | 0.6097 | 0.1884 | 0.30 | 0.6338 |
| 4  | 0.6826 | 0.6656 | 0.1765 | 0.35 | 0.6804 |
| 6  | 0.7231 | 0.6731 | 0.1785 | 0.35 | 0.6827 |
| 7  | 0.7484 | **0.6816** ← mejor | 0.1758 | 0.40 | 0.6852 |
| 10 | 0.8093 | 0.6737 | 0.1849 | 0.40 | 0.6778 |
| 17 | 0.8833 | 0.6697 | 0.2037 ← early stop | 0.30 | 0.6741 |

- **Mejor Val F1: 0.6816** (epoch 7) — nuevo récord, +0.008 sobre v3 (variación aleatoria, mismo config)
- **Con threshold óptimo 0.40:** F1=0.6852
- **Early stopping** en epoch 17 (sin mejora desde epoch 7 — patience=10)
- Duración: **~19h** (27 May 21:02 → 28 May 16:00) — ~66 min/epoch (45 train + 21 eval)
- **Clase 6 completamente fallida** ("Land principally occupied by agriculture"): F1=0.000 — clase rara o ambigua que el modelo nunca predice; tira el F1 macro hacia abajo de forma desproporcionada
- Mejores clases: Marine waters (0.975), Arable land (0.852), Coniferous forest (0.848)
- Peores clases: Coastal wetlands (0.402), Industrial units (0.555), Natural grassland (0.549)
- Log completo: `logs/verode/single/vit_base_patch16_224/train_27052026_210223.log`

### Comparativa de todas las ejecuciones en clúster

| | v1 (sin mejoras) | v2 (LLRD + warmup + early stop) | v3 (label smoothing + mixup) | v3b (stack completo + energía) | v4 (versión actual) |
|---|---|---|---|---|---|
| **Config** | `train_cluster.yaml` | `train_cluster.yaml` + flags | `train_cluster_v3.yaml` | `train_cluster_v3.yaml` | `train_cluster_v3.yaml` |
| **Trace mode** | `--trace deep` | `--trace simple` | `--trace simple` | `--trace simple` | `--trace simple` |
| **LLRD** | No | Sí (decay=0.75) | Sí (decay=0.75) | Sí (decay=0.75) | Sí (decay=0.75) |
| **Warmup** | No | Sí (5 epochs) | Sí (5 epochs) | Sí (5 epochs) | Sí (5 epochs) |
| **Early stopping** | No | Sí (patience=10) | Sí (patience=10) | Sí (patience=10) | Sí (patience=10) |
| **Label smoothing** | No | No | Sí (0.1) | Sí (0.1) | Sí (0.1) |
| **Mixup** | No | No | Sí (α=0.2) | Sí (α=0.2) | Sí (α=0.2) |
| **Dropout** | 0.1 | 0.1 | 0.3 | 0.3 | 0.3 |
| **Weight decay** | 0.05 | 0.05 | 0.1 | 0.1 | 0.1 |
| **Energía medida** | No | No | No | Sí (~35 Wh/eval, ~100 W) | Sí (~35 Wh/eval, ~103 W) |
| **Epochs ejecutados** | 30 | 17 | 16 | 15 | 17 |
| **Duración** | ~45.8h | ~19h | **~18h** | ~17.3h | ~19h |
| **Mejor Val F1** | 0.6588 (epoch 28) | 0.6707 (epoch 7) | 0.6738 (epoch 6) | 0.6708 (epoch 5) | **0.6816 (epoch 7)** |
| **Gap train-val en mejor epoch** | ~0.34 | ~0.11 | ~0.08 | ~0.07 | **~0.13** |

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
- **Para inferencia usar threshold=0.35-0.40:** consistentemente mejora F1 en ~0.004-0.006 sobre threshold=0.5
- El siguiente paso para mejorar resultados es DDP (más datos efectivos por epoch) o tratar las clases raras

**Conclusiones v3/v3b → v4:**
- Mismo config, resultado ligeramente mejor (+0.008) por variación aleatoria — confirma que el techo está en ~0.68
- La clase 6 ("Land principally occupied by agriculture") F1=0.0 es un hallazgo importante: una sola clase rara que el modelo no predice nunca puede tirar el F1 macro varios puntos
- El gap train-val (0.13 en v4 vs 0.07 en v3b) es mayor probablemente por inicialización aleatoria diferente, no por regresión

---

## Estudio single-GPU vs distribuido (apples-to-apples, completado 2026-06-04)

Comparación controlada **mismo modelo / mismo subset (5000/1500) / mismos 3 epochs**, cambiando solo la estrategia de distribución. Es el resultado central del TFG sobre escalado distribuido.

| Escenario | Hardware | Modelo | Train/epoch | Speedup | Eficiencia | Cuello | Val F1 |
|---|---|---|---|---|---|---|---|
| Single-GPU | V100 (Verode) | vit_tiny | ~16s (estable) | 1.00× (baseline) | — | — | 0.252 |
| **Heterogéneo** | V100 + CPU (Verode, gloo) | vit_tiny | ~225s | **0.12×** | ~6% | DDP síncrono + hardware desbalanceado + NFS | 0.278 |
| Single-GPU | 1× T4 (Kaggle) | vit_tiny | 47.5s/ep total | 1.00× | — | — | 0.280 |
| DDP 2 GPU | 2× T4 (Kaggle, NCCL) | vit_tiny | 37.3s/ep total | **1.27×** | 64% | I/O-bound (modelo diminuto) | 0.263 |
| Single-GPU | 1× T4 (Kaggle) | vit_base | 179.3s/ep total | 1.00× | — | — | 0.406 |
| **DDP 2 GPU** | **2× T4 (Kaggle, NCCL)** | **vit_base** | **94.5s/ep total** | **1.90×** | **95%** | **compute-bound → escala casi perfecto** | 0.410 |

**Conclusión demostrada con datos reales:** el speedup del entrenamiento distribuido depende del **ratio cómputo/IO y del balance del hardware**:
- **Compute-bound (vit_base): escala casi linealmente** (1.90×, 95% eficiencia en 2 GPUs reales) — las GPUs trabajan en paralelo de verdad.
- **I/O-bound (vit_tiny): escala poco** (1.27×, 64%) — leer los TIFF domina y añadir GPUs no acelera el disco.
- **Hardware desbalanceado (V100+CPU): penaliza** (0.12×) — el DDP síncrono va al ritmo del nodo más lento (la CPU ~50× más lenta que la V100); la GPU pasa el tiempo esperando.
- El Val F1 es idéntico entre single y DDP en todos los casos → la sincronización de gradientes (incl. la normalización por batch global del heterogéneo) es matemáticamente correcta.
- El **feasibility lo predice** en cada caso: marca I/O-bound (ratio≈23 para vit_tiny en NFS) y recomienda el nº de GPUs (1 si I/O-bound, varias si compute-bound). **Validado en las T4: el `DDPOptimizer` predice vit_base 2-GPU en 1.92× (real 1.90×) y vit_tiny en 1.0× (I/O-bound, real 1.27×).** (Tras corregir un bug que asumía interconexión Gigabit por NFS y daba 0.29×.)

**Limitación de hardware (clave para la memoria):** Verode solo tiene **1 GPU usable** (verode21 V100, CC 7.0). verode16 (M2090, CC 2.0) y verode18 (K40m, CC 3.5) están por debajo del mínimo CC 3.7 de PyTorch 2.x → no hay multi-GPU NCCL real en Verode. Por eso el speedup positivo se midió en Kaggle (2×T4), y en Verode la comparación distribuida es el heterogéneo GPU+CPU (gloo), un resultado negativo bien documentado.

Artefactos: `logs/verode/single/vit_tiny_patch16_224/` (single), `logs/verode/ddp_hetero/vit_tiny_patch16_224/` (heterogéneo), `logs/kaggle/{single,ddp}/{vit_tiny,vit_base}_patch16_224/` (Kaggle) + feasibility en `logs/verode/feasibility/` y `logs/kaggle/feasibility/`.

### Sesión Kaggle 2×T4 — 5 estrategias comparables + validación del feasibility (2026-06-10)

Sesión única en Kaggle (2× Tesla T4) que extiende el estudio: **mismo modelo (vit_base), mismo subset (5000/1500), mismos 15 epochs, mismo batch global (96)**, variando solo la estrategia y la precisión. Primero se corrió el **feasibility completo** (predicción), luego las 5 ejecuciones (validación). Todo `precision=fp32` salvo donde se indica AMP.

| Estrategia | Hardware | Train s/epoch | Speedup | Best Val F1 | Energía/ep train |
|---|---|---|---|---|---|
| Single fp32 | 1×T4 | 194 s | 1.00× | 0.572 | 3.32 Wh (1 GPU) |
| **Model-parallel** (pipeline) | 2×T4 | ~211 s (epoch total) | **1.02×** (no acelera) | 0.534 | — (sin instrumentar) |
| **DDP** (datos, NCCL) | 2×T4 | 99 s | **1.96×** | 0.547 | 1.67 Wh/GPU (≈3.34 total)\* |
| **Single AMP** (Tensor cores) | 1×T4 | 51 s | **3.80×** | 0.550 | 0.85 Wh (1 GPU) |
| **DDP + AMP** | 2×T4 | 32.5 s | **5.97×** | 0.552 | 0.44 Wh/GPU (≈0.88 total)\* |

\* **Estos runs (10/06) se midieron con la instrumentación previa, que medía una sola GPU** (`_PowerSampler` leía el dispositivo 0 y en DDP solo el rank 0 escribía el log). Valores medidos por época de train (épocas estables, descartado el warmup): single fp32 **3.32 Wh** y single AMP **0.85 Wh** (1 GPU haciendo TODO el dato); DDP **1.67 Wh/GPU** y DDP+AMP **0.44 Wh/GPU** (cada GPU hace la MITAD del dato). El "total" de los DDP es esa medida **× 2 GPUs** (válido por simetría: T4 idénticas, reparto 50/50). El model-parallel de esta sesión no llevaba `--fn energy` → sin dato (el "—"). **Corregido el 24/06/2026** (rama `fix/energy-multi-gpu`): `_PowerSampler` mide ahora **el conjunto de GPUs del run** y suma su potencia; `measure_energy` resuelve los dispositivos (en DDP el rank 0 suma todas las GPUs del nodo; acepta lista explícita); el script de model-parallel ya tiene `--fn energy` sobre sus dos GPUs. Los runs **futuros** registran el total real; estos quedan como están.

**Conclusiones (datos reales, una sola sesión → máxima comparabilidad):**
- **Datos escala** (1.96×, ~98% eficiencia, compute-bound). **Modelo NO escala cuando cabe en 1 GPU** (1.02×): el paralelismo de modelo *naive* serializa las etapas (una GPU trabaja mientras la otra espera) → su utilidad es *permitir* modelos que no caben, no acelerar.
- **La precisión (Tensor cores) gana al paralelismo de datos:** Single AMP en 1 GPU (3.80×) supera al DDP fp32 en 2 GPUs (1.96×), y con **~4× menos energía total del trabajo** (0.85 Wh en 1 GPU vs ≈3.34 Wh entre las 2 — ver nota \* de la tabla). La precisión importa más que el nº de GPUs aquí.
- **DDP+AMP se combina pero NO multiplica:** esperado 1.96 × 3.80 = 7.4×, medido **5.97×**. La comunicación de gradientes no se acelera con AMP, así que pesa relativamente más → la eficiencia del DDP cae **98% (fp32) → ~78% (AMP)**. Amdahl puro, conectado con el análisis cómputo/comunicación del feasibility.
- **Val F1 ≈ 0.55 en las 5** → la matemática es correcta en todas las estrategias (el F1 es bajo por ser subset de 5000; la calidad real está en los runs del dataset completo en V100, ~0.68).
- **Model-parallel validado por primera vez en 2 GPUs reales** (en local solo se pudo cruzar `cuda:0→cpu`).
- **Caso de uso REAL del paralelismo de modelo (vit_large, 303M params):** en una sola T4, vit_large da **CUDA OOM a batch 48** (el feasibility lo predijo: marcó OOM en 48/64, solo cabe batch 32 a 13.78 GB). Partido **12/24** entre las 2 T4 con model-parallel, **entrena 1 epoch sin OOM** (~9 GB/GPU, 614 s). Confirma la conclusión: el paralelismo de modelo **no acelera, pero hace posible entrenar modelos que no caben en una sola GPU** — su verdadero motivo de existir. Artefacto: `logs/kaggle/model_parallel/vit_large_patch16_224/`.

**Validación del feasibility (predicho → real):** 3 predicciones acertadas en la T4:
- Tiempo single: ~52 min/15ep predicho → ~54 min real (**+4%**).
- Speedup DDP 2-GPU: **1.94×** predicho → **1.96×** real (**<1%**).
- Speedup precisión FP32→AMP: **3.87×** predicho (`--compare-precision`: 27 vs 103 img/s) → **3.80×** real (**<2%**).
- Specs T4 confirmadas: 40 SMs, **2560 CUDA cores / 320 Tensor cores**.

Artefactos (integrados en el repo): `logs/kaggle/single/vit_base_patch16_224/` (single fp32 `173904`, single AMP `203609`, deep-trace `train_deep_205332`), `logs/kaggle/ddp/vit_base_patch16_224/` (DDP fp32 `190526`, DDP+AMP `211814`), `logs/kaggle/model_parallel/{vit_base,vit_large}_patch16_224/` (C + el OOM-vs-split), feasibility (vit_base `172939` + vit_large `213654`) en `logs/kaggle/feasibility/`.

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

Dependencias principales: `torch`, `timm`, `torchvision`, `torchinfo`, `tqdm`, `rasterio`, `pandas`, `pyarrow`, `pyyaml`, `matplotlib`, `nvidia-ml-py`, `streamlit`, `streamlit-option-menu`, `plotly`

---

## Dashboard web

`src/web/` — interfaz Streamlit profesional para gestionar y analizar el proyecto de principio a fin. **UI en inglés** (post-seminario 3) y **arquitectura modular (SRP)**: `app.py` es un orquestador delgado (~220 líneas: page config + CSS + sidebar + dispatch) que monta el menú de iconos + las 5 páginas y delega en módulos `render(ctx)`. **Tema visual** en `.streamlit/config.toml` (primario azul #1A5276, toolbar mínima) — sin él Streamlit usa su rojo por defecto.

```
.streamlit/config.toml      # tema (primaryColor azul, toolbarMode minimal)
src/web/
  __init__.py
  app.py                    # orquestador: page config + CSS + menú de iconos (streamlit-option-menu)
                            #   + selector de run activo + toggle claro/oscuro + dispatch a tabs/
  ui/
    context.py              # DashboardContext (runs, selected_run, run, refresh_interval) → ctx
    theme.py                # diseño: register_plotly_template(mode) + inject_css(mode) — modo
                            #   claro/oscuro (toggle en la sidebar); paleta COLORS/GOOD/WARN/BAD
    charts.py               # helpers Plotly + estilo: _show, _dl_csv, _metric_fig, _overlay_fig,
                            #   _base_layout, COLORS, _CLASS_GROUPS (única llamada a st.plotly_chart)
    helpers.py              # loaders cacheados (_load_df, _get_runs, _class_gallery,
                            #   _load_val_support, _dataset_meta_path…) + utilidades
  tabs/
    home.py                 # render(ctx) — Overview compacto (1 pantalla): tira de 8 KPIs +
                            #   3 gráficas variadas (barras GPU-h/entorno · tarta split train/val/test ·
                            #   treemap desbalance de clases) + tarjeta del run activo (curvas F1/loss) +
                            #   carrusel de fotos (5/pág, ◀▶, las 19 clases) + tabla "All runs" seleccionable
    run/                    # paquete: Curves / Per-class / Confusions / Batch / Details (una fila de pestañas)
                            #   perclass.py = heatmap clases×(P/R/F1) + support (val); confusions.py = top
                            #   co-activación + matriz 19×19 (avanzado); curves.py = veredicto color-coded
    comparison.py           # Compare unificado: multiselect → resumen + speedup vs baseline + radar + energía + overlays
    feasibility/            # paquete: Predict (predictor analítico) / Compare vs runs (predicho vs real) / Report (benchmark)
                            #   predict.py = formulario → headline + tabla fórmula tiempo + tabla fórmula memoria
                            #   + calidad + escalado 1→8 GPU + coste; validate.py = Compare vs runs; report.py + ddp.py + study.py
    data_models.py          # Import runs (subir zip / apuntar a carpeta → copia a logs/)
  run_import.py             # import_run_archive (zip) / import_run_folder / _dest_relpath (puro, testeable)
  run_registry.py           # descubre runs con rglob (estructura plana y profunda);
                            # RunInfo con env, mode, model, precision (leída del log) y CSV paths;
                            # label compacto con tags [ddp]/[amp]/[deep] (defaults implícitos)
  log_parser.py             # parsea logs --trace simple y --trace deep → DataFrame (fallback)
  batch_parser.py           # lee batch_metrics_*.csv → DataFrame por batch
  perclass_parser.py        # lee perclass_metrics_*.csv → DataFrame por clase
  benchmark_parser.py     # lee feasibility_*.csv → (metadata dict, benchmark DataFrame); bloque #gpu
  confusion_matrix_parser.py # lee confusion_matrix_*.csv → matriz numpy por epoch
  system_monitor.py         # snapshot CPU/RAM/disco/GPU/red; GpuInfo con specs de GPU (usado por el predictor)
```

**Motor analítico de predicción** (`src/performance_model.py`): **único motor** de predicción del proyecto. Predice tiempo/speedup/memoria/cuello/coste **y calidad (F1)** de **cualquier** (estrategia, modelo, GPU, nº GPUs, dataset, batch, precisión) **sin benchmark** (fórmula maestra `T(n,π)`, estimación de `r_c/r_io/π` por specs, modelo de memoria/OOM). Calibrado contra los datos reales de Kaggle (<10%). Expuesto en **Feasibility → Predict** (una pantalla: tiempo + memoria + coste + calidad + puente de calibración `rc_measured`). Derivación en `docs/performance_model.md`.

**Predicción de calidad honesta (`predict_quality`, 18/06/26):** la F1 esperada ya **no** es una tabla histórica plana que ignoraba el tamaño del dataset (devolvía 0.68 para vit_base aunque fuera el subset de 5000, real ≈ 0.55). Ahora usa la ley de escalado por datos `F1_inf(N) = F1_full − k·log10(N_full/N)`, calibrada con dos puntos reales (vit_base full=0.68 / subset-5000=0.55 → k≈0.078; vit_tiny subset-5000 → 0.27, también medido). Curva de aprendizaje saturante estándar, banda de incertidumbre por confianza, etiquetada honestamente como **empirical prior** (no medición — el estudio de convergencia sigue siendo la vía medida). `PerformancePredictor (src/benchmark/predictor.py)` delega en este motor y le pasa el tamaño real del split. Rama: `feature/unified-prediction-engine`.

**`scripts/eval.py` — evaluación en TEST (arreglado 18/06/26):** estaba **roto** (llamaba a `build_model(dropout=…)`, kwarg inexistente → reventaba; por eso nunca generó un `test_*.csv`). Corregido + endurecido (guard de split vacío, `--metadata`, `--max-batches`). Verificado end-to-end. Su salida (`test_*.csv`: per-clase + fila `# aggregate`) ya **se muestra en la web**: `run_registry` descubre los `test_*.csv` por carpeta y **Run results** muestra una tarjeta "Held-out test set" (F1 macro @0.5 y @umbral óptimo, accuracy, clases con F1=0) — el número honesto final, separado de la validación. Parser: `src/web/eval_parser.py`.

**Specs de GPU** (`src/gpu_specs.py`): deriva núcleos CUDA y Tensor de cada GPU a partir de su compute capability (→ arquitectura → cores/SM) × nº de SMs (V100 5120/640, T4 2560/320, RTX 3060 Ti 4864/152). El `benchmark.py` los registra (bloque CSV `#gpu`) y acepta `--device INDEX`; el predictor analítico los usa para estimar `r_c`.

**Modelos de speedup seleccionables** (`src/estimation_models.py`): leyes analíticas Linear / Amdahl / Gustafson (puras, con fórmula), superponibles sobre la estimación compute/IO-aware del feasibility en **Feasibility → Report (DDP)**, con slider de fracción serial *s*.

### Arranque

```bash
uv run streamlit run src/web/app.py
# Abre http://localhost:8501
```

### Estructura (v9 — menú de iconos, 5 secciones, selección de run por tabla)

**Navegación:** menú lateral con iconos (`streamlit-option-menu`) — **5 secciones**. **Regla de diseño: nunca más de un nivel de pestañas dentro de una sección.** El **run activo** es estado compartido (`session_state["run_label"]`): se elige clicando una fila de la tabla del Overview (`st.dataframe(on_select)`) o con el selector compacto de la sidebar (filtro de entorno + etiquetas cortas, sin desbordamiento). System se eliminó (el monitor de hardware no servía con el flujo Kaggle) y "Import runs" vive en su propia sección.

| Sección (menú) | Contenido |
|---|---|
| **Overview** | Dashboard compacto: tira de 8 KPIs · ranking Best F1 · tarjeta del run activo (mini F1/loss + veredicto) · velocidad media/epoch · donut de estrategias · **panel del dataset** (splits, subset usado por los runs recientes, clases frecuentes, **galería de las 19 clases** con info multi-etiqueta) · tabla "All runs" seleccionable con sparklines |
| **Run results** | **Una fila de pestañas**: Curves · Per-class (barras+tendencia) · Confusions (diagnósticos multi-etiqueta) · Batch (selector de vista) · Details (tiempo + config/anomalías/log) |
| **Compare** | **Sección única**: multiselect (≤8 runs) → resumen · speedup vs baseline (tabla+barras+veredictos+validación feasibility+escalado) · radar · energía · overlays |
| **Estimate/Benchmark** (antes "Feasibility"→"Performance") | 3 pestañas, en orden: **Estimate** (antes "Predict"; predictor **analítico**: formulario → headline + **tablas con las fórmulas** de tiempo `max(compute,I/O)+sync` y memoria `weights+grad+Adam+activations+overhead` con los valores puestos + calidad + escalado 1→8 GPU + coste nube; mismo motor que `tfg estimate`) · **Benchmark** (antes "Report"; **medición empírica** generada en terminal con `tfg benchmark`: hardware, throughput, tiempo, escalado distribuido, coste, estudio de convergencia) · **Benchmark vs Run** (antes "Compare vs runs"; predicho vs real con tabla de fórmula por métrica) |
| **Import** | Importar runs entrenados en otra máquina (zip o carpeta → copia a `logs/`) |

**Predicción vs realidad** (rediseño de la antigua "Comparar vs training"): elige el run en la barra lateral → auto-empareja su feasibility (mismo modelo/entorno); muestra estimado vs real con semáforo y un veredicto en lenguaje natural (precisa / optimista por I/O / pesimista). Avisa si el run es distribuido (la estimación es single-GPU → la diferencia incluye el speedup).

**Descarga en todas las pestañas:**
- Gráficas Plotly: ícono de cámara en la barra de herramientas → descarga PNG (escala 2×, client-side)
- Tablas: botón "Descargar CSV" en cada tabla principal

Descubre runs recursivamente en `logs/` (estructura flat legacy y profunda env/mode/model).
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
  - `decorators/confusion.py`: `ConfusionMatrixDecorator` — CSVs de per-class y confusion matrix (sin PNGs; el dashboard genera las gráficas)
  - `decorators/batch_monitor.py`: `BatchMonitorDecorator` — CSV con running_loss, batch_loss (instantánea) y lr por batch; `--batch-log-every N` controla granularidad
  - `decorators/metric_reporters.py`: `LossReporter`, `F1Reporter`, `AccuracyReporter`, `PrecisionRecallReporter`
  - `fn_decorators.py`: `@timed`, `@log_call`, `@measure_energy`, `@retry_on_cuda_oom` — rutean a logger
  - `builder.py`: `TrainingSessionBuilder` — fluent API para montar el stack completo
  - `augmentations.py`: `mixup_batch()` — mezcla de batch compatible con multi-label
- [x] Técnicas anti-overfitting v3: label smoothing, mixup, threshold search, dropout 0.3, weight decay 0.1
- [x] `metrics.py`: métricas extraídas en módulo propio (sin duplicación)
- [x] Flags `--trace / --layers / --fn / --metrics / --inspect / --model` en script de entrenamiento
- [x] Inspección modular: `--inspect model-summary batch-table grad-monitor anomalies`
- [x] Early stopping: `patience` configurable en `EpochController`
- [x] Log con timestamp (DDMMYYYY) a fichero en `logs/{env}/{mode}/{model}/` (ya no se generan PNGs — el dashboard web genera las gráficas de forma interactiva desde los CSVs)
- [x] `benchmark.py`: benchmark train+eval por separado, `--nfs-factor`, auto-save log + CSV en `logs/{env}/feasibility/`
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
- [x] **Web dashboard v2 (20/05/26):** 7 tabs, CSV-driven (epoch_metrics, perclass_metrics, feasibility), Plotly interactivo por clase, pestaña Feasibility, pestaña Time Analysis; `perclass_parser.py`, `benchmark_parser.py`; `benchmark.py` añade `--model` y escribe CSV
- [x] Diagrama de clases v2: DDPTrainer, TracingDecorator con epoch_csv, ConfusionMatrixDecorator con write_csv, ReportFormatter con write_csv, RunInfo con epoch/perclass csv paths, web con 7 tabs (20/05/26)
- [x] **Heatmap 19×19 de confusión — CSV + Plotly interactivo (26/05/26):** `ConfusionMatrixDecorator` genera `confusion_matrix_TIMESTAMP.csv`; `confusion_matrix_parser.py` lee el CSV; sub-tab muestra heatmap Plotly interactivo con hover y selector de epoch
- [x] **Web dashboard v3 (27/05/26):** 9 tabs, interfaz profesional sin emojis; Launcher (lanzar entrenamientos con output en tiempo real); Live Monitor (auto-refresh, GPU via nvidia-smi); mejoras en todas las pestañas (moving average, comparativa multi-run, anomaly detection, etc.)
- [x] **Gestión de carpetas y gitignore (27/05/26):** estructura `{env}/{mode}/{model}/` para logs, plots y checkpoints; feasibility en `{env}/feasibility/`; `run_registry.py` con rglob; `RunInfo` añade `mode` y `model`; `.gitignore` corregido — todos los CSVs y logs bajo `logs/` se commitean
- [x] Diagrama de clases v3: RunInfo con mode/model, web con 9 tabs, confusion_matrix_parser (27/05/26)
- [x] **Multi-model feasibility (27/05/26):** `benchmark.py --model` acepta N modelos separados por espacio (`nargs="+"`) — cada modelo genera su propio par log/CSV con timestamp independiente
- [x] **DDP CPU/gloo support (27/05/26):** `train_ddp.py` lee `backend` del config; `DDPTrainer` omite `device_ids` en CPU; `configs/train_ddp_cpu_test.yaml` con backend gloo, vit_tiny, pretrained=false — permite validar infraestructura multi-nodo sin GPU compatible
- [x] **Fix ZeroDivisionError scheduler (27/05/26):** `T_max = max(1, epochs - warmup_epochs)` en `builder.py` — evita división por cero cuando `epochs ≤ warmup_epochs`
- [x] **Feasibility multi-modelo local (27/05/26):** vit_tiny, vit_small, vit_base, resnet50 con batch-sizes 16 y 32; trace-modes off y simple → 4 pares log/CSV en `logs/local/feasibility/`
- [x] **Entrenamiento local vit_tiny 5 epochs (27/05/26):** Val F1=0.590 (epoch 5, mejorando en todos los epochs), ~11 min/epoch, stack completo (plot, hooks, confusion, batch-monitor, energy, timing) → `logs/local/single/vit_tiny_patch16_224/train_27052026_221827.log`
- [x] **Entrenamiento clúster v4 (27-28/05/26):** vit_base, batch=64, config v3, Val F1=0.6816 (epoch 7) — nuevo récord; early stop epoch 17; clase 6 F1=0.000 detectada como caso problemático → `logs/verode/single/vit_base_patch16_224/train_27052026_210223.log`
- [x] **Eliminación de PNGs del training (02/06/26):** `ConfusionMatrixDecorator` y `PlottingDecorator` ya no generan archivos PNG (matplotlib eliminado); solo generan CSVs. `RunInfo` limpiado — sin `plot_path`, `perclass_paths` ni `confusion_matrix_paths`. `.gitignore` añade `plots/**/*.png` (15 MB de PNGs históricos fuera de git). Feature: `feature/no-png-output`.
- [x] **Dashboard web v6 — español + pantalla inicio grid + descarga (02/06/26):** `app.py` reescrito en español (tecnicismos en inglés); pestaña **Inicio** rediseñada como pantalla principal con cuadrícula (métricas globales, mini curvas, sistema, por clase, tabla runs); helper `_show()` activa barra de herramientas Plotly con descarga PNG en todas las gráficas; helper `_dl_csv()` añade botones "Descargar CSV" en todas las tablas; eliminadas referencias rotas a atributos PNG inexistentes de RunInfo. 14 pestañas: Inicio / Sistema / Dataset / Modelos / Curvas / Por clase / Batch / Comparar / Análisis DDP / Viabilidad / Tiempo / Información / Lanzador / En vivo. Feature: `feature/web-dashboard-es`.
- [x] **Suite de tests ampliada a 132 (02/06/26):** `tests/integration/test_no_png_output.py` (5 tests), `tests/integration/test_web_dashboard_es.py` (21 tests).
- [x] **Batch monitor v2 (02/06/26):** Firma del hook extendida a `(epoch, batch_idx, n_batches, running_loss, batch_loss, lr)`. `BatchMonitorDecorator` ahora registra también la loss instantánea del batch y el LR actual. `log_every=1` por defecto (antes 50). Nuevo flag `--batch-log-every N` en ambos scripts de entrenamiento. Pestaña Batch con 3 sub-tabs: "Por epoch" (selector de métrica, MA, detección picos), "Historia global" (eje x = batch global con límites de epoch), "Learning rate" (curva LR completa con escala log automática). Feature: `feature/batch-live-metrics`. 10 tests nuevos.
- [x] **Feasibility checker v3 (02/06/26):** Perfilado completo del sistema (CPU cores/RAM, GPU compute capability, tipo de disco, detección NFS, medición I/O real con patches TIFF). `DatasetProfiler` calcula `io_bottleneck_ratio` para detectar si el entrenamiento es I/O-bound o compute-bound. `PerformancePredictor` genera curva F1 val+train predicha con banda de incertidumbre (±0.015 F1) basada en datos históricos reales. `DDPOptimizer` calcula speedup real (≠ lineal) incluyendo overhead de sincronización de gradientes, eficiencia por tipo de red (NVLink/PCIe/NFS/Ethernet), y recomienda batch_per_gpu + num_workers. CSV v3 ampliado con bloques `#cpu`, `#disk`, `#dataset`, `#prediction`, `#curve_*`, `#ddp`. Pestaña Viabilidad con 5 sub-tabs: Informe, Análisis DDP (rectángulos de % cómputo/I/O/sync), Predicción F1, Comparar vs training, Ejecutar análisis. Feature: `feature/feasibility-v3`. 20 tests nuevos.
- [x] **Suite de tests en 161 (02/06/26).**
- [x] **DDP heterogéneo corregido y listo (02/06/26):** 3 bugs críticos resueltos: (1) `mixup_batch` devuelve 2 valores, no 4; (2) doble DDP-wrapping eliminado — el builder crea `HeterogeneousDDPTrainer` directamente vía `with_heterogeneous_ddp()`; (3) batch hooks no disparaban — ahora `HeterogeneousDDPTrainer.train_epoch` llama a `self._batch_hooks` con la firma v2. Nuevos métodos en builder: `with_heterogeneous_ddp(local_bs)` y `with_output_mode(mode)`. `train_heterogeneous_ddp.py` reescrito limpio, sin `_find_core/_rewrap`. 11 tests nuevos. **172 tests en verde.** Feature: `feature/fix-heterogeneous-ddp`.
- [x] **Smoke test local + verificación web con Playwright (02/06/26):** feasibility v3 + training vit_tiny 5 epochs (Val F1=0.6292). Verificación visual del dashboard con Playwright + Chrome del sistema → detectó y corrigió 2 bugs: `yaxis` duplicado en la gráfica DDP (rompía todo el dashboard) y orden cronológico de runs (`RunInfo.sort_key` normaliza DDMMYYYY → YYYYMMDD). Features: `feature/fix-chronological-sort`.
- [x] **Tab Dataset arreglado + imágenes por clase (03/06/26):** bug del ndarray en `class_distribution_from_parquet` (las labels de BigEarthNet son ndarray, el `isinstance(x,(list,set))` daba 0 a todo → gráfica vacía); ahora se aplanan y cuentan vectorizado. Escalado de todas las gráficas del tab corregido (márgenes, alturas, automargin). Nueva sección de **imágenes de ejemplo por clase** (`find_example_patches` + `load_rgb_image` cargan bandas B04/B03/B02 con rasterio + stretch de percentiles). Fix del nombre de GPU cortado en Inicio. 11 tests. Feature: `feature/dataset-tab-fixes`.
- [x] **Métricas por batch F1/accuracy/precision (03/06/26):** el batch hook pasa ahora un dict de métricas `(epoch, batch_idx, n_batches, metrics)` en vez de args posicionales. `Trainer` y `HeterogeneousDDPTrainer` computan F1/accuracy/precision por batch. CSV `batch_metrics` v3 con columnas `batch_f1`, `batch_acc`, `batch_prec`. El tab Batch ofrece selector de métrica (loss/F1/accuracy/precision) con eje [0,1] para las no-loss. Feature: `feature/batch-metrics-full`.
- [x] **Feasibility v4 — estudio empírico real (03/06/26):** nuevo módulo `src/training/convergence_study.py` (`ConvergenceStudy`): (1) LR range test (Smith 2017) que barre LRs y mide la loss → recomienda LR óptimo; (2) mini-training real de N steps con datos reales → ajusta power law `loss=a·t⁻ᵇ+c` (con suavizado) → extrapola loss/F1/plateau; (3) gradient noise scale (McCandlish 2018) → batch size crítico. Flags `--convergence-study --study-steps N`. Sub-tab "Estudio real" en la web con las 3 gráficas. La predicción empírica histórica se mantiene en "Predicción F1" para comparar medido vs histórico. Verificado con vit_tiny + SSD: LR sugerido 8.86e-05, throughput real 410 img/s, R² 0.74. Feature: `feature/feasibility-study`.
- [x] **Suite de tests en 212 (03/06/26):** +11 dataset, +14 convergencia, +6 parser estudio, +7 sort cronológico, +5 batch metrics.
- [x] **Training distribuido heterogéneo en Verode (03-04/06/26):** verode21 (V100, rank 0) + verode16 (CPU, rank 1), gloo. Demo vit_tiny subset completado; fix de `train_loss` inflada ~n_clases (dividir por `step_bs × n_clases`); fix `ReduceOp.AVG` no soportado por gloo (usar SUM + dividir). Artefactos en `logs/verode/ddp_hetero/vit_tiny_patch16_224/` (Val F1 0.278).
- [x] **Estudio apples-to-apples single vs distribuido (04/06/26):** baseline single-GPU V100 (`configs/train_demo_single.yaml`, `logs/verode/single/vit_tiny_patch16_224/`) comparable con el heterogéneo. Ver sección "Estudio single-GPU vs distribuido". Conclusión: heterogéneo V100+CPU es **8.6× más lento** que la GPU sola (speedup 0.12×) por el cuello síncrono.
- [x] **Speedup positivo real en Kaggle 2×T4 (04/06/26):** `scripts/export_kaggle_subset.py` + `docs/kaggle_speedup_runbook.md`. vit_tiny 1.27× (64%, I/O-bound) y **vit_base 1.90× (95%, compute-bound)**. Artefactos en `logs/kaggle/{single,ddp}/{vit_tiny,vit_base}_patch16_224/`. Hallazgo: NCCL no permite 2 ranks en 1 GPU ("Duplicate GPU detected") → fix `device = cuda:(local_rank % device_count)` en `train_ddp.py`.
- [x] **Auditoría dashboard — 3 fixes (04/06/26):** (1) parser deep `_parse_deep` reescrito por extracción de campos por nombre (los logs tenían 2 órdenes distintos de `val_f1/best/val_acc` → el run v1 de 30 epochs estaba invisible); (2) pestaña Análisis DDP filtra `mode.startswith("ddp")` → empareja también `ddp_hetero`, etiquetas conscientes del modo (V100+CPU vs 2 GPUs) + aviso cuando speedup<1; (3) parser energía/timing casa `\w*Trainer` (DDPTrainer/HeterogeneousDDPTrainer) y `_load_df` mezcla energía del log al CSV. Borrados 3 runs abortados vacíos.
- [x] **Fix tamaño de dataset en feasibility (04/06/26):** `benchmark.py` lee el conteo real de splits del metadata del config (antes hardcodeaba 237871 → estimaba 47× de más para el subset); escribe bloque `#sizes` (n_train, n_val, nfs_factor). `benchmark_comparison.py` y `benchmark_parser.py` usan ese tamaño real (con fallback al full set). Pestaña Tiempo: la línea de estimación usa el feasibility del mismo modelo que el run.
- [x] **Feasibility en Kaggle 2×T4 + validación estimación-vs-real (04/06/26):** `logs/kaggle/feasibility/` para vit_tiny y vit_base. Reveló que el `DDPOptimizer` predecía **vit_base 2-GPU en 0.29×** cuando el real es **1.90×**.
- [x] **Fix modelo de predicción DDP (04/06/26):** dos bugs en `DDPOptimizer`: (1) `_infer_network_type` asumía interconexión **Gigabit (0.125 GB/s)** solo porque el disco era NFS → all_reduce ~128× inflado. Ahora: **NVLink** para GPUs de datacenter (V100/A100…), **PCIe** para el resto (T4/RTX), Ethernet solo multi-nodo CPU. (2) `_build_scenario` reescrito a nivel de **epoch**: `time = max(compute/n_gpus, io_total) + sync` — el I/O es un total fijo que no escala con nº de GPUs. **Predicción tras el fix: vit_base 2-GPU 1.92× (real 1.90×), vit_tiny 1.0× (I/O-bound, real 1.27×).**
- [x] **Fix `recommend_config` (04/06/26):** antes usaba speedup/n_gpus (= eficiencia) → siempre recomendaba 1 GPU. Ahora recomienda el mayor nº de GPUs con eficiencia ≥75% → compute-bound sugiere escalar (vit_base → 4 GPUs), I/O-bound se queda en 1.
- [x] **Suite de tests en 229 (04/06/26):** +13 respecto a 212 (parser deep ambos formatos, energía distribuida, ddp_hetero distribuido, round-trip #sizes, comparación con tamaño correcto, predicción DDP compute/IO-bound, recomendación). 14 pestañas verificadas sin errores con Playwright.
- [x] **Dashboard web v7 — reorganización 14→6 pestañas + "Predicción vs realidad" rediseñada (04/06/26):** las 14 pestañas planas se anidan en 6 de nivel superior (Inicio · Run · Comparativa · Viabilidad · Datos y modelos · Sistema) usando la colocación-en-creación de contenedores de Streamlit (sin reescribir el contenido). Viabilidad pasa de 6 a 4 sub-tabs: Informe absorbe los escenarios DDP, y "Comparar vs training" se rediseña como **Predicción vs realidad**. 6 pestañas verificadas sin errores con Playwright. Feature: `feature/web-redesign-6tabs`.
- [x] **Predicción vs realidad en 2 secciones + nombres legibles + gráfica simple (04/06/26):** la pantalla separa **(A) En 1 GPU: tiempo estimado vs real** (contra el run single) y **(B) Al distribuir: speedup predicho vs real** (single ÷ ddp; valida el DDPOptimizer: 1.92× predicho vs 1.90× real). Selector explícito "¿Qué comparar?" (entorno · modelo). Los informes de viabilidad se muestran como `entorno · modelo · DD/MM HH:MM` (`_feas_label`) en vez de la fecha cruda. La gráfica de error divergente se sustituye por barras agrupadas estimado-vs-real (Train/Eval/Total, min).
- [x] **Config del run en el log + visible en la web (04/06/26):** los 3 scripts de entrenamiento escriben al inicio del log una línea `Configuración: modelo | batch=…/GPU (global=…) | epochs | lr | train/val` (heterogéneo: batch GPU/CPU + `reparto=GPU %…/CPU %…` usando `len(train_sampler)`). La pestaña **Run → Información** la lee (`_run_config`) y muestra batch size, lr, imágenes y reparto. **Backfill:** los 6 runs de comparación (Kaggle tiny/base single+ddp, Verode single + heterogéneo) llevan esa línea añadida a sus logs con los valores exactos de los configs (marcada `origen=backfill`).

#### Mejoras post-seminario 3 (junio 2026) — propuestas de la reunión 05/06
- [x] **Dashboard web en inglés (09/06/26):** toda la UI traducida a inglés (6 pestañas + sub-pestañas, métricas, gráficas, mensajes, botones) y los comentarios/docstrings de los módulos web. Se conservan en español solo las claves de parseo de logs (`Configuración:`, `threshold óptimo=`, `potencia media`, `RESUMEN`) para casar con logs existentes/backfilleados. Tests del dashboard reescritos a inglés. Ramas: `feature/web-english`.
- [x] **Refactor del monolito → módulos (SRP) (09/06/26):** `app.py` (2972 líneas) partido en orquestador delgado (~127 líneas) + `ui/` (charts, helpers, context) + `tabs/` (home, run, comparison, feasibility, data_models, system), cada uno con `render(ctx)`. Demuestra SRP también en la herramienta de visualización. 6 pestañas + 21 sub-pestañas verificadas con Playwright. Bug latente corregido: `import pandas as pd` inline dentro de `feasibility.render` sombreaba el `pd` del módulo. Rama: `feature/web-split-modules`.
- [x] **Specs de GPU (núcleos CUDA/Tensor) + selección de dispositivo (09/06/26):** `src/gpu_specs.py` deriva núcleos CUDA/Tensor de compute capability × SMs; visibles en System → Monitor y Feasibility → Report; `benchmark.py` los escribe (bloque `#gpu`) y acepta `--device INDEX` (selector en la web). Rama: `feature/gpu-specs`.
- [x] **Modelos de speedup seleccionables (09/06/26):** `src/estimation_models.py` (Linear/Amdahl/Gustafson) superponibles a la estimación compute/IO-aware en Feasibility → Report (DDP), con slider de fracción serial. Rama: `feature/selectable-estimation-models`.
- [x] **Paralelismo de MODELO (pipeline) (09/06/26):** `src/models/model_parallel.py` (`ModelParallelViT`) parte el ViT entre 2 dispositivos (stage 0 + stage 1); el forward reimplementa fielmente `forward_features`+`forward_head` de timm — **verificado numéricamente equivalente al modelo normal en CPU** (test crítico). `scripts/train_model_parallel.py` (bucle propio, 1 proceso; **luego unificado en el builder el 24/06** → ahora usa el mismo stack de decoradores) escribe los artefactos estándar → `logs/{env}/model_parallel/{model}/`; `configs/train_model_parallel_kaggle.yaml` (vit_base, 2×T4) + `docs/model_parallel_runbook.md`. Smoke probado en local cruzando una frontera real `cuda:0`→`cpu` (forward+backward+optimizer). Paralelismo *naive* (sin micro-batches): didáctico — más lento que el de datos en un modelo que cabe en 1 GPU; útil solo para modelos que no caben. **Validado en Kaggle 2×T4 (10/06/26): 1.02× (no acelera) con vit_base** — ver "Sesión Kaggle 2×T4 — 5 estrategias". Rama: `feature/model-parallel`.
- [x] **Selector de precisión = interruptor de Tensor cores (10/06/26):** `src/precision.py` — la precisión numérica decide qué unidades hacen el cómputo: **fp32** → núcleos CUDA; **tf32/amp(fp16)/bf16** → **Tensor cores** (autocast + GradScaler para fp16). `available_precisions()` filtra por compute capability (fp16 en Volta+, tf32/bf16 en Ampere+). Integrado en `Trainer` (+ `DDPTrainer`, builder, `cfg.training.precision`), flag `--precision` en `train_single_gpu.py`. El **feasibility** benchmarkea por precisión y con `--compare-precision` mide FP32 vs Tensor y reporta el speedup (bloques CSV `#precision`/`#precision_cmp`). En la web: selector en **Run analysis** y **Launcher**, y comparación FP32-vs-Tensor en **Feasibility → Report**. **Medido en RTX 3060 Ti (vit_base, batch 32): FP32 68 img/s vs AMP 173 img/s → 2.54× con Tensor cores, y menos VRAM (5.63 → 4.15 GB).** Rama: `feature/precision-tensor-cores`.
- [x] **Vista opcional en español (botón EN/ES) (10/06/26):** la UI se mantiene en **inglés por defecto** (requisito de entrega); `src/web/ui/i18n.py` añade una **capa de traducción global** que parchea los métodos de texto de Streamlit (`DeltaGenerator` + `st.*`) y traduce su etiqueta vía un diccionario EN→ES — sin tocar las ~600 llamadas. Las cadenas no presentes (f-strings dinámicos, títulos de gráficas, identificadores) pasan tal cual. Selector "Language / Idioma" en la barra lateral. `_tr` maneja prefijos markdown (`## …`, `**…**`). Rama: `feature/web-i18n-toggle`.
- [x] **Navegación lateral agrupada (10/06/26):** se sustituye la barra de pestañas superior (con sub-pestañas ocultas) por un **menú lateral siempre visible** agrupado por tarea (ANALYZE: Overview/Run results/Compare · PLAN: Feasibility · DATA & OPS: Data & models/System). `app.py`: `_NAV` + botones de ancho completo que fijan `st.session_state['nav']`; `_PAGES` despacha a los `render(ctx)` de cada módulo. Solo se renderiza la página elegida → mucha menos carga visual; los controles específicos (informe de viabilidad, refresco) aparecen solo en su página. Rama: `feature/web-sidebar-nav`.
- [x] **Auditoría UX — 4 mejoras (10/06/26):** (1) *Inicio* pasa a resumen ejecutivo (fuera el system-status y el snapshot por clase duplicados; KPIs + vistazo del run con veredicto + tabla de runs); (2) **single vs distribuido unificado** en *Compare* (speedup predicho del feasibility + medido, con error/veredicto) y colisión de nombres "DDP analysis" eliminada (`Single vs Distributed — measured speedup` vs `Distributed scaling (predicted)`); (3) **veredictos** de una línea en *Run → Curves* e *Inicio* (mejor epoch, gap train-val, divergencia de val loss); (4) *Feasibility → Report* reorganizado de un megascroll a **4 sub-pestañas por propósito** (Hardware & precision / Dataset I/O & memory / Throughput & time / Distributed scaling). Rama: `feature/web-ux-redesign`.
- [x] **Predicción de coste en la nube (10/06/26, sugerencia del tutor):** `src/cloud_cost.py` — el feasibility ya estima el tiempo total y conoce la GPU, así que predice el **coste** en distintos proveedores: `coste = horas × $/hora`, escalando el tiempo entre GPUs por su throughput FP16 (Tensor cores) relativo. Tabla curada de precios on-demand por GPU (Kaggle/Colab gratis … AWS/GCP/Azure/Lambda/RunPod/Vast) + TFLOPS FP16 por modelo; `estimate_costs()` devuelve filas ordenadas por coste. En la web: sub-pestaña **Cloud cost** en *Feasibility → Report* (elige epochs → tabla ordenada de barato a caro + gráfica de los de pago). Precios editables. Rama: `feature/cloud-cost-prediction`.
- [x] **Suite de tests en 284** (+55 sobre 229: gpu_specs, estimation_models, model_parallel, precision/Tensor-cores, i18n, cloud_cost, dashboard modular). Todo integrado en `develop` y subido a GitHub.
- [x] **Pasada de diseño profesional de la web (10/06/26):** (1) tema propio en `.streamlit/config.toml` — azul corporativo #1A5276 como color primario en vez del rojo por defecto de Streamlit (botón activo, tabs, radios, sliders) + toolbar mínima; (2) el CSS de títulos de app.py estaba muerto (perdía la batalla de especificidad contra el CSS propio de Streamlit) — corregido con scope a `stMarkdownContainer` + `!important`, escala compacta; (3) cabecera consistente en las 6 páginas (título + caption antes de los sub-tabs — Run results y Feasibility aterrizaban directamente en filas de tabs); (4) legends de Plotly dentro del plot (chocaban con la modebar siempre visible) + hover unificado; (5) tabla All runs con gradiente Blues data-driven (el RdYlGn con rango fijo 0.4-0.75 saturaba los runs del subset a rojo alarma); (6) **default comparable en Compare**: el par por defecto prefiere mismo modelo/entorno y traza no-deep que el run DDP elegido (antes emparejaba el single deep-trace → "6.82×/341% eficiencia" engañoso) + avisos de mismatch de modelo/traza; (7) sidebar como menú (botones sin borde, texto a la izquierda) y sin el slider de refresco global duplicado. Verificado con Playwright (6 páginas + toggle ES). Rama: `feature/web-design-polish`.
- [x] **Sidebar sin scroll + tags de precisión en los runs (11/06/26):** (1) la sidebar cabe sin scrollear en un viewport normal — orden nav → selector de run → idioma (al fondo), detalles del run de 8 líneas a 2 (log + CSVs), gaps/divisores compactados por CSS; (2) **`RunInfo.precision`**: `discover_runs` lee `precision=` de la línea `Configuración:` del log (primeros 4 KB) y el label etiqueta los runs Tensor-core (`[amp]`/`[tf32]`/`[bf16]`; fp32 implícito) — los 5 runs de la sesión Kaggle ahora se distinguen a simple vista; (3) label compacto: sin segundos, `[simple]` implícito (solo se etiqueta `[deep]`), modelo sin el sufijo `_patch16_224`; (4) `train_ddp.py` escribe `precision=` en su línea de config (no lo hacía) + backfill en los 2 logs DDP de Kaggle (`190526`=fp32, `211814`=amp, valores documentados aquí); (5) los dropdowns de selectbox copiaban el ancho del control (~200px en la sidebar) y cortaban los tags — el popup no se puede ensanchar (se descentra), así que la lista interior se desborda a la derecha (`overflow: visible` + `min-width` en el `ul`). Verificado con Playwright (sidebar 950=950 sin scroll, dropdown con labels completos). Rama: `feature/web-sidebar-run-tags`.
- [x] **Compare: model-parallel comparable + overlay de 8 runs (11/06/26):** (1) *Single vs Distributed* acepta runs `model_parallel` en el selector de distribuidos (antes solo `ddp*`) — permite "1 GPU vs model-parallel"; etiquetas/veredictos adaptados al modo: para MP la métrica de eficiencia pasa a "Expected (naive pipeline) ≈1×" (el 2× ideal es de paralelismo de DATOS), banner explicativo propio (serialización de etapas + su valor real: vit_large OOM-vs-split), y se omiten la validación del DDPOptimizer y la sección de escalado teórico (son teorías de data-parallel); (2) el default del single ahora también prefiere **misma precisión** que el run distribuido (evitaba emparejar MP fp32 con single AMP → "0.28×" engañoso; con el par correcto da 1.03×, coherente con el 1.02× documentado) + aviso de mismatch de precisión (~3-4× de efecto Tensor cores); (3) *Overlay runs*: hasta **8 runs** (antes 4, no cabían los 6 de Kaggle) y **legends completas bajo cada gráfica** (una fila por run; antes `label[:30]` cortaba justo los tags distintivos) — radar incluido, altura adaptativa al nº de series. Rama: `feature/web-compare-mp-overlay`.
- [x] **Comparación de energía en Overlay runs (11/06/26):** sección "Energy consumption" en *Compare → Overlay runs* — barras horizontales apiladas (train+eval, Wh) con la energía total de cada run seleccionado + veredicto del más/menos eficiente (p. ej. DDP+AMP 8.4 Wh vs single fp32 55.3 Wh → 6.6× menos); el train viene en julios en el log → columna derivada `energy_train_wh`; los runs sin `--fn energy` (p. ej. model-parallel) se excluyen con caption explicativa; el multiselect de métricas por epoch añade `energy_train_wh`/`energy_eval_wh`/`power_train_w` cuando hay datos. Rama: `feature/web-overlay-energy`.
- [x] **Compare unificado en una sola sección (11/06/26):** fuera las 2 sub-pestañas ("Single vs Distributed" / "Overlay runs") — un único multiselect (hasta 8 runs) alimenta todo: resumen (con columnas Mode/Precision), **Speedup analysis generalizado** (todos los runs frente a una **baseline** elegible — default inteligente: single+fp32+no-deep, recalculado al cambiar la selección; el `key` fijo congelaba la elección de la primera renderización), tabla con notas por run (eficiencia vs 2× ideal para ddp, "≈1× expected" para MP, mismatch de modelo/precisión/traza), barras horizontales de speedup, banners pedagógicos (MP/hetero) una sola vez, validación del feasibility (predicho vs medido si hay par single+ddp comparable) y escalado teórico con un punto por run distribuido; después radar + energía + overlays. Reproduce en vivo la tabla de 5 estrategias (1.97×/1.03×/3.69×/5.50× sobre el single fp32). Rama: `feature/web-compare-unified`.
- [x] **Claridad del Speedup analysis + default de sesión (11/06/26):** el selector de baseline confundía (parecía un picker de par de 2 runs) — caption explicativa ("todos los seleccionados arriba se comparan contra UNO — la baseline, que cuenta como 1.00×") + label extendido + aviso si algún run seleccionado no tiene tiempos; el **default del multiselect pasa a ser la última sesión** (runs del mismo entorno y día que el más reciente, máx 8) → Compare abre con las 5 estrategias de Kaggle ya cargadas y la baseline en el single fp32; el "% of ideal 2×" de ddp solo se muestra si la precisión coincide con la baseline (con AMP mezclaba el efecto Tensor cores en la eficiencia DDP). Rama: `feature/web-speedup-clarity`.

#### Mejoras post-reunión 11/06 (4 encargos del tutor)
- [x] **System: fuera Lanzador/En vivo → Importar runs (11/06/26):** el Lanzador y el monitor En vivo no servían con el flujo real (se entrena en Kaggle/Verode, no desde esta máquina). `System` queda en **Monitor** (hardware local) + **Importar runs**: subir el zip que produce un entrenamiento remoto (o apuntar a una carpeta) → `src/web/run_import.py` (puro, testeable: `import_run_archive`/`import_run_folder`/`_dest_relpath` que corta desde el segmento `logs/`, acepta zips de `logs/` o de su contenido, rechaza path traversal) copia los artefactos a `logs/` y el dashboard los descubre. Caches invalidadas tras importar. 6 tests. Rama: `feature/web-import-runs`.
- [x] **Matriz de confusión → diagnósticos multi-etiqueta (11/06/26):** una 19×19 clásica no encaja en multi-label (cada imagen tiene varias de las 19 clases), por eso costaba leerla. El sub-tab **Confusions** muestra lo interpretable de la matriz de co-activación: **recall por clase** (la diagonal, coloreada, con veredicto de las clases que casi nunca detecta), **top confusiones** ("cuando X está presente, el modelo también predice Y"), **perfil por clase**, y la 19×19 completa en un expander con guía de lectura. Quitado el toggle "Absolute" (era erróneo: multiplicaba por la suma de la fila normalizada, no por el soporte real). Helpers testeables `recall_by_class`/`top_confusions`/`confusion_profile` (4 tests). Rama: `feature/web-confusion-multilabel`.
- [x] **Motor analítico del feasibility — `src/performance_model.py` (11/06/26):** predictor de forma cerrada (puro, sin GPU) para tiempo/speedup/memoria/cuello de **cualquier** (estrategia, modelo, GPU, nº GPUs, dataset, batch, precisión) **sin benchmark**, según el brief del tutor. Estima `r_c = MFU·TFLOPS_fp32/FLOPs_train` (MFU=0.17 calibrado), `π` de tabla Tensor-core medida, `r_io` de tabla de disco; fórmula maestra `T(n,π)=φ·[max(compute/n, io)+sync]` por estrategia (single/ddp/model_parallel/heterogéneo) con los regímenes compute/io/sync; modelo de memoria→OOM+batch máx (calibrado a vit_large 13.78 GB@b32 T4); `predict()` unificada con calibración opcional vía `rc_measured`. 16 tests reproducen la tabla real Kaggle 2×T4 (<10%: single +1%, DDP 1.95×, AMP 3.80×, MP 1.00×, vit_tiny I/O-bound, OOM de vit_large). Expuesto en **Viabilidad → Predictor** (formulario + curva 1→8 GPUs). `docs/performance_model.md` con derivación + tabla de validación. Rama: `feature/feasibility-analytic-model`.
- [x] **Rediseño web — hub tipo wandb (11/06/26):** **Inicio (Overview)** pasa a ser un hub: KPIs en tarjetas (`container(border=True)`), fila de **tarjetas de navegación** a cada sección (botón Open → `st.session_state['nav']`), run destacado + run seleccionado en tarjetas con mini F1/loss + veredicto, y tabla "All runs" con **sparkline de Val F1 por run** (`st.column_config.LineChartColumn`) + barra de Best F1 (`ProgressColumn`) + tags mode/precision/env. Verificado con Playwright (EN+ES). Rama: `feature/web-hub-redesign`.
- [x] **Suite de tests en 311** (+26 sobre 285: run_import 6, confusion multi-label 4, performance_model 16; dashboard/i18n actualizados).
- [x] **Rediseño de arquitectura de información + copy neutro (17/06/26):** segundo encargo del tutor sobre la interfaz (seguía liosa). (1) **Menú lateral de iconos** (`streamlit-option-menu`) en vez de botones; las tarjetas "Open" del hub saltan vía `_nav_jump`+`manual_select`. (2) **Eliminado el anidamiento de 3 niveles** (la causa de la confusión): Run results pasa a **una sola fila** (Curves · Per-class · Confusions · Batch · Time · Info) — Confusions y la tendencia suben al primer nivel, Batch usa un selector de vista horizontal; **Feasibility → Report** pasa de 5 sub-sub-pestañas a **expanders plegables** en una página (summary-first). (3) **Pasada de copy a registro neutro/profesional** (fuera "Did the model catch each class?", "what gets confused with what", "the GPU waits for the CPU", "in front of you", "drag the macro-F1 down"…). Herramienta: se mantiene Streamlit (correcto y publicable; hosting gratis en Streamlit Community Cloud). Verificado con Playwright (nav, aplanado, expanders, tarjetas Open, EN/ES). Rama: `feature/web-ia-redesign`.
- [x] **Rediseño compacto + selección por tabla + Feasibility en 3 fases (17/06/26):** tercera iteración tras feedback ("hazla más compacta, útil y publicable"). (1) **Overview compacto tipo dashboard** (inspirado en NanoEdge): KPIs en tarjetas + barra "Best Val F1 by run (top 8)" + tarjeta del run activo con mini F1/loss + tarjetas de sección + **tabla "All runs" seleccionable** (clicar una fila activa ese run en todo el dashboard, vía `st.dataframe(on_select=...)` + guard `_last_table_row` para no pelear con la sidebar). (2) **Selector de run rehecho**: la sidebar muestra el run activo compacto + popover "Change run" (fuera el desplegable que se desbordaba). (3) **System eliminado** (el monitor de hardware no servía con el flujo Kaggle); "Import runs" movido a **Data & runs** → 5 secciones. (4) **Feasibility de 5 pestañas a 3 fases claras**: **Predict** (predictor analítico = la respuesta del profesor), **Validate** (predicho vs real), **Measure (advanced)** (benchmark real: generar informe + verlo + estudio de convergencia, apilados en scroll). (5) **Run results** de 6 a 5 pestañas (Time+Info → "Details"). Rama: `feature/web-compact-redesign`.
- [x] **Overview más denso + dataset y predictor enriquecidos (17/06/26):** continuación del afinado. (1) **Selector de run definitivo**: caja del run activo a nombre completo + filtro por entorno + selectbox con etiquetas cortas distinguibles (`21:43 vit_large [model_parallel]`) — sin popover ni desbordamiento. (2) Las tarjetas "Open" redundantes (la nav ya está en el menú) se sustituyen por **gráficas relevantes**: velocidad media/epoch (8 más rápidos) y **donut de estrategias**. (3) **Tira de 8 KPIs** compacta (Runs, Best F1, Fastest epoch, GPU time, Energy, Models, Environments, Feasibility). (4) **Dataset movido al Overview** (sección Dataset/Models eliminada → la sección queda solo "Import"): splits + **subset usado por los runs recientes** (leído de su config) + clases frecuentes + **galería de las 19 clases** (un patch distinto por clase, rejilla uniforme de 10 col) con **info multi-etiqueta** (media de clases/patch, `+k` y tooltip con todas las clases del patch). (5) El **Predictor** añade **coste en la nube** para la config elegida → predice tiempo+speedup+memoria+coste de una vez (la "fórmula potente" del tutor, completa). Helper cacheado `_class_gallery` (una sola lectura del parquet). Ramas: `feature/web-compact-redesign` (continuación).

#### Arreglos críticos tras auditoría (17/06/26) — rama `feature/critical-fixes`, PR #22 → develop
- [x] **Train F1 insesgado:** `Trainer` y `DeepTracingDecorator` calculan el F1 de entrenamiento sobre las etiquetas **originales**, no las mezcladas por mixup (antes umbralizaban las soft labels a 0.5 → "verdad" inventada). Sesgaba el argumento del gap train-val en las configs con mixup (v3+).
- [x] **Selección de modelo justa:** `EpochController.select_metric` (`f1` por defecto | `f1_optimal`); `training.select_by` en el config. Elige el mejor checkpoint y el early stopping por el F1 al **umbral óptimo** — imprescindible para focal (baja las probabilidades → su F1 a 0.5 sale bajo a propósito).
- [x] **Losses contra el techo ~0.68:** `src/training/losses.py` — `FocalLoss` multi-label + `pos_weight` (`'auto'` del metadata) + factory `build_criterion`. Conectado en el builder (`training.loss` / `focal_gamma` / `pos_weight`) y validado. Configs `train_cluster_focal.yaml` + `train_cluster_bce.yaml` (pareja apples-to-apples) + pareja local de demo. **Demo local vit_tiny (subset): focal redujo las clases con F1=0 de 8 a 6** (rescató Industrial units y Permanent crops). Run grande vit_base/dataset-completo pendiente de Verode libre (runbook: `docs/verode_focal_runbook.md`).
- [x] **Coherencia de diseño:** `Trainer.fit()` lanza `NotImplementedError` — el bucle vive solo en `EpochController` (Template Method). Documentado el invariante "controlador vs aspecto" en `DeepTracingDecorator`.
- [x] **Honestidad documental:** `docs/performance_model.md` separa calibración (in-sample) de validación out-of-sample (los speedups cancelan la MFU → predicción genuina).
- [x] **Proceso:** README real (antes 1 línea) + CI (`.github/workflows/ci.yml`, pytest en cada push/PR). **Suite en 333 tests.**

#### CLI unificado + auditoría de limpieza/SOLID (19/06/26)
- [x] **CLI `tfg`** (`tfg.py` + `src/cli.py`, Typer): un único punto de entrada en terminal — `train` (elige estrategia → script correcto, torchrun para ddp/heterogéneo) · `predict` (predictor analítico, tabla rich) · `feasibility` · `eval` · `runs` (lista runs con Best Val F1 / Test F1) · `dashboard` · `menu` (interactivo). Builders puros testeados + `--dry-run`. Separación **terminal hace / web mira** (W&B/MLflow/TensorBoard). PRs #49–#51.
- [x] **`predict` elige dataset full/subset** en vez de teclear el nº (CLI `--dataset`, menú, y selectbox en la web). `resolve_dataset_n` testeado. PR #53.
- [x] **Fix modelo de memoria** (`performance_model`): activaciones **por modelo** (`act_gb_per_img`, calibrado con vit_base 4.95 GB@b32 3060 Ti / vit_large 13.78 GB@b32 T4) + margen de VRAM usable (0.92). Antes decía que vit_base entraba a batch 64 en 8 GB (real: OOM, solo 32). PR #52.
- [x] **Limpieza del repo:** fuera el tooling MLflow (`scripts/ingest_mlflow.py`, `run_mlflow.sh`) y artefactos locales (`mlflow.db`, `mlartifacts/`, `.venv-mlflow`) — Streamlit es EL dashboard; fuera `dashboard/` (pyc huérfanos del Dash) y `src/evaluation/` (placeholder vacío). PR #54.
- [x] **No-ficheros-enormes (SRP):** `scripts/benchmark.py` (1600 ln) → paquete `src/benchmark/`; `src/web/tabs/run.py` (940 ln) → paquete `run/` (curves/perclass/confusions/batch/details); Predict extraído a `tabs/feasibility_predict.py`. PRs #55–#56.
- [x] **Auditoría funcional completa (19/06):** todos los comandos del CLI (predict sweep, menu, runs, eval/feasibility/train reales — single + DDP torchrun), las 6 secciones de la web + sub-pestañas (Playwright) — **todo verde**.
- [x] **Diagrama de clases (`docs/class_diagram.puml/.svg/.png`) actualizado y regenerado**: paquete `src.feasibility`, `src.cli`, nota de `src.web` (run/ paquete, sin i18n). El `.puml` es muy grande para el endpoint GET de plantuml.com, así que el `.svg`/`.png` se regeneran por **POST a Kroki** (`requests.post("https://kroki.io/plantuml/svg", data=puml)`) — o con `plantuml` local (java). ⚠ El SVG de PlantUML solo lleva fondo blanco como pista CSS (`style="background:#FFFFFF"`), que muchos visores ignoran → zonas vacías transparentes. Se post-procesa inyectando un `<rect fill="#FFFFFF">` que cubre el `viewBox` justo tras el primer `<g>`.

#### Pulido de la web — iteración con el usuario (20-21/06/26)
Sesión de mejoras incrementales sobre el dashboard, decididas una a una con el usuario. Las primeras ya en `main`; el resto en la rama de trabajo `feature/web-polish` (pendiente de PR a develop cuando el usuario cierre la web).
- [x] **Modo oscuro (PR #69):** toggle "Dark mode" en la sidebar. `theme.py` ahora es mode-aware: `register_plotly_template(mode)` (fondo/ejes/texto oscuros) + `inject_css(mode)` (capa de override oscura sobre la base clara). `app.py` lee el modo y colorea el menú de iconos; su `key` incluye el modo para que el componente cacheado se re-renderice. Limitación: el grid de `st.dataframe` se tematiza a nivel de config, no por CSS → se queda claro.
- [x] **Overview rediseñado (PRs #70-73):** la primera impresión pasa a 3 gráficas variadas y dark-aware — **barras GPU-h por entorno** + **tarta del split train/val/test** + **treemap de desbalance de clases** — más la tarjeta del run activo (con mini-curvas F1/loss) y la tabla "All runs". Compactado para entrar en una pantalla (~1180 px): tarjeta activa más ancha, curvas a 108 px, alturas y tabla recortadas. (Iteró por varias versiones: lollipop/scatter → dataset showcase → tiempo/energía → split; el usuario eligió el split del dataset para la tarta.)
- [x] **Per-class → heatmap + support (PRs #75-77):** la vista por epoch pasa de dumbbell (3 puntos solapados) a **heatmap anotado clases×(Precision/Recall/F1)** (color rojo→ámbar→verde, valor en cada celda) → se lee al instante qué clase falla y por qué (precisión vs recall). Añadido **support** (frecuencia de cada clase en validación, columna + hover) derivado del **dataset** (`val_support_from_parquet`) → sale en runs ya entrenados **sin reentrenar**; escalado al tamaño de val del run (full = exacto, subset = aprox + nota).
- [x] **Confusions simplificado (rama):** de 4 secciones a **1 + avanzada** — protagonista "Top label confusions" (off-diagonal de la co-activación) + matriz 19×19 en expander. Quitados recall-by-class (ya en Per-class) y el per-class profile (redundante). Matriz sin `paper_bgcolor="white"` → respeta el modo oscuro.
- [x] **Fix nombres de clase / 18→19 (PR #77 + rama):** una clase usa nombre CORINE completo en el metadata y abreviado en `CLASS_NAMES` → contaba 0 y la galería mostraba 18/19. Alias `_canon_label` aplicado en los conteos del parquet (support, treemap) **y** en `find_example_patches` + `_class_gallery` → galería 19/19 y support correcto.
- [x] **Carrusel de fotos (rama):** la tira de fotos del Overview pasa a carrusel paginado (5/pág, flechas ◀▶, contador "Classes X–Y of 19") que recorre las 19 clases (5·5·5·4), con cicl­ado y rejilla fija de 5 columnas.
- [x] **Curves — veredicto coherente (PR #74):** el aviso de una línea ahora se pone amarillo también cuando la val loss diverge (antes solo por gap train-val > 0.1); texto y color sincronizados.
- [x] **Flujo git restablecido:** `develop` puesto al día con `main` (estaba 124 commits por detrás — toda la sesión previa fui directo a main por error). De aquí en adelante: feature → PR a **develop**, develop → main solo cuando el usuario valida.
- [x] **Dark mode (rama):** toggle "Dark mode" en la sidebar; `theme.py` mode-aware (`register_plotly_template(mode)` + `inject_css(mode)`); `_show()` fuerza `template="tfg"` + `theme=None` (el default `theme="streamlit"` forzaba fondo claro en modo oscuro). El menú de iconos recibe el modo en su `key` para re-renderizar. Limitación: el grid de `st.dataframe` se tematiza por config, no por CSS.
- [x] **Overview rediseñado (rama):** primera impresión con 3 gráficas variadas dark-aware — **barras GPU-h por entorno** + **tarta del split train/val/test** + **treemap de desbalance de clases** — + tarjeta del run activo (mini F1/loss) + tabla "All runs". Compactado para entrar en una pantalla.
- [x] **Compare unificado en una sección (rama):** multiselect (≤8) → resumen (Mode/Precision) + speedup vs baseline + radar + energía + overlays; tabla de configuración fusionada con la de resumen; nombres de run horizontales; per-class heatmap con marcos como Run results. Eliminada la sección **Analysis** (más estética que funcional).
- [x] **Feasibility — análisis profundo + reestructura (rama):** auditoría de cálculo (las estimaciones "muy diferentes" eran: batch no casado → ahora `build_comparison` casa por batch del run; runs AMP comparados contra estimación fp32 → ahora `est = fp32_est / speedup_precisión` medido; DDP sin fila de tiempo → `est = single_est / speedup_predicho`; el resto eran gaps reales de I/O en vit_tiny/NFS, explicados). **Report** adelgazado: fuera "Load distribution per GPU" (heurística a ojo), fuera overlay de scaling-laws (multiselect+Amdahl), fuera gráfica min/época redundante; coste nube al expander usando los mismos epochs. Tabla "Compare vs runs": columna Note legible (quitada Model redundante + ancho large) + default que prioriza single-GPU + caption del requisito de batch benchmarkeado.
- [x] **`tfg predict` enriquecido (rama):** el CLI ya no da solo números finales — muestra **el trabajo** como la tabla de fórmulas de la web: headline + **fórmula de tiempo** `max(compute, I/O) + sync` con cada término + **fórmula de memoria** (weights+grad+Adam + activations + overhead = total vs GPU, fits/OOM, max batch) + calidad + **tabla de escalado 1→8 GPUs** + **coste** (5 proveedores) + **supuestos** (r_c/r_io/params/MFU). Expuestos `t_compute_s/t_io_s/t_sync_s/batch_per_gpu` en `Prediction` (defaults, compatibles). Help de `dashboard` actualizado (Predict pasó al terminal y luego volvió también a la web).
- [x] **Predict de vuelta en la web (rama):** Feasibility pasa a **3 tabs: Predict · Compare vs runs · Report**. `src/web/tabs/feasibility/predict.py` (`_analytic_predictor`) replica el `tfg predict` enriquecido — formulario (modelo/GPU/estrategia/n_gpus/precisión/disco/dataset/batch/epochs) → headline + tablas de fórmula (tiempo + memoria) + calidad con curva + escalado 1→8 GPU (gráfica, solo distribuido) + coste nube + supuestos + expander de calibración con throughput medido. Solo calcula fórmulas, no entrena → respeta "la web mira, el terminal ejecuta".
- [x] **Revisión a fondo del CLI:** todos los comandos verificados (predict/feasibility/train dry-run/eval/runs/menu/dashboard), sin referencias obsoletas (solo importa `eval_parser` y `run_registry` de la web, intactos), 16 tests de CLI en verde.
- [x] **Suite en 367 tests.** README reescrito (sin emojis, requisitos previos + instalación + qué funciona sin GPU/dataset + entrenar de verdad; tutor/cotutor correctos) y luego adelgazado al mínimo (de qué va + requisitos + el menú). Diagrama de clases regenerado (paquete `feasibility/` con `predict.py`). Licencia **MIT** añadida (repo público).
- [x] **Fix de energía multi-GPU (24/06/26, rama `fix/energy-multi-gpu`):** la instrumentación medía **una sola GPU** (`_PowerSampler` leía el dispositivo 0 y en DDP solo el rank 0 logueaba) → los runs DDP de la web mostraban la energía de **1 de 2 GPUs** (infravalorada ~½) y el **model-parallel no medía nada** (su script no tenía `--fn`). Arreglado: (1) `_PowerSampler` acepta una **lista de dispositivos** y **suma su potencia**; (2) `measure_energy(fn, devices, label, logger_name)` resuelve los dispositivos con `_resolve_energy_devices` — en DDP el **rank 0 mide todas las GPUs del nodo** (total real), en single la suya, y acepta lista explícita; (3) `train_model_parallel.py` gana `--fn energy/timing` y envuelve train/eval midiendo **sus dos GPUs** (etiqueta `ModelParallelTrainer.(train|eval)_epoch` para que el parser web la reconozca, logger `model_parallel` para que caiga en el fichero). Formato de log intacto → la web lo lee igual. 5 tests nuevos (incl. sumado multi-GPU con `pynvml` simulado) → **372 tests**. ⚠ Aplica a runs **futuros**; los logs de la sesión Kaggle 10/06 son pre-fix (1 GPU).
- [x] **Model-parallel integrado en la arquitectura de decoradores (24/06/26, rama `feature/model-parallel-decorator-stack`):** el model-parallel era la **única estrategia con bucle propio** → solo generaba curvas (le faltaban per-class, confusión, batch, metric reporters, deep/inspect). Ahora pasa por el **mismo `EpochController` (Template Method) y el mismo stack de decoradores** que single/DDP/heterogéneo. Cómo: (1) hook `Trainer._place_model(model, device)` (por defecto `model.to(device)`); (2) `ModelParallelTrainer(Trainer)` lo sobreescribe para **no mover** el modelo (ya está repartido) y usa `model.output_device` (dev1) para etiquetas/pérdida; (3) `builder.with_model_parallel(devices, split_block)` construye el `ModelParallelViT` + `ModelParallelTrainer` (LLRD sobre `model.base`, mode `model_parallel`); (4) la energía con `--fn energy` mide **ambas GPUs** del reparto (el builder pasa los índices explícitos, deduplicados); (5) `train_model_parallel.py` reescrito para usar el builder → gana `--layers/--metrics/--fn/--batch-log-every` (no `--trace deep`/`--inspect`: torchinfo y los probes de memoria asumen 1 dispositivo). **Verificado con el smoke `cuda:0→cpu`**: genera los 5 artefactos (log + epoch/perclass/confusion/batch CSV) y el parser web lee su energía. +1 test (`_place_model`) + test de energía actualizado → **373 tests**. La frase para la defensa: *el mismo bucle y los mismos decoradores sirven a las 4 estrategias.*
- [x] **Renombrado predict→estimate y feasibility→benchmark (24/06/26, rama `refactor/rename-estimate-benchmark`):** a sugerencia del tutor, nombres con más sentido bajo el contraste **analítico ↔ empírico**. CLI: `tfg estimate` (predictor **analítico**, fórmulas sin GPU) y `tfg benchmark` (**medición empírica** real en la máquina) — vía `@app.command(name=...)`, las funciones internas y los builders (`build_feasibility_cmd`, paquete `src/web/tabs/feasibility/`) **no** cambian (cero churn interno). Web: sección de nav **Feasibility → Performance**; pestañas **Predict → Estimate** y **Report → Benchmark** (la clave de routing `feasibility` se mantiene). Tests de etiquetas y docs actualizados; menú y help reflejan los nombres nuevos. **373 tests.**
- [x] **Consolidación total feasibility→benchmark (24/06/26, rama `refactor/consolidate-benchmark-naming`):** segunda pasada a petición del usuario — el nombre `feasibility` cambiado **en todos lados** (no solo el comando) para coherencia, incluido el diagrama de clases. (1) **Código:** paquete `src/feasibility/` → **`src/benchmark/`**; `scripts/check_feasibility.py` → **`scripts/benchmark.py`**; `src/web/feasibility_parser.py`/`feasibility_comparison.py` → `benchmark_parser.py`/`benchmark_comparison.py`; `src/web/tabs/feasibility/` → `tabs/benchmark/`; funciones (`build_feasibility_cmd`→`build_benchmark_cmd`, `parse_feasibility_csv`→`parse_benchmark_csv`, `_get_feasibility_csvs`→`_get_benchmark_csvs`, `FeasibilityChecker`→`BenchmarkChecker`, `FeasibilityReport`→`BenchmarkReport`…). (2) **Artefactos/logs:** los 58 ficheros `feasibility_*.{csv,log}` → `benchmark_*` y los 3 dirs `logs/{env}/feasibility/` → `logs/{env}/benchmark/` (git mv + `report_formatter` escribe los nombres nuevos + discovery actualizado; **verificado: la web descubre los 28 informes migrados**). (3) **Web:** nav **Performance → Estimate/Benchmark**; pestañas reordenadas a **Estimate · Benchmark · Benchmark vs Run** (antes Compare vs runs). (4) **Diagrama de clases** regenerado con el naming nuevo. Se **mantiene** `performance_model.py` y su API `predict()/predict_quality()/Prediction` — es el motor del Estimate, ya bien nombrado ("predict" = verbo correcto del predictor). **373 tests**; CLI (`tfg benchmark`→`scripts/benchmark.py`) y web verificados.

#### Auditoría de corrección — 4 arreglos (24/06/26, rama `fix/audit-corrections`)
Auditoría a fondo de la matemática de las 3 áreas (run/estimate/benchmark) a petición del usuario, buscando errores silenciosos como el de la energía. **El núcleo estaba correcto**; se encontraron y arreglaron:
- [x] **(real) Escala del gradiente en el DDP heterogéneo:** `HeterogeneousDDPTrainer` usaba `loss = loss_sum / global_bs`; como DDP **promedia** los gradientes (÷world_size) y `loss_sum` suma sobre batch×clases, el gradiente salía escalado por **n_clases/world_size ≈ 9.5×** vs el Trainer single (BCE-mean). **Corregido:** `loss = loss_sum * world_size / (global_bs * n_classes)` → recupera exactamente el gradiente BCE-mean del batch global. **Impacto práctico mínimo:** AdamW es invariante a un escalado constante del gradiente, por eso los runs heterogéneos previos (Val F1 0.278) son válidos y el speedup (0.12×) no se ve afectado; se arregla por rigor. 3 tests nuevos (`test_heterogeneous_grad.py`).
- [x] **(menor) Umbral óptimo per-rank en DDP:** `DDPTrainer.eval_epoch` solo hacía `all_gather` de los `preds` binarios → la búsqueda de threshold era del shard del rank 0, no global (afecta a `select_by=f1_optimal` y al threshold reportado; las métricas @0.5 ya eran globales). **Corregido:** ahora `Trainer.eval_epoch` devuelve `_probs`, DDP hace `all_gather` de las probabilidades y **recalcula el umbral óptimo sobre el set global**.
- [x] **(menor) `nfs_factor` solo al train:** `TimeEstimator` aplicaba el factor NFS al train pero no al eval (que también lee disco). Corregido.
- [x] **(menor) Proyección DDP plana redundante:** el informe de texto mostraba `[DDP×2: ~Xh]` con eficiencia plana 0.85, contradiciendo la sección DDP precisa (compute/IO-aware del `DDPOptimizer`). Quitado el número inline engañoso (la sección DDP precisa se mantiene).
- **Verificado correcto (sin tocar):** métricas macro, Trainer single (mixup/label-smoothing/train-F1 insesgado/threshold), DDP (métricas globales por all_gather, loss SUM/world_size), `performance_model` (fórmula maestra calibrada <10%), `Benchmarker` (sincroniza CUDA → timing fiable), `DDPOptimizer` (compute/n + I/O fijo + sync), energía (suma todas las GPUs), model-parallel. **376 tests.**

### Pendiente
- [ ] (Opcional) Entrenamiento completo en Verode con la versión actual si se quiere un Val F1 de referencia final con todo el stack.
- [ ] (Opcional, **ya implementado, falta correr**) Run focal-vs-BCE en Verode (V100, dataset completo) para confirmar a escala si focal sube el F1 macro / rescata la clase 6. Capacidad y configs listas; solo falta GPU libre.

---

## Próximos pasos y planificación del TFG

### ¿Hacen falta más entrenamientos?
- **Para el núcleo del TFG (comparación single vs distribuido): NO, ya está completo.** Tras la **sesión Kaggle del 10/06** hay un estudio de **5 estrategias comparables** (single/MP/DDP/AMP/DDP+AMP en vit_base, 15 epochs), el **caso OOM-vs-split de vit_large**, el heterogéneo de Verode y **3 predicciones del feasibility validadas (<4% error)**. La historia está completa, validada y es autoconsistente.
- **Opcionales que aportarían algo, no imprescindibles:**
  - Kaggle vit_base con más epochs (p.ej. 10) → curva F1 más vistosa para la memoria; **no cambia** las conclusiones de speedup.
  - Run de referencia final en Verode V100 con el stack actual (config v3) → cierra con la versión actual; el techo seguirá en ~0.68.
  - **(Si hay tiempo) Matriz simétrica: Verode single + heterogéneo con vit_base** sobre el subset (5000/1500, 3 epochs), para completar la cuadrícula {Verode, Kaggle} × {single, distribuido} × {tiny, base}. **Redundante y opcional:** el heterogéneo penaliza por el hardware desbalanceado independientemente del modelo (vit_tiny ya da 0.12×; vit_base daría ~0.05-0.10×, peor pero misma conclusión). La dependencia del modelo (compute-bound vs I/O-bound) ya está aislada en Kaggle (tiny 1.27× vs base 1.90×, mismo hardware homogéneo). El vit_base heterogéneo sobre el dataset COMPLETO es inviable (~16 días, medido); sobre el subset sería ~1-2 h (la CPU haciendo vit_base es el cuello). Usar `train_heterogeneous_ddp_demo.yaml` cambiando el modelo a `vit_base_patch16_224` + un `train_demo_single.yaml` con vit_base de pareja.

### Trabajo futuro (mejoras opcionales, si sobra tiempo)
- **`rico-hdl` — atacar el cuello de I/O (la mejora de mayor impacto):** convertir BigEarthNet-S2 a un formato de lectura rápida (LMDB) con [rico-hdl](https://github.com/rsim-tu-berlin/rico-hdl), uno de los "Additional Links" del dataset. El estudio demostró que con modelos ligeros el cuello es el **I/O del NFS** (vit_tiny escaló solo 1.27× en 2×T4 vs 1.90× de vit_base compute-bound; en Verode el heterogéneo penaliza por I/O + desbalance). Convertir a LMDB aceleraría el data loading → **mejoraría el escalado distribuido de modelos ligeros y el throughput general**. Requiere re-convertir el dataset y adaptar `BigEarthNetDataset`.
- **Clases raras:** `pos_weight` y focal loss **ya implementados** (`src/training/losses.py`, `training.loss`/`pos_weight`) para romper el techo de F1 macro ~0.68 (la clase 6 "Land principally occupied by agriculture" da F1=0). Validado en demo local (clases con F1=0: 8→6 con focal); falta el run grande en Verode para confirmar a escala.
- **Más espectro:** usar las 12 bandas de Sentinel-2 (ahora solo RGB proxy B04/B03/B02) o fusión S1 (radar) + S2 (óptico). Extensión grande, fuera del alcance actual.
- **Recomendación de batch global en `recommend_config`:** afinar la sugerencia de batch/lr (regla de escalado lineal) cuando recomienda varias GPUs.
- **GPUs heterogéneas en `estimate`/`benchmark` (límite conocido):** el predictor analítico y la medición empírica **asumen GPUs homogéneas** (n GPUs idénticas: `compute/n`); reciben un único tipo de GPU × `n_gpus`. El desbalance de hardware se modela y **valida** solo en el caso **GPU+CPU** (estrategia `heterogeneous` en `performance_model` + run de Verode, 0.12×). El **entrenamiento real sí admite GPUs distintas** (DDP da resultados correctos pero va al ritmo de la más lenta; model-parallel se equilibra con `--split-block`; y el `HeterogeneousDistributedSampler` reparte por `compute_weight`, generalizable a 2 GPUs distintas). Extender la **predicción** a conjuntos de GPUs heterogéneas (sumar throughputs en vez de ×n) queda como trabajo futuro — no se implementó porque no hay hardware con 2 GPUs distintas para validarlo, y la filosofía del proyecto es no predecir lo que no se puede contrastar con medidas reales.

### Seminarios pendientes (2) — sin fecha asignada; **ambos a redactar durante junio 2026**
- **Seminario 1:** sugerido — diseño SW (SOLID + Decorator + Template Method, apoyar en `docs/class_diagram.svg`) + entrenamiento single-GPU (resultados v1–v4) + feasibility checker (predicción y recomendaciones).
- **Seminario 2:** sugerido — entrenamiento distribuido: DDP, heterogéneo GPU+CPU, **estudio single vs distribuido** (la tabla de speedups) y **validación del feasibility** (predicho vs real); conclusión compute-bound vs I/O-bound vs hardware desbalanceado.
- *(Cuando el tutor fije fechas, anotarlas aquí.)*

### Memoria final — entrega junio/julio 2026
Estructura sugerida apoyándose en lo ya hecho (figuras desde el dashboard y `docs/class_diagram.svg`):
1. Introducción y objetivos.
2. Dataset BigEarthNet-S2 (estructura, 19 clases CORINE, multi-label, splits).
3. Diseño software: principios SOLID + patrones Decorator (GoF) y Template Method → arquitectura de decoradores (usar diagrama de clases).
4. Entrenamiento single-GPU: LLRD, warmup, cosine, early stopping, regularización v3 (label smoothing, mixup, threshold search); resultados v1–v4 (~0.68 Val F1).
5. Feasibility checker: perfilado de hardware/I/O, predicción de rendimiento, estudio empírico de convergencia, optimizador DDP; validación estimación-vs-real.
6. Entrenamiento distribuido: DDP (NCCL/gloo), DDP heterogéneo (sampler proporcional + normalización de gradiente por batch global), comparación single vs distribuido (tabla de 3 escenarios), speedup en GPUs reales (Kaggle 2×T4).
7. Conclusiones: el escalado distribuido depende del ratio cómputo/IO y del balance del hardware (compute-bound escala ~lineal, I/O-bound no, hardware desbalanceado penaliza); el feasibility lo predice. Limitación: Verode solo tiene 1 GPU usable.
8. Trabajo futuro: rico-hdl (I/O), clases raras, multi-banda/multi-sensor.

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
| `Duplicate GPU detected: rank X and rank Y both on CUDA device` (NCCL) | NCCL exige 1 rank = 1 GPU física; no se pueden 2 procesos en la misma GPU | Usar 2 GPUs reales (Kaggle 2×T4) o gloo CPU. `train_ddp.py` mapea `cuda:(local_rank % device_count)` para multi-GPU real |
| `Cannot use ReduceOp.AVG with Gloo` | gloo no soporta AVG en all_reduce (sí NCCL) | `ddp_trainer.py` usa `ReduceOp.SUM` y divide por `world_size` |
| `git clone` pide usuario en Kaggle/CI | El repo es privado | Token de GitHub (fine-grained, Contents:Read) vía Kaggle Secrets: `https://x-access-token:$GH_TOKEN@github.com/...` |
| Feasibility estima ~47× de más para un subset | `dataset_train` hardcodeado a 237871 | Corregido: lee el conteo real del metadata del config y escribe bloque `#sizes` |
| Run `--trace deep` invisible en el dashboard (0 epochs) | Regex posicional no casaba el orden de campos de la línea RESUMEN | Corregido: `_parse_deep` extrae cada campo por nombre |

---

## Comandos útiles

### Local
```bash
# Feasibility checker (genera .log + .csv)
uv run python scripts/benchmark.py --batch-sizes 16 32 --epochs 5
uv run python scripts/benchmark.py --model resnet50 --batch-sizes 32 64

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
.venv/bin/python scripts/benchmark.py \
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
