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
  .venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
  ```

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
- Cabeza personalizada: `Dropout(0.1) → Linear(768, 19)`
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
├── Trainer                        # lógica pura, sin prints ni logging
└── TrainerDecorator               # base OOP: delega todos los métodos al trainer envuelto
    ├── LossReporter               # metric reporter: train_loss / val_loss
    ├── F1Reporter                 # metric reporter: train_f1 / val_f1
    ├── AccuracyReporter           # metric reporter: train_acc / val_acc
    ├── PrecisionRecallReporter    # metric reporter: val_precision / val_recall
    ├── PlottingDecorator          # aspecto: guarda curvas PNG tras cada epoch
    ├── LayerHooksDecorator        # aspecto: forward hooks en capas Linear
    └── EpochController            # Template Method: define fit() con hooks _on_*
        └── TracingDecorator       # controlador: logging a consola y/o fichero
            └── DeepTracingDecorator  # controlador: hereda TracingDecorator + trazado profundo
```

### Ficheros

```
src/training/
  base_trainer.py          # ABC con train_epoch, eval_epoch, save_checkpoint, fit
  trainer.py               # implementación pura, usa metrics.py
  metrics.py               # f1_score, precision, recall, accuracy, eta_str
  logger_setup.py          # setup_logger() con formato timestamp
  fn_decorators.py         # decoradores @ de Python: timed, log_call, measure_energy, retry_on_cuda_oom
  decorators/
    base.py                # TrainerDecorator + EpochController
    tracing.py             # TracingDecorator (consola o fichero según logger=)
    deep_tracing.py        # DeepTracingDecorator (hereda TracingDecorator)
    plotting.py            # PlottingDecorator (aspecto, guarda PNG)
    layer_hooks.py         # LayerHooksDecorator (aspecto, forward hooks)
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

--layers [plot] [hooks]    Decoradores de aspecto (combinables):
                             plot  → PlottingDecorator, PNG en plots/training_FECHA.png
                             hooks → LayerHooksDecorator, activaciones cada 5 epochs
                                     (ignorado con --trace deep, que tiene sus propios hooks)

--fn [timing] [energy]     Decoradores @ de Python (combinables):
                             timing → @timed en train_epoch y eval_epoch
                             energy → @measure_energy en train_epoch y eval_epoch
                             (con --trace deep solo aplica a eval_epoch)

--metrics [loss] [f1] [accuracy] [precision_recall]
                           Metric reporters individuales (solo para --trace off/simple):
                             sin args (--metrics) → desactiva todos
                             por defecto → todos activos
```

### Ejemplos

```bash
# Solo consola
uv run python scripts/train_single_gpu.py --trace off

# Log a fichero + gráficas
uv run python scripts/train_single_gpu.py --trace simple --layers plot

# Solo F1 y loss en pantalla
uv run python scripts/train_single_gpu.py --trace simple --metrics loss f1

# Trazado profundo + medición de energía
uv run python scripts/train_single_gpu.py --trace deep --fn energy

# Todo junto
uv run python scripts/train_single_gpu.py --trace deep --layers plot hooks --fn energy timing

# Test rápido 1 epoch con todo activado
uv run python scripts/train_single_gpu.py --epochs 1 --trace deep --layers plot hooks --fn energy timing

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
```

**Resultados conocidos en RTX 3060 Ti:**
- batch_size=32: ~65 imgs/s, 4.95 GB VRAM ← óptimo local
- batch_size=64: OOM (necesita ~11.5 GB)
- `--trace deep` añade ~22% overhead vs off

**En V100 32 GB (clúster):** pendiente de ejecutar tras liberar el nodo.

---

## Configuración

### `configs/train.yaml` — local
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
  batch_size: 32          # 64 OOM en RTX 3060 Ti (8 GB)
  lr: 0.0001              # OJO: no usar 1e-4, se parsea como string en YAML
  weight_decay: 0.0001
  log_batch_every: 50     # DeepTracingDecorator: tabla cada N batches
checkpoint:
  dir: "checkpoints/single_gpu"
```

### `configs/train_cluster.yaml` — clúster VERODE
Igual que `train.yaml` pero con rutas del clúster:
```yaml
data:
  root: "/home/bejeque/alu0101317038/datasets/bigearthnet/BigEarthNet-S2"
  metadata: "/home/bejeque/alu0101317038/datasets/bigearthnet/metadata.parquet"
```

---

## Resultados de entrenamiento

### Local — RTX 3060 Ti, batch_size=32, 30 epochs (completado 2026-05-01/02)

| Epoch | Train F1 | Val F1 | Val Loss |
|-------|----------|--------|----------|
| 1  | 0.587 | 0.593 | 0.168 |
| 4  | 0.718 | 0.657 | 0.155 |
| 9  | 0.825 | **0.659** ← mejor | 0.183 |
| 30 | 0.947 | 0.654 | 0.674 |

- **Mejor Val F1: 0.6586** (epoch 9) — guardado en `checkpoints/single_gpu/checkpoint_epoch_009.pt`
- Duración: ~32.5 horas (~65 min/epoch)
- Sobreajuste claro a partir del epoch 9: train loss → 0.0001, val loss sigue subiendo
- Log completo: `logs/train_local.log`

### Clúster — V100 32 GB (pendiente)
- Job en cola (Slurm), esperando que libere verode21 (nhernang lleva ~3 días)
- Comando lanzado desde tmux:
```bash
/opt/soft/slurm/20.11.04/bin/srun \
    --partition=batch --nodelist=verode21 --gres=gpu:1 --time=72:00:00 \
    --job-name=single_gpu_vit \
    bash -c "
cd ~/tfg-distributed-transformers && \
.venv/bin/python scripts/check_feasibility.py \
  --batch-sizes 16 32 64 128 --epochs 30 \
  2>&1 | tee ~/logs/feasibility_cluster.log && \
.venv/bin/python scripts/train_single_gpu.py \
  --config configs/train_cluster.yaml \
  --batch-size 64 \
  --trace deep \
  2>&1 | tee ~/logs/train_cluster.log
"
```

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
.venv/bin/python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
```

Dependencias principales: `torch`, `timm`, `torchvision`, `torchinfo`, `tqdm`, `rasterio`, `pandas`, `pyarrow`, `pyyaml`, `matplotlib`, `nvidia-ml-py`

---

## Git workflow

```
main ← feature/xxx
```

- Rama activa: `feature/refactor-decorators`
- **No añadir Co-Authored-By en los commits**

---

## Estado actual del proyecto

### Completado
- [x] Pipeline de datos: `BigEarthNetDataset` con metadata.parquet
- [x] Modelo: `BigEarthViT` (ViT + cabeza multi-label)
- [x] Entrenamiento single-GPU: `Trainer` + scheduler cosine + checkpoints
- [x] Arquitectura de decoradores: Decorator (GoF) + Template Method
  - `decorators/`: `TracingDecorator`, `DeepTracingDecorator`, `PlottingDecorator`, `LayerHooksDecorator`
  - `decorators/metric_reporters.py`: `LossReporter`, `F1Reporter`, `AccuracyReporter`, `PrecisionRecallReporter`
  - `fn_decorators.py`: `@timed`, `@log_call`, `@measure_energy`, `@retry_on_cuda_oom`
- [x] `metrics.py`: métricas extraídas en módulo propio (sin duplicación)
- [x] Flags `--trace / --layers / --fn / --metrics` en script de entrenamiento
- [x] Log con timestamp a fichero + gráficas PNG por epoch
- [x] `check_feasibility.py` con benchmark, estimaciones y análisis de memoria
- [x] Entrenamiento local completado: 30 epochs, Val F1 = 0.6586
- [x] Test completo con todo el stack activo: 1 epoch `--trace deep --layers plot hooks --fn energy timing` (06/05/26)
- [x] Acceso al clúster VERODE (ULL) con V100 32 GB
- [x] Dataset en el clúster: 549 488 patches verificados
- [x] Entorno Python en el clúster: uv + PyTorch cu118 instalado
- [x] `configs/train_cluster.yaml` con rutas del clúster
- [x] Job en cola Slurm (esperando verode21)

### Pendiente inmediato
- [ ] Esperar que el job del clúster arranque y verificar CUDA + resultados
- [ ] Ejecutar `check_feasibility.py` en V100 para calibrar batch_size óptimo

### Pendiente futuro
- [ ] Implementar entrenamiento distribuido (PyTorch DDP) con múltiples V100
- [ ] Proyección multi-GPU en feasibility checker
- [ ] Merge `feature/refactor-decorators` → `main`

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

---

## Comandos útiles

### Local
```bash
# Feasibility checker
uv run python scripts/check_feasibility.py --batch-sizes 16 32 64 --epochs 30

# Test rápido (1 epoch)
uv run python scripts/train_single_gpu.py --epochs 1 --batch-size 32 --trace deep

# Entrenamiento completo
uv run python scripts/train_single_gpu.py --config configs/train.yaml --trace simple

# Con gráficas y medición de energía
uv run python scripts/train_single_gpu.py --trace simple --layers plot --fn energy

# Ver log en tiempo real
tail -f logs/train_*.log
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
srun --partition=short --gres=gpu:1 --time=02:00:00 --pty bash

# Verificar CUDA tras reinstalar torch
cd ~/tfg-distributed-transformers
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# Ver log del entrenamiento en tiempo real
tail -f ~/logs/train_cluster.log
```
