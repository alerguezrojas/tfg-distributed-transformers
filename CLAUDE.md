# CLAUDE.md — tfg-distributed-transformers

Contexto completo del proyecto para continuar el trabajo en cualquier máquina.

---

## Sobre el proyecto

**TFG:** "Entrenamiento distribuido de modelos abiertos de aprendizaje automático basados en Transformers"
**Tutor:** Paco Almeida (Universidad de La Laguna)
**Alumno:** Alejandro Rodríguez Rojas
**Entrega:** junio/julio 2026
**Repo:** https://github.com/alerguezrojas/tfg-distributed-transformers

El objetivo es demostrar la aplicación de principios SOLID y patrones de diseño (especialmente Decorator) al ciclo de entrenamiento de un ViT sobre BigEarthNet-S2, y escalar más adelante a entrenamiento distribuido con PyTorch DDP.

---

## Hardware

- **Local:** NVIDIA RTX 3060 Ti (8 GB VRAM), NVIDIA driver 580-open, kernel 6.8
- **Dataset:** SSD externo montado en `/media/alejandro/SSD/` (ext4, ~120 GB)
- **Futuro:** VMs con GPU en la universidad para entrenamiento distribuido

---

## Dataset: BigEarthNet-S2 v2.0

- **Ruta local:** `/media/alejandro/SSD/datasets/bigearthnet/BigEarthNet-S2/`
- **Metadata:** `/media/alejandro/SSD/datasets/bigearthnet/metadata.parquet`
- **Estructura de directorios:** `root/scene_id/patch_id/*.tif`
  - `scene_id` = `patch_id` sin los dos últimos segmentos (`_row_col`)
- **Tamaño:** 480 038 patches (splits: train / validation / test)
  - Train: 237 871 | Val: 122 342 | Test: 119 825
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
    ├── MetricsLoggerDecorator   # epoch-level, prints simples + ETA
    ├── BatchMetricsDecorator    # tqdm por batch (white-box, solo didáctico)
    ├── LayerHooksDecorator      # forward hooks en Linear layers (solo didáctico)
    ├── TracingDecorator         # logging estructurado a fichero + ETA
    └── DeepTracingDecorator     # trazado máximo (ver abajo)
```

**Nota:** `TensorBoardDecorator` fue eliminado. `BatchMetricsDecorator` y `LayerHooksDecorator` se mantienen solo por valor didáctico del TFG (muestran la progresión del patrón), no se usan en producción.

Ficheros:
- `src/training/base_trainer.py` — contrato abstracto
- `src/training/trainer.py` — implementación pura
- `src/training/trainer_decorators.py` — decoradores nivel 1–4
- `src/training/deep_tracing.py` — decorador de máxima profundidad
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
- **Tabla por bloque**: patch_embed + `attn.proj` de cada uno de los 12 bloques + head
- **Alertas de anomalías**: neuronas muertas, gradiente explosivo/evanescente, update ratio anómalo

Todos los tensores se calculan en GPU con `.detach().float()` y solo se transfiere el escalar final con `.item()` para no saturar la VRAM.

### Script de entrenamiento

`scripts/train_single_gpu.py` — flag `--trace` con tres modos:

```python
# --trace off   → MetricsLoggerDecorator  (sin hooks, máxima velocidad)
# --trace simple → TracingDecorator        (timestamps + log a fichero)
# --trace deep   → DeepTracingDecorator    (trazado completo por capa)
trainer = DeepTracingDecorator(
    Trainer(model, optimizer, scheduler, device, checkpoint_dir),
    logger=logger,
    log_every=cfg["training"].get("log_batch_every", 100),
)
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
uv run python scripts/check_feasibility.py --batch-sizes 16 32 64 --epochs 10 30

# Solo algunos modos
uv run python scripts/check_feasibility.py --batch-sizes 32 --trace-modes off deep
```

**Resultado conocido en RTX 3060 Ti:**
- batch_size=32 óptimo: ~65 imgs/s, 4.95 GB VRAM
- batch_size=64 OOM (necesita ~11.5 GB)
- `--trace deep` añade ~22% overhead vs off

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
  batch_size: 64
  lr: 0.0001          # OJO: no usar notación 1e-4 en YAML, se parsea como string
  weight_decay: 0.0001
  log_batch_every: 50
  log_top_n_layers: 10

checkpoint:
  dir: "checkpoints/single_gpu"
```

**Importante:** En YAML, `1e-4` se parsea como string. Usar siempre `0.0001`.

---

## Gestión de dependencias

```bash
# Instalar entorno
uv sync

# Ejecutar scripts
uv run python scripts/train_single_gpu.py --config configs/train.yaml --trace simple
uv run python scripts/check_feasibility.py --batch-sizes 16 32 64

# Añadir dependencia
uv add <paquete>
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

### Pendiente
- [ ] Entrenamiento completo 30 epochs (batch_size=32, --trace simple)
- [ ] Proyección multi-GPU en feasibility checker
- [ ] Visualización de attention maps
- [ ] Implementar entrenamiento distribuido (PyTorch DDP)
- [ ] Solicitar recursos GPU universitarios

---

## Errores conocidos y soluciones

| Error | Causa | Solución |
|-------|-------|----------|
| `lr '<=' not supported between float and str` | `1e-4` en YAML se parsea como string | Usar `0.0001` en el YAML |
| `property 'model' has no setter` | Intentar poner `model` como `@property` abstracta en BaseTrainer | No declarar propiedades en BaseTrainer; usar `__getattr__` en TrainerDecorator |
| `CUDA out of memory` con batch_size=64 | ViT-B necesita ~11.5 GB para activaciones con batch 64 | Usar batch_size=32 (4.95 GB) |
| `CUDA out of memory` en hooks | `.float()` en GPU de tensores grandes | Calcular en GPU con `.detach().float()`, transferir solo el escalar con `.item()` |
| Hooks muy lentos (6 GB/batch transferidos) | `.detach().cpu()` copia tensor entero a RAM | Usar `.detach().float().mean().item()` — solo transfiere 4 bytes |
| nvidia driver no funciona con kernel 6.8 | Driver 470 incompatible | Actualizar a `nvidia-driver-580-open` |

---

## Comandos útiles

```bash
# Análisis de viabilidad previo
uv run python scripts/check_feasibility.py --batch-sizes 16 32 64 --epochs 30

# Entrenamiento rápido (test, 1 epoch)
uv run python scripts/train_single_gpu.py --epochs 1 --batch-size 32 --trace deep

# Entrenamiento completo
uv run python scripts/train_single_gpu.py --config configs/train.yaml --batch-size 32 --trace simple

# Ver logs en tiempo real
tail -f logs/train_*.log

# Limpiar caché Python si hay comportamientos raros
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Estado del repo
git log --oneline -10
git status
```
