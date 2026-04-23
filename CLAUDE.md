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
- **Métricas:** macro F1 + sample-averaged accuracy

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
    ├── MetricsLoggerDecorator   # epoch-level, prints simples
    ├── BatchMetricsDecorator    # tqdm por batch (white-box)
    ├── LayerHooksDecorator      # forward hooks en Linear layers
    ├── TensorBoardDecorator     # SummaryWriter por época
    ├── TracingDecorator         # logging estructurado + grad norm
    └── DeepTracingDecorator     # trazado máximo (ver abajo)
```

Ficheros:
- `src/training/base_trainer.py` — contrato abstracto
- `src/training/trainer.py` — implementación pura
- `src/training/trainer_decorators.py` — decoradores nivel 1–5
- `src/training/deep_tracing.py` — decorador de máxima profundidad
- `src/training/logger_setup.py` — `setup_logger()` con formato timestamp
- `src/training/python_decorators.py` — decoradores Python `@` (contraste didáctico)

### DeepTracingDecorator (nivel más profundo)

El decorador activo en producción. Registra:
- **Forward hooks** en todos los módulos hoja → `act_mean`, `act_std`, `act_max`, `dead_ratio`
- **Backward hooks** (`register_full_backward_hook`) → `grad_norm`, `grad_max`, `vanishing`, `exploding`
- **Parameter hooks** (`param.register_hook`) → `weight_norm`, `grad_norm`, `update_ratio`
- **GPU memory** (`torch.cuda.memory_allocated`) por step
- **Learning rate** por grupo del optimizer
- **torchinfo** summary al inicio
- **Alertas de anomalías**: neuronas muertas (dead_ratio > 0.5), gradiente explosivo (norm > 10), gradiente evanescente (norm < 1e-7), update ratio anómalo

Todos los tensores se mueven a CPU (`.detach().cpu().float()`) antes de calcular estadísticas para no saturar la VRAM.

### Script de entrenamiento actual

`scripts/train_single_gpu.py` usa:
```python
trainer = DeepTracingDecorator(
    Trainer(model, optimizer, scheduler, device, checkpoint_dir),
    logger=logger,
    log_every=cfg["training"].get("log_batch_every", 100),
    log_top_n_layers=cfg["training"].get("log_top_n_layers", 8),
)
trainer.fit(train_loader, val_loader, epochs=N)
```

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
uv run python scripts/train_single_gpu.py --config configs/train.yaml

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
- Branch actual de trabajo: `feature/design-patterns`
- Commits anteriores relevantes:
  - `f146072` feat: apply Decorator pattern for metrics logging
  - `106850a` Merge PR #3 feature/training-single-gpu
  - `504b701` Merge PR #1 feature/data-pipeline

---

## Estado actual del proyecto

### Completado
- [x] Pipeline de datos: `BigEarthNetDataset` con metadata.parquet
- [x] Modelo: `BigEarthViT` (ViT + cabeza multi-label)
- [x] Entrenamiento single-GPU: `Trainer` + `Scheduler` + checkpoints
- [x] Patrón Decorator completo (niveles 1–5 en `trainer_decorators.py`)
- [x] Decoradores Python `@` (`@timed`, `@log_call`, `@retry_on_cuda_oom`)
- [x] `DeepTracingDecorator` con trazado a nivel neurona/capa
- [x] `setup_logger` con salida a consola y fichero
- [x] Script integrado con `DeepTracingDecorator`

### Pendiente
- [ ] Test de `DeepTracingDecorator` (1 epoch para verificar output)
- [ ] Commit de `deep_tracing.py` + `train_single_gpu.py` actualizados en `feature/design-patterns`
- [ ] PR feature/design-patterns → develop → main
- [ ] Entrenamiento completo 30 epochs
- [ ] Visualización de attention maps
- [ ] Implementar entrenamiento distribuido (PyTorch DDP)
- [ ] Solicitar recursos GPU universitarios

---

## Errores conocidos y soluciones

| Error | Causa | Solución |
|-------|-------|----------|
| `lr '<=' not supported between float and str` | `1e-4` en YAML se parsea como string | Usar `0.0001` en el YAML |
| `property 'model' has no setter` | Intentar poner `model` como `@property` abstracta en BaseTrainer | No declarar propiedades en BaseTrainer; usar `__getattr__` en TrainerDecorator |
| `'BatchMetricsDecorator' has no attribute 'model'` | Decorator sin delegación | `TrainerDecorator.__getattr__` resuelve toda la cadena |
| `CUDA out of memory` en hooks | `.float()` en GPU de tensores grandes con 280 hooks | Mover a CPU primero: `.detach().cpu().float()` |
| nvidia driver no funciona con kernel 6.8 | Driver 470 incompatible | Actualizar a `nvidia-driver-580-open` |

---

## Comandos útiles

```bash
# Entrenamiento rápido (test, 1 epoch)
uv run python scripts/train_single_gpu.py --epochs 1 --batch-size 64

# Entrenamiento completo
uv run python scripts/train_single_gpu.py --config configs/train.yaml

# TensorBoard (si se usa TensorBoardDecorator)
uv run tensorboard --logdir runs/

# Ver logs en tiempo real
tail -f logs/train.log

# Estado del repo
git log --oneline -10
git status
```
