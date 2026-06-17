# Entrenamiento distribuido de Transformers (TFG)

[![CI](https://github.com/alerguezrojas/tfg-distributed-transformers/actions/workflows/ci.yml/badge.svg)](https://github.com/alerguezrojas/tfg-distributed-transformers/actions/workflows/ci.yml)

Trabajo de Fin de Grado — **"Entrenamiento distribuido de modelos abiertos de
aprendizaje automático basados en Transformers"** (Universidad de La Laguna).

Se entrena un **Vision Transformer (ViT-Base)** para clasificación multi-etiqueta
sobre **BigEarthNet-S2** (imágenes Sentinel-2, 19 clases CORINE) y se estudia su
**escalado a entrenamiento distribuido** (PyTorch DDP, paralelismo de modelo,
clúster heterogéneo GPU+CPU). El proyecto demuestra además la aplicación de
**principios SOLID** y de los patrones de diseño **Decorator (GoF)** y
**Template Method** al bucle de entrenamiento.

---

## Resultados principales

**Calidad del modelo (V100, dataset completo, 30 epochs con early stopping):**
mejor **Val F1 macro ≈ 0.68** (ViT-Base + LLRD + warmup + label smoothing + mixup).
El techo está dominado por las clases raras (varias con F1≈0); ver
`configs/train_cluster_focal.yaml` para el experimento que lo ataca (focal loss /
`pos_weight`).

**Escalado distribuido (mismo modelo / subset / epochs, solo cambia la estrategia):**

| Estrategia | Hardware | Speedup | Eficiencia | Cuello de botella |
|---|---|---:|---:|---|
| DDP (datos), ViT-Base | 2× T4 (NCCL) | **1.90×** | 95% | compute-bound → escala ~lineal |
| DDP (datos), ViT-Tiny | 2× T4 (NCCL) | 1.27× | 64% | I/O-bound (modelo diminuto) |
| Paralelismo de modelo | 2× T4 | 1.02× | — | etapas serializadas (permite modelos que no caben) |
| Heterogéneo GPU+CPU | V100 + CPU (gloo) | 0.12× | ~6% | DDP síncrono al ritmo del nodo lento |

**Conclusión:** el speedup distribuido depende del ratio cómputo/IO y del balance
del hardware; el *feasibility checker* lo **predice** y se validó contra lo medido
(<4% de error en las T4).

---

## Arquitectura

El bucle de entrenamiento se define **una sola vez** en `EpochController`
(Template Method). Las capas transversales (logging, gráficas, hooks, matriz de
confusión, monitor por batch, reporters de métricas) se añaden como **decoradores
GoF** que envuelven al `Trainer`, montados por un `TrainingSessionBuilder` fluido.

```
TracingDecorator / DeepTracingDecorator   ← controlador (define fit() vía Template Method)
  └── metric reporters (loss/f1/acc/prec)  ← aspecto
        └── aspectos (plot, hooks, confusion, batch-monitor)
              └── Trainer / DDPTrainer / HeterogeneousDDPTrainer   ← lógica pura
```

Diagrama de clases completo: [`docs/class_diagram.svg`](docs/class_diagram.svg).

---

## Uso

Requiere [uv](https://docs.astral.sh/uv/). `uv sync` crea el entorno.

```bash
# Tests (CPU, sin GPU ni dataset)
uv run pytest -q

# Entrenamiento single-GPU
uv run python scripts/train_single_gpu.py --config configs/train_v3.yaml --trace simple --layers plot

# Atacar el techo de F1 macro (clases raras)
uv run python scripts/train_single_gpu.py --config configs/train_cluster_focal.yaml --trace simple

# Análisis de viabilidad previo (predice tiempo/speedup/memoria/coste)
uv run python scripts/check_feasibility.py --batch-sizes 32 64 --epochs 30

# Entrenamiento distribuido (DDP)
torchrun --nproc_per_node=2 scripts/train_ddp.py --config configs/train_ddp_verode.yaml --trace simple

# Dashboard interactivo (analiza y compara todos los runs)
uv run streamlit run src/web/app.py
```

---

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `src/training/` | `Trainer`, decoradores, builder, métricas, **losses** (BCE/pos_weight/focal), DDP |
| `src/models/` | `BigEarthViT` (ViT + cabeza multi-etiqueta), paralelismo de modelo |
| `src/data/` | `BigEarthNetDataset` (Sentinel-2 RGB proxy + multi-hot) |
| `src/performance_model.py` | predictor analítico de tiempo/speedup/memoria sin benchmark |
| `src/web/` | dashboard Streamlit modular (orquestador + `tabs/` + `ui/`) |
| `scripts/` | entrenamiento (single / DDP / heterogéneo / model-parallel), feasibility, eval |
| `configs/` | configs de entrenamiento (local / Verode / Kaggle / distribuido) |
| `tests/` | suite (unit + integración), CPU-only |
| `docs/` | diagrama de clases, runbooks, derivación del modelo de rendimiento |

> Contexto operativo detallado (clúster VERODE, dataset, historial de runs,
> decisiones): [`CLAUDE.md`](CLAUDE.md).

---

## Autoría

Alejandro Rodríguez Rojas — Universidad de La Laguna. Tutor: Paco Almeida.
