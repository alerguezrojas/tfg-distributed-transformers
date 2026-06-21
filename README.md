<div align="center">

# Entrenamiento distribuido de modelos abiertos de aprendizaje automático basados en Transformers

**Trabajo de Fin de Grado · Grado en Ingeniería Informática · Universidad de La Laguna**

[![CI](https://github.com/alerguezrojas/tfg-distributed-transformers/actions/workflows/ci.yml/badge.svg)](https://github.com/alerguezrojas/tfg-distributed-transformers/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-367%20passing-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)

**Autor:** Alejandro Rodríguez Rojas · **Tutor:** Francisco Carmelo Almeida Rodríguez · **Cotutor:** Daniel Suárez Labena

</div>

---

## Resumen

Entrenamiento de un **Vision Transformer** para la clasificación multi-etiqueta de imágenes Sentinel-2
(**BigEarthNet-S2**, 19 clases CORINE) y estudio, con medidas reales, de **cómo escala al distribuirlo**:
paralelismo de datos (PyTorch DDP), de modelo y clúster heterogéneo GPU+CPU. La conclusión central,
validada con datos, es que **el speedup distribuido depende del ratio cómputo/E-S y del balance del
hardware, y puede predecirse de antemano con un error inferior al 10 %**. Todo el ciclo se diseña
aplicando **principios SOLID** y los patrones **Decorator (GoF)**, **Template Method**, **Builder** y
**Facade**.

## Qué incluye

Cuatro herramientas que cubren el ciclo completo, desacopladas (el terminal *ejecuta*, la web
*visualiza*) y operadas desde un único comando, `tfg`:

| Herramienta | Qué hace |
|---|---|
| **Entrenador** (`src/training/`) | Entrena en single-GPU, **DDP**, paralelismo de modelo y **GPU+CPU heterogéneo**; con LLRD, warmup, *early stopping*, *label smoothing*, *mixup*, *focal loss* y medición de energía. |
| **Predictor analítico** (`src/performance_model.py`) | Estima **tiempo, memoria/OOM, coste en la nube y calidad (F1)** de cualquier configuración **sin ejecutar nada**, con fórmulas calibradas con datos reales. |
| **Análisis de viabilidad** (`src/feasibility/`) | *Benchmark* real en la máquina: mide **throughput, E-S de disco, memoria y escalado**, recomienda *batch*/precisión/nº de GPUs y **valida el predictor** (predicho vs. medido). Incluye un estudio empírico de convergencia (LR range test, *gradient noise scale*). |
| **Dashboard** (`src/web/`, Streamlit) | Visualiza y **compara** entrenamientos, **predice** configuraciones y contrasta **predicho vs. real**. |

## Resultados

**Calidad** (V100, dataset completo, con *early stopping*): mejor **Val F1 macro ≈ 0,68** (ViT-Base +
regularización). El techo lo imponen las clases raras y el uso de un proxy RGB de 3 bandas.

**Escalado distribuido** (mismo modelo / subconjunto / épocas, cambiando solo la estrategia):

| Estrategia | Hardware | Speedup | Eficiencia | Cuello de botella |
|---|---|---:|---:|---|
| DDP (datos), ViT-Base | 2× T4 (NCCL) | **1,90×** | 95 % | *compute-bound* → escala ~lineal |
| DDP (datos), ViT-Tiny | 2× T4 (NCCL) | 1,27× | 64 % | *I/O-bound* (modelo diminuto) |
| Paralelismo de modelo | 2× T4 | 1,02× | — | etapas serializadas (permite modelos que no caben) |
| Heterogéneo GPU+CPU | V100 + CPU (gloo) | 0,12× | ~6 % | DDP síncrono al ritmo del nodo lento |

El **análisis de viabilidad acertó cada caso** con error < 10 % (p. ej. DDP en 2 GPU: 1,92× predicho
vs. 1,90× real; aceleración por *Tensor cores* FP32→AMP: 3,87× vs. 3,80×).

## Arquitectura

El bucle de entrenamiento se define **una sola vez** en `EpochController` (**Template Method**); el
logging, las gráficas, los *hooks*, la matriz de confusión y los *reporters* de métricas se añaden como
**decoradores GoF** que envuelven al `Trainer` sin modificarlo, montados por un `TrainingSessionBuilder`
(**Builder**). El benchmark de viabilidad se coordina con una **Facade**.

```
TracingDecorator / DeepTracingDecorator   ← controlador (define fit() vía Template Method)
  └── metric reporters (loss/f1/acc/prec)  ← aspecto
        └── aspectos (plot, hooks, confusion, batch-monitor)
              └── Trainer / DDPTrainer / HeterogeneousDDPTrainer   ← lógica pura
```

Diagrama de clases completo: **[`docs/class_diagram.svg`](docs/class_diagram.svg)**.

## Instalación y uso

**Requisitos:** [uv](https://docs.astral.sh/uv/) (gestor de paquetes de Python) y git. Una GPU NVIDIA y
el dataset solo hacen falta para entrenar de verdad: las pruebas, el predictor y el dashboard funcionan
solo con CPU.

```bash
# 1. Instalar uv (Linux/macOS; en Windows PowerShell: irm https://astral.sh/uv/install.ps1 | iex)
curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Clonar e instalar el entorno (uv crea .venv con todas las dependencias y, si falta, Python 3.12)
git clone https://github.com/alerguezrojas/tfg-distributed-transformers.git
cd tfg-distributed-transformers && uv sync
```

**Sin GPU ni dataset** (el repo ya trae los *logs* reales que alimentan el dashboard):

```bash
uv run pytest -q                 # suite de 367 pruebas (solo CPU, ~10 s)
uv run tfg.py dashboard          # dashboard web → http://localhost:8501
uv run tfg.py predict --model vit_base_patch16_224 --gpu "Tesla T4" --strategy ddp --n-gpus 2 --precision amp
uv run tfg.py runs               # lista los entrenamientos incluidos
uv run tfg.py --help             # todos los comandos
```

**Con GPU + dataset** (BigEarthNet-S2, ~63 GB, [Zenodo 10891137](https://zenodo.org/records/10891137);
ajusta las rutas `data.root`/`data.metadata` en el `config`):

```bash
uv run tfg.py train --strategy single --config configs/train_v3.yaml
uv run tfg.py train --strategy ddp --n-gpus 2 --config configs/train_demo_ddp.yaml
uv run tfg.py feasibility --model vit_base_patch16_224 --batch-sizes 32,64   # benchmark real
uv run tfg.py eval --checkpoint <ruta.pt> --split test                       # número final en test
```

> Cualquier comando admite `--dry-run`: imprime el comando exacto sin ejecutarlo (útil para el clúster).

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `tfg.py` + `src/cli.py` | CLI unificado (Typer): train / predict / feasibility / eval / runs / dashboard / menu |
| `src/training/` | `Trainer`, decoradores, *builder*, métricas, *losses* (BCE / `pos_weight` / focal), DDP |
| `src/models/` · `src/data/` | `BigEarthViT` + paralelismo de modelo · `BigEarthNetDataset` (Sentinel-2) |
| `src/performance_model.py` | predictor analítico (tiempo / speedup / memoria / coste / calidad) |
| `src/feasibility/` | paquete SRP del *benchmark* (probes, analyzer, predictor, optimizer, Facade) |
| `src/web/` | dashboard Streamlit modular |
| `scripts/` · `configs/` · `logs/` | entrenamiento y benchmark · configuraciones · resultados reales |
| `tests/` · `docs/` | suite solo-CPU · diagrama de clases, *runbooks* y derivación del modelo de rendimiento |

**Pila:** PyTorch 2.x · timm · Streamlit · Plotly · Typer · uv · pytest. Los entrenamientos se hicieron
en local (RTX 3060 Ti), en el clúster **VERODE** de la ULL (Tesla V100) y en **Kaggle** (2× Tesla T4).

## Autoría

**Alejandro Rodríguez Rojas** — Grado en Ingeniería Informática, Universidad de La Laguna.
Tutor: **Francisco Carmelo Almeida Rodríguez**. Cotutor: **Daniel Suárez Labena**. Curso 2025/2026.
