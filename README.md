<div align="center">

# Entrenamiento distribuido de modelos abiertos de aprendizaje automático basados en Transformers

**Trabajo de Fin de Grado · Grado en Ingeniería Informática · Universidad de La Laguna**

Clasificación multi-etiqueta de cobertura terrestre con *Vision Transformers* sobre **BigEarthNet-S2**,
y estudio sistemático de su **escalado distribuido** (datos, modelo y hardware heterogéneo).

[![CI](https://github.com/alerguezrojas/tfg-distributed-transformers/actions/workflows/ci.yml/badge.svg)](https://github.com/alerguezrojas/tfg-distributed-transformers/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-367%20passing-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)

**Autor:** Alejandro Rodríguez Rojas · **Tutor:** Francisco Carmelo Almeida Rodríguez · **Cotutor:** Daniel Suárez Labena

</div>

---

## Resumen

Este proyecto entrena un **Vision Transformer (ViT-Base)** para la clasificación multi-etiqueta de
imágenes Sentinel-2 (**BigEarthNet-S2**, 19 clases CORINE) y estudia, con datos reales, **cómo escala
ese entrenamiento al distribuirlo**: paralelismo de datos (PyTorch DDP), paralelismo de modelo y un
clúster heterogéneo GPU+CPU. Más allá del modelo, el trabajo aporta **dos herramientas de ingeniería**:
un **predictor analítico** que estima —sin entrenar— el tiempo, la memoria, el coste y la calidad de
cualquier configuración, y un **dashboard** para analizar y comparar resultados. Todo el ciclo se diseña
aplicando **principios SOLID** y los patrones **Decorator (GoF)**, **Template Method**, **Builder** y
**Facade**.

La conclusión central, demostrada y validada con medidas reales, es que **el speedup del entrenamiento
distribuido depende del ratio cómputo/E-S y del balance del hardware**, y que ese comportamiento puede
**predecirse de antemano** con un error inferior al 10 %.

## Tabla de contenidos

- [Aportaciones principales](#aportaciones-principales)
- [Resultados](#resultados)
- [Arquitectura del software](#arquitectura-del-software)
- [Puesta en marcha](#puesta-en-marcha)
- [Cómo probarlo](#cómo-probarlo)
- [Entrenar y evaluar de verdad](#entrenar-y-evaluar-de-verdad)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Pila tecnológica y pruebas](#pila-tecnológica-y-pruebas)
- [Autoría](#autoría)

## Aportaciones principales

El proyecto se articula en **tres herramientas** que cubren el ciclo completo, conectadas pero
desacopladas (el terminal *ejecuta*, la web *visualiza* — el modelo de W&B/MLflow/TensorBoard):

| Herramienta | Qué hace |
|---|---|
| **Predictor analítico** (`src/performance_model.py`) | Estima **tiempo, memoria/OOM, coste en la nube y calidad (F1)** de cualquier (modelo, GPU, estrategia, batch, precisión, dataset) **sin ejecutar nada**, a partir de fórmulas calibradas con datos reales. |
| **Entrenador** (`scripts/` + `src/training/`) | Entrena en **single-GPU, DDP, paralelismo de modelo y GPU+CPU heterogéneo**, con LLRD, warmup, *early stopping*, *label smoothing*, *mixup*, *focal loss* y medición de energía. |
| **Dashboard** (`src/web/`, Streamlit) | Visualiza y **compara** entrenamientos, **predice** configuraciones, contrasta **predicho vs. real** y explora el dataset. |

Todo se opera desde **un único punto de entrada en terminal**, `tfg`, y el código se estructura
siguiendo SOLID (sin ficheros monolíticos: la lógica vive en paquetes con una responsabilidad por módulo).

## Resultados

**Calidad del modelo** (V100, dataset completo, 30 épocas con *early stopping*): mejor
**Val F1 macro ≈ 0,68** (ViT-Base + LLRD + warmup + label smoothing + mixup). El techo lo imponen las
**clases raras** (varias con F1 ≈ 0) y el uso de un proxy RGB de 3 bandas; ver
`configs/train_cluster_focal.yaml` para el experimento que lo ataca (*focal loss* / `pos_weight`).

**Escalado distribuido** (mismo modelo / subconjunto / épocas, cambiando solo la estrategia):

| Estrategia | Hardware | Speedup | Eficiencia | Cuello de botella |
|---|---|---:|---:|---|
| DDP (datos), ViT-Base | 2× T4 (NCCL) | **1,90×** | 95 % | *compute-bound* → escala ~lineal |
| DDP (datos), ViT-Tiny | 2× T4 (NCCL) | 1,27× | 64 % | *I/O-bound* (modelo diminuto) |
| Paralelismo de modelo | 2× T4 | 1,02× | — | etapas serializadas (permite modelos que no caben) |
| Heterogéneo GPU+CPU | V100 + CPU (gloo) | 0,12× | ~6 % | DDP síncrono al ritmo del nodo lento |

**Validación del predictor:** las estimaciones se contrastaron contra los entrenamientos reales con un
**error < 10 %** (p. ej. speedup DDP en 2 GPU: **1,92× predicho vs. 1,90× real**; aceleración por *Tensor
cores* FP32→AMP: 3,87× predicho vs. 3,80× real).

## Arquitectura del software

El bucle de entrenamiento se define **una sola vez** en `EpochController` (**Template Method**). Las
capas transversales (logging, gráficas, *hooks*, matriz de confusión, monitor por *batch*, *reporters*
de métricas) se añaden como **decoradores GoF** que envuelven al `Trainer` sin modificarlo, montados por
un `TrainingSessionBuilder` (**Builder**). El análisis de viabilidad se coordina con una **Facade**
(`FeasibilityChecker`).

```
TracingDecorator / DeepTracingDecorator   ← controlador (define fit() vía Template Method)
  └── metric reporters (loss/f1/acc/prec)  ← aspecto
        └── aspectos (plot, hooks, confusion, batch-monitor)
              └── Trainer / DDPTrainer / HeterogeneousDDPTrainer   ← lógica pura
```

Diagrama de clases completo: **[`docs/class_diagram.svg`](docs/class_diagram.svg)**.

## Puesta en marcha

### Requisitos previos

| Requisito | Para qué | Obligatorio |
|---|---|---|
| **git** | Clonar el repositorio | Sí |
| **[uv](https://docs.astral.sh/uv/)** | Gestor de paquetes y entornos de Python | Sí |
| **Python 3.12** | Intérprete | `uv` lo instala automáticamente si falta |
| **GPU NVIDIA + CUDA** | Entrenar y medir rendimiento | No (las pruebas, el predictor y el dashboard funcionan solo con CPU) |
| **Dataset BigEarthNet-S2** | Entrenar/evaluar con datos reales | No (el repo ya incluye los *logs* de los entrenamientos para que el dashboard funcione) |

### Instalación (3 pasos)

```bash
# 1. Instalar uv (si no lo tienes). Linux/macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
#    (Windows PowerShell: irm https://astral.sh/uv/install.ps1 | iex)

# 2. Clonar el repositorio
git clone https://github.com/alerguezrojas/tfg-distributed-transformers.git
cd tfg-distributed-transformers

# 3. Crear el entorno con todas las dependencias
uv sync
```

`uv sync` crea un entorno aislado en `.venv/` y descarga PyTorch, timm, Streamlit, etc. No necesitas
activar el entorno: cada comando se ejecuta con `uv run …`, que lo usa automáticamente.

## Cómo probarlo

Lo siguiente **funciona sin GPU y sin descargar el dataset** (el repositorio ya trae los resultados de
los entrenamientos reales en `logs/`):

```bash
# Ejecutar toda la suite de pruebas (solo CPU, ~10 s, no necesita dataset)
uv run pytest -q

# Predecir una configuración SIN entrenar: tiempo, memoria, coste y F1 esperada,
# con las fórmulas detrás de cada número
uv run tfg.py predict --model vit_base_patch16_224 --gpu "Tesla T4" \
    --strategy ddp --n-gpus 2 --precision amp

# Abrir el dashboard web (ya poblado con los entrenamientos del repo)
uv run tfg.py dashboard          # abre http://localhost:8501

# Listar los entrenamientos incluidos · menú interactivo guiado
uv run tfg.py runs
uv run tfg.py menu

# Ver la ayuda de cualquier comando
uv run tfg.py --help
uv run tfg.py predict --help
```

## Entrenar y evaluar de verdad

Esto **requiere una GPU NVIDIA y el dataset**. BigEarthNet-S2 (~63 GB) se descarga del
[registro de Zenodo 10891137](https://zenodo.org/records/10891137); ajusta las rutas `data.root` y
`data.metadata` en el `config` que uses (ver `configs/`).

```bash
# Entrenar — la estrategia decide cómo se lanza (torchrun para DDP, etc.)
uv run tfg.py train --strategy single --config configs/train_v3.yaml
uv run tfg.py train --strategy ddp --n-gpus 2 --config configs/train_demo_ddp.yaml

# Evaluar un checkpoint en el conjunto de test (número honesto final)
uv run tfg.py eval --checkpoint checkpoints/local/checkpoint_epoch_009.pt --split test

# Benchmark de viabilidad real en esta máquina (mide throughput de verdad)
uv run tfg.py feasibility --model vit_base_patch16_224 --batch-sizes 32,64
```

> Cualquier comando admite `--dry-run`: imprime el comando exacto sin ejecutarlo, útil para copiarlo en
> el clúster o en Kaggle. Los scripts de `scripts/` siguen funcionando por separado; `tfg` solo los unifica.

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `tfg.py` + `src/cli.py` | **CLI unificado** (Typer): train / predict / feasibility / eval / runs / dashboard / menu |
| `src/training/` | `Trainer`, decoradores, *builder*, métricas, *losses* (BCE / `pos_weight` / focal), DDP |
| `src/models/` | `BigEarthViT` (ViT + cabeza multi-etiqueta) y paralelismo de modelo |
| `src/data/` | `BigEarthNetDataset` (Sentinel-2, proxy RGB + multi-hot) |
| `src/performance_model.py` | predictor analítico (tiempo / speedup / memoria / coste / **calidad**) |
| `src/feasibility/` | paquete SRP del *benchmark* (probes, analyzer, predictor, optimizer, formatter, *checker* Facade) |
| `src/web/` | dashboard Streamlit modular (orquestador + `tabs/` en paquetes + `ui/`) |
| `scripts/` | entrenamiento (single / DDP / heterogéneo / model-parallel), `check_feasibility`, `eval` |
| `configs/` | configuraciones (local / Verode / Kaggle / distribuido) |
| `logs/` | resultados reales de los entrenamientos (alimentan el dashboard sin necesitar el dataset) |
| `tests/` | suite (unitarios + integración), **solo CPU** |
| `docs/` | diagrama de clases, *runbooks* y derivación del modelo de rendimiento |

## Pila tecnológica y pruebas

**PyTorch 2.x · timm · Streamlit · Plotly · Typer · uv · pytest.** El dataset (BigEarthNet-S2) es
externo; los entrenamientos se realizaron en local (RTX 3060 Ti), en el clúster **VERODE** de la ULL
(Tesla V100) y en **Kaggle** (2× Tesla T4).

La calidad se cuida con una suite de **367 pruebas** (unitarias + integración, ejecutables sin GPU ni
dataset) y **integración continua** en cada *push* / *pull request*:

```bash
uv run pytest -q
```

## Autoría

**Alejandro Rodríguez Rojas** — Grado en Ingeniería Informática, Universidad de La Laguna.
Tutor: **Francisco Carmelo Almeida Rodríguez**. Cotutor: **Daniel Suárez Labena**. Curso 2025/2026.

> Documentación operativa detallada (clúster, dataset, historial de experimentos y decisiones de
> diseño): [`CLAUDE.md`](CLAUDE.md).
