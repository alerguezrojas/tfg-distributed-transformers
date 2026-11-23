# AGENTS.md

Instrucciones para agentes de código (Claude Code, Codex, Copilot, Cursor…) que trabajen en este
repositorio. Lo imprescindible está aquí; el detalle vive en `docs/` y se consulta bajo demanda.

## Qué es

**ParaViT-Lab** (`paravit`): framework para entrenar Vision Transformers (timm) de forma
distribuida y **predecir** su coste (tiempo, memoria, energía) antes de entrenar. Nació como TFG
(Universidad de La Laguna; autor Alejandro Rodríguez Rojas, tutor Francisco Almeida). La versión
entregada está etiquetada como **`v1.0-tfg`** (12/07/2026).

**Fase actual (desde oct. 2026):** trabajo de continuación, sin prisa, con vistas a una
**publicación**. Las líneas de trabajo están en [`docs/roadmap.md`](docs/roadmap.md): soporte
multi-dataset y mejora del predictor analítico con medidas de más máquinas.

## Comandos

```bash
uv sync                                  # entorno (Python 3.12, gestionado con uv)
uv run pytest -q                         # tests: solo CPU, sin GPU ni dataset (~1,5 min)
uv run paravit.py --help                 # CLI: train | estimate | benchmark | eval | predict | compare-preds | select-ensemble | runs | report | dashboard | menu
uv run paravit.py estimate --model vit_base_patch16_224 --gpu "Tesla T4" --strategy ddp --n-gpus 2
uv run paravit.py train --strategy single --config configs/train.yaml --dry-run   # ver el comando sin ejecutar
uv run streamlit run src/web/app.py      # panel web (solo lectura de logs/)
```

Entrenar de verdad necesita GPU y el dataset (rutas en `configs/*.yaml`). Para probar el código
basta un modelo pequeño: `--model vit_tiny_patch16_224 --epochs 1`.

## Mapa del código

| Ruta | Responsabilidad |
|---|---|
| `src/data/` | Fichas de dataset (`DatasetSpec`) y su registro por nombre (`data.name`): BigEarthNet, MedMNIST y PANDA; `image_io.py` lee imágenes sueltas (`predict`) |
| `src/models/` | `TimmClassifier` (backbone + cabeza), registro de backbones no timm (`backbones.py`: `BackboneSpec`, `hf:`; [guía](docs/modelos/README.md)), optimizador LLRD, `ModelParallelViT` |
| `src/training/` | `Trainer` y subclases por estrategia (DDP, heterogéneo, model-parallel), `TrainingSessionBuilder`, tareas (`tasks.py`), pérdidas, métricas |
| `src/training/strategies.py`, `launch.py` | Registro de estrategias (`StrategySpec`: lanzamiento, entorno, loaders, `with_*`, línea `Configuración:`) y lanzador común (`build_parser`, `main`) que usan los `scripts/train_*.py` y la CLI |
| `src/training/decorators/` | Decoradores GoF: controladores (`EpochController` → tracing/deep) y aspectos (plot, hooks, confusion, batch-monitor, reporters) |
| `src/training/events.py` | Observadores del bucle (`TrainingListener`) y flujo de eventos `events_*.jsonl` ([esquema](docs/eventos.md)) |
| `src/training/checkpoints.py` | Política de checkpoints: carpeta por run, `best.pt`/`last.pt` (`checkpoint.keep`), escritura atómica y `resolve_checkpoint` |
| `src/training/fn_decorators.py` | Decoradores `@` de Python: `timed`, `measure_energy`, `retry_on_cuda_oom` |
| `src/performance_model.py` | **Predictor analítico** (puro: sin torch, sin GPU, sin I/O). Fórmula maestra en [`docs/performance_model.md`](docs/performance_model.md); con AMP, throughput de tres términos (cómputo + tráfico de memoria + coste por paso, `THROUGHPUT_FITS`, §8) |
| `src/analysis/` | Análisis puro (NumPy) de los datos de inspección: curvas, calibración, umbrales, confusión, dinámica, proyección, `compare.py`/`ensemble.py` (comparar y elegir conjuntos de runs), `throughput_model.py`; lo usa la web |
| `src/benchmark/` | Benchmark empírico en la máquina real (Facade `BenchmarkChecker`) |
| `src/cli.py`, `src/cli_menu.py`, `src/config_io.py` | CLI Typer, menú interactivo y carga de YAML; lanzadores `paravit.py` / `tfg.py` (alias) |
| `src/web/` | Dashboard Streamlit: `app.py` orquesta, `tabs/*` con `render(ctx)`, parsers de logs/CSV; lógica pura del selector y los enlaces en `run_select.py`, rutas de datos desde los perfiles en `data_paths.py` |
| `scripts/` | Puntos de entrada por estrategia (`train_single_gpu.py`, `train_ddp.py`, …) + `benchmark.py`, `eval.py`, `predict.py`, `select_ensemble.py`, `compare_predictions.py`, `predictor_architecture_study.py` |
| `src/slurm.py`, `cluster/` | `paravit train --launcher slurm` (rendezvous `c10d`, mismo comando en todos los nodos), plantillas `cluster/slurm/*.sbatch` y receta Apptainer; [guía](cluster/README.md) |
| `kaggle/` | Kernels de Kaggle (2 × T4): mosaicos y entrenamientos de PANDA con DDP, y `benchmark_t4/`; [guía](kaggle/README.md) |
| `configs/` | YAML de entrenamiento (índice en `configs/README.md`) |
| `logs/{env}/[{dataset}/]{mode}/{model}/` | Artefactos de cada run (se commitean); `checkpoints/<env>/.../<sello>/{best,last}.pt` no |
| `tests/unit`, `tests/integration` | pytest, todo en CPU |
| `CONTRIBUTING.md`, `.github/` | Normas de trabajo, plantillas de PR e issues, CI, CodeQL, Dependabot |

## Invariantes de diseño (no romper)

- **El bucle de entrenamiento vive solo en `EpochController.fit()`** (Template Method). `Trainer.fit()`
  lanza `NotImplementedError` a propósito. Las cuatro estrategias comparten el mismo bucle y los
  mismos decoradores; una estrategia nueva es una subclase de `Trainer` + un `with_*` en el builder + un `StrategySpec`
  registrado en `src/training/strategies.py` (la CLI, el menú y el lanzador salen del registro).
- **Nuevo comportamiento transversal = nuevo decorador**, no `if`s dentro de `Trainer`.
- **El núcleo no asume un dataset.** Clases, canales y lector vienen de la `DatasetSpec`; lo que
  depende del tipo de problema (pérdida, sigmoide/softmax, métricas, umbral) vive en la `Task`.
  Un dataset nuevo se añade sin tocar el entrenamiento ([guía](docs/datasets/README.md)); un
  backbone no timm, igual, con un `BackboneSpec` ([guía](docs/modelos/README.md)).
- **`tests/integration/test_behaviour_golden.py` fija números exactos** del entrenamiento. Si
  falla, el comportamiento ha cambiado: se investiga, no se reescriben los valores.
- **El terminal hace, la web mira:** la web nunca entrena ni toca la GPU; lee `logs/`.
- **`performance_model.py` se mantiene puro** (testeable sin hardware). Toda constante calibrada
  lleva un comentario con el run del que sale.
- **El formato de los logs es un contrato con la web.** Las claves en español que parsea
  (`Configuración:`, `threshold óptimo=`, `potencia media`, `RESUMEN`) no se cambian sin actualizar
  los parsers de `src/web/` y sus tests.
- **No predecir lo que no se puede contrastar:** cada capacidad nueva del predictor necesita
  medidas reales que la validen, y se separa calibración (in-sample) de validación (out-of-sample).

## Convenciones

- **Git:** `main ← develop ← feature/*`. Las ramas salen de `develop` y hacen PR a `develop`; nunca
  directo a `main`. `develop → main` por PR, como versión: CI en verde, revisión del agente
  revisor en los cambios del núcleo, sección del `CHANGELOG.md` con fecha, versión en
  `pyproject.toml` y etiqueta `vX.Y.Z` (el autor lo delegó el 20/11/2026). GitHub borra la rama al
  mergear.
- **Commits:** Conventional Commits en español (`feat:`, `fix(web):`, `docs:`, `chore:`…), cortos.
  **Sin** `Co-Authored-By` ni "Generated with…" en commits ni PRs.
- **Máquina:** las rutas dependientes del equipo van en `configs/machines/<perfil>.yaml`
  (`--machine`, `PARAVIT_MACHINE` o host; `src/machine.py`), no en las configs de experimento; todo
  punto de entrada carga la config con `prepare_config` (perfil + `--set`) y `paravit doctor`
  diagnostica la máquina.
- **Idioma:** documentación en español (registro canario: "ustedes", no "vosotros"); UI de la web en
  inglés. En el código, seguir el idioma del fichero que se edita.
- **YAML:** escribir los floats en decimal (`lr: 0.0001`); `1e-4` se parsea como string.
- **Nombres de artefactos:** `tipo_DDMMYYYY_HHMMSS.{log,csv}`.
- **Definición de hecho:** `uv run pytest -q` en verde, tests para la lógica nueva, docs afectadas
  al día y entrada en `CHANGELOG.md` (sección *Unreleased*).
- **Calidad:** SonarCloud analiza `main` (triple A). No introducir comparaciones de float con `==`,
  `torch.load` sin `weights_only=True` ni literales repetidos.

## Trampas conocidas

- Clúster VERODE: el nodo de login no tiene AVX2 (usar siempre `srun`); `uv sync` instala torch cu13,
  incompatible con el driver → reinstalar cu118. Detalle en [`docs/infraestructura.md`](docs/infraestructura.md).
- NCCL no admite dos ranks en la misma GPU; gloo no admite `ReduceOp.AVG`.
- Los tests y la CI usan torch CPU; no asumir CUDA en tests.
- PC del autor con 15 GB de RAM: la batería de tests llega a ~4,5 GB; nunca dos a la vez y, en
  local, con `systemd-run --user --scope -p MemoryMax=5G` (el 23/11 `systemd-oomd` cerró la sesión).

## Continuidad

El estado del proyecto vive en el repositorio, no en ningún chat. Al terminar un bloque de trabajo,
cualquier agente actualiza la sección **"Estado actual"** de [`docs/roadmap.md`](docs/roadmap.md),
las casillas del roadmap y `CHANGELOG.md`. Quien retome el proyecto empieza por ahí.

Los agentes especializados del proyecto están en `.claude/agents/` (revisor, implementador,
documentador, investigador, explorador); `CLAUDE.md` explica cuándo usar cada uno.

## Documentación

Índice completo en [`docs/README.md`](docs/README.md). Lo más consultado:
[arquitectura](docs/arquitectura.md) · [modelo de rendimiento](docs/performance_model.md) ·
[infraestructura](docs/infraestructura.md) · [datasets](docs/datasets/README.md) · [modelos](docs/modelos/README.md) · [roadmap](docs/roadmap.md) ·
[guía de uso](docs/guia_de_uso.md) · [primeros pasos](docs/primeros_pasos.md) (para quien se incorpora) ·
[CONTRIBUTING](CONTRIBUTING.md).
