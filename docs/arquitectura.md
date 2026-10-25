# Arquitectura del sistema

> Referencia técnica de cómo está construido ParaViT-Lab: modelo, decoradores, CLI, estrategias
> distribuidas, benchmark, artefactos, configuraciones y panel web. El diagrama de clases
> completo está en [`class_diagram.svg`](class_diagram.svg).

## Datasets y tareas

El núcleo de entrenamiento no asume ningún dataset. Dos piezas concentran todo lo que
depende de los datos (detalle y guía para añadir uno nuevo en [`datasets/`](datasets/)):

- **`DatasetSpec`** (`src/data/spec.py`): la ficha de un dataset. Incluye las clases, los
  canales de entrada, el tipo de tarea, las claves obligatorias de `data:`, el lector de
  cada split y sus transformaciones, y opcionalmente el recuento por clase. Se registra con
  un nombre en `src/data/registry.py` y el config la elige con `data.name` (por defecto
  `bigearthnet`).
- **`Task`** (`src/training/tasks.py`, patrón Strategy): `MultiLabelTask`, `BinaryTask` y
  `MultiClassTask`. Decide la pérdida, el suavizado de etiquetas, el paso de logits a
  predicciones, las métricas y si se busca un umbral óptimo.

```
config (data.name) ──► registry ──► DatasetSpec ──► Task
                                        │              │
               scripts / eval / benchmark (datos)      Trainer, DDPTrainer, Heterogeneous,
               builder (num_classes, in_chans)         DeepTracing, eval, benchmark
```

El `TrainingSessionBuilder` lee la ficha, crea la tarea y se la pasa al trainer de
cualquier estrategia. Las métricas de validación se calculan en un único sitio,
`Trainer._evaluate`, que DDP reutiliza sobre las probabilidades reunidas de todos los
procesos. El `ConfusionMatrixDecorator` recibe los nombres de clase de la ficha.

## Modelo

- `TimmClassifier` (`src/models/vit.py`; alias retrocompatibles `BigEarthModel` y
  `BigEarthViT`) con `build_model()`: cualquier backbone de **timm** más la cabeza
  `Dropout(p) → Linear(embed_dim, num_classes)`.
- `num_classes` e `in_chans` vienen de la ficha del dataset. Con `in_chans != 3`, timm adapta
  la primera capa de los pesos preentrenados.
- El forward devuelve logits crudos; la tarea decide cómo interpretarlos.
- Modelo de referencia del TFG: `vit_base_patch16_224` preentrenado en ImageNet (85 813 267
  parámetros, embed_dim = 768).
- `model.dropout` (por defecto 0.1) es la probabilidad del `Dropout` de la cabeza. Hasta el
  17/10/2026 no se aplicaba (siempre 0.1); ver la nota en [resultados_tfg.md](resultados_tfg.md).

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
    ├── PlottingDecorator          # aspecto: acumula métricas por epoch (CSV; la web pinta las curvas)
    ├── LayerHooksDecorator        # aspecto: forward hooks en capas Linear
    ├── ConfusionMatrixDecorator   # aspecto: CSV por clase + matriz de co-activación N×N tras cada eval
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
    plotting.py            # PlottingDecorator (aspecto, métricas → CSV)
    layer_hooks.py         # LayerHooksDecorator (aspecto, forward hooks)
    confusion.py           # ConfusionMatrixDecorator — perclass_metrics_*.csv + confusion_matrix_*.csv
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
  - `PlottingDecorator` — acumula métricas tras cada eval epoch (sin PNG desde 02/06/26; la web genera las gráficas); expone `_record_train_result()` para recibir métricas de train cuando `DeepTracingDecorator` gestiona el bucle directamente
  - `LayerHooksDecorator` — captura activaciones de capas Linear cada N epochs
  - `ConfusionMatrixDecorator` — dos CSVs por eval epoch: F1/prec/rec por clase (`perclass_metrics_*.csv`) + matriz 19×19 normalizada (`confusion_matrix_*.csv`) (celda (i,j) = P(predice j | verdadero es i), diagonal = recall)
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

### Observadores del bucle: flujo de eventos

El controlador también admite **observadores** (patrón *Observer*): objetos `TrainingListener`
(`src/training/events.py`) que se suscriben con `add_listener` y reciben el ciclo de vida del run
(`on_fit_start`, `on_epoch_start`, `on_epoch_end`, `on_early_stop`, `on_fit_end` y
`on_fit_error` si el entrenamiento lanza una excepción). Solo observan: reciben los mismos valores
que los hooks `_on_*` cuando el bucle ya los ha usado, y si uno falla se emite un aviso sin parar el
entrenamiento. A diferencia de los decoradores, que envuelven `train_epoch`/`eval_epoch`, un
observador ve el run entero, incluido su final o su fallo.

El builder suscribe en rank 0 un `JsonlEventSink`, que escribe `events_<ts>.jsonl` (esquema en
[eventos.md](eventos.md)) siempre que `--trace` no sea `off`. Si el monitor de batches está activo,
registra además su `on_batch` como *batch hook* del `Trainer`, con la misma cadencia que el CSV.
Es la fuente estructurada de la vista en vivo de la web.

Con el mismo flujo de eventos, el builder suscribe un `SystemSampler`
(`src/training/system_sampler.py`, otro `TrainingListener`) **antes** del sink, de modo que se
detiene antes de escribirse `run_end`. Al empezar la primera época lanza un hilo *daemon* que cada
`monitor.system_interval_s` segundos (5 por defecto; 0 lo desactiva) lee NVML (`pynvml`) y `psutil`
y añade un evento `system` con `sink.emit`. Solo lee: no toca torch ni CUDA. Sin NVML o sin GPU
reporta `gpus: []`, y un error de lectura o escritura se convierte en un único aviso.

El rendimiento por batch lo mide `StepTimer` (`src/training/throughput.py`), compartido por los tres
bucles que poseen una época (`Trainer` —y con él DDP y paralelismo de modelo—,
`HeterogeneousDDPTrainer` y `_train_epoch_deep`). Solo usa `time.perf_counter()`, sin
`cuda.synchronize`: el `loss.item()` del propio paso ya sincroniza la GPU. Los *batch hooks* reciben
`data_time_s`, `step_time_s` y `samples_per_s`, y el diccionario de `train_epoch` añade
`samples_per_s` y `data_wait_frac`; el CSV de épocas no cambia de columnas.

### Monitores de inspección

Con `--monitor` (o `monitor.enabled`), el builder suscribe dos observadores más
(`src/training/monitor.py`, contrato en [inspeccion.md](inspeccion.md)):

- `PredictionMonitor`: elige con su propia semilla un subconjunto de validación (equilibrado por
  clase si el dataset expone `targets`) y, tras cada época, guarda sus probabilidades, la pérdida
  por imagen, la entrada de la última capa lineal (*embedding*) y Grad-CAM de un panel, además de
  las probabilidades de toda la validación. Trabaja en `on_epoch_end`, fuera del tiempo medido, con
  el modelo desenvuelto de DDP y en modo `eval`, y restaura el modo al terminar.
- `LayerStatsMonitor`: *batch hook* que mide por grupo de capas la norma del gradiente, la de los
  pesos y el ratio actualización/peso (el paso real de AdamW reconstruido de su estado), e
  histogramas de pesos por época.

Ninguno consume números aleatorios globales ni acumula gradientes en los parámetros
(Grad-CAM usa `torch.autograd.grad` respecto a la activación): un test comprueba que los pesos
finales son idénticos con el monitor activado o no.

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

## Análisis de la inspección (`src/analysis/`)

Paquete **puro** (solo NumPy: sin torch, sin Streamlit, sin GPU) con el que la web analiza en CPU
los `.npz` del directorio `inspect_<ts>/` ([contrato](inspeccion.md)). El terminal guarda datos en
bruto; aquí se calculan las vistas. Un módulo por pregunta:

| Módulo | Funciones |
|---|---|
| `io.py` | `load_monitor_set`, `list_epochs`, `load_epoch` (dataclasses `MonitorSet` y `EpochData`; las claves opcionales ausentes quedan en `None`) |
| `curves.py` | `roc_curve`, `pr_curve`, `per_class_curves` (por clase + macro y micro; ≤ 1000 puntos por curva) |
| `thresholds.py` | `threshold_sweep`: F1, precisión, recall, sensibilidad y especificidad frente al umbral (misma definición que `training/metrics.py`) |
| `calibration.py` | `calibration`: ECE, MCE, Brier y diagrama de fiabilidad; `calibration_curve_stats` (pendiente e intercepto), `per_class_calibration`, `adaptive_ece` |
| `bootstrap.py` | `bootstrap_ci` (cualquier métrica) y `metrics_with_ci` (AUC, AP, F1, sensibilidad, especificidad, exactitud, ECE y Brier con IC95; remuestreo vectorizado por pesos) |
| `operating_point.py` | Punto de operación por clase: umbral fijo, sensibilidad objetivo, Youden, VPP/VPN según la prevalencia |
| `subgroups.py` | `subgroup_metrics`: métricas (con IC opcional) por valor de un array de grupos y brecha máx-mín |
| `confusion.py` | `confusion`: matriz de conteos, índices por celda y TP/FP/FN/TN por clase |
| `dynamics.py` | `cartography`, `prob_of_true`, `sample_correct`, `flips`, `worst_errors` |
| `projection.py` | `project_2d`: PCA por SVD (determinista); UMAP solo si `umap-learn` está instalado |

Todas reciben la tarea (`multilabel`, `binary` o `multiclass`) y las probabilidades tal como se
guardan (float16 incluido). Una clase sin positivos o sin negativos da NaN, nunca una excepción.
Los tests usan un generador de directorios sintéticos
(`tests/fixtures/make_inspect_fixture.py`) que cumple exactamente el contrato.

La página **Inspector** de la web (`src/web/tabs/inspector/`, un módulo por vista) la consume:
`src/web/inspector_data.py` reúne los helpers puros (tablas de muestras, matrices de capas,
superposición de Grad-CAM, descubrimiento de runs) y `EventsFollower` guarda, acotados, los
eventos `system`, `layers`, `histograms`, `monitor_start` e `inspection`. Solo la vista elegida
se calcula en cada refresco. `tests/fixtures/make_inspector_run.py` genera un run completo
(eventos + `inspect_<ts>/`) para los tests.

---

## Informe de un run (`src/reporting/`)

`paravit report <run>` genera un HTML autocontenido a partir de `events_*.jsonl` e `inspect_<ts>/`
(solo lectura, CPU). Un módulo por responsabilidad: `data.py` (resolver el argumento: fichero,
carpeta o sello; leer los eventos), `evaluation.py` (estadística con `src/analysis/` para la época
elegida), `figures.py` (plotly), `tripod.py` (lista TRIPOD+AI y su estado por ítem), `render.py`
(HTML, tablas y estilo, sin plantillas ni recursos externos) y `report.py` (secciones y
`build_report`/`write_report`). Reutiliza `parse_event`, `system_frames`, `energy_wh` y los
helpers de imagen de `src/web/`. Tests: `tests/unit/test_reporting.py`.

---

## CLI unificado (`paravit`)

**El sistema/framework se denomina `ParaViT-Lab`** (*Parallel/Parallax ViT Lab*); el comando es `paravit` y `tfg` se mantiene como **alias retrocompatible** (`tfg.py` sigue funcionando). Renombrado el 26/06/2026 (rama `feature/rename-paravit-cli`).

`src/cli.py` (Typer) + lanzador `paravit.py` en la raíz (alias: `tfg.py`) → **un único punto de entrada en terminal** para todo lo que usa la máquina. Separación tipo W&B/MLflow/TensorBoard: **el terminal hace** (toca la GPU), **la web mira** (visualización de solo lectura).

```bash
uv run paravit.py --help
uv run paravit.py estimate --model vit_base_patch16_224 --gpu "Tesla T4" --strategy ddp --n-gpus 2 --precision amp
uv run paravit.py train --strategy {single|ddp|model-parallel|heterogeneous} --config ... [--dry-run]
uv run paravit.py benchmark --model vit_base_patch16_224 --batch-sizes 32,64 --epochs 30
uv run paravit.py eval --checkpoint checkpoints/local/checkpoint_epoch_009.pt --split test
uv run paravit.py dashboard
```

- Subcomandos: `train` (elige estrategia → lanza el script correcto, con `torchrun` para ddp/heterogéneo), `estimate` (antes `predict`; predictor **analítico** en terminal, sin GPU — tabla rich), `benchmark` (antes `feasibility`; **medición empírica** real en la máquina), `eval` (test split), `runs` (lista los entrenamientos de `logs/` con Best Val F1 / Test F1), `dashboard` (streamlit), `menu` (interactivo guiado — pregunta los parámetros, ideal para la defensa). **Renombrados el 24/06 (analítico/empírico)** y consolidados después: los builders y el paquete web también pasaron a `benchmark` (`build_benchmark_cmd`, `src/web/tabs/benchmark/`).
- Builders puros `build_train_cmd` / `build_benchmark_cmd` / `build_eval_cmd` + helper `_run_row` (testeados: `tests/unit/test_cli.py`). `paravit train --strategy model-parallel` también reenvía `--layers/--fn` (arreglado 25/06). `--dry-run` imprime el comando sin ejecutarlo (para copiarlo en Verode/Kaggle). Opciones de lista por coma (`--batch-sizes 32,64`, `--layers plot,confusion`).
- Los scripts de `scripts/` siguen funcionando por separado; `paravit` solo los unifica. Dependencia nueva: `typer`. Rama: `feature/unified-cli`.

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
                             plot         → PlottingDecorator (métricas por epoch)
                             hooks        → LayerHooksDecorator, activaciones cada 5 epochs
                             confusion    → ConfusionMatrixDecorator, CSV por clase + matriz 19×19 tras cada eval
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

## Benchmark — medición empírica (antes "Feasibility Checker")

`scripts/benchmark.py` (antes `check_feasibility.py`) — **medición empírica** previa al entrenamiento. Usa datos sintéticos (sin tocar el dataset) para medir throughput real y estimar tiempos. Comando: `paravit benchmark`.

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
- `BenchmarkChecker` — Facade que coordina todo

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

Genera dos artefactos en `logs/{env}/benchmark/`:
- `benchmark_DDMMYYYY_HHMMSS.log` — informe de texto legible
- `benchmark_DDMMYYYY_HHMMSS.csv` — CSV estructurado con filas `#meta` (modelo/hardware) y filas de benchmark; consumido por la sección Estimate/Benchmark del dashboard

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
                      # batch_metrics_*.csv, confusion_matrix_*.csv, events_*.jsonl
    ddp/{model}/      # ídem para runs distribuidos
    benchmark/        # benchmark_*.log + benchmark_*.csv
  verode/
    single/{model}/
    ddp/{model}/
    ddp_hetero/{model}/   # DDP heterogéneo GPU+CPU (gloo)
    benchmark/
  kaggle/
    single/{model}/       # baseline 1×T4
    ddp/{model}/          # DDP 2×T4 (NCCL) — speedup positivo
    model_parallel/{model}/
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

## Dashboard web

`src/web/` — interfaz Streamlit profesional para gestionar y analizar el proyecto de principio a fin. **UI en inglés** (post-seminario 3) y **arquitectura modular (SRP)**: `app.py` es un orquestador delgado (~220 líneas: page config + CSS + sidebar + dispatch) que monta el menú de iconos + las 6 páginas y delega en módulos `render(ctx)`. **Tema visual** en `.streamlit/config.toml` (primario azul #1A5276, toolbar mínima) — sin él Streamlit usa su rojo por defecto.

```
.streamlit/config.toml      # tema (primaryColor azul, toolbarMode minimal)
src/web/
  __init__.py
  app.py                    # orquestador: page config + CSS + menú de iconos (streamlit-option-menu)
                            #   + selector de run activo + toggle claro/oscuro + dispatch a tabs/
  ui/
    context.py              # DashboardContext (runs, selected_run, run) → ctx
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
    live.py                 # Live: sigue un run en curso (events_*.jsonl) con st.fragment(run_every=…);
                            #   estado/ETA, KPIs, curvas por época, loss por batch, config y entorno
    comparison.py           # Compare unificado: multiselect → resumen + speedup vs baseline + radar + energía + overlays
    benchmark/              # paquete: Estimate (predict.py, analítico) / Benchmark (report.py + ddp.py + study.py) / Estimate vs Benchmark vs Run (validate.py)
                            #   predict.py = formulario → headline + tabla fórmula tiempo + tabla fórmula memoria
                            #   + calidad + escalado 1→8 GPU + coste; validate.py = Compare vs runs; report.py + ddp.py + study.py
    data_models.py          # Import runs (subir zip / apuntar a carpeta → copia a logs/)
  run_import.py             # import_run_archive (zip) / import_run_folder / _dest_relpath (puro, testeable)
  events_reader.py          # lector incremental de events_*.jsonl (EventsFollower, RunState) y
                            # descubrimiento de runs activos; puro (sin Streamlit)
  run_registry.py           # descubre runs con rglob (estructura plana y profunda);
                            # RunInfo con env, mode, model, precision (leída del log), CSV paths y events_path;
                            # label compacto con tags [ddp]/[amp]/[deep] (defaults implícitos)
  log_parser.py             # parsea logs --trace simple y --trace deep → DataFrame (fallback)
  batch_parser.py           # lee batch_metrics_*.csv → DataFrame por batch
  perclass_parser.py        # lee perclass_metrics_*.csv → DataFrame por clase
  benchmark_parser.py     # lee benchmark_*.csv → (metadata dict, benchmark DataFrame); bloque #gpu
  confusion_matrix_parser.py # lee confusion_matrix_*.csv → matriz numpy por epoch
  system_monitor.py         # snapshot CPU/RAM/disco/GPU/red; GpuInfo con specs de GPU (usado por el predictor)
```

**Pestaña Live.** Sigue un run en curso a partir de su flujo de eventos (`events_*.jsonl`, esquema
en [eventos.md](eventos.md)). `EventsFollower` guarda el desplazamiento en bytes y solo lee lo
añadido; descarta la última línea si no parsea y la reintenta. El estado (`running`, `completed`,
`early_stopped`, `failed`, `interrupted`, `stalled`) sale de `run_end` o, sin él, de que el fichero
lleve más de 3 minutos sin cambiar (un entrenamiento vivo escribe un `heartbeat` cada 30 s). Los
tipos desconocidos o de versión mayor se ignoran. La web sigue siendo de solo lectura.

**Motor analítico de predicción** (`src/performance_model.py`): **único motor** de predicción del proyecto. Predice tiempo/speedup/memoria/cuello/coste/**energía** **y calidad (F1)** de **cualquier** (estrategia, modelo, GPU, nº GPUs, dataset, batch, precisión) **sin benchmark** (fórmula maestra `T(n,π)`, estimación de `r_c/r_io/π` por specs, modelo de memoria/OOM). **Energía = potencia efectiva × tiempo** (29/06/26): `GpuSpec.train_power_w` calibrada de los logs `--fn energy` (T4 64W, V100 103W, RTX 3060 Ti 180W; resto = 0.65×TDP, `power_calibrated=False`), potencia independiente de precisión (AMP ahorra por *tiempo*), eval=0.9×train, `power_working_gpus` (single/heterogéneo=1, ddp/MP=n), MP×0.83 (idle de tubería); `predict()` también estima el tiempo de **eval** → total/epoch. Calibrado contra los datos reales de Kaggle (<10% en compute-bound). Expuesto en la pestaña **Estimate** (tiempo + memoria + **energía** + coste + calidad + puente `rc_measured`). Derivación en `docs/performance_model.md`.

**Predicción de calidad honesta (`predict_quality`, 18/06/26):** la F1 esperada ya **no** es una tabla histórica plana que ignoraba el tamaño del dataset (devolvía 0.68 para vit_base aunque fuera el subset de 5000, real ≈ 0.55). Ahora usa la ley de escalado por datos `F1_inf(N) = F1_full − k·log10(N_full/N)`, calibrada con dos puntos reales (vit_base full=0.68 / subset-5000=0.55 → k≈0.078; vit_tiny subset-5000 → 0.27, también medido). Curva de aprendizaje saturante estándar, banda de incertidumbre por confianza, etiquetada honestamente como **empirical prior** (no medición — el estudio de convergencia sigue siendo la vía medida). `PerformancePredictor (src/benchmark/predictor.py)` delega en este motor y le pasa el tamaño real del split. Rama: `feature/unified-prediction-engine`.

**`scripts/eval.py` — evaluación en TEST (arreglado 18/06/26):** estaba **roto** (llamaba a `build_model(dropout=…)`, kwarg inexistente → reventaba; por eso nunca generó un `test_*.csv`). Corregido + endurecido (guard de split vacío, `--metadata`, `--max-batches`). Verificado end-to-end. Su salida (`test_*.csv`: per-clase + fila `# aggregate`) ya **se muestra en la web**: `run_registry` descubre los `test_*.csv` por carpeta y **Run results** muestra una tarjeta "Held-out test set" (F1 macro @0.5 y @umbral óptimo, accuracy, clases con F1=0) — el número honesto final, separado de la validación. Parser: `src/web/eval_parser.py`.

**Specs de GPU** (`src/gpu_specs.py`): deriva núcleos CUDA y Tensor de cada GPU a partir de su compute capability (→ arquitectura → cores/SM) × nº de SMs (V100 5120/640, T4 2560/320, RTX 3060 Ti 4864/152). El `benchmark.py` los registra (bloque CSV `#gpu`) y acepta `--device INDEX`; el predictor analítico los usa para estimar `r_c`.

**Modelos de speedup seleccionables** (`src/estimation_models.py`): leyes analíticas Linear / Amdahl / Gustafson (puras, con fórmula), superponibles sobre la estimación compute/IO-aware del feasibility en **Estimate/Benchmark → Benchmark (DDP)**, con slider de fracción serial *s*.

### Arranque

```bash
uv run streamlit run src/web/app.py
# Abre http://localhost:8501
```

### Estructura (v9 — menú de iconos, 5 secciones, selección de run por tabla)

**Navegación:** menú lateral con iconos (`streamlit-option-menu`) — **5 secciones**. **Regla de diseño: nunca más de un nivel de pestañas dentro de una sección.** El **run activo** es estado compartido (`session_state["run_label"]`): se elige clicando una fila de la tabla del Overview (`st.dataframe(on_select)`) o con el selector compacto de la sidebar (filtro de entorno, filtro de dataset cuando hay más de uno + etiquetas cortas, sin desbordamiento). System se eliminó (el monitor de hardware no servía con el flujo Kaggle) y "Import runs" vive en su propia sección.

| Sección (menú) | Contenido |
|---|---|
| **Overview** | Dashboard compacto: tira de 8 KPIs · ranking Best F1 · tarjeta del run activo (mini F1/loss + veredicto) · velocidad media/epoch · donut de estrategias · **panel del dataset** (splits, subset usado por los runs recientes, clases frecuentes, **galería de las 19 clases** con info multi-etiqueta) · tabla "All runs" seleccionable con sparklines |
| **Run results** | **Una fila de pestañas**: Curves · Per-class (barras+tendencia) · Confusions (diagnósticos multi-etiqueta) · Batch (selector de vista) · Details (tiempo + config/anomalías/log) |
| **Compare** | **Sección única**: multiselect (≤8 runs) → resumen · speedup vs baseline (tabla+barras+veredictos+validación feasibility+escalado) · radar · energía · overlays |
| **Estimate/Benchmark** (antes "Feasibility"→"Performance") | 3 pestañas, en orden: **Estimate** (antes "Predict"; predictor **analítico**: formulario → headline + **tablas con las fórmulas** de tiempo `max(compute,I/O)+sync`, memoria `weights+grad+Adam+activations+overhead` y **energía `potencia × tiempo`** con los valores puestos + calidad + escalado 1→8 GPU + coste nube; mismo motor que `paravit estimate`) · **Benchmark** (antes "Report"; **medición empírica** generada en terminal con `paravit benchmark`: hardware, throughput, tiempo, escalado distribuido, coste, estudio de convergencia) · **Estimate vs Benchmark vs Run** (antes "Benchmark vs Run"; comparación a **3 fuentes** — analítico / benchmark / run real — con **gráficas de barras agrupadas por métrica** (tiempo, energía, F1) en rojo/verde/azul, selector de fuente por gráfica, scorecard y scatter de calibración; el benchmark solo modela single+DDP, así que para MP/heterogéneo se ocultan sus barras y manda el analítico) |
| **Import** | Importar runs entrenados en otra máquina (zip o carpeta → copia a `logs/`) |

**Predicción vs realidad** (rediseño de la antigua "Comparar vs training"): elige el run en la barra lateral → auto-empareja su feasibility (mismo modelo/entorno); muestra estimado vs real con semáforo y un veredicto en lenguaje natural (precisa / optimista por I/O / pesimista). Avisa si el run es distribuido (la estimación es single-GPU → la diferencia incluye el speedup).

**Descarga en todas las pestañas:**
- Gráficas Plotly: ícono de cámara en la barra de herramientas → descarga PNG (escala 2×, client-side)
- Tablas: botón "Descargar CSV" en cada tabla principal

Descubre runs recursivamente en `logs/` (estructura flat legacy y profunda env/mode/model).
Compatible con `--trace simple`, `--trace deep` y logs legacy.
