# Changelog

Cambios notables del proyecto. Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).

## [Unreleased]

### Añadido
- Muestreo de sistema (`src/training/system_sampler.py`): un `TrainingListener` que, mientras el
  run está vivo, añade al flujo de eventos un evento `system` cada `monitor.system_interval_s`
  segundos (5 por defecto; 0 lo desactiva) con, por GPU visible, uso, memoria, potencia,
  temperatura y reloj SM (NVML) y, del equipo y del proceso, CPU y RAM (`psutil`). Respeta
  `CUDA_VISIBLE_DEVICES`; sin NVML o sin GPU emite `gpus: []`; los errores se avisan una vez y
  nunca detienen el entrenamiento. Lo cablea el builder en rank 0 junto al `JsonlEventSink`. Las
  GPU se identifican por UUID (NVML y CUDA numeran distinto), de modo que `index` es `cuda:N`;
  cada entrada lleva `uuid`. La medida de energía (`_PowerSampler`) usa el mismo criterio.
- Rendimiento por batch (`src/training/throughput.py`, `StepTimer`): espera del DataLoader y tiempo
  de paso con `time.perf_counter()` (sin `cuda.synchronize`). Los *batch hooks* reciben
  `data_time_s`, `step_time_s` y `samples_per_s`, y el resultado de `train_epoch` añade
  `samples_per_s` y `data_wait_frac`, que salen en `epoch_end.train`. Lo usan los tres bucles
  (`Trainer`, heterogéneo y `--trace deep`); el CSV de épocas no cambia de columnas.
- Paquete `src/analysis/` (solo NumPy, sin torch ni Streamlit) para analizar los datos de
  inspección en la web: curvas ROC/PR por clase con macro y micro, barrido de umbral (misma
  definición de F1 que `training/metrics.py`), calibración (ECE, MCE, Brier, diagrama de
  fiabilidad), matriz de confusión con índices por celda, cartografía de datos, *flips* entre
  épocas, peores errores, proyección 2D (PCA; UMAP opcional) y lector del directorio
  `inspect_<ts>/`. Generador de fixtures sintéticos que cumplen el contrato
  (`tests/fixtures/make_inspect_fixture.py`). Descrito en `docs/arquitectura.md`.
- Contrato de la inspección del entrenamiento (`docs/inspeccion.md`): eventos de sistema,
  rendimiento por batch, monitor de predicciones por muestra, Grad-CAM y capas; sección `monitor:`
  validada en la config; `JsonlEventSink.emit()` para otros productores del flujo; los datos de
  inspección (`logs/**/inspect_*/`) no se versionan.
- Fuentes PlantUML de los diagramas del diseño multi-dataset en `docs/diagramas/multidataset/`.
- Flujo de eventos de cada run (`events_<ts>.jsonl`, esquema v1 en `docs/eventos.md`): inicio
  con configuración y huella del entorno (host, versiones, commit, GPUs), épocas, batches (con
  el monitor de batches), parada temprana y fin, también si el entrenamiento falla. Lo escribe
  rank 0 con *flush* por línea, para seguir el run en vivo; un hilo aparte escribe un `heartbeat`
  cada 30 s, y un error de escritura nunca detiene el entrenamiento. `EpochController` admite observadores
  (`TrainingListener`, `add_listener`); el bucle y sus resultados no cambian.
- Tests de `LayerHooksDecorator`, de los metric reporters y de `batch-table` + `batch-monitor`.
- Auditoría de observabilidad (`docs/roadmap.md`) y revisión del sector en
  `docs/research/observabilidad_entrenamiento.md`: seguimiento en vivo, formato de eventos e
  interpretabilidad de ViT en imagen médica. Esquema v1 del flujo de eventos en `docs/eventos.md`.
- Web: pestaña **Live** para seguir un entrenamiento en curso. Lee de forma incremental (por
  desplazamiento de bytes) el flujo `logs/<env>/…/events_<sello>.jsonl` (esquema v1: `run_start`,
  `epoch_start`, `batch`, `epoch_end`, `early_stop`, `run_end`) y muestra estado (running,
  completed, early_stopped, failed, interrupted, stalled), progreso y ETA, KPIs, curvas por época, loss por
  batch y la config y el entorno del run. Se refresca con `st.fragment(run_every=…)`, con
  interruptor Live/Pause e intervalo de 2/5/10/30 s; la web sigue siendo solo de lectura. Lector
  puro en `src/web/events_reader.py`; `RunInfo.events_path` asocia el JSONL al run por el sello.
- MedMNIST con 3 semillas: media ± desviación en test (`docs/datasets/medmnist.md`). `--seed` en
  `train_single_gpu.py` (se escribe en el log) y `scripts/summarize_runs.py` para agregar los
  resultados de test de runs repetidos.
- Perfilado automático de modelos (`src/model_profile.py`): `paravit estimate` y la pestaña
  *Estimate* de la web funcionan con cualquier modelo de timm, no solo con los seis de la
  tabla. Parámetros y FLOPs se miden con `torch.utils.flop_counter` en el dispositivo `meta`
  (sin pesos ni GPU; con CPU como respaldo) y coinciden con la tabla a ≤3,6 %. La memoria de
  activaciones se estima por regla y se marca como no calibrada. Nuevas opciones del CLI:
  `--in-chans`, `--num-classes` e `--img-size`.
  La búsqueda en la tabla es exacta (`wide_resnet50_2` ya no se confunde con `resnet50`), un
  modelo de la tabla con otro `in_chans`/`img_size` se perfila, y los modelos perfilados avisan
  de que la MFU (tiempo y energía) no está calibrada para su arquitectura y de que la confianza
  de la calidad esperada es baja.
- `docs/research/estado_del_arte_prediccion.md`: revisión del trabajo relacionado sobre predicción del
  rendimiento del entrenamiento (Paleo, Habitat, Daydream, DS-Analyzer, NeuSight…) y posibles contribuciones.
- Web: curva de AUC de validación en *Run results → Curves* (con la mejor época marcada) y
  *Best Val AUC* en las métricas y en el veredicto, solo para runs cuyo `epoch_metrics_*.csv`
  trae `val_auc`; los runs anteriores se ven igual que antes. En *Compare*, `val_auc` se puede
  superponer y la tabla resumen muestra *Best Val AUC* si algún run seleccionado lo tiene.
- Web: filtro por dataset en el selector de runs de la barra lateral (solo aparece cuando hay
  runs de más de un dataset).
- Test de humo de la web con `streamlit.testing` (AppTest): renderiza las cinco secciones con un
  run de BigEarthNet y otro de MedMNIST como run activo.
- Arnés de trabajo con IA versionado (`docs/flujo_ia.md`): agentes de proyecto con modelo por
  rol (`.claude/agents/`), skills (`/estado`, `/cerrar-bloque`, `/revisar-pr`, `/nuevo-dataset`),
  hooks (estado del proyecto al abrir sesión, ruff al editar Python, bloqueo de commits con
  atribución de IA) y permisos de proyecto (`.claude/settings.json`).
- Sección "Estado actual" en `docs/roadmap.md` para retomar el proyecto desde cualquier sitio.
- ruff como dependencia de desarrollo (`[tool.ruff]`: solo errores reales), lint en CI,
  `.pre-commit-config.yaml` y plantilla de PR con checklist.
- `training.select_by: auc`: elige el checkpoint y el early stopping por el AUC de validación
  (sin umbral), útil cuando las clases son tan raras que el F1 a 0,5 se queda en cero. El AUC de
  validación se calcula en cada época y se escribe como `val_auc` en `epoch_metrics_*.csv`.
- El checkpoint guarda el umbral óptimo elegido en validación (`metrics.optimal_threshold`).
- MedMNIST: cuatro datasets médicos (`medmnist_breast`, `medmnist_derma`, `medmnist_path`,
  `medmnist_chest`) que cubren las tareas binaria, multiclase y multi-etiqueta, con 1 o 3
  canales. Incluye script de descarga con verificación MD5 y configs locales.
- AUC macro (ROC) en `eval.py`, definido como en el evaluador de MedMNIST.
- Los runs de datasets distintos de BigEarthNet se guardan en `logs/{env}/<dataset>/...` (la
  ruta de BigEarthNet no cambia) y la línea `Configuración:` del log incluye `dataset=`.
- Soporte multi-dataset: ficha de dataset (`DatasetSpec`) y registro por nombre; el config
  elige el dataset con `data.name` (por defecto `bigearthnet`, sin cambios para los configs
  existentes). Guía para añadir datasets en `docs/datasets/README.md`.
- Tareas `multilabel`, `binary` y `multiclass` (`src/training/tasks.py`): pérdida, suavizado,
  predicciones, métricas y búsqueda de umbral según el tipo de problema.
- El modelo acepta un número de canales de entrada distinto de 3 (`in_chans`).
- Tests de caracterización que fijan los números exactos del entrenamiento y test de punta a
  punta con un dataset multiclase de un canal y otro binario.
- `AGENTS.md`, `docs/roadmap.md` y este `CHANGELOG.md`.

### Cambiado
- Web: se eliminan `DashboardContext.refresh_interval` (nadie lo usaba) y los comentarios
  engañosos sobre un refresco en vivo que no existía.
- `eval.py` aplica a test el umbral elegido en **validación** (el del checkpoint, o ajustado sobre
  val en checkpoints antiguos); antes lo ajustaba sobre el propio test, lo que inflaba el "F1 con
  umbral óptimo". `--threshold-from test` mantiene el comportamiento anterior. El CSV de test
  indica `threshold_source` y la web lo muestra.
- AUC por rangos vectorizado (de ~32 s a ~0,2 s a escala de BigEarthNet).
- Trainers (single, DDP, heterogéneo, model-parallel), deep tracing, `eval.py`, benchmark y
  estudio de convergencia dejan de asumir BigEarthNet. Para BigEarthNet los resultados son
  idénticos: verificado en GPU (vit_tiny, subset, semilla fija) con single, DDP, paralelismo de
  modelo y heterogéneo GPU+CPU, comparando todos los CSV y el log contra el código anterior.
- `model.num_classes` pasa a ser opcional: sale de la ficha del dataset; si se indica, debe
  coincidir. El validador comprueba la pérdida y `pos_weight` según la tarea.
- `src/data/dataset.py` pasa a `src/data/bigearthnet.py`; `BigEarthModel` pasa a
  `TimmClassifier` (se mantienen los alias antiguos).
- Las métricas de validación se calculan en un único sitio (`Trainer._evaluate`), que DDP
  reutiliza; antes estaban duplicadas.
- `eval.py` usa la misma búsqueda de umbral que el entrenamiento: en caso de empate prefiere 0,5
  (antes elegía el umbral más bajo). En tareas sin umbral no escribe `f1_opt`/`threshold`.
- Cambios mínimos de comportamiento, documentados:
  - el trainer heterogéneo predice con `> 0.5` como los demás (antes `>= 0.5`); solo difiere si
    un logit es exactamente 0;
  - `pos_weight` se aplica también con los alias de la pérdida (`bcewithlogits`), que antes lo
    ignoraban sin avisar.
- El validador rechaza `select_by: f1_optimal` en tareas sin umbral y nombres de dataset en
  mayúsculas.

### Corregido
- `--trace deep` con `batch-table` ya alimenta a `--layers batch-monitor`: el bucle propio de
  `DeepTracingDecorator` no llamaba a los batch hooks del `Trainer` y `batch_metrics_*.csv`
  quedaba solo con la cabecera. Llama a los hooks con el mismo diccionario que
  `Trainer.train_epoch`; ningún cálculo cambia (los tests de caracterización siguen igual).
- Docstrings que afirmaban que `plot` y `confusion` guardan PNGs (`scripts/train_single_gpu.py`,
  `ConfusionMatrixDecorator`, `PlottingDecorator`): `plot` solo acumula el histórico en memoria y
  `confusion` escribe `perclass_metrics_*.csv` y `confusion_matrix_*.csv`.
- `LayerHooksDecorator` escribe en el logger `trainer` si tiene handlers (como los decoradores
  `@fn`) y en stdout si no; el texto es el mismo.
- `psutil` declarado en las dependencias (lo importan `src/web/system_monitor.py` y las sondas del
  benchmark; solo llegaba de forma transitiva vía `ipykernel`).
- El hook guardián de commits solo examina el mensaje del `git commit`: antes bloqueaba
  comandos que mencionaban "Co-Authored-By" en otro sitio (por ejemplo, el cuerpo de un PR).
- Web: `use_container_width` (obsoleto en Streamlit) pasa a `width="stretch"`; la columna
  "Best epoch" usa un solo tipo y ya no provoca trazas de Arrow en Compare y Details.
- Imports sin usar eliminados en `src/` y `tests/` (ruff F401/F541).
- El builder no pasaba la pérdida configurada al trainer heterogéneo, así que `pos_weight` se
  ignoraba en esa estrategia. Ningún run usó `pos_weight` en heterogéneo.
- `model.dropout` del config no llegaba al modelo: todos los entrenamientos usaban 0.1, también
  los configs "v3" que pedían 0.3. Los resultados del TFG llevan una nota con la corrección.
- El resumen de torchinfo (`--inspect model-summary`) y el análisis del benchmark asumían 3
  canales de entrada; ahora los leen del modelo.
- Documentación para agentes estandarizada: `CLAUDE.md` mínimo que importa `AGENTS.md`; el
  antiguo `CLAUDE.md` (1300 líneas) se reparte en `docs/` (`arquitectura.md`,
  `infraestructura.md`, `resultados_tfg.md`, `historial_tfg.md`, `datasets/`), con un índice en
  `docs/README.md`. Corregidas varias referencias obsoletas (PNGs que ya no se generan,
  `feasibility` → `benchmark`).

## [v1.0-tfg] - 2026-07-12

Versión entregada como Trabajo de Fin de Grado (ULL, julio de 2026). Entre otras cosas incluye:

- Entrenamiento de ViT sobre BigEarthNet-S2 con una arquitectura de decoradores (Decorator +
  Template Method) común a cuatro estrategias: single-GPU, DDP, paralelismo de modelo y DDP
  heterogéneo GPU+CPU.
- Predictor analítico de tiempo, memoria, energía y coste (`paravit estimate`) y benchmark empírico
  (`paravit benchmark`).
- CLI `paravit` y dashboard Streamlit.
- Estudio de escalado en Kaggle 2×T4 y VERODE; mejor modelo con F1 macro de 0,620 en test.

El detalle completo está en [`docs/historial_tfg.md`](docs/historial_tfg.md) y los resultados
en [`docs/resultados_tfg.md`](docs/resultados_tfg.md).

[Unreleased]: https://github.com/alerguezrojas/tfg-distributed-transformers/compare/v1.0-tfg...HEAD
[v1.0-tfg]: https://github.com/alerguezrojas/tfg-distributed-transformers/releases/tag/v1.0-tfg
