# Changelog

Cambios notables del proyecto. Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).

## [Unreleased]

### Experimentos

- PANDA, tres semillas de la base a 1.024 px en Kaggle (2 × T4, DDP) frente a las tres locales a
  512 px. El QWK de test no mejora de forma significativa (0,804 ± 0,017 frente a 0,794 ± 0,012;
  *bootstrap* +0,010 [−0,010; +0,030]). Sí mejoran la exactitud (+0,030) y el AUC de ISUP ≥ 1
  (+0,014). Se sigue a 512 px. Logs en `logs/kaggle/panda/ddp/` y análisis en
  `docs/datasets/panda.md`.

## [1.2.1] - 2026-11-21

### Seguridad

- setuptools 81 → 84.2.1 (cierra GHSA-xmj3-9gf7-h52w y GHSA-h35f-9h28-mq5c / PYSEC-2026-3447): la
  auditoría de la CI ya no ignora ningún aviso. Comprobado con torch 2.15, tensorboard 2.20 y la
  suite. `uv.lock` recoge la versión 1.2.0 del proyecto.

## [1.2.0] - 2026-11-20

### Añadido
- PANDA a 1024 px en Kaggle (2 × T4, DDP): control a 512 px igual que en local (0,801 frente a
  0,794 ± 0,012) y base a 1024 px con QWK 0,821 en test, el mejor run hasta ahora (una semilla; la
  ordinal a 1024 da 0,806). Logs en `logs/kaggle/panda/`, tablas en `docs/datasets/panda.md` y
  contraste del predictor en `docs/performance_model.md` (subestima el tiempo en T4: −40 % a 512,
  −16 % a 1024).
- Kernel de entrenamiento de Kaggle por lotes (`BATCHES`/`BATCH` en `kaggle/panda_train/run.py`);
  lote `seeds_1024` con las semillas 43 y 44 de la base a 1024 px.
- `--seed` y `--data-root` en `paravit train` y en todos los scripts de entrenamiento (DDP,
  heterogéneo y paralelismo de modelo; antes solo en el de una GPU), con la sobrescritura de la
  config en un único sitio (`src/training/config_overrides.py`). El menú los ofrece en las opciones
  avanzadas.
- Incorporación de colaboradores: `CONTRIBUTING.md` (ramas protegidas, ramas largas `tfg/<nombre>`,
  commits, definición de hecho, pre-commit, revisión y releases), `docs/primeros_pasos.md` (guía
  paso a paso con un primer entrenamiento en BreastMNIST, mapa del código, protocolo de
  comparación de modelos y licencias), plantillas de incidencias (fallo y propuesta; las
  vulnerabilidades van al aviso privado) y sección «Colaborar» en el README. La plantilla de PR
  añade tests, credenciales y licencias.
- Entrenamiento en Kaggle (2 × T4): kernels en `kaggle/` para generar los mosaicos de PANDA desde
  los datos de la competición y para entrenar con DDP y evaluar en test una lista de experimentos
  (`timings.csv` para contrastar el predictor). La partición se comprueba por md5 contra la local.
  `scripts/kaggle_push.sh` rellena el usuario de Kaggle al lanzar, para no publicarlo en el repo.
- Backbones que no son de timm: `BackboneSpec` y registro por nombre (`src/models/backbones.py`),
  adaptador genérico `hf:<repo>` de Hugging Face `transformers` (pooling `model.pooling`, media de
  tokens por defecto) e I-JEPA ViT-H/14 (`ijepa_vith14_1k`, `ijepa_vith14_22k`, licencia CC BY-NC 4.0
  con aviso al construirlo). Los nombres que no están en el registro siguen siendo timm, sin
  cambios. `transformers` es opcional (`uv sync --extra hf`).
- `model.freeze_backbone: true` (sonda lineal): solo se entrena la cabeza; el validador comprueba
  también `model.pooling` y rechaza los backbones no timm con paralelismo de modelo.
- Guía [docs/modelos/README.md](docs/modelos/README.md): cómo añadir un modelo, sonda lineal frente
  a ajuste fino, protocolo de comparación y licencias. Un nombre mal escrito (ni timm ni registrado)
  da un error que lista los backbones registrados. Probado con `transformers` 5.12 real (modelo
  I-JEPA diminuto desde config; el test se salta si no está instalado).
- Adaptador `hf:`: sin *pooler* ni `mask_token` entrenable (DDP fallaba por parámetros sin
  gradiente), la media excluye `[CLS]` y *register tokens*, `eval.py` y el benchmark respetan
  `model.pooling`, y Grad-CAM funciona con `freeze_backbone`. La línea `Configuración:` añade
  `freeze_backbone=` y `backbone=`; la validación del predictor ignora las sondas lineales.

- La web conoce el dataset y la métrica de cada run. Cada entrenamiento escribe un manifiesto
  `run_<ts>.json` (lo de `run_start` más el comando; unos KB, se versiona) por el mismo observador
  que los eventos, y `scripts/backfill_run_manifests.py` lo genera para los runs viejos (de sus
  eventos o de la línea `Configuración:`; se ejecutó sobre `logs/`). El evento `run_start` gana
  `ordinal`. Un lector común (`src/web/run_meta.py`) saca dataset, tarea, métrica principal,
  pérdida, semilla, hospital, color y resolución del manifiesto, de los eventos o del log.
- Etiquetas de run que distinguen semilla, pérdida, hospital, color y resolución
  (`… vit_small [amp] · ce_emd · seed 43 · train: karolinska · color 0.2 · 512px`), con los
  segundos si dos runs coincidirían; la barra lateral las muestra enteras.
- Overview, Run results y Compare usan la métrica principal de cada run (QWK en PANDA, AUC en
  MedMNIST, F1 en BigEarthNet), con su curva (incluida `val_qwk`) y la pérdida real en los títulos.
  Los indicadores del Overview son por dataset. Los paneles de BigEarthNet solo salen para este
  dataset; los demás tienen una vista genérica (tarea, clases, tamaños, hospitales, distribución de
  clases y miniaturas del panel del monitor, si el run guardó `inspect_*`).
- *Compare* agrupa los runs que solo difieren en la semilla: media ± desviación en la tabla y curvas
  con banda; trabaja con un dataset cada vez, hasta 12 runs, con columnas de semilla, hospital,
  color y resolución.

### Corregido
- Calidad (SonarCloud, PR a `main`): la línea de comando del manifiesto se compone con una función
  propia de presentación (`command_line`, nunca se ejecuta) en vez de `shlex.join`, que el análisis
  marcaba como inyección de comandos; la media de tiempos de *Compare* detecta NaN con `math.isnan`
  y no con `s == s`; `manifest_from_log` baja de complejidad.
- **`--seed` en DDP cambia también el orden de los datos**: el `DistributedSampler` (y el del
  heterogéneo) barajaba siempre con la semilla 0, así que las repeticiones con otra semilla solo
  cambiaban la inicialización y los aumentos. Ahora usan `training.seed`, igual en todas las
  ranks (`sampler_seed`). Afecta a la comparación entre semillas de los runs DDP anteriores.
- El manifiesto `run_<ts>.json` sanea los valores como los eventos (un NaN o una fecha en el YAML
  ya no impiden escribirlo) y no deja un `.tmp` en `logs/` si la escritura falla. La ayuda de
  `--data-root` aclara que solo cambia `data.root`.
- **Checkpoints DDP y de paralelismo de modelo cargables en `eval.py`**: `Trainer` guardaba el
  `state_dict` del envoltorio (prefijos `module.` de DDP y `base.` de `ModelParallelViT`), así que
  `scripts/eval.py` no cargaba esos checkpoints (falló en los tres runs de Kaggle del 20/11). Ahora
  se guardan los pesos del clasificador suelto y la carga (evaluación y reanudación) acepta los
  checkpoints antiguos con prefijo. El kernel de Kaggle copia solo los logs que escribe.
- `--epochs 1` ya no falla con los configs que traen `warmup_epochs: 1`: si `--epochs` deja
  `epochs <= warmup_epochs`, se baja `warmup_epochs` a `epochs - 1` (mínimo 0) y se avisa por el log.
  Un YAML con esa combinación sigue fallando en el validador.
- Documentación desfasada: el recuento de pruebas del README y de la guía de uso (411 → más de 1.200),
  el tiempo de la batería en `AGENTS.md` (~10 s → ~1,5 min) y la ruta de los logs en el mapa del código
  (lleva el dataset).
- «Fastest epoch» del Overview marcaba 0,0 min porque mezclaba datasets (MedMNIST tarda ~1,4 s por
  época): ahora es del dataset en curso y se muestra en segundos si dura menos de un minuto.
- Los soportes por clase de BigEarthNet ya no se aplican a runs de otro dataset (Per-class) y la
  descripción de Confusions deja de decir «19 clases» fuera de BigEarthNet. La vista *Live* nombra
  el QWK.

### Seguridad
- torch 2.11 → 2.15 (torchvision 0.29, CUDA 13.2), que corrige CVE-2025-3000. Validado en la
  RTX 3060 Ti local: suite completa en verde (golden incluido) y PANDA a 512 px con el mismo
  tiempo por época (~82 s) y el mismo QWK que 2.11 dentro del ruido. Verode sigue con su torch
  2.7.1+cu118 instalado aparte (ver `docs/infraestructura.md`). La auditoría ya no ignora el aviso.
- Dependencias con vulnerabilidades conocidas actualizadas en `uv.lock` (pillow 13, tornado,
  starlette, urllib3, jupyter-server, jupyterlab, mistune, filelock y otras; streamlit sube a
  1.67 porque lo exige pillow 13). torch se queda en 2.11 hasta validar 2.13 en las GPU.
- El job `audit` de la CI pasa a bloquear: cero avisos salvo cuatro aceptados con su motivo en
  `ci.yml` (torch.jit, que no se usa, y dos de setuptools al construir paquetes).

## [1.1.0] - 2026-11-20

Primera versión posterior al TFG: soporte multi-dataset (fichas `DatasetSpec` y tareas), MedMNIST,
PANDA con clases ordenadas y QWK, observabilidad del entrenamiento (flujo de eventos, vista *Live*,
monitor de predicciones e *Inspector* en la web), evaluación clínica en test y perfilado automático
de modelos de timm en el predictor.

### Añadido
- El Inspector de la web lee las predicciones de test: si el run tiene `test_preds_<ts>.npz`,
  ofrece la fuente «Test (eval)» junto a «Validation (monitor)» en Curves, Clinical, Calibration y
  Subgroups; en tareas ordinales, Clinical muestra la tabla de cortes «grado ≥ k». `eval
  --save-predictions` sin ruta (`--save-predictions-auto` en `paravit.py eval`) guarda ese fichero
  en la carpeta de logs del run con su sello, deducido del checkpoint con la estructura del
  *builder* (`checkpoint.dir` y `output.env`; carpeta con sello o `checkpoint_epoch_*.pt` antiguos, con
  aviso si es ambiguo) o el de la evaluación si no se puede. El `.npz` guarda `threshold_source` y la
  web avisa si el umbral se eligió sobre el propio test. La web solo abre ficheros dentro de `logs/` y carga el `.npz` con `allow_pickle=False`.
- Seguridad y calidad del repositorio: `SECURITY.md` (aviso privado de GitHub, alcance y reglas),
  Dependabot semanal agrupado (`uv` y `github-actions`, hacia `develop`), análisis CodeQL de Python
  (`.github/workflows/codeql.yml`), `permissions: contents: read` y `concurrency` en CI, job
  `audit` con `pip-audit` sobre `uv export` (informativo por ahora: el lock tiene avisos en
  dependencias de desarrollo) y hook de gitleaks en `pre-commit`. Historial escaneado: sin secretos.
- PANDA, mejoras y generalización entre hospitales (18-19/11/2026, 14 runs): con tres semillas,
  base 0,794 ± 0,012 y pérdida ordinal 0,802 ± 0,009 de QWK en test, sin mejora clara (con una
  sola semilla parecía +0,025); el color tampoco mejora. Cortes clínicos (ISUP ≥ 1 y ≥ 2) y
  entrenamiento en un hospital evaluado en el otro (QWK 0,81 → 0,55 y 0,75 → 0,37; ni la ordinal
  ni el color evitan la caída). Runs en `logs/local/panda/` y tablas en `docs/datasets/panda.md`.
- Pérdida ordinal `ce_emd` para clases ordenadas (`DatasetSpec.ordinal`): entropía cruzada +
  `training.emd_weight` × EMD al cuadrado entre la distribución predicha y la real (Hou et al.
  2017), compatible con mixup y suavizado; misma cabeza y predicción. Config
  `configs/train_panda_ordinal.yaml`. El builder pasa a `build_criterion` la tarea de la ficha
  (antes la reconstruía por nombre y perdía `ordinal`).
- Evaluación en test más clínica: `eval --save-predictions RUTA.npz` guarda probabilidades,
  objetivos, umbral, clases, tarea y variables de grupo por muestra (`save_predictions` /
  `load_predictions` en `src/analysis/io.py`); si el dataset expone `groups()` imprime n, exactitud,
  F1, AUC (y QWK) por grupo, y en tareas ordinales una tabla de cortes «grado ≥ k» con sensibilidad,
  especificidad, VPP, VPN y AUC (`src/analysis/clinical.py`). Las líneas anteriores de `eval.py` no
  cambian (las secciones nuevas van después).
- PANDA: `data.color_jitter` (aumentos de color solo en train, `ColorJitter` con hue = s/4;
  0 = desactivado, idéntico a antes), `data.providers` (filtra la partición por centro y split,
  para entrenar en un hospital y evaluar en el otro; `class_counts` lo respeta) y las configs
  `train_panda_color.yaml`, `train_panda_radboud.yaml` y `train_panda_karolinska.yaml`.
- El `PredictionMonitor` escribe `groups.npz` cuando el dataset de validación expone `groups()`
  (también envuelto en `Subset`), alineado con `monitor_set.npz`: la vista *Subgroups* ya tiene
  datos en PANDA (variable `data_provider`).
- Revisión sobre medición clínica del cáncer de próstata (`docs/research/medicion_clinica_prostata.md`):
  grado Gleason/ISUP, variabilidad entre patólogos, métricas de los modelos de IA con cifras publicadas y
  lista de requisitos para un dataset clínico nuevo.
- PANDA, línea base (02/11/2026): mosaicos del dataset completo y primer entrenamiento a 512 px
  (ViT-S, RTX 3060 Ti): QWK 0,781 en test, 0,745/0,760 por centro. Run y tabla en
  `docs/datasets/panda.md`.
- Dataset PANDA (`panda`, multiclase ordinal ISUP 0-5, QWK, 512 px): ficha y lector (`src/data/panda.py`), mosaicos de
  tiles con más tejido (`src/data/wsi.py`, puro NumPy), `scripts/prepare_panda.py` (idempotente,
  paralelo, tolerante a TIFF ilegibles, `--make-splits`), partición 70/15/15 estratificada por
  centro y grado (semilla 42, por biopsia), `configs/train_panda.yaml`, ficha
  `docs/datasets/panda.md` y dependencias `tifffile` e `imagecodecs`. El Dataset expone
  `data_provider` para el futuro `groups.npz`. La ficha declara clases ordenadas (QWK y
  `select_by: qwk`) y el config entrena a 512 px (`model.img_size`). La partición ordena cada
  estrato por una huella `sha256(semilla:id)`, estable si se quitan unas pocas biopsias.
- Clases ordenadas: `DatasetSpec.ordinal` (solo multiclase) añade el kappa cuadrático ponderado
  (`metrics.quadratic_weighted_kappa`, `qwk`) a las métricas de la tarea, la columna `val_qwk` del
  CSV por época (NaN en los demás datasets), `select_by: qwk` y la línea `QWK` de `eval.py`. La
  tarea sigue llamándose `multiclass` aguas abajo (inspección, web).
- `model.img_size`: el *backbone* de timm se construye para otro lado de entrada (p. ej. 512 px
  para los mosaicos de PANDA) en las cuatro estrategias y en `eval.py`; los resúmenes de torchinfo
  usan ese tamaño (`input_size`), igual que el benchmark (modelo y lotes sintéticos). Con CNNs la
  clave se ignora al construir (aceptan cualquier tamaño). Sin la clave, el comportamiento no cambia.
- Selección del mejor checkpoint: una métrica indefinida (NaN) en la primera época ya no deja el run
  sin checkpoint, y un mejor valor negativo (QWK por debajo del azar) se informa en los eventos en
  vez de salir como *n/a*.
- Revisión bibliográfica del dataset PANDA (`docs/research/panda.md`): formato, licencia, métrica,
  soluciones ganadoras, ViT, problemas conocidos e implicaciones para el proyecto.
- Roadmap: el dataset del tutor es PANDA (biopsias de próstata, grado ISUP); estado de la
  descarga, primer prototipo de mosaicos y decisiones pendientes.
- Diagrama genérico del patrón Factory (`docs/diagramas/multidataset/factory.puml`), junto a los
  de Strategy y Registry.
- Alertas (`src/training/alerts.py`): reglas puras y un `AlertMonitor` que escribe eventos `alert`
  (`level` warning/critical, `code`, `message`, `epoch`, `batch`, `value`, `threshold`, `group`)
  para `nan_loss`, `loss_divergence`, `overfitting`, `grad_explosion`,
  `update_ratio_out_of_range`, `dataloader_bound`, `gpu_memory_high`, `gpu_temp_high` y
  `throughput_drop`. Una vez por (código, grupo) y se re-arma al desaparecer la condición;
  umbrales en `monitor.alerts` (validados), activas por defecto con el flujo de eventos. No
  fuerzan cálculo por batch ni alteran el entrenamiento (test de invariancia). `JsonlEventSink`
  gana `subscribe` para observar los eventos que otros productores emiten. Contrato en
  `docs/inspeccion.md`.
- Alertas: `grad_explosion` también sobre la norma total del gradiente antes del recorte
  (`Trainer.last_grad_norm`, solo con *batch hooks*; los tres bucles), ignora los pasos que el
  `GradScaler` salta, y el par (código, grupo) solo se da por emitido tras escribir el evento.
- Histogramas de gradientes por grupo de capas (gradientes del último batch de cada época) en el
  evento `histograms`, que ahora lleva `kind: "weights" | "gradients"`.
- `pytest.ini`: `tmp_path_retention_policy = failed` y `tmp_path_retention_count = 1`, para que
  los directorios temporales de los tests (con checkpoints) no llenen `/tmp`.
- Web, Inspector: vistas clínicas. **Curves** con intervalos de confianza *bootstrap* (macro y por
  clase; 200/1000/2000 remuestras, semilla fija, cacheados); nueva vista **Clinical** (umbral
  manual, óptimo de validación, sensibilidad objetivo o Youden; sensibilidad, especificidad, VPP,
  VPN, F1 y 2×2 con IC, prevalencia editable, ROC con el punto); **Calibration** con pendiente,
  intercepto, ECE adaptativa y por clase; **Subgroups** (solo con el `groups.npz` opcional,
  documentado en `docs/inspeccion.md`, `load_groups` en `src/analysis/io.py`); **Alerts** y banner
  de alertas en Inspector y Live a partir de los eventos `alert` (con `group`); selector
  pesos/gradientes en **Layers** (`histograms.kind`). Helpers en `src/web/inspector_stats.py`.
- `paravit report <run>` (`src/reporting/`): informe HTML autocontenido de un run (un solo
  fichero, plotly.js una vez, imágenes en base64) con resumen y entorno, curvas de entrenamiento,
  rendimiento y GPU, alertas, evaluación con IC95 *bootstrap* (semilla anotada), ROC/PR, calibración
  (pendiente e intercepto), puntos de operación, peores errores, Grad-CAM, dinámica y lista
  TRIPOD+AI con cada ítem documentado, parcial o pendiente. `<run>` es un `events_*.jsonl`, la
  carpeta del run o un sello. Degrada indicando qué flag activar. `report_*.html` se ignora en git.
- Estadística para publicación en `src/analysis/` (solo NumPy): intervalos de confianza
  *bootstrap* (`bootstrap_ci`, `metrics_with_ci` con AUC, AP, F1, sensibilidad, especificidad,
  exactitud, ECE y Brier; ruta vectorizada), pendiente e intercepto de calibración, ECE por clase
  y ECE adaptativo, punto de operación clínico (sensibilidad objetivo, Youden, VPP/VPN por
  prevalencia) y métricas por subgrupo con brecha máx-mín. No cambia ninguna función existente.
  Descrito en `docs/inspeccion.md` y `docs/arquitectura.md`.
- `docs/research/inventario_herramientas_observabilidad.md`: inventario funcionalidad a
  funcionalidad de 14 herramientas de observabilidad, matriz comparativa, requisitos de imagen
  médica y lista priorizada; la segunda ola de la fase 4 sale de ahí.
- Web: página **Inspector** (system, performance, predictions, errors, curves, calibration,
  dynamics, embeddings, explain, layers) para analizar un run en vivo y a posteriori a partir de
  `events_*.jsonl` e `inspect_<ts>/`. `EventsFollower` reconoce `system`, `layers`, `histograms`,
  `monitor_start` e `inspection` (acotados en memoria). Helpers puros en
  `src/web/inspector_data.py`; fixture `tests/fixtures/make_inspector_run.py`.
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
- Monitor de predicciones (`--monitor` en los scripts y en `paravit train`, o `monitor.enabled`):
  en cada época guarda en `inspect_<ts>/` las probabilidades, la pérdida por imagen, los
  *embeddings* y Grad-CAM (ViT y CNN) de un subconjunto fijo de validación con miniaturas, y las
  probabilidades de toda la validación; `LayerStatsMonitor` registra normas de gradiente y peso,
  ratio de actualización e histogramas por grupo de capas. No altera el entrenamiento (test de
  pesos idénticos). Coste en DermaMNIST (512 muestras, 16 mapas): ~0,6 s por época, fuera del
  tiempo medido.
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
- Checkpoints por run (20/11/2026): cada run guarda en su carpeta
  `checkpoints/.../<modelo>/<DDMMYYYY_HHMMSS>/` y por defecto solo conserva `best.pt` y `last.pt`
  (antes, un `checkpoint_epoch_NNN.pt` por época mejorada, todos los runs del modelo mezclados; llenó
  el disco el 19/11/2026). Evita mezclar runs y deja dos ficheros por run (~0,5 GB con ViT-S), pero
  los runs se acumulan y hay que limpiar los viejos a mano. Nueva clave `checkpoint.keep` (`best_last` | `all`, validada) y escritura
  atómica. `eval`, `--resume` y el menú entienden ficheros antiguos, `best.pt`/`last.pt` y carpetas
  de run (`resolve_checkpoint`). `last.pt` guarda la época global, el `GradScaler`, la mejor métrica y
  la paciencia, y `--resume` los restaura; si falla su escritura se avisa y se sigue. Mismas reglas en las cuatro estrategias (DDP: solo rank 0).
- `.gitignore`: `checkpoints/runs` y `checkpoints/local` (copias de checkpoints y predicciones de
  test por run; en el equipo local son enlaces al SSD).
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
- Importación de runs en la web: la comprobación de que el destino queda dentro de `logs/` usaba
  un prefijo de texto (aceptaba `logs_evil/` vía enlace simbólico); ahora usa `is_relative_to`.
- La línea `Configuración:` del log muestra la pérdida real de la tarea cuando el YAML no define `training.loss`
  (`ce` en multiclase, antes `bce`), con el helper `configured_loss_name`.
- `PrecisionRecallReporter` imprime también `qwk` (datasets ordinales) y `auc` por época; el parser de la web ignora
  esas líneas.
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

[Unreleased]: https://github.com/alerguezrojas/tfg-distributed-transformers/compare/v1.2.1...HEAD
[1.2.1]: https://github.com/alerguezrojas/tfg-distributed-transformers/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/alerguezrojas/tfg-distributed-transformers/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/alerguezrojas/tfg-distributed-transformers/compare/v1.0-tfg...v1.1.0
[v1.0-tfg]: https://github.com/alerguezrojas/tfg-distributed-transformers/releases/tag/v1.0-tfg
