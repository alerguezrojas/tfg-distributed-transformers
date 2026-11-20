# Inventario de herramientas de observabilidad del entrenamiento y de las decisiones del modelo

> Referencia de diseño para el análisis en vivo del entrenamiento y de la inferencia de ParaViT-Lab.
> Complementa la comparativa de alto nivel de
> [observabilidad_entrenamiento.md](observabilidad_entrenamiento.md) (trackers, formatos de eventos,
> métodos de explicación) con un inventario **funcionalidad a funcionalidad**. Fecha de la revisión:
> 25/10/2026. Las afirmaciones se apoyan en la documentación oficial, el código fuente de cada
> herramienta o Crossref/arXiv/PMC (enlace en cada sección). Lo que dicen la **documentación o los autores** se distingue de
> la **interpretación** de esta revisión, marcada así: *(interpretación)*. Lo que no se ha podido
> comprobar figura como *no verificado* (`n.v.` en la matriz).

## 0. Convenciones y alcance

- **Granularidad:** *batch* (cada paso), *época*, *muestra* (una fila por imagen), *run* (un valor
  por ejecución). **Momento:** *vivo* (mientras entrena) o *posterior* (se calcula o se mira
  después).
- **Marcas de la matriz (sección 2):** `Sí` = la herramienta lo trae de serie y consta en la fuente
  citada; `P` = parcial (hay una pieza, hay que montarlo a mano o solo vale para un caso); `No` = no
  consta en la documentación consultada o queda fuera del propósito de la herramienta
  *(interpretación)*; `n.v.` = no he podido verificarlo.
- **Aviso sobre `No`:** demostrar una ausencia con documentación es poco fiable. Un `No` significa
  "no lo encontré donde debía estar", no "imposible".
- **Estado del proyecto:** las columnas de estado de la sección 4 se basan en
  [`docs/inspeccion.md`](../inspeccion.md), [`docs/eventos.md`](../eventos.md) y la lista de ficheros
  de `src/web/`. **No he leído el código de entrenamiento** para confirmarlo.
- **Neptune** (†) cerró su servicio el 5/03/2026 (sección 1.4): solo se incluye como referencia
  histórica de diseño.
- Las docs de W&B viven ahora en `docs.coreweave.com`; algunas páginas devolvieron "Access
  Restricted" de forma intermitente. Donde ocurrió, la fuente es el código del SDK en GitHub, y se
  indica.

---

## 1. Inventario por herramienta

### 1.1 Weights & Biases (W&B)

Fuentes: [métricas de sistema](https://docs.coreweave.com/models/ref/python/experiments/system-metrics) ·
[ajuste del intervalo](https://docs.coreweave.com/support/models/articles/how-can-i-change-how-frequently-to-log-s.md) ·
[`wandb_watch.py`](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_watch.py) ·
[media](https://docs.coreweave.com/models/track/log/media) ·
[tablas](https://docs.coreweave.com/models/tables) ·
[`table.py`](https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/table.py) ·
[gráficas personalizadas](https://docs.coreweave.com/models/track/log/plots) ·
[`wandb/integration/sklearn/plot`](https://github.com/wandb/wandb/tree/main/wandb/integration/sklearn/plot) ·
[Sweeps](https://docs.coreweave.com/models/sweeps) ·
[coordenadas paralelas](https://docs.coreweave.com/models/app/features/panels/parallel-coordinates) ·
[importancia de parámetros](https://docs.wandb.ai/ref/app/features/panels/parameter-importance) ·
[alertas](https://docs.coreweave.com/models/runs/alert).

| Vista / función | Qué se ve | Granularidad | Momento |
|---|---|---|---|
| Métricas de sistema | Ver lista exacta abajo | Serie temporal, cada 15 s (ver discrepancia) | Vivo |
| `wandb.watch(models, criterion, log="gradients", log_freq=1000, idx, log_graph=False)` | Histogramas de gradientes (por defecto), de parámetros (`"parameters"`) o de ambos (`"all"`); opcionalmente el grafo del modelo. `log=None` lo desactiva. `log_freq` está en *batches* | Por capa, cada 1000 batches por defecto | Vivo (ver 1.1.1: el vivo no consta en doc oficial) |
| Media: `wandb.Image` | Imagen con `caption`, máscaras de segmentación (opacidad ajustable) y cajas con filtros. Un **deslizador de pasos** en el panel permite comparar la misma imagen a lo largo del entrenamiento. Recomendación de la doc: menos de 50 imágenes por paso | Por muestra y paso | Vivo |
| Media: otros | `Video`, `Audio` (hasta 100 clips por paso), `Histogram`, `Object3D`, `Molecule`, `Html`; el texto va en `Table` | Por paso | Vivo |
| Tables (`wandb.Table`) | Rejilla bidimensional con tipos primitivos, listas/diccionarios y medios (imágenes). Usos citados por la doc: "comparar cómo rinden distintos modelos sobre el mismo test" y "consultar para encontrar ejemplos mal clasificados con frecuencia". Filtrar, ordenar y agrupar en la interfaz. Se puede comparar entre runs en el espacio de trabajo | Por muestra | Vivo (ver 1.1.1) |
| `Table(log_mode=...)` | `IMMUTABLE` (por defecto, solo se registra una vez), `MUTABLE` (se vuelve a registrar creando una versión de artefacto) e `INCREMENTAL` (cada registro añade solo los datos nuevos). Constantes del código: `MAX_ROWS = 10000`, `MAX_ARTIFACT_ROWS = 200000` | Por muestra y paso | Vivo |
| Gráficas personalizadas `wandb.plot` | `line`, `scatter`, `bar`, `histogram`, `line_series`, `pr_curve`, `roc_curve`, `confusion_matrix` | Por época o final | Vivo |
| `wandb.sklearn` (código del SDK) | 15 funciones: `plot_classifier`, `plot_clusterer`, `plot_regressor`, `plot_summary_metrics`, `plot_learning_curve`, `plot_feature_importances`, `plot_class_proportions`, `plot_calibration_curve` ("traza cuán bien calibradas están las probabilidades de un clasificador"), `plot_roc`, `plot_precision_recall`, `plot_confusion_matrix`, `plot_elbow_curve`, `plot_silhouette`, `plot_residuals`, `plot_outlier_candidates` | Por run o final | Posterior |
| Sweeps | Búsqueda bayesiana, en rejilla y aleatoria (doc). Hyperband con parada temprana: lo afirma la página comercial de Sweeps, no la doc técnica | Por run | Vivo |
| Coordenadas paralelas | Cada línea es un run; los ejes son hiperparámetros de la config y métricas registradas; se ajustan tooltips, escala log, gradiente de color, orden de ejes | Por run | Posterior |
| Importancia de parámetros | Ordena los hiperparámetros por cuánto predicen la métrica elegida; según la doc usa un modelo basado en árboles (no lineal) y muestra también la correlación | Por run | Posterior |
| Alertas | `run.alert(title, text, level, wait_duration)` a Slack o correo; interruptores para "run terminado" y "run caído" (sale sin `run.finish()`); *Automations* para umbrales de métricas | Por evento | Vivo |

**Métricas de sistema (nombres exactos de la doc).** Muestreo: *"wandb automatically logs system
metrics every 15 seconds"* en la página de referencia actual.

- CPU y proceso: `cpu` (CPU del proceso), `proc.cpu.threads`.
- Memoria: `proc.memory.rssMB`, `proc.memory.percent`, `memory_percent`, `proc.memory.availableMB`.
- Disco (ruta por defecto `/`): `disk.{path}.usagePercent`, `disk.{path}.usageGB`, `disk.in`, `disk.out`.
- Red: `network.sent`, `network.recv`.
- GPU NVIDIA (por índice; hay variantes por proceso `gpu.process.{i}.*`): `gpu.{i}.gpu` (utilización),
  `gpu.{i}.memory` (utilización de memoria), `gpu.{i}.memoryAllocated`,
  `gpu.{i}.memoryAllocatedBytes`, `gpu.{i}.temp`, `gpu.{i}.powerWatts`, `gpu.{i}.powerPercent`,
  `gpu.{i}.smClock`, `gpu.{i}.memoryClock`, `gpu.{i}.graphicsClock`,
  `gpu.{i}.correctedMemoryErrors`, `gpu.{i}.unCorrectedMemoryErrors`,
  `gpu.{i}.encoderUtilization`.
- Otros aceleradores: AMD (`gpu`, `memoryAllocated`, `temp`, `powerWatts`, `powerPercent`), Apple
  ARM, Graphcore IPU, Google TPU (`tensorcoreUtilization`, `dutyCycle`, `hbmCapacity*`, latencias
  de transferencia y colectivas), AWS Trainium (`neuroncore_utilization` y desglose de memoria) y
  OpenMetrics (endpoints Prometheus externos filtrados por expresión regular).

**Discrepancia de la propia documentación.** El artículo de soporte sobre el intervalo dice que el
ajuste es `_stats_sampling_interval` con valor por defecto `10.0` (segundos), pero su ejemplo usa
`x_stats_sampling_interval`. Una página antigua, citada en un resultado de búsqueda que no leí
entera, decía "cada 2 s, promediado en 30 s" *(no verificado)*. **Conclusión:** el intervalo por
defecto depende de la versión del cliente; hay que fijarlo explícitamente si importa.

#### 1.1.1 Lo que no consta en la documentación oficial leída

- Que el panel se actualice en vivo mientras corre el run: lo afirman cursos y tutoriales
  (resultados de búsqueda), no una página oficial que haya leído.
- Un hilo de la comunidad de W&B cuenta que una tabla no se puede actualizar fila a fila durante un
  entrenamiento largo y que la respuesta fue volver a registrarla en cada paso *(fuente no
  oficial, no verificado)*. Las constantes de `table.py` y el modo `INCREMENTAL` sugieren que hoy
  se puede ir añadiendo, pero no he podido comprobar cómo se muestra en la interfaz.
- Suavizado, agrupación y bandas de las curvas: existen, pero no las he leído en la doc.

### 1.2 TensorBoard (y su Profiler)

Fuentes: [README](https://github.com/tensorflow/tensorboard/blob/master/README.md) ·
[primeros pasos](https://www.tensorflow.org/tensorboard/get_started) ·
[`torch.utils.tensorboard`](https://docs.pytorch.org/docs/2.14/tensorboard.html) ·
[imágenes y matriz de confusión](https://www.tensorflow.org/tensorboard/image_summaries) ·
[escalares con Keras](https://www.tensorflow.org/tensorboard/scalars_and_keras) ·
[plugin PR curves](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/README.md) ·
[Embedding Projector](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin) ·
[HParams](https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams) ·
[guía del Profiler](https://www.tensorflow.org/guide/profiler) ·
[perfilado con Keras](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras).

| Vista | Qué se ve | Granularidad | Momento |
|---|---|---|---|
| Scalars / Time Series | Pérdida y métricas por época; velocidad de entrenamiento, tasa de aprendizaje y otros escalares. Selector de runs (comparar). Parámetro de **suavizado** (se baja a 0 para ver el valor crudo). Registro por *batch* posible con escalares propios dentro de `train_step`, y acumulado para una curva más suave | Batch o época según lo que se escriba | Vivo si se activa "Reload data" |
| Images | Último PNG de cada etiqueta (filas = etiquetas, columnas = runs); un **deslizador** recorre pasos anteriores (la guía lo usa para ver la matriz de confusión época a época). Brillo, contraste y tamaño real | Por imagen y paso | Vivo |
| Histograms / Distributions | Distribución de un tensor a lo largo del tiempo (pesos, sesgos); *Distributions* resume con líneas de percentiles (mínimo, mediana, máximo) | Por tensor y paso | Vivo |
| Graphs | Grafo del modelo | Por run | Posterior |
| Projector | Embeddings de alta dimensión desde un *checkpoint* con metadatos (`metadata.tsv`), búsqueda de términos y vecinos próximos. La doc leída no cita PCA/t-SNE/UMAP ni *sprites* *(no verificado en esa página)* | Por muestra | Posterior |
| PR Curves | Una curva por problema binario (multiclase = una por clase); selector de paso; el tooltip da **umbral, precisión, exhaustividad y conteos TP/FP/TN/FN** en el punto. Con operación en streaming, acumula conteos entre *batches*. `num_thresholds` sin valor por defecto documentado | Por clase y paso | Vivo |
| HParams | *Table View*, *Parallel Coordinates View* y *Scatter Plot View* (hiperparámetros frente a métricas; la doc lo llama "Scatter Plot View", no "matrix") | Por run | Posterior |
| Text, Audio, Mesh | Texto con Markdown, audio reproducible, mallas 3D (`add_mesh`) | Por paso | Vivo |
| Escritura (`SummaryWriter`) | `add_scalar(s)`, `add_histogram`, `add_image(s)`, `add_figure`, `add_video`, `add_audio`, `add_text`, `add_graph`, `add_embedding`, `add_pr_curve`, `add_custom_scalars`, `add_mesh`, `add_hparams`. `flush_secs` por defecto 120 s | — | — |

**Vivo.** Desde la 2.3.0 TensorBoard ya no recarga solo cada 30 s: hay que activar "Reload data" en
los ajustes (README). `--reload_multifile=true` (experimental) sondea todos los ficheros activos.

**Sistema.** TensorBoard **no monitoriza el sistema de forma continua**; solo el Profiler, bajo
demanda *(interpretación: no consta ningún monitor continuo en las páginas leídas)*.

**TensorFlow Profiler (pestaña Profile).** Herramientas descritas en la guía:

| Herramienta | Qué muestra |
|---|---|
| Overview page | Resumen de alto nivel con recomendaciones; *Performance summary* (desglose del tiempo de paso), *Step-time graph* (el área roja es el tiempo que los dispositivos esperan datos) y las 10 operaciones TensorFlow principales en el dispositivo |
| Input pipeline analyzer | Si la aplicación está limitada por la entrada (*input bound*) y dónde falla el `tf.data`; secciones *Summary*, *Device-side* y *Host-side* (lectura bajo demanda o anticipada, preprocesado, encolado hacia el dispositivo) |
| TensorFlow Stats | Rendimiento de cada operación en host y dispositivo, con gráficos de tarta y tabla exportable a CSV |
| Trace viewer | Línea de tiempo de operaciones por dispositivo (pistas *Step*, *TensorFlow Ops*, *XLA Ops*) y por hilos del host. Límite de 1 millón de eventos por dispositivo |
| GPU Kernel Stats | Estadísticas de cada *kernel* CUDA y la operación que lo originó: nombre, registros y memoria compartida, dimensiones de bloque y rejilla, si es elegible o usa Tensor Cores |
| Memory profile | Uso de memoria del dispositivo: pico de *heap*, fragmentación definida como `1 - mayor bloque libre / memoria libre total`, línea temporal (pila, *heap*, libre) y tabla de asignaciones activas en el pico |
| Pod viewer | Desglose de un paso de entrenamiento en todos los trabajadores |
| tf.data Bottleneck Analysis | Experimental; marca la transformación más lenta del camino crítico |

**Captura.** Modo programático (`profile_batch='10,15'` en el *callback* de Keras, o
`tf.profiler.experimental.start/stop`; la guía recomienda no perfilar más de 10 pasos a la vez) y
modo de muestreo bajo demanda (servidor gRPC y botón *Capture Profile*).

**Perfilado de PyTorch.** `torch.profiler` ofrece `ProfilerActivity` (CPU/CUDA/XPU…), `schedule`
(`wait`, `warmup`, `active`, `repeat`), `profile_memory`, `record_shapes`, `with_flops`
(operaciones matmul y conv2d), `export_chrome_trace` y `tensorboard_trace_handler`
([API](https://docs.pytorch.org/docs/2.14/profiler.html)). El plugin de TensorBoard para PyTorch
mostraba *Overview* (resumen de GPU, desglose del tiempo de paso donde se vio el coste del
DataLoader), *Operator*, *Kernel* (incluye ocupación estimada), *Trace*, *Memory*, *Distributed*
(DDP con NCCL o Gloo: solapamiento de cómputo y comunicación) y *Module*; pero **el tutorial oficial
declara ahora que "la integración de TensorBoard con el profiler de PyTorch está obsoleta"** y
recomienda Perfetto o `chrome://tracing`
([fuente](https://docs.pytorch.org/tutorials/_sources/intermediate/tensorboard_profiler_tutorial.rst.txt)).
La referencia de la API sigue describiendo `tensorboard_trace_handler` sin avisar de ello.

### 1.3 Comet ML

Fuentes: [página de un experimento](https://www.comet.com/docs/v2/guides/comet-ui/experiment-management/single-experiment-page/) ·
[matriz de confusión](https://www.comet.com/docs/v2/guides/experiment-management/log-data/confusion-matrix/) ·
[paneles](https://www.comet.com/docs/v2/guides/comet-ui/experiment-management/visualizations/overview/) ·
[API de `Experiment`](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/) ·
[gráfico de coordenadas paralelas](https://www.comet.com/docs/v2/guides/comet-ui/experiment-management/visualizations/parallel-coordinate-chart/).

| Vista | Qué se ve | Granularidad | Momento |
|---|---|---|---|
| Panels | Paneles integrados: *Bar Chart*, *Data Panel*, *Image Panel*, *Line Chart*, *Logged Curve*, *Parallel Coordinate Chart*, *Scalar Chart*, *Scatter Chart*, *Video Panel*, *3D Panel*, más paneles Python propios; 30+ paneles destacados de la comunidad y casi 300 públicos | Por run | Vivo |
| Metrics | Última, mínima y máxima de cada métrica con su paso y hora | Por paso/época | Vivo |
| Graphics | Imágenes, activos y vídeos; se pueden buscar, ordenar, agrupar, recorrer por paso y comparar entre experimentos | Por muestra y paso | Vivo |
| **Confusion Matrix** | Con `log_confusion_matrix(y_true, y_predicted, labels, images=…, max_examples_per_cell=25, title=…)`. La doc dice que se pueden asociar **imágenes, texto o URL a cada celda**; en la interfaz se cambia entre conteos y porcentajes, se inspeccionan los ejemplos de la celda y se navega entre matrices. Desde sept. 2025 admite anotaciones (p. ej. cajas) sobre las imágenes de ejemplo | Por época o final | Posterior (se registra cuando se llama) |
| Histograms | `log_histogram_3d` (histogramas en el tiempo; requieren paso); la integración con PyTorch los usa para pesos y gradientes | Por tensor y paso | Vivo |
| Embedding Projector | `log_embedding` (marcado como experimental en la doc) | Por muestra | Posterior |
| Otros registros | `log_curve` (series temporales), `log_table`, `log_text`, `log_html`, `log_audio`, `log_video`, `log_model`, `log_code`, `log_dataset_info`, `log_system_info`, `send_notification` | — | — |
| Parallel Coordinate Chart | Se elige una variable objetivo (p. ej. pérdida) y se ve cómo la influyen los parámetros u otras métricas entre experimentos; líneas coloreadas por el valor objetivo; interactivo desde oct. 2025 | Por run | Posterior |

**Sistema.** Pestaña *System Metrics*: GPU por dispositivo (utilización de cómputo y del bus de
memoria, memoria libre/usada/porcentaje, potencia, temperatura), CPU (`sys.compute.overall` de todos
los núcleos y `sys.compute.utilized` del proceso), memoria (disponible y usada), red (bytes/s de
entrada y salida) y disco (uso total y E/S de lectura y escritura), más pares clave-valor del
entorno (usuario, Python, PID, SO, procesador, IP, nombre de host). Solo registra las GPU visibles
(`CUDA_VISIBLE_DEVICES`). **La documentación no da la frecuencia de muestreo** *(no verificado)*.

**Alertas.** `send_notification` avisa por correo cuando termina un experimento (referencia
antigua, solo experimentos en línea). Las alertas con umbral y Slack que aparecen en la doc son de
otros productos de Comet (Opik y Model Production Monitoring), no del seguimiento de experimentos.

### 1.4 Neptune (†)

Fuentes: [portada de la doc](https://docs.neptune.ai/) · [documentación 2.x, métricas de sistema](https://docs-legacy.neptune.ai/logging/system_metrics/) ·
[anuncio de cierre](https://support.neptune.ai/en/articles/13003335-acquisition-shutdown-news) (la
página no se pudo abrir por un error TLS; el contenido se confirmó con la portada de la doc y con
la búsqueda).

- **Estado:** la portada de la doc dice que "tras su adquisición por OpenAI, los servicios de
  Neptune se interrumpirán de forma permanente" y que la aplicación alojada deja de funcionar el
  **5 de marzo de 2026**, con borrado de los datos. Otras fuentes de prensa dan fechas distintas por
  un día (4 o 5 de marzo): irrelevante, ya pasó.
- **Qué ofrecía (doc 2.x):** hardware en pestaña *Monitoring* (memoria, CPU, GPU solo NVIDIA con su
  memoria y su potencia) y salida de consola, bajo el espacio de nombres `monitoring` de cada run;
  se desactiva con `capture_hardware_metrics=False`. La portada actual cita gráficas con bandas de
  error, tratamiento de NaN e infinitos, informes, paneles y consultas entre proyectos. La doc no
  da el intervalo de muestreo.
- **Uso en este inventario:** referencia histórica; no se puntúa más allá de lo anterior.

### 1.5 Aim

Fuentes: [explorers](https://aimstack.readthedocs.io/en/latest/ui/pages/explorers.html) ·
[gestión de runs](https://aimstack.readthedocs.io/en/latest/ui/pages/run_management.html) ·
[configurar runs](https://aimstack.readthedocs.io/en/latest/using/configure_runs.html) ·
[`configs.py`](https://github.com/aimhubio/aim/blob/main/aim/ext/resource/configs.py) ·
[`stat.py`](https://github.com/aimhubio/aim/blob/main/aim/ext/resource/stat.py) ·
[repositorio](https://github.com/aimhubio/aim).

| Vista | Qué se ve | Granularidad | Momento |
|---|---|---|---|
| Metrics Explorer | "Buscar y comparar miles de métricas". Agrupar por color, trazo o gráfico; **agregación** (media, mediana, mín., máx., con bandas); **suavizado** (media móvil exponencial o centrada, interpolación lineal o cúbica) | Paso/época | Vivo |
| Images Explorer | Consultar y comparar imágenes intermedias; agrupa por parámetros del run, paso e índice; tamaño, orden y tooltips; sin agregación ni suavizado | Muestra y paso | Vivo |
| Params Explorer | Coordenadas paralelas de alta dimensión; agrupar por color, trazo o gráfico | Run | Posterior |
| Scatters Explorer | Correlación entre el último valor de una métrica y un hiperparámetro; tendencia lineal o LOESS | Run | Posterior |
| Página de un run | *Overview* (parámetros, métricas, sistema, CLI, variables de entorno, paquetes, Git), *Params*, *Metrics*, *System*, **Distributions** (distribuciones de gradientes, pesos y sesgos por capa a lo largo de los pasos), *Images*, *Audios*, *Texts*, *Figures* (Plotly y matplotlib), *Terminal Logs* (casi en tiempo real) y *Settings* | Mixta | Vivo |
| Runs Explorer | Búsqueda con Aim QL, **Live Update**, archivar/borrar en lote, exportar CSV | Run | Vivo |
| Alertas | El README habla de "alertas en tiempo real sobre el progreso del entrenamiento" y la hoja de ruta lista "notificar runs atascados" (agosto 2022). No he leído la doc de uso | Evento | Vivo |

**Sistema.** Código fuente: `DEFAULT_SYSTEM_TRACKING_INT = 10` (segundos, ajustable con
`system_tracking_interval`; `None` lo apaga). Métricas: `cpu`, `memory_percent`,
`p_memory_percent` (memoria del proceso), `disk_percent` (raíz) y, por GPU cuando NVML lo permite,
`gpu` (utilización), `gpu_memory_percent`, `gpu_temp`, `gpu_power_watts`. No hay red ni relojes.

**Mantenimiento.** La última versión en PyPI es de mayo de 2025 (según
[la revisión previa](observabilidad_entrenamiento.md)); la licencia es Apache 2.0. La hoja de ruta
del README da Figures y Audios Explorer como entregados, y Distributions y Text Explorer como
pendientes o en curso (la hoja de ruta es de 2021-2023): no he comprobado qué es hoy un *explorer*
y qué una pestaña de run.

### 1.6 ClearML

Fuentes: [pestañas de un experimento](http://clear.ml/docs/latest/docs/webapp/webapp_exp_track_visual/) ·
[comparar experimentos](https://clear.ml/docs/latest/docs/webapp/webapp_exp_comparing/) ·
[SDK de `Task`](https://clear.ml/docs/latest/docs/clearml_sdk/task_sdk/) ·
[`resource_monitor.py`](https://github.com/clearml/clearml/blob/master/clearml/utilities/resource_monitor.py) ·
[alertas en Slack](https://clear.ml/docs/latest/docs/guides/services/slack_alerts/).

| Vista | Qué se ve | Granularidad | Momento |
|---|---|---|---|
| Scalars | Una gráfica por métrica con todas sus variantes; tabla *Summary* de valores únicos y tabla con primero/último/mín./máx./media de cada serie. **Suavizado** (media móvil exponencial, gaussiana o móvil) con "mostrar originales". **Eje X:** iteraciones, tiempo desde el inicio o tiempo de reloj | Iteración | Vivo (la doc no da el intervalo de refresco) |
| Plots | Gráficas no temporales: 2D/3D, tablas Pandas/CSV y Plotly; la última por métrica y un deslizador de iteración | Iteración | Vivo |
| Debug Samples | Imágenes, audio y vídeo; filtro por métrica; la iteración más reciente primero | Muestra e iteración | Vivo |
| Console | Todo lo impreso por stdout/stderr, descargable | Línea | Vivo |
| Hyperparameters | Argumentos de línea de órdenes (argparse, click, hydra), variables de entorno, grupos de `Task.connect()` y definiciones de TensorFlow | Run | — |
| Artifacts, Info | Artefactos y modelos con ruta, tamaño y hash; propiedades de tiempo de ejecución (SO, CUDA, núcleos, número y tipo de GPU, memoria) | Run | — |
| Comparar | Dos o más tareas: campos lado a lado con diferencias resaltadas; escalares en tabla (último/mín./máx.) o curvas superpuestas; gráficas comparadas; **Parallel Coordinates** y **Scatter Plot** de hiperparámetros frente a métricas; *debug samples* de distintas iteraciones sincronizados | Run | Posterior |

**Sistema (`ResourceMonitor`, valores por defecto del constructor).** `sample_frequency_per_sec=2.0`
y `report_frequency_sec=30.0`: se muestrea 2 veces por segundo y se informa cada 30 s *(los
valores pueden cambiarse por configuración; no he comprobado la ruta de configuración)*. Métricas
de máquina (título `:monitor:machine` en el código; la doc habla de `:resource monitor:`):
`cpu_usage`, `memory_used_gb`, `memory_free_gb`, `disk_free_percent`, `cpu_temperature` (solo con
sensor `coretemp`), `network_tx_mbs`, `network_rx_mbs`, `io_read_mbs`, `io_write_mbs`. Métricas de GPU
(`:monitor:gpu`): `gpu_N_temperature`, `gpu_N_utilization`, `gpu_N_mem_used_gb`, `gpu_N_mem_usage`,
`gpu_N_mem_free_gb`, `gpu_N_power_draw`. `Task.init` incluye `auto_resource_monitoring=True`.

**Alertas.** Servicio de monitorización de Slack (`clearml.automation.monitor`): avisa de tareas
fallidas (y, opcionalmente, completadas) con la salida reciente de consola; solo para servidores
propios y con un agente en la cola `services`. No hay umbrales de métricas en lo leído.

**Licencia.** SDK Apache 2.0; servidor SSPL (ver la revisión previa).

### 1.7 Voxel51 FiftyOne

Fuentes: [evaluación](https://docs.voxel51.com/user_guide/evaluation/index.html) ·
[Brain](https://docs.voxel51.com/brain/index.html) ·
[panel de evaluación](https://docs.voxel51.com/getting_started/model_evaluation/02_advanced_analysis.html) ·
[repositorio](https://github.com/voxel51/fiftyone) (Apache 2.0).

FiftyOne trabaja **a posteriori y por muestra** sobre un dataset con etiquetas y predicciones
guardadas; no sigue un entrenamiento en curso *(interpretación)*.

| Vista / método | Qué da | Granularidad |
|---|---|---|
| Evaluación | Regresión, clasificación, detección, polígonos y segmentación (instancia y semántica) en imagen y vídeo. Informes de clasificación, matrices de confusión y curvas PR; tablas de precisión, exhaustividad y F1 por clase con medias micro, macro y ponderada. **No cita ROC ni calibración** | Dataset |
| Resultados por muestra | Con un `eval_key` guarda en cada muestra conteos de FP/FN y exactitud (ejemplo de detección: `eval_fp`, `eval == "fp"`). `use_subset()` calcula métricas por subconjuntos (etiqueta, atributo, vista guardada) | Muestra |
| Matriz de confusión interactiva | Un clic en una celda selecciona los objetos (verdad o predicción) correspondientes; se actualiza con la vista actual | Muestra |
| Panel de evaluación (App) | Todas las evaluaciones de un dataset, estadísticas de resumen, métricas por clase con filtro, matrices de confusión, histogramas (confianza, IoU), cambio entre evaluaciones para compararlas y filtros por FP/TP/FN y confianza | Muestra |
| `compute_visualization` | Proyección 2D de embeddings (UMAP por defecto, t-SNE, PCA o manual) con selección por lazo en la App | Muestra |
| `compute_similarity` | Índice de similitud para buscar ejemplos parecidos | Muestra |
| `compute_uniqueness`, `compute_representativeness` | Cuán único es cada ejemplo (sin etiquetas) y cuáles son los más comunes | Muestra |
| **`compute_hardness`** | "Muestras para las que el modelo no está seguro de qué etiqueta asignar"; necesita predicciones con **logits**; escribe un escalar `hardness` por muestra. La fórmula no está documentada ("incorpora la confianza y los logits en un modelo ajustado") | Muestra |
| **`compute_mistakenness`** | Posibles errores de anotación: ranking de las muestras en las que el modelo está seguro pero discrepa de la etiqueta humana; necesita etiquetas y predicciones y, para máxima eficacia, logits; escribe `mistakenness` (y en detección `mistakenness_loc`, `possible_missing`, `possible_spurious`). Usa un "modelo de puntuación propietario": **método no documentado** | Muestra |
| Duplicados y fugas | Duplicados exactos, casi duplicados y *leaky splits* ("posibles fugas entre particiones") | Muestra |

### 1.8 Ultralytics YOLO

Fuentes: [guía de métricas](https://docs.ultralytics.com/guides/yolo-performance-metrics/) ·
[modo train](https://docs.ultralytics.com/modes/train/) ·
[API de `plotting`](https://docs.ultralytics.com/reference/utils/plotting/) ·
[entrenar con datos propios (YOLOv5)](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/).

Artefactos del directorio de resultados, tal como los describe la documentación (**orientados a
detección**; los nombres con prefijo `Box*` cambian a `Mask*` o `Pose*` en segmentación y pose; no he
comprobado el conjunto para clasificación):

| Artefacto | Qué contiene |
|---|---|
| `results.csv` y `results.png` | El CSV se actualiza **después de cada época** y se dibuja como `results.png` **al terminar** (doc de YOLOv5; la API de `plotting` confirma que `plot_results` guarda `results.png` junto al CSV) |
| `confusion_matrix.png` | Conteos por clase |
| `confusion_matrix_normalized.png` | La misma matriz en proporciones, para comparar clases |
| `BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png` | Precisión, exhaustividad y F1 **frente al umbral de confianza** |
| `BoxPR_curve.png` | Compromiso precisión-exhaustividad, útil con clases desequilibradas |
| `val_batchX_labels.jpg` y `val_batchX_pred.jpg` | Verdad y predicción de los mismos *batches* de validación, para comparar a ojo |
| `labels.jpg` | Histogramas de clases y estadísticas de cajas del conjunto de entrenamiento |
| `tune_scatter_plots.png`, `tune_fitness.png` | Evolución de la búsqueda de hiperparámetros |
| `images.jpg` | Cuadrícula de imágenes con etiquetas (nombre por defecto de `plot_images`) |

Las métricas son P, R, IoU, AP, mAP50, mAP50-95 y F1. Se pueden enviar a **Comet, ClearML,
TensorBoard, MLflow, W&B y DVCLive** (`yolo settings tensorboard=True`). No he podido confirmar el
nombre exacto `train_batch*.jpg`: la doc que leí no lo cita *(no verificado)*.

### 1.9 PyTorch Lightning

Fuentes: [`LearningRateMonitor`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.LearningRateMonitor.html) ·
[`DeviceStatsMonitor`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.DeviceStatsMonitor.html) ·
[`ModelSummary`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelSummary.html) ·
[`ThroughputMonitor`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ThroughputMonitor.html) ·
[profilers](https://lightning.ai/docs/pytorch/stable/tuning/profiler_basic.html) ·
[depurar](https://lightning.ai/docs/pytorch/latest/debug/debugging_intermediate.html).

| Componente | Qué registra | Granularidad |
|---|---|---|
| `LearningRateMonitor` | `lr` siempre; opcionalmente momento (`log_momentum`) y decaimiento de pesos (`log_weight_decay`), ambos desactivados por defecto. `logging_interval`: `'step'`, `'epoch'` o `None` (cada planificador a su ritmo). `log_key_prefix` | Paso o época |
| `DeviceStatsMonitor` | CPU vía `psutil`: `cpu_percent`, `cpu_vm_percent`, `cpu_swap_percent`. GPU CUDA: **estadísticas del asignador de PyTorch en bytes** (asignada, reservada, activa, por *pool*, eventos como OOM). **"La utilización de cómputo de la GPU no se registra por defecto."** TPU: HBM libre/usada/porcentaje. Claves con prefijo `DeviceStatsMonitor.{hook}/…`; `cpu_stats` (`None`: solo si el acelerador es CPU), `filter_keys` | En los *hooks* de inicio y fin de *batch* de entrenamiento, validación y test |
| `ModelSummary` | Resumen de todas las capas de un `LightningModule`; `max_depth` por defecto 1, `0` lo apaga. La página no detalla las columnas *(no verificado)* | Run |
| `ThroughputMonitor` | Calcula y registra el rendimiento con `Throughput` (requiere `batch_size_fn`, opcionalmente `length_fn`); los FLOPs dependen de un atributo `flops_per_batch` del módulo (se puede medir con `measure_flops()`). Supone un tamaño de *batch* constante. **Los nombres exactos de las métricas no están en la página leída** | Paso |
| Normas de gradiente | `from lightning.pytorch.utilities import grad_norm` dentro de `on_before_optimizer_step`, registrando con `self.log_dict`: la norma 2 de cada capa; si crece, es indicio de gradientes que explotan. Los gradientes ya están sin escalar con precisión mixta, y el *hook* se llama antes del recorte | Paso |
| Profilers | `SimpleProfiler` (`"simple"`: duración media y total de las acciones clave de callbacks, DataModules y LightningModule); `AdvancedProfiler` (`"advanced"`, sobre cProfile: llamadas, tiempo acumulado y por llamada de cada función); `PyTorchProfiler` (para personalizar el perfilado; la página no detalla sus salidas) | Run |

### 1.10 Captum y Captum Insights

Fuentes: [notas de la versión](https://github.com/pytorch/captum/releases) ·
[doc de Insights en la v0.8.0](https://raw.githubusercontent.com/pytorch/captum/v0.8.0/docs/captum_insights.md) ·
[artículo](https://arxiv.org/abs/2009.07896).

- **Captum Insights** era "un *widget* de visualización de interpretabilidad construido sobre
  Captum", que mostraba atribución de características en imágenes, texto y otros tipos de entrada.
  **Fue retirado en la v0.9.0** ("Captum Insights se ha retirado en la v0.9.0"; el *frontend*, el
  *backend* y los extras de empaquetado se eliminaron); quien lo necesite debe fijar la etiqueta
  `v0.8.0`.
- **Captum** (biblioteca) sigue vivo: métodos de atribución como `IntegratedGradients`, `Saliency`,
  `DeepLift`, `NoiseTunnel`, `Occlusion`, `GradCAM` (detalle y costes en la revisión previa).
- **Granularidad:** por muestra y a posteriori; no hay vista en vivo.

### 1.11 Cleanlab

Fuentes: [`filter`](https://docs.cleanlab.ai/master/cleanlab/filter.html) ·
[repositorio](https://github.com/cleanlab/cleanlab) (Apache 2.0) ·
Northcutt, Jiang y Chuang, "Confident Learning: Estimating Uncertainty in Dataset Labels", *JAIR*,
2021, [arXiv:1911.00068](https://arxiv.org/abs/1911.00068).

- `find_label_issues(labels, pred_probs, filter_by, frac_noise, multi_label, return_indices_ranked_by)`
  devuelve una máscara booleana o los índices **ordenados por gravedad** según una puntuación de
  calidad de etiqueta por muestra.
- **Condición clave de la doc:** "los problemas de etiquetado devueltos son más precisos cuando se
  calculan con `pred_probs` **fuera de muestra**"; para obtenerlas **se puede usar validación
  cruzada**.
- Multi-etiqueta: el módulo `cleanlab.multilabel_classification` (etiquetas como lista de listas),
  con `find_multilabel_issues_per_class` para señalar qué clase concreta es la errónea.
- La interfaz `Datalab` (README) detecta también duplicados y valores atípicos y genera un informe
  (`lab.report()`).
- **Granularidad:** por muestra; a posteriori.

### 1.12 Deepchecks y Evidently

Fuentes: Deepchecks, [checks de evaluación de visión](https://docs.deepchecks.com/0.17/api/generated/deepchecks.vision.checks.model_evaluation.html)
y [repositorio](https://github.com/deepchecks/deepchecks) (licencia **AGPL 3.0**); Evidently,
[visión general](https://docs.evidentlyai.com/docs/library/overview),
[métricas de clasificación](https://docs.evidentlyai.com/metrics/explainer_classification) y
[repositorio](https://github.com/evidentlyai/evidently) (Apache 2.0).

**Deepchecks (visión, doc 0.17):** `ClassPerformance` (resume métricas por clase),
`ConfusionMatrixReport`, `SingleDatasetPerformance`, `SimpleModelComparison` (compara con un modelo
simple), `WeakSegmentsPerformance` ("busca segmentos con bajo rendimiento"), `PredictionDrift`
(`TrainTestPredictionDrift` está obsoleto), `MeanAveragePrecisionReport` y
`MeanAverageRecallReport`. No he podido abrir la guía de inicio de clasificación (404); las
condiciones pasa/falla de las *suites* y la exportación de informes **no están verificadas**.

**Evidently:** *Reports* que calculan y visualizan métricas (exportables a JSON, diccionario o
HTML) y *Test Suites* (un Report con condiciones: pasa, falla, aviso o error). `ClassificationPreset`
y, en las métricas de clasificación: exactitud, precisión, exhaustividad, F1, ROC AUC y LogLoss;
**representación de clases**; **matriz de confusión**; **calidad por clase**; **separación de
clases** (dispersión de probabilidades predichas, correctas e incorrectas, por clase: "refleja la
exactitud y la calidad de la calibración" y sirve para elegir un umbral por clase);
**distribución de probabilidades**; **curva ROC**; **curva PR**; **tabla PR** (por umbrales, de 5 en
5 % de predicciones) y **calidad por característica** (para ver segmentos de bajo rendimiento). No
hay un diagrama de fiabilidad propio. Panel de monitorización autoalojable que guarda varias
evaluaciones en el tiempo; las **alertas** son de Evidently Cloud.

**Granularidad:** ambos operan sobre un dataset de evaluación completo, a posteriori *(sin vista
de un entrenamiento en curso; interpretación)*.

### 1.13 Dataset Cartography

Swayamdipta, Schwartz, Lourie, Wang, Hajishirzi, Smith y Choi, "Dataset Cartography: Mapping and
Diagnosing Datasets with Training Dynamics", *EMNLP 2020*, pp. 9275-9293, DOI
[10.18653/v1/2020.emnlp-main.746](https://doi.org/10.18653/v1/2020.emnlp-main.746)
([arXiv:2009.10795](https://arxiv.org/abs/2009.10795),
[ACL Anthology](https://aclanthology.org/2020.emnlp-main.746/)).

Según los autores, dos medidas obtenidas **en una sola ejecución** (la **confianza** del modelo en la
clase verdadera y su **variabilidad** entre épocas) separan tres regiones: **ambiguas** (las que más
contribuyen a la generalización fuera de distribución), **fáciles de aprender** (las más numerosas;
importantes para la optimización) y **difíciles de aprender** (que a menudo corresponden a errores de
etiquetado). Granularidad: por muestra y época; análisis posterior sobre el registro de
probabilidades. Es una técnica de PLN; la revisión previa no halló aplicaciones directas a imagen
médica *(no verificado como ausencia)*.

---

## 2. Tablas de síntesis

### 2.1 Métricas de sistema y frecuencia de muestreo por defecto

| Herramienta | GPU | CPU / RAM / disco / red | Muestreo por defecto | Fuente |
|---|---|---|---|---|
| W&B | Utilización, memoria (% y bytes), temperatura, potencia (W y %), relojes SM/memoria/gráficos, errores ECC corregidos y no corregidos, codificador | CPU y memoria del proceso, hilos, disco (uso y E/S), red enviada y recibida | 15 s (referencia) o 10,0 s (`_stats_sampling_interval`): inconsistente | doc oficial |
| Comet | Utilización (cómputo y bus de memoria), memoria libre/usada/%, potencia, temperatura | CPU global y del proceso, memoria, red (bytes/s), disco (uso y E/S) | No documentada | doc oficial |
| Neptune† | Memoria y potencia (NVIDIA) | Memoria, CPU | No documentada | doc 2.x |
| Aim | `gpu`, `gpu_memory_percent`, `gpu_temp`, `gpu_power_watts` | `cpu`, `memory_percent`, `p_memory_percent`, `disk_percent` | 10 s | código fuente |
| ClearML | Temperatura, utilización, memoria (GB y %), memoria libre, potencia | CPU, memoria usada/libre, disco libre, temperatura de CPU, red (MB/s), E/S de disco (MB/s) | Muestreo 2 por segundo, informe cada 30 s | código fuente |
| Lightning | Memoria del asignador CUDA (no la utilización) | CPU, RAM y swap vía `psutil` | En cada *hook* de *batch* (sin reloj) | doc oficial |
| TensorBoard | Solo con el Profiler bajo demanda | — | Captura puntual (≤ 10 pasos recomendados) | guía del Profiler |
| FiftyOne, Ultralytics, Captum, Cleanlab, Deepchecks, Evidently, Cartography | No monitorizan sistema | — | — | — |

*(Interpretación)* Entre todas, solo W&B documenta relojes y errores ECC. Ninguna presenta la
potencia como una serie pensada para integrar energía: W&B y Aim muestrean cada 10-15 s, y ClearML
muestrea a 2 Hz pero informa cada 30 s (no he verificado qué se almacena de esas muestras). El
proyecto (`system_interval_s: 5`, con `sm_clock_mhz`, potencia y temperatura) ya cubre lo que ofrecen
las demás; para energía conviene medir también el coste de muestrear más rápido.

### 2.2 Matriz de funcionalidades

Leyenda: `Sí` verificado · `P` parcial · `No` no consta / fuera de propósito · `n.v.` no verificado ·
† descontinuada. Columnas: **W&B** · **TB** (TensorBoard) · **Comet** · **Nept.**† · **Aim** ·
**CML** (ClearML) · **F51** (FiftyOne) · **Ultr.** (Ultralytics) · **Lightn.** (Lightning) · **Capt.**
(Captum/Insights) · **Clean.** (Cleanlab) · **Deepch.** · **Evid.** (Evidently) · **Cart.** (Cartography).

| Funcionalidad | W&B | TB | Comet | Nept.† | Aim | CML | F51 | Ultr. | Lightn. | Capt. | Clean. | Deepch. | Evid. | Cart. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Actualización en vivo mientras corre el run** | P | P | Sí | P | Sí | P | No | P | P | No | No | No | P | No |
| **Métricas escalares** | | | | | | | | | | | | | | |
| Escalares por paso/época y curvas superpuestas | Sí | Sí | Sí | Sí | Sí | Sí | No | Sí | P | No | No | No | P | No |
| Suavizado, bandas y eje X alternativo | n.v. | P | n.v. | P | Sí | Sí | No | No | No | No | No | No | No | No |
| **Sistema / hardware** | | | | | | | | | | | | | | |
| GPU: utilización, memoria, potencia, temperatura | Sí | No | Sí | Sí | Sí | Sí | No | No | P | No | No | No | No | No |
| Relojes y errores ECC de GPU | Sí | No | No | No | No | No | No | No | No | No | No | No | No | No |
| CPU, RAM, disco y red | Sí | No | Sí | P | P | Sí | No | No | P | No | No | No | No | No |
| Intervalo de muestreo configurable | Sí | No | n.v. | n.v. | Sí | Sí | No | No | P | No | No | No | No | No |
| **Rendimiento, throughput, DataLoader** | | | | | | | | | | | | | | |
| Perfil de operadores y *kernels*, traza temporal | No | P | No | No | No | No | No | No | Sí | No | No | No | No | No |
| Espera de la tubería de datos / DataLoader | No | P | No | No | No | No | No | No | P | No | No | No | No | No |
| Throughput (muestras/s) y FLOPs | No | No | No | No | No | No | No | No | Sí | No | No | No | No | No |
| Perfil de memoria (pico, línea temporal) | No | P | No | No | No | No | No | No | P | No | No | No | No | No |
| Resumen de arquitectura (capas, parámetros) | P | P | P | n.v. | No | n.v. | No | No | Sí | No | No | No | No | No |
| **Gradientes y pesos** | | | | | | | | | | | | | | |
| Histogramas de pesos/gradientes por capa | Sí | Sí | Sí | n.v. | Sí | n.v. | No | No | P | No | No | No | No | No |
| Norma del gradiente por capa | n.v. | No | n.v. | No | No | No | No | No | Sí | No | No | No | No | No |
| LR, momento, decaimiento de pesos | P | P | P | P | P | P | No | n.v. | Sí | No | No | No | No | No |
| **Predicciones por muestra** | | | | | | | | | | | | | | |
| Galería de imágenes con predicción y deslizador de pasos | Sí | Sí | Sí | n.v. | Sí | Sí | Sí | P | No | P | No | No | No | No |
| Tabla por muestra (probabilidades, filtro, orden) | Sí | No | P | n.v. | No | P | Sí | No | No | No | Sí | No | No | Sí |
| Trayectoria de cada muestra entre épocas | P | No | No | No | No | No | No | No | No | No | No | No | No | Sí |
| **Curvas de evaluación** | | | | | | | | | | | | | | |
| Matriz de confusión | Sí | P | Sí | n.v. | n.v. | P | Sí | Sí | No | No | P | Sí | Sí | No |
| Matriz de confusión con ejemplos por celda | No | No | Sí | No | No | No | Sí | No | No | No | No | No | No | No |
| Curva precisión-exhaustividad | Sí | Sí | P | n.v. | No | P | Sí | Sí | No | No | No | No | Sí | No |
| Curva ROC | Sí | No | P | n.v. | No | P | No | No | No | No | No | No | Sí | No |
| Curvas por clase | P | Sí | n.v. | n.v. | No | n.v. | P | Sí | No | No | No | Sí | Sí | No |
| P/R/F1 frente al umbral (punto de operación) | No | Sí | No | No | No | No | No | Sí | No | No | No | No | Sí | No |
| **Calibración** | | | | | | | | | | | | | | |
| Diagrama de fiabilidad / ECE | P | No | No | No | No | No | No | No | No | No | No | No | P | No |
| Histograma de confianza / distribución de probabilidades | P | P | No | No | No | No | Sí | No | No | No | No | No | Sí | No |
| **Errores y ejemplos difíciles** | | | | | | | | | | | | | | |
| Peores ejemplos / mal clasificados | Sí | No | Sí | No | No | No | Sí | P | No | No | Sí | P | P | Sí |
| Dureza por muestra / posible error de etiqueta | No | No | No | No | No | No | Sí | No | No | No | Sí | No | No | Sí |
| Rendimiento por subgrupos o segmentos | P | No | No | No | No | No | P | No | No | No | No | Sí | P | P |
| **Embeddings** | | | | | | | | | | | | | | |
| Proyección interactiva de embeddings | n.v. | Sí | Sí | No | No | n.v. | Sí | No | No | No | No | No | No | No |
| Similitud, duplicados y fugas entre particiones | No | No | No | No | No | No | Sí | No | No | No | Sí | No | No | No |
| **Explicabilidad** | | | | | | | | | | | | | | |
| Atribución / saliencia por muestra | No | No | No | No | No | No | No | No | No | Sí | No | No | No | No |
| Visor interactivo de atribuciones | No | No | No | No | No | No | No | No | No | P | No | No | No | No |
| **Comparación de runs** | | | | | | | | | | | | | | |
| Comparar runs (curvas, tabla, diferencias de configuración) | Sí | Sí | Sí | Sí | Sí | Sí | P | No | No | No | No | No | P | No |
| Agregación entre semillas (media y banda) | n.v. | No | No | P | Sí | n.v. | No | No | No | No | No | No | No | No |
| **Hiperparámetros** | | | | | | | | | | | | | | |
| Coordenadas paralelas, dispersión, importancia | Sí | Sí | Sí | n.v. | Sí | Sí | No | No | No | No | No | No | No | No |
| Búsqueda de hiperparámetros integrada | Sí | No | n.v. | n.v. | No | n.v. | No | P | No | No | No | No | No | No |
| **Informes y exportación** | | | | | | | | | | | | | | |
| Informes compartibles | n.v. | No | P | Sí | No | n.v. | No | No | No | No | P | n.v. | Sí | No |
| Exportación de datos (CSV, JSON) | n.v. | P | P | n.v. | Sí | n.v. | n.v. | Sí | No | No | No | No | Sí | No |
| **Alertas y pruebas** | | | | | | | | | | | | | | |
| Alertas (caída, umbrales, atasco) | Sí | No | P | n.v. | P | P | No | No | No | No | No | n.v. | P | No |
| Pruebas con condición pasa/falla | No | No | No | No | No | No | No | No | No | No | No | n.v. | Sí | No |

**Notas sobre celdas no triviales** *(las marcas `P` y `n.v.` merecen explicación)*:

- *En vivo.* W&B: lo afirman tutoriales, no la doc. TensorBoard: hay que activar "Reload data" (README).
  Comet: la integración con Ludwig dice que las métricas aparecen en vivo en la pestaña de gráficas.
  Aim: *Live Update* en el explorador de runs. ClearML: un vídeo de la doc lo muestra; la doc no da
  el intervalo. Neptune: la pestaña *Monitoring* era en vivo (doc 2.x). Ultralytics: el CSV se
  actualiza por época, pero `results.png` solo se dibuja al terminar.
- *Escalares, Lightning (P).* `self.log` envía a un logger (TensorBoard, W&B, etc.); no hay panel propio.
- *Fiabilidad, W&B (P).* `plot_calibration_curve` existe en `wandb.sklearn`; por su firma está
  pensado para estimadores de scikit-learn, así que con un modelo de PyTorch habría que montarlo a
  mano *(interpretación; no he leído la firma)*.
- *Curvas, Comet/ClearML (P).* Aceptan curvas propias (`log_curve`, Plotly en *Plots*), pero no
  traen una "curva ROC" integrada que haya podido confirmar.
- *Trayectoria por muestra, W&B (P).* `Table(log_mode="INCREMENTAL")` permite ir añadiendo filas;
  el modo en que la interfaz las compara entre pasos *no está verificado*.
- *Subgrupos, FiftyOne (P).* `use_subset()` calcula métricas por subconjuntos definidos por etiqueta
  o atributo; no es un análisis automático de segmentos débiles (Deepchecks sí lo es).
- *Perfil, TensorBoard (P).* El Profiler de TensorFlow está completo; el plugin de PyTorch está
  oficialmente obsoleto (sección 1.2).
- *Captum Insights (P).* Retirado en la v0.9.0.
- *Cleanlab, informes (P).* `Datalab.report()` imprime un informe; no hay exportación
  verificada.

---

## 3. Imagen médica: qué se considera imprescindible

Esta sección separa lo que **dicen las fuentes** de lo que **se recomienda** *(interpretación)*. No
existe una lista única y normativa de "visualizaciones imprescindibles": lo que hay son guías de
**comunicación de resultados** (STARD, TRIPOD+AI, CLAIM) y trabajos sobre **evaluación** (calibración,
métricas, subgrupos, saliencia). La traducción a funcionalidades de una herramienta es de esta
revisión.

| Necesidad | Qué debería mostrar la herramienta *(interpretación)* | Qué dice la fuente |
|---|---|---|
| **Calibración** | Diagrama de fiabilidad con intervalos de confianza, ECE (y MCE/Brier), pendiente e intercepto de calibración, **por clase** en multi-etiqueta; sobre un conjunto de validación suficientemente grande | Van Calster et al. proponen una jerarquía de calibración (**media**, **débil** —intercepto y pendiente—, **moderada** —curva de calibración flexible— y **fuerte**, que califican de meta utópica), señalan que la calibración se evalúa mucho menos que la discriminación y que los métodos flexibles de aprendizaje automático (no solo la regresión) también pueden dar modelos mal calibrados (parafraseado del resumen); exigen muestras suficientemente grandes para curvas precisas. Guo et al. (ICML 2017) muestran que las redes modernas suelen estar mal calibradas y que el **escalado de temperatura** es "sorprendentemente eficaz" para corregirlo |
| **Sensibilidad y especificidad en un umbral** | Punto de operación marcado sobre ROC y PR, **fijado en validación y aplicado a test**; tabla de sensibilidad, especificidad, VPP y VPN con intervalos; para multi-etiqueta, una por clase | STARD 2015 es "la lista actualizada de elementos esenciales para comunicar estudios de precisión diagnóstica" (título; no he leído la lista). TRIPOD+AI (27 ítems) pide **estimaciones del rendimiento con intervalos de confianza, incluidos los subgrupos clave** (ítem 23a) y que se describa cómo se abordó la equidad (ítem 14); la calibración aparece como ejemplo de medida a especificar (ítem 12e), no como obligación |
| **Curvas por clase y elección de métrica** | ROC y PR por clase; la métrica principal elegida según el problema, no por costumbre | Maier-Hein et al., *Metrics Reloaded* (*Nature Methods* 21, 195-212, 2024): un marco, construido con un proceso Delphi, que guía la elección de métricas mediante una "huella del problema" y avisa de errores frecuentes; cubre clasificación a nivel de imagen. Saito y Rehmsmeier (*PLOS ONE*, 2015, ya citado en la revisión previa): con clases desequilibradas la curva PR es más informativa que la ROC |
| **Análisis de errores y subgrupos** | Peores ejemplos con su confianza, matriz de confusión con los ejemplos de cada celda, rendimiento por subgrupos clínicamente relevantes (centro, sexo, edad, subtipo) con intervalos | Oakden-Rayner et al., "Hidden Stratification Causes Clinically Meaningful Failures in Machine Learning for Medical Imaging" ([arXiv:1909.12475](https://arxiv.org/abs/1909.12475), resumen para ML4H en NeurIPS 2019): un modelo puede rendir bien en conjunto y fallar en subconjuntos importantes poco frecuentes; encuentran **diferencias relativas de rendimiento superiores al 20 %** en subconjuntos clínicamente importantes y concluyen que evaluarlo debe ser parte central de cualquier despliegue clínico |
| **Calidad de etiquetas** | Muestras con alta confianza en contra de la etiqueta; muestras difíciles de aprender | Northcutt et al. (JAIR 2021, *confident learning*); Swayamdipta et al. (EMNLP 2020, regiones difíciles = a menudo errores de etiquetado); FiftyOne `mistakenness` (método no documentado) |
| **Mapas de saliencia** | Con cautela: como **herramienta de depuración**, con la capa, el objetivo y el método documentados, y con pruebas de cordura | Arun et al., *Radiology: Artificial Intelligence* 3(6):e200267, 2021 (DOI [10.1148/ryai.2021200267](https://doi.org/10.1148/ryai.2021200267)): ocho métodos de saliencia probados en radiografías de tórax según cuatro criterios (utilidad de localización, sensibilidad a la aleatorización de pesos, repetibilidad y reproducibilidad); **ninguno** pasó todos y la mayoría rindió peor que la línea base, y las redes de localización (U-Net, RetinaNet) los superaron; recomiendan detección o segmentación si el objetivo es localizar. Adebayo et al., "Sanity Checks for Saliency Maps" ([arXiv:1810.03292](https://arxiv.org/abs/1810.03292); sede de publicación no comprobada): algunos métodos son independientes del modelo y de los datos. Ghassemi, Oakden-Rayner y Beam, "The false hope of current approaches to explainable artificial intelligence in health care", *Lancet Digital Health* 3(11):e745-e750, 2021 (DOI [10.1016/S2589-7500(21)00208-9](https://doi.org/10.1016/S2589-7500(21)00208-9)): según los resúmenes localizados, defienden la validación rigurosa por encima de exigir explicabilidad *(solo leí resúmenes, no el texto completo)* |
| **Comunicación y reproducibilidad** | Informe exportable por run con configuración, entorno, semilla, métricas con intervalos y curvas | CLAIM 2024 (Tejani et al., *Radiology: Artificial Intelligence* 6(4):e240300, DOI [10.1148/ryai.240300](https://doi.org/10.1148/ryai.240300)), actualización por proceso Delphi de la lista de verificación de IA en imagen médica; TRIPOD+AI (Collins et al., *BMJ* 385:e078378, 2024, DOI [10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378)) |
| **Utilidad clínica** (opcional) | Análisis de curvas de decisión (beneficio neto frente al umbral) | Vickers y Elkin, "Decision Curve Analysis: A Novel Method for Evaluating Prediction Models", *Medical Decision Making* 26(6):565-574, 2006, DOI [10.1177/0272989X06295361](https://doi.org/10.1177/0272989X06295361). No he leído el contenido, solo he verificado la ficha |

*(Interpretación)* Lo que de verdad separa a una herramienta "de laboratorio" de una "de
publicación médica" son tres cosas que ninguna de las herramientas generales inventariadas trae
resueltas: **calibración con intervalos y por clase**, **intervalos de confianza en todas las
métricas** y **rendimiento por subgrupo**. Hay que construirlas.

Complementos ya verificados en la revisión previa:
van der Velden et al. (*Medical Image Analysis* 79, 102470, 2022; los mapas de saliencia son la
forma más común de XAI en imagen médica, pero su encuesta llega solo hasta octubre de 2020) y
Barekatain y Glocker (MICCAI 2025, taller; [arXiv:2510.12021](https://arxiv.org/abs/2510.12021), dos
datasets).

---

## 4. Lista priorizada para ParaViT-Lab

**Cómo leerla.** Prioridad 1 = primero. *Coste*: bajo (días de trabajo o menos, solo CPU o
instrumentación ligera), medio (varios días o una pieza nueva), alto (infraestructura o un
entrenamiento extra por cada run). *Invariante* "el terminal hace, la web mira": **Sí** significa
que la GPU solo se usa en el terminal y la web lee ficheros y calcula en CPU (con NumPy). *Estado*,
según `docs/inspeccion.md`, `docs/eventos.md` y `src/web/` (no he leído el código de
entrenamiento): **Cont.** = contemplado en el diseño actual, **Parc.** = parcialmente, **Falta**.

| # | Funcionalidad | Referentes del sector | Coste | Invariante | Estado |
|---|---|---|---|---|---|
| 1 | **Predicciones por muestra con identificador estable**, por época, en un panel fijo de validación (probabilidades, pérdida) y probabilidades de toda la validación | W&B Tables, FiftyOne, Cartography | Bajo | Sí: se guardan en el terminal; la web lee `.npz` | Cont. |
| 2 | **Curvas ROC y PR por clase**, barrido de umbral (sensibilidad, especificidad, F1) y punto de operación fijado en validación | TB PR curves, Ultralytics, Evidently | Bajo | Sí (CPU en la web) | Cont. |
| 3 | **Calibración**: diagrama de fiabilidad, ECE/MCE/Brier; añadir pendiente e intercepto, por clase en multi-etiqueta y con intervalos | Evidently (parcial), Van Calster, Guo | Bajo | Sí | Cont. (faltan intercepto, pendiente e intervalos) |
| 4 | **Intervalos de confianza (bootstrap)** en AUC, sensibilidad, especificidad, ECE y F1, también por subgrupo | TRIPOD+AI ítem 23a | Bajo-medio | Sí (CPU) | Falta |
| 5 | **Errores**: peores ejemplos con confianza y matriz de confusión con los ejemplos de cada celda | Comet, FiftyOne | Bajo | Sí | Cont. (`worst_errors`, `confusion` con índices) |
| 6 | **Rendimiento por subgrupo y por clase** con metadatos (centro, sexo, edad, subtipo) | Deepchecks *weak segments*, Oakden-Rayner | Medio | Sí | Falta (depende del dataset que envíe el tutor) |
| 7 | **Sistema en vivo por GPU** (utilización, memoria, potencia, temperatura, relojes) y CPU/RAM, con intervalo configurable | W&B, ClearML, Aim | Bajo | Sí (hilo en el terminal) | Cont. (`system_interval_s: 5`) |
| 8 | **Desglose del paso**: espera del DataLoader, tiempo de paso, muestras/s (y FLOPs si se quiere) | TB Input pipeline analyzer, Lightning `ThroughputMonitor` | Bajo | Sí | Cont. (`data_time_s`, `step_time_s`, `samples_per_s`) |
| 9 | **Métricas por *rank* en DDP**: tiempo por rank y diferencias entre ellos (rezagados) y cómputo frente a comunicación | Vista *Distributed* del plugin de PyTorch (obsoleto), Pod viewer | Medio | Sí; cambia el "solo rank 0" actual del flujo de eventos: ver aviso abajo | Falta |
| 10 | **Estadísticas por grupo de capas**: normas de gradiente y pesos, ratio de actualización, histogramas de pesos **y de gradientes** | `wandb.watch` (por defecto cada 1000 batches), Aim *Distributions*, Lightning `grad_norm` | Bajo-medio | Sí | Parc. (hay pesos; no se citan histogramas de gradientes) |
| 11 | **Mapas Grad-CAM del panel** con documentación de capa y objetivo, más una **prueba de cordura** (aleatorizar pesos y comprobar que el mapa cambia) | Captum, `pytorch-grad-cam`; Arun, Adebayo | Medio | Sí (se generan en el terminal, fuera del tiempo medido) | Cont. (Grad-CAM); falta la prueba de cordura |
| 12 | **Embeddings del panel y proyección 2D** (PCA siempre, UMAP opcional), coloreada por clase y por acierto | TB Projector, FiftyOne, Comet | Medio | Sí (la proyección se hace en CPU) | Cont. |
| 13 | **Cartografía y cambios de decisión** (confianza, variabilidad, corrección; aciertos que pasan a fallo) | Swayamdipta et al., Toneva et al. | Bajo | Sí | Cont. (`cartography`, `flips`) |
| 14 | **Comparación de runs** con diferencias de configuración y **agregación por semilla** (media y banda) | Aim, ClearML, W&B | Medio | Sí | Parc. (hay pestaña `comparison`; no he comprobado la agregación) |
| 15 | **Alertas**: pérdida NaN/infinita, divergencia, estancamiento y run caído, a fichero o webhook | `run.alert` de W&B, servicio de Slack de ClearML | Bajo | Sí (lo emite el terminal) | Parc. (existe *heartbeat* y estado "parado") |
| 16 | **Informe exportable de un run** (HTML o Markdown con configuración, entorno, semilla, curvas y tablas) y lista de comprobación TRIPOD+AI/CLAIM | Evidently *Reports*, Neptune†, CLAIM | Medio | Sí (generado en CPU) | Falta |
| 17 | **Problemas de etiquetado**: cartografía como método barato; opcionalmente probabilidades **fuera de muestra** por validación cruzada, que es lo que Cleanlab exige para mayor precisión | Cleanlab, FiftyOne `mistakenness` | Medio (cartografía) / **Alto** (k-fold) | Sí (el k-fold se entrena en el terminal) | Parc. (solo cartografía) |
| 18 | **Duplicados y fuga entre particiones** (hash o embeddings), en particular por paciente | FiftyOne *leaky splits*, Cleanlab | Medio | Sí | Falta |
| 19 | **Perfil puntual de un paso** con `torch.profiler` (`schedule` con `wait`, `warmup`, `active`) exportado como traza Chrome/Perfetto, en un comando aparte, fuera del run medido | TB Profiler, Lightning profilers | Medio | Sí (se captura en el terminal; la web enlaza o el usuario abre Perfetto) | Falta |
| 20 | **Exportadores opcionales** del flujo de eventos a TensorBoard (y, si hace falta, MLflow) | TensorBoard, MLflow | Bajo | Sí | Falta (decidido como opcional en la auditoría) |

**Por qué este orden** *(interpretación)*: del 1 al 6 son lo que más se acerca al encargo del tutor
("ver las decisiones del modelo") y a lo que exige una publicación médica; casi no cuestan
cómputo, porque reutilizan salidas que la validación ya calcula. Del 7 al 9 son lo que sostiene la
tesis del proyecto (predecir tiempo, memoria y energía); el 9 es el hueco mayor frente a la literatura
de rendimiento distribuido. Del 10 al 13 son análisis ricos que ya están diseñados. Del 14 al 20 son
de producto y publicación.

**Avisos de coste y de medición** *(interpretación)*:

- Todo lo que necesita una pasada hacia delante (panel, embeddings, Grad-CAM) debe seguir ocurriendo
  **fuera del tiempo medido**, como ya establece `docs/inspeccion.md`. Una captura por época del
  panel completo puede ser notable en datasets pequeños y modelos pequeños; medirlo con el campo
  `seconds` de los eventos `inspection`.
- El punto 9 choca con la decisión actual de escribir el flujo solo desde rank 0. Una opción es que
  cada rank escriba su propio fichero con pocas métricas (tiempos), o un `all_gather` de escalares
  al final de cada época; no he evaluado el coste de ninguna.
- Registrar potencia con intervalos cortos añade carga al propio muestreo. Habría que medirla antes
  de bajar de 5 s si el objetivo es energía.

**Funcionalidades que NO encajan con el invariante** *(y por qué se descartan)*:

- **Inferencia interactiva desde la web** (subir una imagen y obtener una predicción o un mapa):
  exigiría que la web cargue el modelo y use la GPU. Se sustituye por `paravit predict` y
  `paravit explain` en el terminal, con la web solo mostrando los resultados guardados.
- **Adoptar W&B o ClearML como interfaz**: ya descartados por licencia e infraestructura en la
  revisión previa; además duplicarían la web.
- **Captum Insights**: retirado en la v0.9.0.

---

## 5. Lo que no se ha podido verificar

- **W&B:** que su panel se actualice en vivo (solo fuentes no oficiales); suavizado y bandas de las
  curvas; el comportamiento en la interfaz de las tablas incrementales; el contenido de la página de
  Hyperband y de importancia de parámetros (leídas a través de un resumen de búsqueda de la página
  oficial, no del texto completo); el intervalo de muestreo por defecto real (15 s en la
  referencia, 10 s en el artículo de soporte).
- **Comet:** intervalo de muestreo de las métricas de sistema; curva ROC integrada; el *Optimizer*
  de hiperparámetros; qué hace `log_table` en la interfaz.
- **Neptune:** casi todo, por el cierre del servicio; la página del anuncio no se pudo abrir (error
  TLS).
- **Aim:** qué es hoy un *explorer* y qué una pestaña; el funcionamiento de las alertas; el estado
  de mantenimiento.
- **ClearML:** la ruta de configuración de la frecuencia del monitor de recursos; histogramas y
  proyección de embeddings; el intervalo de refresco de la web.
- **FiftyOne:** ROC y calibración (la doc leída no los cita); la fórmula de `hardness` y de
  `mistakenness`, que no están documentadas.
- **Ultralytics:** el nombre `train_batch*.jpg`; el conjunto de artefactos para clasificación.
- **Lightning:** columnas de `ModelSummary`; nombres exactos de las métricas de
  `ThroughputMonitor`; salidas de `PyTorchProfiler`.
- **Deepchecks:** condiciones de las *suites*, exportación de informes y alertas.
- **Cleanlab:** exportación de informes.
- **Fuentes médicas:** STARD 2015 y Vickers y Elkin (solo se verificó la ficha bibliográfica);
  Ghassemi et al. (solo resúmenes); Adebayo et al. (solo arXiv, la sede de publicación no está
  comprobada); la lista de ítems de CLAIM 2024 (verificada la ficha, no el contenido).
- **Costes absolutos** en milisegundos de cada instrumentación, en ViT sobre las GPU del proyecto:
  hay que medirlos en este proyecto.

## 6. Fuentes verificadas para las referencias médicas y de datos

| Referencia | Enlace comprobado |
|---|---|
| Van Calster et al., "Calibration: the Achilles heel of predictive analytics", *BMC Medicine* 17:230, 2019 | [PMC6912996](https://pmc.ncbi.nlm.nih.gov/articles/PMC6912996) · DOI [10.1186/s12916-019-1466-7](https://doi.org/10.1186/s12916-019-1466-7) |
| Collins et al., "TRIPOD+AI statement…", *BMJ* 385:e078378, 2024 | [PMC11019967](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019967/) · DOI [10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378) (Crossref) |
| Maier-Hein et al., "Metrics reloaded: recommendations for image analysis validation", *Nature Methods* 21(2):195-212, 2024 | [arXiv:2206.01653](https://arxiv.org/abs/2206.01653) · DOI [10.1038/s41592-023-02151-z](https://doi.org/10.1038/s41592-023-02151-z) (Crossref) |
| Arun et al., "Assessing the Trustworthiness of Saliency Maps for Localizing Abnormalities in Medical Imaging", *Radiology: AI* 3(6):e200267, 2021 | [PMC8637231](https://pmc.ncbi.nlm.nih.gov/articles/PMC8637231) |
| Oakden-Rayner et al., "Hidden Stratification Causes Clinically Meaningful Failures in Machine Learning for Medical Imaging" | [arXiv:1909.12475](https://arxiv.org/abs/1909.12475) |
| Ghassemi et al., "The false hope of current approaches to explainable artificial intelligence in health care", *Lancet Digit Health* 3(11):e745-e750, 2021 | DOI [10.1016/S2589-7500(21)00208-9](https://doi.org/10.1016/S2589-7500(21)00208-9) (Crossref) |
| Tejani et al., "Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update", *Radiology: AI* 6(4):e240300 | [PMC11304031](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031) · DOI [10.1148/ryai.240300](https://doi.org/10.1148/ryai.240300) |
| Bossuyt et al., "STARD 2015: an updated list of essential items for reporting diagnostic accuracy studies", *BMJ* 351:h5527, 2015 | DOI [10.1136/bmj.h5527](https://doi.org/10.1136/bmj.h5527) (Crossref y Europe PMC) |
| Vickers y Elkin, "Decision Curve Analysis: A Novel Method for Evaluating Prediction Models", *Med Decis Making* 26(6):565-574, 2006 | DOI [10.1177/0272989X06295361](https://doi.org/10.1177/0272989X06295361) (Crossref) |
| Guo et al., "On Calibration of Modern Neural Networks", ICML 2017 | [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) |
| Adebayo et al., "Sanity Checks for Saliency Maps" | [arXiv:1810.03292](https://arxiv.org/abs/1810.03292) |
| Northcutt, Jiang y Chuang, "Confident Learning: Estimating Uncertainty in Dataset Labels", *JAIR*, 2021 | [arXiv:1911.00068](https://arxiv.org/abs/1911.00068) |
| Swayamdipta et al., "Dataset Cartography…", EMNLP 2020, pp. 9275-9293 | [ACL Anthology](https://aclanthology.org/2020.emnlp-main.746/) · DOI [10.18653/v1/2020.emnlp-main.746](https://doi.org/10.18653/v1/2020.emnlp-main.746) |
| Yang et al., "MedMNIST v2…", *Scientific Data* 10:41, 2023 (solo ficha) | DOI [10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8) (Crossref) |
