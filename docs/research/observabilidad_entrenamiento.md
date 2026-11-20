# Observabilidad del entrenamiento: seguimiento en vivo y análisis de las decisiones del modelo

> Revisión para la petición del tutor de "poder analizar más detalladamente la inferencia en la
> parte de traza, para poder observar en tiempo real las decisiones que toman los modelos durante
> los entrenamientos", posiblemente con vistas en vivo desde la web. Fecha de la revisión:
> 25/10/2026. Cada referencia se ha comprobado contra la página oficial, el DOI, arXiv o
> Crossref. Versiones y fechas de paquetes: PyPI el 25/10/2026. Lo que afirman los **autores o la
> documentación** se distingue de la **interpretación** de esta revisión, que se marca así:
> *(interpretación)*. Lo que no se ha podido comprobar se indica como *no verificado*.

## 1. Seguimiento de experimentos y métricas en vivo

### 1.1 Comparativa

| | TensorBoard | MLflow | Weights & Biases | Aim | ClearML |
|---|---|---|---|---|---|
| **Autoalojado / offline** | Sí, local (`tensorboard --logdir`). | Sí: `mlflow server`; por defecto SQLite desde la 3.7.0 (antes `./mlruns`). | Offline con `WANDB_MODE=offline` y subida posterior con `wandb sync`. Servidor propio ("Self-Managed"): la documentación dice que "algunas funciones requieren licencia Enterprise". | Sí, "self-hosted" según su README. | Sí: servidor de código abierto (Docker Compose o Kubernetes). |
| **Licencia** | Apache 2.0 (2.21.0, 29/06/2026). | Apache 2.0 (3.17.2, 21/10/2026). | Cliente `wandb` MIT (0.31.0, 16/10/2026). Servidor: ver la fila anterior; condiciones académicas *no verificadas*. | Apache 2.0. Última versión en PyPI: 3.29.1 (08/05/2025). | SDK Apache 2.0 (2.1.12, 19/08/2026). **Servidor SSPL v1.0**; depende de MongoDB y Elasticsearch. |
| **Integración con PyTorch** | `torch.utils.tensorboard.SummaryWriter`: `add_scalar`, `add_image`, `add_histogram`, `add_embedding`. Escritura asíncrona. | Manual: `mlflow.log_metrics({...}, step=)`, `mlflow.log_image(key=, step=)`. La página de MLflow sobre PyTorch dice que el autolog "funciona con Lightning" y que para bucles propios se use el registro manual (pero su ejemplo inicial lo usa en un bucle plano: es ambigua). | `wandb.init()` y `wandb.log()`. | Integraciones listadas con PyTorch Ignite y PyTorch Lightning. | PyTorch, Ignite y Lightning listados como frameworks soportados. |
| **DDP** | La documentación de `SummaryWriter` no trata DDP. | La documentación consultada no trata DDP. | Documentado: (a) solo rank 0; (b) un run por proceso con `group`; (c) un run compartido (SDK >= 0.19.9, servidor >= 0.68). | No consultado. | El README no menciona DDP. |
| **Qué se ve en vivo** | Escalares, imágenes, histogramas, embeddings. Desde la 2.3.0 ya no recarga solo cada 30 s: hay que activar "Reload data" en los ajustes. `SummaryWriter` vuelca cada 2 min por defecto (`flush_secs`) salvo que se llame a `flush()`. | Gráficas de métricas, métricas de sistema (GPU, memoria, potencia) con `enable_system_metrics_logging()` (cada 10 s; requiere `psutil` y `nvidia-ml-py`). El refresco automático de gráficas de runs activos *no está confirmado en la documentación oficial consultada*. | Panel web en vivo (no verificado explícitamente en las páginas leídas). | Anuncia "live updates" (hoja de ruta, oct. 2021) y alertas en tiempo real. | Panel web propio. *No verificado en detalle.* |
| **Coste de adopción** | Bajo. `tensorboard>=2.20.0` ya está en `pyproject.toml`, pero el código no lo usa. | Medio: dependencia pesada, servidor y base de datos, esquema propio de runs. | Alto para un proyecto autoalojado: nube propia del proveedor o servidor con licencia. | Medio; el ritmo de versiones parece parado (*interpretación*). | Alto: tres servicios (web, API, ficheros) más MongoDB y Elasticsearch. |

Contexto del sector: CoreWeave completó la compra de Weights & Biases el 5/05/2025 y su
documentación está ahora alojada en `docs.coreweave.com` (la antigua `docs.wandb.ai` redirige
allí). No hay indicio en lo consultado de que cambie el cliente, pero es un factor de riesgo para
depender de un SaaS en un proyecto académico *(interpretación)*.

### 1.2 Qué encaja con este proyecto

*(interpretación, apoyada en `AGENTS.md`)* Ninguna de las cinco herramientas aporta lo que pide el
tutor, que es ver **las decisiones del modelo** (predicciones por muestra, calibración, mapas de
atención) y no solo curvas de pérdida. Todas guardan escalares e imágenes sueltas; lo específico
hay que generarlo igualmente. Además:

- El proyecto ya tiene un dashboard Streamlit propio, "el terminal hace, la web mira", y el
  formato de logs es un contrato con la web. Adoptar MLflow o ClearML duplicaría la interfaz.
- Un formato propio, estructurado y versionado (JSONL) es lo más barato, funciona offline y se
  puede leer también con `pandas`, lo cual encaja con la publicación (reproducibilidad).
- TensorBoard y MLflow tienen sentido como **exportadores opcionales**, sin cambiar el núcleo:
  encajan con el invariante de "comportamiento transversal = decorador". MLflow, si se quiere
  comparar muchos runs entre máquinas; TensorBoard, por ser casi gratis (ya es dependencia).
- W&B y ClearML se descartan por licencia o peso de infraestructura para un proyecto académico
  autoalojado. Aim queda como alternativa solo si se confirma que sigue mantenido.

Sobre DDP, las fuentes oficiales consultadas solo dan la regla de rank 0 de forma explícita en
W&B (opción "solo rank 0") y, para guardar checkpoints, en el tutorial de DDP de PyTorch (el
código guarda con `if rank == 0:`). Que TensorBoard/MLflow se escriban solo desde rank 0 es la
convención habitual, pero **no la dice la documentación consultada** de esas dos herramientas.

## 2. Formato de eventos y vista en directo

### 2.1 Formatos

| Formato | Qué es (según la fuente) | Pros / contras para este proyecto *(interpretación)* |
|---|---|---|
| **JSON Lines** | UTF-8, sin BOM; cada línea es un valor JSON válido; separador `\n`; extensión `.jsonl`; el tipo MIME `application/jsonl` "aún no está estandarizado". | Se añade línea a línea (ideal para seguir el final del fichero), legible a mano y con `pandas.read_json(lines=True)` (admite `chunksize`, según la documentación de [pandas](https://pandas.pydata.org/docs/reference/api/pandas.read_json.html)). Sin esquema impuesto: hay que versionarlo. |
| **tfevents (TensorBoard)** | Mensajes `Event` de protobuf con `wall_time` (double), `step` (int64) y una unión (`file_version`, `graph_def`, `summary`, `session_log`...). | Estándar de facto, pero binario y solo útil con TensorBoard o `EventAccumulator`. La API de DataFrame de la documentación oficial solo funcionaba con `tensorboard.dev` (que parece cerrado; *no verificado*). |
| **Métricas de MLflow** | `log_metric(clave, valor, step=)`; el `step` es un entero de 64 bits, puede ser negativo, desordenado o con huecos. | Muy cómodo si ya se usa MLflow; no sirve para eventos con estructura (probabilidades por muestra). `log_image(key, step)` guarda imágenes en el tiempo. |

### 2.2 Cómo hace una interfaz el "seguimiento en directo"

- **`st.fragment(run_every=...)`.** Según la documentación de Streamlit, un fragmento con
  `run_every` (número de segundos, `timedelta` o cadena tipo `"10s"`) se vuelve a ejecutar solo, "aunque
  el usuario no interactúe", y solo se repinta ese fragmento, no toda la página. Con `run_every=None`
  el temporizador se apaga; el tutorial oficial usa un interruptor en `session_state` para
  iniciar y detener la lectura en vivo y exige `streamlit>=1.37.0` (el proyecto pide >= 1.57).
  Limitaciones documentadas: los valores devueltos se ignoran en las re-ejecuciones de fragmento
  (se comparte estado con `session_state`) y **no se admiten caché y fragmento en la misma
  función**. *(Interpretación)* Hay que leer el fichero con un desplazamiento (`offset`) guardado
  en `session_state` y añadir solo lo nuevo, ignorando una última línea incompleta; así el coste
  no crece con el tamaño del log.
- **`streamlit-autorefresh`** (MIT): componente que, desde el navegador, hace que el servidor
  vuelva a ejecutar **toda** la app cada `interval` ms. Útil en versiones antiguas, peor que el
  fragmento porque repinta todo.
- **WebSocket.** Streamlit ya usa un WebSocket por pestaña entre navegador y servidor
  (documentación de arquitectura), pero eso no se expone para datos propios.
- **Server-Sent Events (SSE).** Canal de un solo sentido servidor -> navegador sobre HTTP, con
  `EventSource`, `text/event-stream` y reconexión automática (MDN). Solo compensa si algún día se
  separa el servicio de eventos de Streamlit; no hace falta ahora *(interpretación)*.

## 3. Interpretar las decisiones de un ViT en imagen médica

### 3.1 Métodos

| Método | Referencia verificada | Qué necesita | Coste por muestra *(interpretación a partir de las definiciones)* |
|---|---|---|---|
| **Attention rollout** | Abnar y Zuidema, "Quantifying Attention Flow in Transformers", ACL 2020, pp. 4190-4197. DOI [10.18653/v1/2020.acl-main.385](https://doi.org/10.18653/v1/2020.acl-main.385). Estiman cuánta atención llega a las entradas; según los autores, correlaciona mejor que la atención cruda con la importancia por ablación y con gradientes. | Las matrices de atención de cada capa. | Una pasada hacia delante más productos de matrices N x N por capa. Sin pasada hacia atrás. |
| **Grad-CAM** | Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization", ICCV 2017 (DOI [10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74)) e IJCV 128(2):336-359, 2020 (DOI [10.1007/s11263-019-01228-7](https://doi.org/10.1007/s11263-019-01228-7)). | Activaciones y gradientes de una capa; en ViT, una `reshape_transform` para volver a la rejilla de parches. | Una pasada hacia delante y una hacia atrás. |
| **Integrated Gradients** | Sundararajan, Taly y Yan, "Axiomatic Attribution for Deep Networks", ICML 2017, PMLR 70:3319-3328 ([enlace](https://proceedings.mlr.press/v70/sundararajan17a.html), arXiv [1703.01365](https://arxiv.org/abs/1703.01365)). | Línea base (imagen de referencia). | En Captum, `n_steps=50` por defecto; la documentación indica que se evalúan `pasos x ejemplos` puntos con pasadas hacia delante y atrás. Es decir, del orden de 50 veces el coste de Grad-CAM. |
| **Transformer attribution (Chefer et al.)** | Chefer, Gur y Wolf, "Transformer Interpretability Beyond Attention Visualization", CVPR 2021, pp. 782-791, DOI [10.1109/CVPR46437.2021.00084](https://doi.org/10.1109/CVPR46437.2021.00084) (arXiv [2012.09838](https://arxiv.org/abs/2012.09838)). Relevancia con Deep Taylor Decomposition propagada por capas, con conservación de la relevancia total. | Código propio con propagación de relevancia. Repo oficial con ejemplos ViT y DeiT; última actualización fechada en abril de 2021. | Una pasada hacia delante y una de propagación; más intrusivo que los anteriores. Los mismos autores publicaron una versión más general sin LRP: [arXiv 2103.15679](https://arxiv.org/abs/2103.15679) (el *venue* no se ha podido confirmar). |

### 3.2 Qué es estándar en imagen médica

- Según la encuesta de van der Velden et al. (*Medical Image Analysis* 79, 102470, 2022; DOI
  [10.1016/j.media.2022.102470](https://doi.org/10.1016/j.media.2022.102470), arXiv
  [2107.10912](https://arxiv.org/abs/2107.10912)), la explicación visual (mapas de saliencia) "es
  la forma más común de XAI en análisis de imagen médica", y Grad-CAM aparece en una gran
  parte de los trabajos de su tabla de mapas de saliencia. **Matiz importante**: la encuesta
  llega solo hasta octubre de 2020, antes de la oleada de ViT, así que describe sobre todo CNN.
- Para ViT en imagen médica hay trabajo reciente, no una norma asentada. Barekatain y Glocker,
  "Evaluating the Explainability of Vision Transformers in Medical Imaging" (taller *Interpretability
  of Machine Intelligence in Medical Image Computing*, MICCAI 2025; arXiv
  [2510.12021](https://arxiv.org/abs/2510.12021)), comparan ViT, DeiT, DINO y Swin con *Gradient
  Attention Rollout* y Grad-CAM en células sanguíneas y ecografía de mama. Sus autores concluyen
  que DINO con Grad-CAM da las explicaciones más fieles y localizadas, y que Grad-CAM produce
  mapas más precisos que el rollout, más disperso. Son dos conjuntos de datos: no generalizar.
- Wall et al., "Grad-CAM for Vision Transformers: A Systematic Taxonomy and Audit of Methodological
  Ambiguity in Explainable AI" (arXiv [2608.05258](https://arxiv.org/abs/2608.05258), agosto de
  2026, preprint): de 175 artículos que aplican Grad-CAM a ViT, la mayoría no explica cómo se
  adaptó. *(Interpretación)* Si se usa Grad-CAM en ViT, hay que documentar la capa, el objetivo
  del gradiente y la reconstrucción espacial elegidos, para que sea reproducible.
- Reseña general (preprint, sin revisión por pares conocida): Lai, "Interpretable Medical Imagery
  Diagnosis with Self-Attentive Transformers: A Review of Explainable AI for Health Care", arXiv
  [2309.00252](https://arxiv.org/abs/2309.00252).

### 3.3 Librerías

| Librería | Licencia | Notas verificadas |
|---|---|---|
| **pytorch-grad-cam** (jacobgil), [repo](https://github.com/jacobgil/pytorch-grad-cam), paquete `grad-cam` 1.5.7 (21/08/2026) | MIT | GradCAM, HiResCAM, ScoreCAM, AblationCAM, EigenCAM, LayerCAM, FullGrad y otros. ViT y Swin con `reshape_transform` (capa de ejemplo `model.blocks[-1].norm1`). Soporta lotes en todos los métodos; ScoreCAM y AblationCAM necesitan muchas pasadas hacia delante; `aug_smooth` multiplica el tiempo por 6. Incluye métricas de evaluación (ROAD). |
| **Captum** (PyTorch), [repo](https://github.com/pytorch/captum) | BSD-3-Clause | `IntegratedGradients`, `LayerIntegratedGradients`, `Occlusion`, `GradCAM`, `DeepLift`, etc. Requiere Python >= 3.10 y PyTorch >= 2.3; versión 0.9.0 (17/04/2026). Captum Insights se retiró tras la v0.8.0. Artículo: Kokhlikyan et al., arXiv [2009.07896](https://arxiv.org/abs/2009.07896). |
| **Transformer-Explainability** (Chefer), [repo](https://github.com/hila-chefer/Transformer-Explainability) | MIT | Ejemplos para ViT y DeiT; no se declara mantenimiento. |

### 3.4 Trampa técnica con timm (verificada en el código)

En `timm/layers/attention.py`, `Attention` usa `F.scaled_dot_product_attention` cuando
`fused_attn` está activo, y entonces **no se crea el tensor `attn`**: no hay pesos de atención que
capturar. La variable de entorno `TIMM_FUSED_ATTN` (por defecto 1; 0 = apagado) se lee al importar,
y `set_fused_attn(False)` existe en `timm.layers.config`. *(Interpretación)* Para rollout hay que
apagarla **antes de crear el modelo** (el flag se fija en `__init__`), y apagarla cambia el
tiempo y la memoria del modelo. Como el proyecto predice precisamente tiempo y memoria, las
explicaciones deben generarse en una pasada de evaluación aparte, con otra instancia del modelo,
y nunca dentro del entrenamiento medido. Grad-CAM no necesita atención explícita, solo
activaciones y gradientes de una capa, por lo que evita este problema.

## 4. Monitorización por muestra durante el entrenamiento

Esta parte es la más cercana a "observar las decisiones del modelo". No existe un estándar
con nombre para el "panel fijo": es una práctica de diseño propuesta aquí *(interpretación)*,
combinando ideas verificadas.

- **Panel fijo de validación.** Un conjunto pequeño y fijo (con semilla) de muestras de validación,
  estratificado por clase, del que se guarda **en cada época** el vector de probabilidades (no la
  imagen ni el mapa). Permite dibujar trayectorias de confianza, ver cuándo cambia la decisión y
  qué muestras oscilan. El coste es una pasada hacia delante de N muestras; la validación ya las
  procesa, así que basta con registrar sus salidas. Contar "eventos de olvido" (paso de
  acierto a fallo) está respaldado por Toneva et al., "An Empirical Study of Example Forgetting
  during Deep Neural Network Learning" (ICLR 2019, arXiv [1812.05159](https://arxiv.org/abs/1812.05159)).
- **Calibración.** Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017, arXiv
  [1706.04599](https://arxiv.org/abs/1706.04599)), definen el diagrama de fiabilidad y el ECE
  (error de calibración esperado) y usan M = 15 intervalos en su Tabla 1. Minderer et al.,
  "Revisiting the Calibration of Modern Neural Networks" (NeurIPS 2021, arXiv
  [2106.07998](https://arxiv.org/abs/2106.07998)), afirman que los modelos recientes sin convoluciones
  figuran entre los mejor calibrados: es una hipótesis comprobable con las medidas de este
  proyecto. TorchMetrics ofrece `BinaryCalibrationError` y `MulticlassCalibrationError` (normas
  `l1` = ECE, `l2`, `max`; 15 intervalos por defecto) pero **su documentación no menciona una
  versión multietiqueta** *(interpretación)*: para ChestMNIST habría que calcular un ECE binario
  por clase. TorchMetrics no es hoy dependencia del proyecto.
- **Curvas ROC y PR por época.** Para clases desequilibradas, Saito y Rehmsmeier (*PLOS ONE* 10(3),
  e0118432, 2015, DOI [10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432))
  sostienen que la curva precisión-exhaustividad es más informativa que la ROC. Se pueden calcular
  desde las mismas salidas guardadas, sin pasada adicional.
- **Peores ejemplos.** Ordenar la validación por la probabilidad asignada a la clase verdadera y
  guardar los índices de los K peores por época. Es barato y se enlaza con los mapas de la sección 3.
- **Dataset cartography.** Swayamdipta et al., "Dataset Cartography: Mapping and Diagnosing Datasets
  with Training Dynamics", EMNLP 2020, pp. 9275-9293, DOI
  [10.18653/v1/2020.emnlp-main.746](https://doi.org/10.18653/v1/2020.emnlp-main.746). Según los
  autores, la *confianza* (probabilidad de la clase verdadera) y su *variabilidad* entre épocas,
  obtenidas en **una sola ejecución**, separan muestras fáciles, ambiguas y difíciles de aprender;
  las difíciles suelen corresponder a errores de etiquetado. Es una técnica pensada para PLN: una
  búsqueda breve no encontró aplicaciones directas a imagen médica (*no verificado como ausencia*).
  *(Interpretación)* Para el conjunto de entrenamiento exigiría registrar la probabilidad por
  muestra con un identificador estable; con aumentos de datos y dropout la señal es más ruidosa
  que en un panel de validación sin aumentos. Un estadístico relacionado: AUM, Pleiss et al.,
  "Identifying Mislabeled Data using the Area Under the Margin Ranking" (NeurIPS 2020, arXiv
  [2001.10528](https://arxiv.org/abs/2001.10528)).

## 5. Recomendación priorizada para ParaViT-Lab

1. **Flujo de eventos JSONL, escrito solo desde rank 0, como fuente de verdad.** Un nuevo
   decorador (no `if`s en `Trainer`) que añada una línea por evento (`run_start`, `epoch_end`,
   `panel`, `run_end`) con `schema_version`, `step`, `wall_time` y la configuración del run.
   Convive con los logs actuales (que son un contrato con la web). Sin dependencias nuevas.
2. **Vista en directo en la web con `st.fragment(run_every=...)`.** Lectura incremental por
   desplazamiento, con interruptor de directo/pausa, intervalo de unos segundos y descarte de la
   línea incompleta. No hace falta SSE ni WebSockets propios.
3. **Panel fijo de validación + calibración + PR/ROC por época**, todo derivado de las salidas
   que la validación ya calcula (coste casi nulo). Es lo que más se parece a "ver las decisiones
   del modelo" sin añadir una pasada extra. Empezar con MedMNIST (BreastMNIST, PathMNIST...), que
   ya está soportado.
4. **Mapas de explicación fuera del bucle medido**: un comando de terminal (por ejemplo
   `paravit explain`) que cargue un checkpoint, genere Grad-CAM con `pytorch-grad-cam` (MIT) para
   el panel fijo y guarde las imágenes en `logs/`; la web solo las muestra (respeta "la web nunca
   toca la GPU"). Rollout y Chefer, solo como complemento y con `TIMM_FUSED_ATTN=0` en una
   instancia aparte. Documentar la capa y el objetivo elegidos (Wall et al.). Advertir en la
   memoria de que los mapas no sustituyen una validación clínica.
5. **Exportadores opcionales, más tarde**: TensorBoard (casi gratis, ya es dependencia) y, si
   hace falta comparar muchos runs entre máquinas, MLflow. Descartar W&B y ClearML por licencia e
   infraestructura.

## 6. Lo que no se ha podido verificar

- Que la interfaz de MLflow refresque sola las gráficas de un run en curso (no consta en la
  documentación oficial consultada).
- Las condiciones de licencia del servidor de W&B para uso académico y el comportamiento en
  vivo de su panel (solo se confirmó lo citado arriba).
- El estado de mantenimiento de Aim (solo consta que su última versión en PyPI es de mayo de 2025)
  y la integración de ClearML y Aim con DDP.
- El *venue* de [arXiv 2103.15679](https://arxiv.org/abs/2103.15679) (la página solo muestra arXiv).
- El coste absoluto (ms por muestra) de cada método en ViT-Base con una GPU concreta: aquí solo se
  dan costes relativos deducidos de las definiciones; habría que medirlo en este proyecto.
- Una aplicación de dataset cartography a imagen médica (no encontrada en una búsqueda breve).
- La documentación de TorchMetrics sobre ECE multietiqueta (la página consultada no la menciona).

## Fuentes principales

- Streamlit: [`st.fragment`](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment),
  [guía de fragmentos](https://docs.streamlit.io/develop/concepts/architecture/fragments),
  [tutorial de inicio y parada](https://docs.streamlit.io/develop/tutorials/execution-flow/start-and-stop-fragment-auto-reruns),
  [arquitectura](https://docs.streamlit.io/develop/concepts/architecture/architecture),
  [`streamlit-autorefresh`](https://github.com/kmcgrady/streamlit-autorefresh).
- MLflow: [autoalojamiento](https://mlflow.org/docs/latest/self-hosting/),
  [PyTorch](https://mlflow.org/docs/latest/ml/deep-learning/pytorch/),
  [métricas de sistema](https://mlflow.org/docs/latest/ml/tracking/system-metrics/),
  [API de seguimiento](https://mlflow.org/docs/latest/ml/tracking/tracking-api/),
  [API Python](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html),
  [PyPI](https://pypi.org/project/mlflow/), [repo](https://github.com/mlflow/mlflow).
- TensorBoard: [`torch.utils.tensorboard`](https://docs.pytorch.org/docs/2.14/tensorboard.html),
  [README](https://github.com/tensorflow/tensorboard/blob/master/README.md),
  [`event.proto`](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/event.proto),
  [API de DataFrame](https://www.tensorflow.org/tensorboard/dataframe_api),
  [PyPI](https://pypi.org/project/tensorboard/).
- W&B: [DDP](https://docs.coreweave.com/guides/track/log/distributed-training/),
  [alojamiento](https://docs.coreweave.com/guides/hosting/),
  [variables de entorno](https://docs.coreweave.com/guides/track/environment-variables/),
  [`wandb sync`](https://docs.coreweave.com/ref/cli/wandb-sync/),
  [PyPI](https://pypi.org/project/wandb/),
  [compra por CoreWeave](https://investors.coreweave.com/news/news-details/2025/CoreWeave-Completes-Acquisition-of-Weights--Biases/default.aspx).
- Aim: [repo](https://github.com/aimhubio/aim), [versiones](https://github.com/aimhubio/aim/releases),
  [documentación](https://aimstack.readthedocs.io/en/latest/). ClearML:
  [SDK](https://github.com/clearml/clearml), [servidor](https://github.com/clearml/clearml-server).
- Formatos y tiempo real: [JSON Lines](https://jsonlines.org/),
  [Server-Sent Events (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events).
- PyTorch DDP: [tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html).
- timm: [`attention.py`](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/attention.py),
  [`config.py`](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/config.py).
- Captum: [`IntegratedGradients`](https://captum.ai/api/integrated_gradients.html).
- TorchMetrics: [CalibrationError](https://lightning.ai/docs/torchmetrics/stable/classification/calibration_error.html).
