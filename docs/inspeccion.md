# Inspección del entrenamiento: datos que se capturan y cómo se guardan

Contrato entre lo que captura el entrenamiento (terminal) y lo que analiza la web. Amplía el
[flujo de eventos](eventos.md) (esquema v1: los tipos y campos nuevos no cambian la versión) con:

1. **Sistema** (siempre que haya flujo de eventos): uso de GPU, memoria, potencia, temperatura y
   relojes por GPU; CPU y RAM del proceso y del equipo.
2. **Rendimiento por batch:** espera del DataLoader, tiempo de paso y muestras/s.
3. **Monitor de predicciones** (opcional, `monitor.enabled`): un subconjunto fijo de validación
   cuyas predicciones, pérdida, *embeddings* y mapas Grad-CAM se guardan **en cada época**, más
   las probabilidades de toda la validación para curvas y calibración.
4. **Capas** (opcional, `monitor.layers`): normas de gradientes y pesos, ratio de actualización e
   histogramas de pesos y de gradientes por grupo de capas.
5. **Alertas** (siempre que haya flujo de eventos, `monitor.alerts`): avisos de problemas del run
   (pérdida NaN, divergencia, sobreajuste, gradientes, cuello de I/O, GPU al límite).

Diseño y alternativas: [auditoría de observabilidad](roadmap.md). Inventario de lo que ofrecen
las herramientas del sector: [research/inventario_herramientas_observabilidad.md](research/inventario_herramientas_observabilidad.md).

## Principios

- **No altera el entrenamiento.** Ninguna captura consume números aleatorios globales, cambia el
  modo del modelo de forma persistente ni acumula gradientes en los parámetros. Con el monitor
  activado o no, los pesos y las métricas son idénticos (hay un test que lo comprueba).
- **Fuera del tiempo medido.** El monitor de predicciones trabaja en `on_epoch_end`, después de
  medir el tiempo de la época, y la energía solo envuelve `train_epoch`/`eval_epoch`.
- **Solo rank 0.** En DDP se usa el modelo desenvuelto (`model.module`) para no lanzar colectivas.
- **El terminal hace, la web mira.** El entrenamiento guarda datos en bruto; la web calcula
  curvas, calibración, dinámica y proyecciones en CPU con `src/analysis/` (puro, sin torch).

## Configuración

```yaml
monitor:
  enabled: false          # monitor de predicciones (también con --monitor en los scripts)
  samples: 512            # tamaño del subconjunto fijo de validación
  every: 1                # cada cuántas épocas
  seed: 0                 # semilla propia del subconjunto (no toca la global)
  thumbnail: 64           # lado de las miniaturas guardadas (px)
  embeddings: true        # vector de la penúltima capa de cada muestra
  gradcam: 16             # nº de muestras del panel con mapa Grad-CAM (0 = desactivado)
  full_val_max: 4000000   # guarda probabilidades de toda la validación si N×C ≤ este valor
  layers: true            # estadísticas por grupo de capas (cada log_batch_every batches)
  system_interval_s: 5    # muestreo de sistema (también sin monitor; 0 = desactivado)
  alerts:                 # alertas (también sin monitor); todas las claves son opcionales
    enabled: true
    loss_divergence_factor: 2.0   # pérdida de validación > k × la mejor
    overfit_epochs: 3             # validación sube N épocas seguidas mientras la de train baja
    grad_explosion_factor: 100.0  # norma de gradiente de un grupo > k × su mediana reciente
    grad_history: 20              # muestras de `layers` que forman esa mediana
    update_ratio_min: 0.000001    # ratio actualización/peso fuera de [min, max]...
    update_ratio_max: 0.1
    update_ratio_patience: 3      # ...en N muestras de `layers` seguidas
    data_wait_frac: 0.3           # fracción de la época esperando datos
    gpu_mem_pct: 95.0             # memoria de GPU usada (%)
    gpu_temp_c: 85.0              # temperatura de GPU (°C)
    throughput_drop_factor: 0.5   # muestras/s < k × mediana de las épocas anteriores
```

`--monitor` en `train_single_gpu.py`, `train_ddp.py`, `train_heterogeneous_ddp.py` y el de
paralelismo de modelo equivale a `monitor.enabled: true`.

## Eventos nuevos

| `type` | Campos | Cuándo |
|---|---|---|
| `system` | `gpus`: lista de `{index, uuid, name, util_pct, mem_used_mb, mem_total_mb, power_w, temp_c, sm_clock_mhz}`; `cpu_pct` (equipo), `ram_used_pct`, `proc_cpu_pct`, `proc_rss_mb` | Cada `system_interval_s` s desde un hilo aparte (sin NVML: `gpus` vacío) |
| `monitor_start` | `dir` (relativo al log), `samples`, `panel`, `classes`, `full_val`, `embedding_dim` | Al empezar el run, si el monitor está activo |
| `inspection` | `epoch`, `file` (relativo al log), `seconds` (lo que tardó la captura) | Tras cada época inspeccionada |
| `layers` | `epoch`, `batch`, `groups`: lista de `{name, grad_norm, weight_norm, update_ratio}` | Cada `log_batch_every` batches, si `monitor.layers` |
| `histograms` | `epoch`, `kind` (`"weights"` o `"gradients"`), `groups`: lista de `{name, edges (33), counts (32)}` | Al final de cada época, si `monitor.layers`: un evento de pesos y otro de gradientes del **último batch** de la época (los grupos sin gradiente o con NaN/inf no aparecen). Los eventos anteriores a `kind` eran de pesos |
| `alert` | `level` (`"warning"` o `"critical"`), `code`, `message` (inglés), `epoch`, `batch`, `value`, `threshold`, `group` (ver [Alertas](#alertas)) | Cuando una regla salta por primera vez, o de nuevo tras dejar de cumplirse |

**`batch`** añade a `metrics`: `data_time_s` (espera del DataLoader en ese batch), `step_time_s`
(resto del paso), `samples_per_s`. **`epoch_end.train`** añade `samples_per_s` y
`data_wait_frac` (fracción del tiempo de entrenamiento esperando datos: >0,3 indica cuello de I/O).

### Cómo se miden

- **Rendimiento por batch** (`src/training/throughput.py`, `StepTimer`): `data_time_s` es el
  tiempo desde que acaba la iteración anterior hasta tener el batch; `step_time_s`, el resto del
  paso (transferencia, forward, backward, optimizador y las lecturas que sincronizan la GPU).
  `samples_per_s = muestras del batch / (data_time_s + step_time_s)`. Se usa
  `time.perf_counter()` y no hay `cuda.synchronize`, porque `loss.item()` ya sincroniza. Los
  *batch hooks* (escribir el CSV o el evento) se ejecutan después y no cuentan en ninguno de los
  dos. A nivel de época, `data_wait_frac = Σ data_time_s / Σ (data_time_s + step_time_s)` (en
  [0, 1]) y `samples_per_s = muestras / Σ (data_time_s + step_time_s)`. Las claves salen en
  `epoch_end.train` solas; el CSV de épocas no cambia. Los `batch` solo se escriben con el monitor
  de batches (sin él no se calculan métricas por batch). Los cuatro bucles lo comparten: `Trainer`
  (también DDP y paralelismo de modelo), heterogéneo y el bucle de `--trace deep` con `batch-table`.
  - En DDP las cifras son **por rank** (el flujo lo escribe el rank 0): la capacidad global es
    aproximadamente `samples_per_s × world_size`. En DDP y heterogéneo, la espera al rank más
    lento (el *all-reduce*) cuenta como `step_time_s`, así que el `data_wait_frac` del rank 0 no
    refleja un cuello de I/O de otro rank.
  - El primer batch de cada época incluye el arranque de los *workers* del DataLoader (si no son
    persistentes), así que `data_wait_frac` puede salir alto con épocas muy cortas aunque los
    batches siguientes no esperen.
- **Sistema** (`src/training/system_sampler.py`, `SystemSampler`): hilo *daemon* que arranca en
  el primer `on_epoch_start` (cuando el sink ya ha escrito `run_start`) y se para (con `join` de
  5 s) en `on_fit_end`/`on_fit_error`. La primera muestra llega tras un intervalo, y una muestra
  en curso al parar se descarta, así que `system` no precede a `run_start` ni sigue a `run_end`.
  - `mem_used_mb` y `mem_total_mb` están en MiB (1024²); `power_w` en vatios; `cpu_pct` y
    `ram_used_pct` son del equipo; `proc_cpu_pct` y `proc_rss_mb` son del proceso de
    entrenamiento (pueden superar 100 con varios hilos) y **no** incluyen a los *workers* del
    DataLoader, que son procesos aparte.
  - Una consulta que la GPU no soporta (potencia, reloj…) da `null` en ese campo.
  - **GPUs visibles:** NVML numera por bus PCI y CUDA según `CUDA_DEVICE_ORDER` (por defecto,
    la más rápida primero), así que el índice de NVML puede no ser `cuda:N`. El builder pasa las
    UUID de las GPU que ve torch (`torch.cuda.get_device_properties(i).uuid`, que ya aplica
    `CUDA_VISIBLE_DEVICES` y `CUDA_DEVICE_ORDER`) y el probe las resuelve con
    `nvmlDeviceGetHandleByUUID` en ese orden: `index` coincide con `cuda:N` y con el orden de
    `env.devices` en `run_start`, y `uuid` (`GPU-…`) identifica la tarjeta. Sin torch/CUDA se
    usa la numeración de NVML filtrada a mano por `CUDA_VISIBLE_DEVICES` (índices, UUID o vacío =
    ninguna; se corta en la primera entrada inválida, como CUDA), donde `index` es la posición
    entre las visibles. Los UUID de MIG no se interpretan. El mismo criterio (índice de torch →
    UUID → NVML) lo aplica la medida de energía (`_PowerSampler`).
  - Errores: sin `pynvml` o sin driver, `gpus: []`; un fallo al leer o escribir se avisa una vez y
    el hilo sigue. Escribir eventos nunca detiene el entrenamiento.
  - Solo rank 0 y solo si hay flujo de eventos (`--trace` distinto de `off`). En DDP/heterogéneo se
    muestrean todas las GPU visibles para ese proceso, no solo la del rank 0.

## Alertas

`src/training/alerts.py`. Reglas puras (`check_*`, sin torch) y un `AlertMonitor`
(`TrainingListener`) que las evalúa con lo que ya produce el run y escribe un evento `alert` por
cada hallazgo nuevo. **Contrato del evento** (esquema v1; los campos de abajo siempre están):

```json
{"v": 1, "type": "alert", "time": 1761400061.9, "level": "warning", "code": "loss_divergence",
 "message": "Validation loss 2.1 is 2.6x the best so far (0.81).", "epoch": 7, "batch": null,
 "value": 2.1, "threshold": 1.62, "group": null}
```

| Campo | Tipo | Significado |
|---|---|---|
| `level` | `"warning"` o `"critical"` | `critical` solo para `nan_loss` |
| `code` | str | Identificador estable de la regla (tabla siguiente) |
| `message` | str | Frase en inglés legible, con las cifras |
| `epoch` | int o null | Época en que se detectó (la del bucle); `null` si no aplica |
| `batch` | int o null | Batch dentro de la época, si la regla mira batches (`nan_loss` con batch hook, `grad_explosion`, `update_ratio_out_of_range`); `null` si es de época o de sistema |
| `value` | float o null | Valor medido. `null` si no es representable en JSON (NaN/inf) |
| `threshold` | float o null | Umbral que se ha superado (ya multiplicado por el factor, p. ej. `2 × mejor`) |
| `group` | str o null | Instancia dentro del código: grupo de capas, `gpu<N>` o `train`/`val`; `null` si es global. Campo aditivo (no estaba en el contrato mínimo): un lector que no lo conozca lo ignora |

| `code` | Nivel | Regla (umbral por defecto) | Fuente |
|---|---|---|---|
| `nan_loss` | critical | La pérdida es NaN o infinita en un batch o en la media de la época (`group` `train` o `val`) | `epoch_end`; el batch hook si ya hay |
| `loss_divergence` | warning | Pérdida de validación > 2 × la mejor de las épocas anteriores | `epoch_end` |
| `overfitting` | warning | La de validación sube 3 épocas seguidas mientras la de train baja | `epoch_end` |
| `grad_explosion` | warning | Norma del gradiente de un grupo (`group` = capa) o del modelo entero (`group` `total`) > 100 × su mediana reciente (mínimo 5 muestras). Un gradiente no finito con pérdida finita es un paso que el `GradScaler` (fp16) salta: se ignora | `layers`; norma total del batch hook |
| `update_ratio_out_of_range` | warning | Ratio actualización/peso fuera de [1e-6, 1e-1] en 3 muestras seguidas | `layers` |
| `dataloader_bound` | warning | `data_wait_frac` de la época > 0,3 | `epoch_end.train` |
| `gpu_memory_high` | warning | Memoria usada > 95 % del total (`group` `gpu<N>`) | `system` |
| `gpu_temp_high` | warning | Temperatura > 85 °C (`group` `gpu<N>`) | `system` |
| `throughput_drop` | warning | Muestras/s de la época < 50 % de la mediana de las anteriores (mínimo 2) | `epoch_end.train` |

Reglas de emisión:
- Una alerta se emite **una vez por (`code`, `group`)** y se **re-arma** solo cuando la condición
  deja de cumplirse (una evaluación sin hallazgo para ese par). Mientras persiste no se repite.
  Una regla que afecta a muchos grupos (p. ej. ratio de actualización) emite un evento por grupo.
- `update_ratio_out_of_range` y `grad_explosion` por capa solo se evalúan si hay eventos `layers`
  (`monitor.layers`). `grad_explosion` con `group` `total` usa la norma global del gradiente
  **antes del recorte** (`Trainer.last_grad_norm`, lo que devuelve `clip_grad_norm_`; sin
  `grad_clip` se calcula, y solo si hay *batch hooks*), así que `grad_clip` no la oculta.
  `gpu_*` necesita el muestreo de sistema.
- **Las normas y los histogramas por capa son posteriores al recorte** (`grad_clip`): con el
  recorte activo el gradiente por grupo está acotado y por eso la regla de explosión mira
  también la norma total previa. Los histogramas de gradientes omiten los grupos con NaN/inf.
- El NaN por batch usa el *batch hook* **solo si el `Trainer` ya tiene alguno** (monitor de
  batches o de capas): no fuerza cálculo por batch. Sin hooks se detecta al final de la época. Un
  run con NaN no se detiene: sigue o termina como lo haría sin alertas.
- Los eventos `system` y `layers` llegan al `AlertMonitor` porque este se suscribe al sink
  (`JsonlEventSink.subscribe`); ni el muestreador ni el monitor de capas saben de alertas. Una
  alerta de GPU sale desde el hilo del muestreador; que no aparezca después de `run_end` depende
  de que el muestreador se pare a tiempo (espera de hasta 5 s en `on_fit_end`).
- Todo error de una regla o de escritura se convierte en un aviso; nunca detiene el
  entrenamiento ni cambia ningún número (test de invariancia de pesos y de RNG).
- Se cablea en rank 0 con el sink, también sin `--monitor`. `monitor.alerts.enabled: false` las
  desactiva. Las claves de `monitor.alerts` se validan (claves conocidas, tipos y mínimos).
- Con LLRD y *warmup* los grupos más bajos del ViT pueden rondar el límite inferior del ratio
  (1e-6) en las primeras épocas: el umbral se ajusta con `update_ratio_min`.

## Directorio de inspección

`logs/<env>/<subdir>/inspect_<ts>/`, junto al log del run. **No se versiona** (`.gitignore`):
puede ocupar decenas de MB. Ficheros NumPy comprimidos (`np.savez_compressed`):

**`monitor_set.npz`** (una vez):

| Clave | Forma y tipo | Significado |
|---|---|---|
| `indices` | `(M,)` int64 | Índices de las muestras dentro de la partición de validación |
| `targets` | `(M, C)` uint8 | Etiquetas reales (0/1; *one-hot* en multiclase) |
| `thumbs` | `(M, t, t, 3)` uint8 | Miniaturas para la web (gris replicado a 3 canales; rango por imagen) |
| `panel` | `(P,)` int64 | Posiciones (en `0..M-1`) de las muestras con Grad-CAM |
| `classes` | `(C,)` str | Nombres de las clases |
| `task` | `()` str | `multilabel`, `binary` o `multiclass` |

**`epoch_<EEE>.npz`** (una por época inspeccionada):

| Clave | Forma y tipo | Significado |
|---|---|---|
| `probs` | `(M, C)` float16 | Probabilidades del subconjunto (sigmoide o *softmax* según la tarea) |
| `loss` | `(M,)` float32 | Pérdida por muestra (BCE media por clase o entropía cruzada) |
| `embeddings` | `(M, D)` float16 | Entrada de la última capa lineal (opcional) |
| `cam` | `(P, h, w)` uint8 | Grad-CAM de la clase predicha, normalizado a 0-255 (opcional) |
| `cam_class` | `(P,)` int64 | Clase explicada en cada mapa |
| `val_probs` | `(N, C)` float16 | Probabilidades de toda la validación (si `N×C ≤ full_val_max`) |
| `val_targets` | `(N, C)` uint8 | Etiquetas reales alineadas con `val_probs` |
| `threshold` | `()` float32 | Umbral óptimo de validación de la época (o 0,5) |

**`groups.npz`** (opcional; *previsto: lo escribirá el monitor cuando el dataset exponga
metadatos*; hoy la web lo lee si existe, p. ej. escrito a mano o con otro script). Una clave por
variable de agrupación (sexo, centro, edad...), cada una un vector 1-D de `M` valores **alineado
con `monitor_set.npz`** (la fila `i` es la muestra `i` del subconjunto):

| Clave | Forma y tipo | Significado |
|---|---|---|
| `<nombre>` (p. ej. `sex`, `age`, `centre`) | `(M,)` cadenas o números | Etiqueta de grupo de cada muestra. Las numéricas con muchos valores distintos (edad) la web las parte en cuantiles; los datos faltantes se codifican como su propio valor (`"NA"`) |

Si alguna variable no tiene exactamente `M` entradas, el fichero se ignora entero (una variable
desalineada daría cifras falsas). Sin `groups.npz` la vista *Subgroups* no aparece.

En DDP, `val_probs` sale de las probabilidades ya reunidas de todos los ranks (el orden no es el
del dataset, pero cada fila está alineada con su `val_targets`; puede haber duplicados del relleno
del `DistributedSampler`).

**Grad-CAM:** ViT/DeiT, sobre la entrada de la atención del último bloque (`norm1`, tokens de
parche sin los de prefijo: con cabeza de token CLS, la salida del último bloque no llega a los
*logits* y su gradiente es cero);
CNN, sobre el último mapa de activación 4D. Se calcula con `torch.autograd.grad` respecto a la
activación, así que no toca los `.grad` de los parámetros. Funciona también con paralelismo de
modelo (los *hooks* van en el último bloque, en la segunda GPU), Swin (NHWC), ConvNeXt y
EfficientNet. Modelos sin capa compatible: sin `cam` (se avisa una vez); si Grad-CAM falla, se
desactiva y el resto de la captura se sigue guardando.

El monitor necesita el flujo de eventos: con `--trace off` no se activa (y se avisa). Las
`probs` del subconjunto se calculan con la misma precisión (autocast) que la validación. El
monitor de capas registra un *batch hook*, así que el `Trainer` calcula sus métricas por batch en
cada paso (trabajo pequeño en CPU; no cambia ningún número).

## Análisis en la web (`src/analysis/`, puro: NumPy, sin torch)

| Función | Devuelve |
|---|---|
| `roc_curve(scores, target)` / `pr_curve(scores, target)` | Puntos de la curva y AUC / AP, por clase |
| `per_class_curves(probs, targets)` | ROC y PR de cada clase + macro y micro |
| `threshold_sweep(probs, targets, task)` | F1, precisión, recall, sensibilidad y especificidad frente al umbral |
| `calibration(probs, targets, task, bins=15)` | ECE, MCE, Brier y diagrama de fiabilidad (top-label en multiclase; por etiqueta agregada en multi-etiqueta) |
| `confusion(probs, targets, task, threshold)` | Matriz de confusión e índices de las muestras de cada celda |
| `cartography(prob_true_by_epoch, correct_by_epoch)` | Confianza, variabilidad y corrección por muestra; regiones fácil / ambigua / difícil (Swayamdipta et al., 2020) |
| `prob_of_true(probs, targets, task)` / `sample_correct(probs, targets, task)` | Probabilidad de la verdad y acierto por muestra: entradas de `cartography` (acierto multi-etiqueta = coincidencia exacta) |
| `load_monitor_set(dir)` / `list_epochs(dir)` / `load_epoch(dir, epoch)` | Lector de este directorio (`src/analysis/io.py`) |
| `flips(probs_prev, probs_curr, targets, task)` | Muestras que pasan de acierto a fallo y al revés entre épocas |
| `project_2d(embeddings, method="pca")` | Proyección 2D (PCA siempre; UMAP si `umap-learn` está instalado) |
| `worst_errors(probs, targets, loss, task, k)` | Las *k* muestras con mayor pérdida y sus etiquetas |
| `load_groups(dir, n=None)` | Lee el `groups.npz` opcional (`None` si no existe; `ValueError` si una variable no tiene `n` entradas) |

### Estadística para publicación

Lo que ninguna herramienta general del [inventario](research/inventario_herramientas_observabilidad.md)
trae (secciones 3 y 4): intervalos de confianza en todas las métricas, calibración completa y
rendimiento por subgrupo (TRIPOD+AI, ítem 23a). Todo NumPy, sin dependencias nuevas.

| Función | Devuelve |
|---|---|
| `bootstrap_ci(metric_fn, probs, targets, n_boot=1000, alpha=0.05, seed=0, stratified=True)` | Intervalo percentil (Efron 1979) de cualquier `metric_fn(probs, targets) -> float`: `estimate`, `ci=(lo, hi)`, `n_boot` válidas, `n_degenerate` descartadas (métrica no finita, p. ej. AUC con una sola clase) y la distribución. Remuestreo por muestra, estratificado por clase en binaria y multiclase |
| `metrics_with_ci(probs, targets, task, threshold=0.5, n_boot=1000, ..., metrics=None)` | `estimate` e IC95 de AUC y AP (macro), F1, sensibilidad, especificidad (macro, al umbral), exactitud, ECE y Brier. Ruta vectorizada por pesos: 1000 remuestras de 10 000 × 10 en ~3 s en un núcleo; coincide remuestra a remuestra con las funciones del paquete |
| `calibration_curve_stats(probs, targets, task)` | Pendiente e intercepto de calibración (regresión logística de `y` sobre `logit(p)`, Newton/IRLS; Cox 1958, Van Calster et al. 2019) con error estándar, más el intercepto con pendiente fija a 1 (`intercept_citl`) y el ajuste por clase. Multiclase: top-label; multi-etiqueta: por elemento |
| `per_class_calibration(probs, targets, task, bins=15)` | ECE, MCE y Brier de la probabilidad de cada clase (SCE de Nixon et al. 2019) y su diagrama de fiabilidad por clase |
| `adaptive_ece(probs, targets, task, bins=15, classwise=False)` | ECE con bins de igual masa (Nixon et al. 2019); `classwise=True` da el ACE por clase |
| `operating_point(probs, targets, task, threshold, prevalence=None)` | Conteos, sensibilidad, especificidad, VPP y VPN por clase a un umbral fijo; la prevalencia por defecto es la observada |
| `threshold_for_sensitivity(probs, targets, task, target=0.90)` / `youden_threshold(...)` | Por clase, el umbral más alto que alcanza la sensibilidad objetivo (o que maximiza J de Youden, 1950) y la especificidad, VPP y VPN resultantes. El umbral cae entre dos puntuaciones distintas: vale igual con `>` y `>=` |
| `operating_points(probs, targets, task, threshold, sensitivities=(0.90, 0.95))` | Los tres criterios en una llamada |
| `predictive_values(sensitivity, specificity, prevalence)` | VPP y VPN por Bayes para otra prevalencia (Altman y Bland, 1994) |
| `subgroup_metrics(probs, targets, task, groups, threshold=0.5, min_size=20, n_boot=0)` | AUC, F1, sensibilidad, especificidad, ECE y `n` por valor de `groups` (sexo, centro, edad en tramos...), IC bootstrap opcional, `overall` y brecha máx-mín. Los grupos con menos de `min_size` muestras se marcan (`small`), no llevan IC y quedan fuera de la brecha |

Decisiones de método:
- El IC es el **percentil** simple: es optimista con pocas decenas de muestras o con una métrica
  pegada a su límite (AUC ≈ 1). Con multi-etiqueta se remuestrea la muestra completa (no hay una
  clase única por muestra para estratificar).
- Una remuestra es *degenerada* si la métrica no es finita; solo puede ocurrir en AUC/AP (ninguna
  clase conserva positivos y negativos). Se descarta y se cuenta.
- La ECE de `calibration` mide la confianza de la etiqueta predicha; la de `per_class_calibration`,
  la fiabilidad de la probabilidad de cada clase, que es lo que necesita una clase rara. Ambas
  conviven, no se sustituyen.
- La brecha por subgrupo es descriptiva (diferencia de estimaciones puntuales): para decir si
  supera el ruido hay que mirar los intervalos.
- Los umbrales de punto de operación deben fijarse en validación y aplicarse a test.

### Vistas clínicas del Inspector

La web usa estas funciones en cuatro sitios (`src/web/inspector_stats.py` reúne los helpers puros):

- **Curves:** AUC, AP, F1, sensibilidad, especificidad y exactitud macro con IC95 (`metrics_with_ci`),
  y opcionalmente IC por clase (un *bootstrap* por columna; en multiclase solo AUC y AP). Control de
  remuestras (200 / 1000 / 2000), semilla fija 0, resultados cacheados por (fichero, mtime,
  ajustes). Cada intervalo lleva la nota «95% CI, percentile bootstrap, n=…».
- **Clinical:** punto de operación de una clase (uno contra el resto en multiclase, sobre su
  probabilidad): umbral manual, el óptimo de validación del run (`threshold` del `.npz`),
  sensibilidad objetivo (0,80 / 0,90 / 0,95 / 0,99) o Youden. Muestra sensibilidad, especificidad,
  VPP, VPN, F1 y exactitud con IC, la matriz 2×2 y la ROC con el punto marcado. La prevalencia de
  VPP/VPN es la observada o la que indique el usuario (Bayes; con prevalencia impuesta el remuestreo
  se estratifica por etiqueta). Los IC son *condicionales al umbral*: si este se eligió con los
  mismos datos (sensibilidad objetivo, Youden) no recogen la incertidumbre de esa elección.
- **Calibration:** pendiente e intercepto (CITL) con interpretación (pendiente < 1 = sobreconfianza,
  > 1 = infraconfianza; se compara con 1 usando el IC de Wald), ECE adaptativa, ECE por clase (tabla
  con MCE, Brier, ACE, pendiente e intercepto por clase) y la pendiente en la evolución por época.
- **Subgroups:** solo si existe `groups.npz`. Métricas por grupo con IC opcional y brecha máx-mín;
  se calculan sobre el subconjunto monitorizado (el que tiene los grupos alineados), así que los
  grupos pequeños son ruidosos (`min_size` los marca y los excluye de la brecha).
- **Alerts:** lista cronológica de los eventos `alert` y banner con las alertas activas en la
  cabecera del Inspector y de Live. Como no hay evento de «resuelta», *activa* se define por
  recencia: la última de cada (`code`, `group`) emitida en la época actual o la anterior (con el
  run terminado, todas las que saltaron). Rojo = `critical`, ámbar = `warning`.
- **Layers:** si el run trae histogramas de gradientes (`kind: "gradients"`) aparece un selector
  Weights / Gradients.
