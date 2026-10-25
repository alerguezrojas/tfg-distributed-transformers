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
   histogramas de pesos por grupo de capas.

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
```

`--monitor` en `train_single_gpu.py`, `train_ddp.py`, `train_heterogeneous_ddp.py` y el de
paralelismo de modelo equivale a `monitor.enabled: true`.

## Eventos nuevos

| `type` | Campos | Cuándo |
|---|---|---|
| `system` | `gpus`: lista de `{index, name, util_pct, mem_used_mb, mem_total_mb, power_w, temp_c, sm_clock_mhz}`; `cpu_pct` (equipo), `ram_used_pct`, `proc_cpu_pct`, `proc_rss_mb` | Cada `system_interval_s` s desde un hilo aparte (sin NVML: `gpus` vacío) |
| `monitor_start` | `dir` (relativo al log), `samples`, `panel`, `classes`, `full_val`, `embedding_dim` | Al empezar el run, si el monitor está activo |
| `inspection` | `epoch`, `file` (relativo al log), `seconds` (lo que tardó la captura) | Tras cada época inspeccionada |
| `layers` | `epoch`, `batch`, `groups`: lista de `{name, grad_norm, weight_norm, update_ratio}` | Cada `log_batch_every` batches, si `monitor.layers` |
| `histograms` | `epoch`, `groups`: lista de `{name, edges (33), counts (32)}` de los pesos | Al final de cada época, si `monitor.layers` |

**`batch`** añade a `metrics`: `data_time_s` (espera del DataLoader en ese batch), `step_time_s`
(resto del paso), `samples_per_s`. **`epoch_end.train`** añade `samples_per_s` y
`data_wait_frac` (fracción del tiempo de entrenamiento esperando datos: >0,3 indica cuello de I/O).

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

En DDP, `val_probs` sale de las probabilidades ya reunidas de todos los ranks (el orden no es el
del dataset, pero cada fila está alineada con su `val_targets`; puede haber duplicados del relleno
del `DistributedSampler`).

**Grad-CAM:** ViT/DeiT, sobre la entrada de la atención del último bloque (`norm1`, tokens de
parche sin los de prefijo: con cabeza de token CLS, la salida del último bloque no llega a los
*logits* y su gradiente es cero);
CNN, sobre el último mapa de activación 4D. Se calcula con `torch.autograd.grad` respecto a la
activación, así que no toca los `.grad` de los parámetros. Modelos sin capa compatible o
paralelismo de modelo: sin `cam` (y se avisa una vez).

## Análisis en la web (`src/analysis/`, puro: NumPy, sin torch)

| Función | Devuelve |
|---|---|
| `roc_curve(scores, target)` / `pr_curve(scores, target)` | Puntos de la curva y AUC / AP, por clase |
| `per_class_curves(probs, targets)` | ROC y PR de cada clase + macro y micro |
| `threshold_sweep(probs, targets, task)` | F1, precisión, recall, sensibilidad y especificidad frente al umbral |
| `calibration(probs, targets, task, bins=15)` | ECE, MCE, Brier y diagrama de fiabilidad (top-label en multiclase; por etiqueta agregada en multi-etiqueta) |
| `confusion(probs, targets, task, threshold)` | Matriz de confusión e índices de las muestras de cada celda |
| `cartography(prob_true_by_epoch, correct_by_epoch)` | Confianza, variabilidad y corrección por muestra; regiones fácil / ambigua / difícil (Swayamdipta et al., 2020) |
| `flips(probs_prev, probs_curr, targets, task)` | Muestras que pasan de acierto a fallo y al revés entre épocas |
| `project_2d(embeddings, method="pca")` | Proyección 2D (PCA siempre; UMAP si `umap-learn` está instalado) |
| `worst_errors(probs, targets, loss, task, k)` | Las *k* muestras con mayor pérdida y sus etiquetas |
