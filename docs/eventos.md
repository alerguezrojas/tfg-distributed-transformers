# Flujo de eventos del entrenamiento (`events_*.jsonl`)

Cada run escribe, además del log de texto y los CSV de siempre, un **flujo de eventos**
estructurado: un fichero JSON Lines con una línea por cosa que ocurre (empieza el run, termina
una época, para, falla…). Es la fuente estructurada del run para la vista en vivo de la web, el
análisis por muestra y las medidas del predictor. Motivación y alternativas en la
[auditoría de observabilidad](roadmap.md#auditoría-de-observabilidad-25102026).

## Fichero

- Ruta: `logs/<env>/<subdir>/events_<DDMMYYYY_HHMMSS>.jsonl`, junto al `train_*.log` del mismo
  run y con el mismo sello de tiempo (`<subdir>` es `mode/model` o `dataset/mode/model`).
- Lo escribe **solo el rank 0**. Una línea JSON por evento, con *flush* tras cada línea para
  que se pueda leer mientras el run sigue en marcha.
- Quien lo lea debe descartar una **última línea incompleta** (puede estar a medio escribir) y
  volver a leerla en la siguiente pasada.
- Se escribe siempre que `--trace` no sea `off`, igual que `epoch_metrics_*.csv`.

## Campos comunes

Todas las líneas llevan:

| Campo | Tipo | Significado |
|---|---|---|
| `v` | int | Versión del esquema (ahora `1`) |
| `type` | str | Tipo de evento (tabla siguiente) |
| `time` | float | Momento del evento, en segundos Unix |

**Compatibilidad hacia delante:** un lector ignora, sin fallar, los tipos que no conoce y las
líneas con `v` mayor que la suya. Añadir campos o tipos nuevos no cambia la versión; quitar o
cambiar el significado de un campo sí.

## Tipos de evento (v1)

| `type` | Campos | Cuándo |
|---|---|---|
| `run_start` | `run_id` (el sello), `dataset`, `task` (`multilabel`/`binary`/`multiclass`), `classes`, `model`, `mode` (`single`/`ddp`/`ddp_hetero`/`model_parallel`, como la carpeta de `logs/`), `world_size`, `epochs` (planificadas), `select_metric` (`f1`/`f1_optimal`/`auc`), `config` (YAML completo), `env` (`host`, `python`, `torch`, `timm`, `git_commit` o `null`, `devices`) | Al empezar `fit()` |
| `epoch_start` | `epoch`, `epochs` | Al empezar cada época |
| `batch` | `epoch` (la del bucle, como en `epoch_start`, también tras `--resume`), `batch`, `n_batches`, `metrics` (`batch_loss`, `running_loss`, `lr`, `batch_f1`, `batch_acc`…) | Cada `log_batch_every` batches, si está activo el monitor de batches |
| `epoch_end` | `epoch`, `epochs`, `train` (métricas), `val` (`loss`, `f1`, `accuracy`, `precision`, `recall`, `auc` y, si la tarea busca umbral, `optimal_threshold` y `f1_at_optimal_threshold`), `epoch_time_s`, `best`, `is_best` | Al terminar cada época |
| `early_stop` | `epoch`, `best` | Si salta la parada temprana |
| `heartbeat` | `epoch` (en curso o `null`), `elapsed_s` | Cada 30 s mientras el run está vivo, desde un hilo aparte: no añade trabajo al bucle |
| `run_end` | `status` (`completed`/`early_stopped`/`failed`/`interrupted`), `epochs_run`, `best`, `total_time_s`, `error` (si falló o se interrumpió) | Al terminar `fit()`, también si lanza una excepción; `interrupted` es Ctrl-C o `SystemExit` |

Un run sin `run_end` cuyo fichero lleva varios minutos sin cambiar se considera **parado**
(proceso muerto o máquina caída). Gracias al `heartbeat`, un run vivo escribe al menos cada 30 s
aunque una época dure una hora. `best` es `null` mientras no ha terminado ninguna época.

Escribir eventos nunca detiene el entrenamiento: un error de escritura (disco lleno, NFS caído)
se convierte en un aviso.

### Ejemplo

```json
{"v": 1, "type": "run_start", "time": 1761400000.0, "run_id": "25102026_120000", "dataset": "medmnist_derma", "task": "multiclass", "classes": ["..."], "model": "vit_tiny_patch16_224", "mode": "single", "world_size": 1, "epochs": 15, "select_metric": "auc", "config": {"...": "..."}, "env": {"host": "pc", "python": "3.12.3", "torch": "2.9.0", "timm": "1.0.20", "git_commit": "aee08bf", "devices": ["NVIDIA GeForce RTX 3060 Ti"]}}
{"v": 1, "type": "epoch_start", "time": 1761400001.2, "epoch": 1, "epochs": 15}
{"v": 1, "type": "epoch_end", "time": 1761400061.9, "epoch": 1, "epochs": 15, "train": {"loss": 0.81, "f1": 0.42, "accuracy": 0.70}, "val": {"loss": 0.74, "f1": 0.47, "accuracy": 0.73, "precision": 0.55, "recall": 0.45, "auc": 0.90}, "epoch_time_s": 60.7, "best": 0.90, "is_best": true}
{"v": 1, "type": "run_end", "time": 1761400950.0, "status": "completed", "epochs_run": 15, "best": 0.93, "total_time_s": 950.0}
```

## Tipos ampliados

Los eventos de sistema, rendimiento por batch, monitor de predicciones y capas (`system`,
`monitor_start`, `inspection`, `layers`, `histograms`), las alertas (`alert`) y los datos por época que referencian se
describen en [inspeccion.md](inspeccion.md). Previstos para la fase 4: `io` / `power` (medidas del
predictor).

## Evento `alert`

Lo emiten las reglas de salud del entrenamiento y lo muestran el banner de Live y del Inspector y
la vista *Alerts*: `{"type": "alert", "level": "warning"|"critical", "code": str, "message": str,
"epoch": int|null, "batch": int|null, "value": float|null, "threshold": float|null, "group":
str|null}`. Códigos: `nan_loss` (el único `critical`), `loss_divergence`, `overfitting`,
`grad_explosion`, `update_ratio_out_of_range`, `dataloader_bound`, `gpu_memory_high`,
`gpu_temp_high`, `throughput_drop`. `group` es el nombre de la capa, `gpu<N>` o `train`/`val`;
`value` es `null` si no es finito. Una alerta por (`code`, `group`). Los lectores antiguos la
ignoran (tipo desconocido).
