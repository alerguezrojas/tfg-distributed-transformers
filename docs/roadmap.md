# Roadmap posterior al TFG

> Plan de trabajo desde octubre de 2026, acordado tras la reunión con el tutor. Se avanza sin
> prisa, una fase cada vez. Objetivo a medio plazo: una **publicación** sobre la predicción del
> rendimiento del entrenamiento distribuido.

## Diagnóstico de partida (auditoría del 17/10/2026)

La base de software es sólida: decoradores, cuatro estrategias, CLI, web y tests. Los dos puntos
débiles de cara a publicar son:

1. **Todo asume BigEarthNet:** el dataset, la tarea (multi-etiqueta), las métricas, el benchmark y
   el predictor.
2. **El predictor está calibrado con pocos puntos:** casi todo sale de vit_base en una T4, en una
   sola sesión y con una repetición por configuración.

Las dos líneas se refuerzan entre sí: cada dataset tiene un **perfil de I/O distinto** (tamaño,
formato, decodificación), y el I/O es justo donde el predictor falla más (+130 % en vit_tiny).

### Acoplamiento a BigEarthNet

| Sitio | Qué asume |
|---|---|
| `src/data/dataset.py` (ahora `bigearthnet.py`) | 19 clases CORINE, TIFF por banda (rasterio), `/10000`, normalización ImageNet, rotaciones de 90° |
| `src/models/vit.py` | 3 canales de entrada (no pasa `in_chans`) |
| `src/training/trainer.py`, `metrics.py` | Multi-etiqueta fijo: `sigmoid > 0.5`, F1 macro, búsqueda de umbral |
| `src/training/losses.py`, `decorators/confusion.py`, `scripts/eval.py` | Importan o duplican la lista de clases |
| `src/benchmark/benchmarker.py`, `checker.py` | Etiquetas sintéticas de 19 clases a 224² |
| `src/performance_model.py` | `N_FULL_TRAIN`, proporción val/train de 0,51, prior de calidad anclado a BigEarthNet |
| `src/web/` | ~35 referencias a nombres o grupos de clases |

### Limitaciones del predictor (`src/performance_model.py`)

1. **Una sola MFU (0,17)**, calibrada con vit_base en T4 y aplicada a toda GPU y modelo (de ahí el
   −24 % en ResNet50 y el +18 % en vit_large).
2. **Tabla cerrada de 6 modelos:** con cualquier otro, `predict()` devuelve `None`. *(Resuelto en la
   fase 5: perfilado automático; las activaciones siguen sin calibrar fuera de la tabla.)*
3. **`r_io` en img/s de BigEarthNet:** mezcla disco, decodificación y transforms, y no modela
   `num_workers` ni la caché de RAM.
4. **Sincronización simplificada:** `8P/β` por batch, sin solapamiento con el backward, sin factor
   de ring all-reduce, sin latencia y sin multinodo.
5. **Constantes ad hoc** calibradas con un solo run (heterogéneo `/50` y `×2.0`, MP `×0.25`,
   NFS φ=1,3, potencia MP 0,83).
6. **Solo estimaciones puntuales**, sin intervalos de incertidumbre.
7. **El prior de calidad (F1)** no se transfiere a otros datasets.

## Fases

### Fase 1 — Higiene (en curso)
- [x] Etiqueta `v1.0-tfg` + GitHub Release.
- [x] Ramas locales mergeadas eliminadas; borrado automático de ramas al mergear activado en GitHub.
- [x] Ramas remotas ya mergeadas eliminadas (solo quedan `main` y `develop`).
- [x] `AGENTS.md` + `CLAUDE.md` mínimo + `CHANGELOG.md`; el antiguo `CLAUDE.md` repartido en `docs/`.
- [ ] (Opcional) Integración con Zenodo para tener un DOI de cada release.

### Fase 2 — Abstracción dataset/tarea (hecha, 17/10/2026)
- [x] `DatasetSpec` (clases, canales, tarea, lector, transforms) + registro elegible con `data.name`.
- [x] `Task` (multi-etiqueta / multiclase / binaria): define la pérdida, la activación, las
      métricas y si hay búsqueda de umbral.
- [x] BigEarthNet como primera implementación, con **tests de caracterización** y comparación
      en GPU de las cuatro estrategias con resultados idénticos.
- [x] Quitar las listas de clases duplicadas (eval, confusion, web) y leerlas de la ficha.
- [x] `in_chans` configurable en el modelo (las 12 bandas necesitan además un lector multibanda).
- [x] Corregido: `model.dropout` del YAML no se aplicaba (la cabeza usaba siempre 0.1).
- [x] Corregido: el builder no pasaba la pérdida configurada al trainer heterogéneo (`pos_weight` se perdía).
- [ ] La web (galería y estadísticas del dataset) sigue siendo específica de BigEarthNet.
- [ ] El benchmark aún toma de BigEarthNet los tamaños de split por defecto y el perfil de I/O
      (ficheros `*.tif`); se generaliza junto con el modelo de I/O en la fase 5.

### Fase 3 — Nuevos datasets (en curso)
- [x] MedMNIST: BreastMNIST (binaria, cáncer de mama), DermaMNIST (multiclase, melanoma),
      PathMNIST (multiclase, cáncer colorrectal) y ChestMNIST (multi-etiqueta); descarga con MD5,
      configs y comparación con las referencias oficiales ([ficha](datasets/medmnist.md)).
- [x] AUC macro en la evaluación de test (métrica principal en imagen médica).
- [x] El umbral óptimo se elige en **validación** (se guarda en el checkpoint) y `eval.py` lo
      aplica a test; antes lo buscaba sobre el propio test (el 0,631 del TFG está aclarado en el
      README; reevaluar el checkpoint v4 en Verode con el umbral de validación, que fue 0,40).
- [x] `select_by: auc` y AUC de validación en cada época (`val_auc` en el CSV); los configs de
      MedMNIST lo usan.
- [ ] Repeticiones (≥3 semillas) y AUC también en validación para los runs de MedMNIST.
- [ ] Siguiente dataset médico grande: NIH ChestX-ray14 a resolución completa o PatchCamelyon.
- [ ] NIH ChestX-ray14 (multi-etiqueta, 14 clases, PNG grandes: I/O dominado por la decodificación).
- [ ] Un dataset de cáncer: PatchCamelyon (binaria, HDF5, parches de 96²) o ISIC (multiclase).
- [ ] Una ficha por dataset en `docs/datasets/`.

### Fase 4 — Instrumentación y registro de medidas
- [ ] Esquema estándar de cada medida: huella de hardware y software, configuración, throughput
      estable, pico de VRAM, potencia y **espera del DataLoader** (un decorador nuevo).
- [ ] Script de barrido con sondas cortas (N pasos en régimen estable) para muchas configuraciones.
- [ ] Entorno portable para HPC (contenedor Apptainer) y plantillas Slurm multinodo.

### Fase 5 — Predictor v2 (caja gris)
- [x] FLOPs y parámetros automáticos para cualquier modelo de timm (`torch.utils.flop_counter`):
      `src/model_profile.py`; la memoria de activaciones de estos modelos se estima por regla
      (sin calibrar). Detalle en [performance_model.md §7](performance_model.md).
- [ ] Modelo de I/O separado: almacenamiento (bytes/s), decodificación en CPU × workers y caché.
- [ ] Comunicación: ring all-reduce, latencia, solapamiento y multinodo.
- [ ] Calibración por microbenchmarks en la máquina destino; intervalos de incertidumbre.
- [ ] Validación con calibración y validación separadas (dejando fuera GPU, modelo o dataset) y baselines (lineal, Amdahl, roofline).

### Fase 6 — Campaña en máquinas nuevas y publicación
- [ ] Barridos con ≥3 repeticiones por configuración.
- [ ] Revisión del trabajo relacionado (Paleo, Daydream, Habitat…).
- [ ] Dataset de medidas y herramienta publicados con DOI.

### Ideas aparcadas
FSDP/ZeRO · pipeline con micro-batches (GPipe/1F1B) · gradient checkpointing · GPUs heterogéneas en
el predictor · formato de lectura rápida (LMDB/WebDataset) · focal loss a escala en VERODE.
