# Roadmap posterior al TFG

> Plan de trabajo desde octubre de 2026, acordado tras la reunión con el tutor. Se avanza sin
> prisa, una fase cada vez. Objetivo a medio plazo: una **publicación** sobre la predicción del
> rendimiento del entrenamiento distribuido.

## Estado actual

> Actualizado el 25/10/2026. Se reescribe al cerrar cada bloque de trabajo (ver
> `CLAUDE.md`, "Continuidad entre sesiones"); el hook de inicio de sesión de Claude Code lo muestra.

**Reunión con el tutor (20/10/2026):** se le presentaron la etiqueta `v1.0-tfg`, el diseño
multi-dataset, MedMNIST y el trabajo con agentes de IA. **El tutor enviará el dataset médico
concreto** en el que centrar el trabajo; hasta entonces, MedMNIST sirve de banco de pruebas y el
diseño debe seguir admitiendo datasets nuevos sin tocar el entrenamiento. Pidió además **analizar
con más detalle la inferencia en las trazas**, para ver en tiempo real las decisiones del modelo
durante el entrenamiento, posiblemente desde la web. Auditoría y plan en "Auditoría de
observabilidad (25/10/2026)" y en la fase 4.

**Hecho:**
- Fases 1 (higiene) y 2 (abstracción de dataset y tarea, verificada en GPU con las cuatro
  estrategias).
- Fase 3, primer bloque: MedMNIST (cuatro datasets médicos) con **3 semillas**. DermaMNIST y
  PathMNIST superan a todas las referencias oficiales; BreastMNIST y ChestMNIST, dentro del rango.
  AUC en validación y test, selección por AUC, umbral elegido en validación, `--seed` y
  `scripts/summarize_runs.py`.
- Predictor: perfilado automático de cualquier modelo de timm (fase 5, primer punto), revisado
  por el agente revisor y corregido.
- Web: AUC de validación y filtro por dataset. Estado del arte sobre predicción del rendimiento
  (`docs/research/`).
- Arnés de trabajo con IA (agentes, skills, hooks, permisos, lint en CI y pre-commit).
- Corregidos los fallos de `model.dropout` (todos los runs del TFG usaron 0.1) y del heterogéneo
  con `pos_weight`.
- Limpieza de trazas (deuda de la auditoría): docstrings de `plot`/`confusion` al día,
  `LayerHooksDecorator` al logger, tests de los metric reporters, `psutil` declarado y
  `--trace deep` con `batch-table` ya alimenta a `batch-monitor`.

**En curso (fase 4, observabilidad):**
- Flujo de eventos estructurado (`events_*.jsonl`), la base común de la vista en vivo y de las
  medidas del predictor ([esquema](eventos.md)).
- Pestaña *Live* de la web, que sigue un entrenamiento en curso leyendo ese flujo.

**Siguiente paso recomendado:** el panel fijo de validación (probabilidades por época de un
conjunto fijo de imágenes), las predicciones por muestra en test con calibración y curvas ROC/PR,
y después `paravit predict` y `paravit explain` (Grad-CAM). Orden y motivos en la auditoría.

**Decisiones pendientes del autor o del tutor:**
- **El dataset médico definitivo** (lo enviará el tutor). Si fuera de segmentación, detección o
  3D, haría falta ampliar la tarea y el modelo (ver los límites en
  [datasets/README.md](datasets/README.md)).
- Si se acepta `pytorch-grad-cam` (MIT) como dependencia para los mapas de explicación.
- Elegir el enfoque de la contribución entre los tres de
  [`docs/research/estado_del_arte_prediccion.md`](research/estado_del_arte_prediccion.md).
- Reevaluar el checkpoint v4 del TFG en Verode con el umbral de validación (fue 0,40).
- Máquinas nuevas a las que dará acceso el tutor (aún sin concretar).
- En cada clon nuevo, instalar el pre-commit una vez (ya instalado en el equipo local):
  `uvx pre-commit install --hook-type pre-commit --hook-type commit-msg`.

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

## Auditoría de observabilidad (25/10/2026)

Petición del tutor: ver con detalle qué decide el modelo durante el entrenamiento, en tiempo real
y desde la web. Inventario del código y revisión del sector en
[`research/observabilidad_entrenamiento.md`](research/observabilidad_entrenamiento.md).

**Diagnóstico:**

1. **No se guarda nada por muestra.** Las predicciones de validación se reducen a métricas
   globales, por clase y a la matriz de confusión; `eval.py` tampoco guarda predicciones por
   imagen. No hay forma de ver *qué* imágenes falla el modelo ni con qué confianza.
2. **No hay inferencia fuera del entrenamiento.** No existe un comando para clasificar imágenes
   sueltas ni para explicar una predicción (Grad-CAM, *attention rollout*).
3. **Todo es texto y CSV, sin formato estructurado.** La web reconstruye cada run con
   expresiones regulares sobre el log; las claves en español son un contrato frágil.
4. **La web no es "en vivo".** Lee los ficheros al interactuar o cuando caduca la caché (30-60 s);
   `refresh_interval` está definido pero nadie lo usa, y el monitor del sistema se retiró.
5. **Deuda menor:** docstrings que prometen PNG que no se generan, `LayerHooksDecorator` que solo
   imprime por consola, decoradores sin tests, `tensorboard` declarado y sin uso (se mantiene para un futuro exportador), y `psutil`
   usado y sin declarar.

**Decisión de diseño:** un único **flujo de eventos** (`events_*.jsonl`, solo rank 0, una línea
por evento y versionado) como fuente estructurada de cada run. Sirve a la vez para la vista en
vivo, para el análisis por muestra (eventos de panel) y para las medidas de la fase 4
(espera del DataLoader, potencia por GPU). Convive con los logs actuales, que siguen siendo el
contrato con la web existente. Los exportadores a TensorBoard o MLflow, si llegan, serán otros
consumidores del mismo flujo. W&B y ClearML se descartan por licencia e infraestructura.

**Orden:** (1) flujo de eventos y vista *Live*; (2) panel fijo de validación, predicciones por
muestra en test, calibración (ECE) y ROC/PR, con su pestaña en la web; (3) `paravit predict`;
(4) `paravit explain` con Grad-CAM fuera del bucle medido (los mapas se generan en el terminal y
la web solo los muestra). Lo que dependa del tipo de problema se resuelve en la `Task`, de modo
que funcione con cualquier dataset, incluido el que envíe el tutor.

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
- [x] Repeticiones (3 semillas) y AUC en validación para MedMNIST: Derma y Path superan a todas las
      referencias oficiales; Breast y Chest, dentro del rango ([ficha](datasets/medmnist.md)).
- [ ] Siguiente dataset médico grande: NIH ChestX-ray14 a resolución completa o PatchCamelyon.
- [ ] NIH ChestX-ray14 (multi-etiqueta, 14 clases, PNG grandes: I/O dominado por la decodificación).
- [ ] Un dataset de cáncer: PatchCamelyon (binaria, HDF5, parches de 96²) o ISIC (multiclase).
- [ ] Una ficha por dataset en `docs/datasets/`.

### Fase 4 — Instrumentación, observabilidad e inferencia
- [ ] Flujo de eventos `events_*.jsonl` (esquema v1 en [eventos.md](eventos.md)): inicio con
      configuración y huella del entorno, épocas, batches, parada temprana y fin (también si falla).
- [ ] Web: pestaña *Live* que sigue un run en curso con lectura incremental del flujo.
- [x] Limpieza de la deuda de las trazas (docstrings, `LayerHooksDecorator`, *batch hooks* con
      `batch-table`, dependencias).
- [ ] Panel fijo de validación: probabilidades de un conjunto fijo de imágenes en cada época
      (eventos `panel`) para ver cómo evolucionan las decisiones del modelo.
- [ ] Predicciones por muestra en test (`eval.py --save-predictions`), calibración (ECE, diagrama
      de fiabilidad), curvas ROC/PR y peores errores, con su pestaña en la web.
- [ ] `paravit predict`: clasificar imágenes sueltas o una carpeta con un checkpoint.
- [ ] `paravit explain`: mapas Grad-CAM del panel fijo fuera del bucle medido; la web los muestra.
- [ ] Esquema estándar de cada medida: huella de hardware y software, configuración, throughput
      estable, pico de VRAM, potencia y **espera del DataLoader** (nuevos eventos del flujo).
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
