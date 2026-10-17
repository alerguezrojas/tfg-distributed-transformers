---
name: nuevo-dataset
description: Añade un dataset nuevo al proyecto siguiendo el patrón DatasetSpec + registro — lector, ficha, config, tests, descarga verificada y documentación. Úsala cuando se quiera entrenar con un dataset que aún no está registrado.
---

Sigue `docs/datasets/README.md` (sección "Cómo añadir un dataset"). En resumen:

1. Investiga el dataset (agente **investigador** si hace falta): licencia, cita, splits oficiales,
   formato, tarea (multilabel / binary / multiclass), canales y resultados de referencia publicados.
2. Crea `src/data/<nombre>.py`: un `Dataset` que devuelva `(imagen (C,H,W), objetivo float (K,))`
   en el formato de su tarea, y registra su `DatasetSpec` (nombre en minúsculas). Añade el módulo
   a `_BUILTIN_MODULES` en `src/data/registry.py`.
3. Si hay descarga, hazla verificable (MD5/SHA) en `scripts/download_<nombre>.py`.
4. Tests sin red ni dataset real (ficheros sintéticos con el formato oficial): forma de las
   muestras, codificación de etiquetas, `class_counts` y splits.
5. Config `configs/<nombre>_<entorno>.yaml` con `data.name` y validación OK.
6. Ficha `docs/datasets/<nombre>.md` (origen, licencia, citas, descarga, referencias oficiales y
   resultados propios con semillas) y fila en la tabla de `docs/datasets/README.md`.
7. Entrena y evalúa una vez en test con el umbral de validación; resultados con ≥3 semillas
   (`--seed`, `scripts/summarize_runs.py`) antes de compararlos con la literatura.
8. Cierra con la skill **cerrar-bloque**.
