# Datasets

ParaViT-Lab no asume un dataset concreto: cada uno se describe con una **ficha**
(`DatasetSpec`, en `src/data/spec.py`) que se registra con un nombre. El config elige el
dataset con `data.name`; si no lo indica, se usa `bigearthnet`, así que los configs
anteriores siguen funcionando sin cambios.

| Nombre (`data.name`) | Tarea | Clases | Canales | Ficha |
|---|---|---|---|---|
| `bigearthnet` (por defecto) | multi-etiqueta | 19 | 3 | [bigearthnet.md](bigearthnet.md) |
| `medmnist_breast` | binaria | 1 | 1 | [medmnist.md](medmnist.md) |
| `medmnist_derma` | multiclase | 7 | 3 | [medmnist.md](medmnist.md) |
| `medmnist_path` | multiclase | 9 | 3 | [medmnist.md](medmnist.md) |
| `medmnist_chest` | multi-etiqueta | 14 | 1 | [medmnist.md](medmnist.md) |

## Qué decide la ficha y qué decide la tarea

**La ficha del dataset** dice qué hay en los datos:
- los nombres de las clases, en el orden de las salidas del modelo;
- el número de canales de entrada;
- el tipo de tarea;
- las claves obligatorias del bloque `data:` del YAML;
- cómo se lee cada split y con qué transformaciones;
- opcionalmente, el recuento de positivos por clase, que habilita `pos_weight: auto`.

**La tarea** (`src/training/tasks.py`) dice cómo se aprende y se mide ese tipo de problema:

| Tarea | Ejemplo | Salida | Pérdida (`training.loss`) | Accuracy | Umbral óptimo |
|---|---|---|---|---|---|
| `multilabel` | BigEarthNet, radiografías de tórax | sigmoide por clase | `bce` (por defecto), `focal` | media por etiqueta | sí |
| `binary` | ¿hay tumor? | una sigmoide | `bce` (por defecto), `focal` | media por etiqueta | sí |
| `multiclass` | tipo de lesión de piel | softmax | `ce` (por defecto) | acierto exacto | no |

Las etiquetas viajan siempre como tensores `float` de forma `(N, C)`: multi-hot en
multi-etiqueta, one-hot en multiclase y `(N, 1)` en binaria. Así el mixup, los trainers y
los decoradores no necesitan saber qué tarea es.

## Cómo añadir un dataset

1. **Crear `src/data/<nombre>.py`** con un `torch.utils.data.Dataset` que devuelva
   `(imagen, etiqueta)`: la imagen como tensor `(canales, H, W)` y la etiqueta como `float`
   de forma `(C,)`, en el formato de su tarea.
2. **Definir y registrar la ficha** al final del mismo fichero:

   ```python
   from src.data.registry import register_dataset
   from src.data.spec import DatasetSpec

   def _build(data_cfg: dict, split: str, transform):
       return MiDataset(data_cfg["root"], split=split, transform=transform)

   MI_DATASET = register_dataset(DatasetSpec(
       name="mi_dataset",
       description="Descripción corta",
       classes=("clase_a", "clase_b", "clase_c"),
       task="multiclass",
       in_channels=1,
       required_keys=("root",),
       build=_build,
       transforms=get_transforms,      # split -> transformaciones de ese split
   ))
   ```

3. **Añadir el módulo a `_BUILTIN_MODULES`** en `src/data/registry.py`.
4. **Escribir su ficha** en `docs/datasets/<nombre>.md` (origen, licencia, descarga,
   estructura, splits) y añadirla a la tabla de arriba.
5. **Crear un config** con `data.name: mi_dataset` y las claves de `required_keys`.
   `model.num_classes` es opcional: si se indica, el validador comprueba que coincide con
   la ficha.

No hace falta tocar el entrenamiento, las estrategias distribuidas ni los decoradores.
El test `tests/integration/test_new_dataset_end_to_end.py` comprueba precisamente eso, con
un dataset multiclase de un canal y otro binario.

## Limitaciones actuales

- El **DDP heterogéneo** (GPU+CPU) normaliza una pérdida sigmoide elemento a elemento, así
  que solo admite tareas `multilabel` y `binary`.
- La pérdida **focal** y `pos_weight` existen solo para tareas sigmoide.
- La **web** sigue siendo específica de BigEarthNet en la galería de clases y las
  estadísticas del dataset; las curvas y métricas de cualquier run sí se ven bien. Los runs de
  otros datasets se guardan en `logs/{env}/<dataset>/{mode}/{model}/` y la web los etiqueta con
  el nombre del dataset.
- El **predictor analítico** (`src/performance_model.py`) sigue calibrado con BigEarthNet
  (tamaños de split, rendimiento de lectura de TIFF, prior de calidad). Generalizarlo es la
  fase 5 del [roadmap](../roadmap.md).
