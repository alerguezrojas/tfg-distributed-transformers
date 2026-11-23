# Entrenar en Kaggle

Kaggle ofrece GPU gratuitas (2 × NVIDIA T4 de 16 GB, unas 30 h a la semana y sesiones de hasta
12 h) y tiene PANDA como datos de su competición, así que no hay que subir los 400 GB de
imágenes. Cada carpeta de aquí es un *kernel* (script) de Kaggle:

| Carpeta | Máquina | Qué hace | Salida |
|---|---|---|---|
| `panda_mosaicos/` | CPU | Mosaicos de las 10.616 biopsias (nivel 1, 16 trozos de 256 px) y `splits.csv`, con la misma receta que `scripts/prepare_panda.py` en local | `mosaics_l1_16x256/`, `splits.csv`, `train.csv` |
| `panda_train/` | 2 × T4 | Mide el ancho de banda de all-reduce entre las dos T4 (`scripts/measure_allreduce.py`), entrena el lote `BATCH` de `BATCHES` en `run.py` (DDP en las dos T4 o, si el experimento lo indica, una sola T4) y evalúa cada uno en test con `--save-predictions` | `results/<nombre>/`, `logs/kaggle/` (solo lo que escribe el kernel), `timings.csv`, `allreduce.log` |
| `benchmark_t4/` | 1 × T4 | `paravit benchmark` (lotes sintéticos, sin datos ni I/O) de las 13 arquitecturas de la criba de PANDA a 512 px con AMP: datos fuera de muestra para el predictor en una GPU distinta de la local | `throughput.csv`, `bench/<modelo>.log`, `gpu.txt` |

`panda_train` usa como entrada la salida de `panda_mosaicos`, que solo hay que ejecutar una vez.
El `md5` de `splits.csv` sale en el log y debe coincidir con el local
(`76dcfaff305092dbe98b8387df19a894`): así los runs de Kaggle y los locales usan la misma partición.

## Cómo se lanza

Hace falta la CLI de Kaggle con un token (`~/.kaggle/`, nunca en el repo) y haber aceptado las
reglas de la competición PANDA en la web de Kaggle.

```bash
scripts/kaggle_push.sh kaggle/panda_mosaicos     # una vez
scripts/kaggle_push.sh kaggle/panda_train        # cada tanda de experimentos
scripts/kaggle_push.sh kaggle/benchmark_t4       # throughput en T4 (~20 min, sin datos)
uvx --with six kaggle kernels status <usuario>/paravit-panda-train
uvx --with six kaggle kernels output <usuario>/paravit-panda-train -p <carpeta>
```

Los `kernel-metadata.json` dicen `{user}` en lugar del usuario de Kaggle, para no publicarlo en el
repo; `scripts/kaggle_push.sh` lo rellena en una copia temporal con el usuario de la CLI (o con
`KAGGLE_USERNAME`). Los kernels son privados y clonan la rama `develop` del repositorio público
(variable `PARAVIT_REF` en `run.py` para fijar otra rama o etiqueta).

## Reglas

- Cada experimento es `(nombre, config base, cambios[, estrategia])`; la estrategia es `ddp` (por
  defecto, 8 imágenes por GPU) o `single` (una T4 con las 16; `single()` en `run.py`).
- Mismas condiciones que en local salvo lo que se quiera medir: lote global 16 (8 por GPU),
  semilla del YAML, `--precision amp --monitor`. El experimento `panda_ddp_512` es el control
  que comprueba que Kaggle con DDP reproduce los resultados locales a 512 px.
- Cada envío ejecuta un lote (`BATCH`). Una sesión dura como mucho 12 h y un run a 1024 px tarda
  ~2,9 h, así que caben tres o cuatro. Los lotes ya ejecutados se quedan en `BATCHES` como registro
  de lo que se lanzó, y los nombres no se repiten entre lotes.
- Los resultados se descargan (`kaggle kernels output <usuario>/paravit-panda-train -p <dir del
  SSD>`), se copian sus logs a `logs/kaggle/` (con `scripts/backfill_run_manifests.py` para los
  manifiestos) y se documentan en la ficha del dataset (`docs/datasets/panda.md`), igual que los
  runs locales. Si la evaluación en test falla, se repite en local con el `best.pt` descargado y
  la config de `results/<nombre>/config.yaml` cambiando `data.root`.
- `timings.csv` guarda el tiempo real de cada entrenamiento para contrastar el predictor
  (`paravit estimate`), separando la calibración de la validación.
