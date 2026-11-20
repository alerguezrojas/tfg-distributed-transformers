# Entrenar en Kaggle

Kaggle ofrece GPU gratuitas (2 × NVIDIA T4 de 16 GB, unas 30 h a la semana y sesiones de hasta
12 h) y tiene PANDA como datos de su competición, así que no hay que subir los 400 GB de
imágenes. Cada carpeta de aquí es un *kernel* (script) de Kaggle:

| Carpeta | Máquina | Qué hace | Salida |
|---|---|---|---|
| `panda_mosaicos/` | CPU | Mosaicos de las 10.616 biopsias (nivel 1, 16 trozos de 256 px) y `splits.csv`, con la misma receta que `scripts/prepare_panda.py` en local | `mosaics_l1_16x256/`, `splits.csv`, `train.csv` |
| `panda_train/` | 2 × T4 | Entrena con DDP la lista `EXPERIMENTS` de `run.py` y evalúa cada uno en test con `--save-predictions` | `results/<nombre>/`, `logs/`, `timings.csv` |

`panda_train` usa como entrada la salida de `panda_mosaicos`, que solo hay que ejecutar una vez.
El `md5` de `splits.csv` sale en el log y debe coincidir con el local
(`76dcfaff305092dbe98b8387df19a894`): así los runs de Kaggle y los locales usan la misma partición.

## Cómo se lanza

Hace falta la CLI de Kaggle con un token (`~/.kaggle/`, nunca en el repo) y haber aceptado las
reglas de la competición PANDA en la web de Kaggle.

```bash
scripts/kaggle_push.sh kaggle/panda_mosaicos     # una vez
scripts/kaggle_push.sh kaggle/panda_train        # cada tanda de experimentos
uvx --with six kaggle kernels status <usuario>/paravit-panda-train
uvx --with six kaggle kernels output <usuario>/paravit-panda-train -p <carpeta>
```

Los `kernel-metadata.json` dicen `{user}` en lugar del usuario de Kaggle, para no publicarlo en el
repo; `scripts/kaggle_push.sh` lo rellena en una copia temporal con el usuario de la CLI (o con
`KAGGLE_USERNAME`). Los kernels son privados y clonan la rama `develop` del repositorio público
(variable `PARAVIT_REF` en `run.py` para fijar otra rama o etiqueta).

## Reglas

- Mismas condiciones que en local salvo lo que se quiera medir: lote global 16 (8 por GPU),
  semilla del YAML, `--precision amp --monitor`. El experimento `panda_ddp_512` es el control
  que comprueba que Kaggle con DDP reproduce los resultados locales a 512 px.
- Los resultados se descargan, se copian sus logs a `logs/kaggle/` y se documentan en la ficha
  del dataset (`docs/datasets/panda.md`), igual que los runs locales.
- `timings.csv` guarda el tiempo real de cada entrenamiento para contrastar el predictor
  (`paravit estimate`), separando la calibración de la validación.
