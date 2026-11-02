# PANDA (biopsias de próstata)

*Prostate cANcer graDe Assessment*: 10 616 biopsias de próstata teñidas con H&E, escaneadas como
imágenes de lámina completa (WSI), con el **grado ISUP 0-5** de cada una, el hospital de origen
(`radboud` o `karolinska`) y la puntuación de Gleason. Es el dataset del tutor para el trabajo de
continuación. Tarea **multiclase** de 6 clases (`ISUP 0` … `ISUP 5`), 3 canales.

> Bulten, W., Kartasalo, K., Chen, P.-H. C., et al. *Artificial intelligence for diagnosis and
> Gleason grading of prostate cancer: the PANDA challenge*. Nature Medicine 28, 154-163 (2022).
> <https://doi.org/10.1038/s41591-021-01620-2>. Competición de Kaggle:
> <https://www.kaggle.com/c/prostate-cancer-grade-assessment>.

**Licencia:** CC BY-NC-SA 4.0 (solo uso no comercial; las obras derivadas se comparten con la
misma licencia). Revisión bibliográfica completa en [../research/panda.md](../research/panda.md).

## Formato original

`train.csv` (`image_id`, `data_provider`, `isup_grade`, `gleason_score`), `train_images/<id>.tiff`
y `train_label_masks/<id>_mask.tiff`. Cada TIFF es piramidal con tres niveles (reducción 1, 4 y
16), de unos 25-33 k px de alto en el nivel 0. El conjunto de test de Kaggle está oculto, así que
la partición es propia. El zip pesa 193 GB.

## Descarga

```bash
# Token de Kaggle en ~/.kaggle/access_token
uvx --with six kaggle competitions download -c prostate-cancer-grade-assessment -p /media/alejandro/SSD/datasets/panda
```

## Preparación (offline)

Una WSI no cabe en un ViT. Se reduce a **un mosaico por biopsia**: nivel 1 (4× reducido), cortado
en tiles de 256 px, se conservan los 16 con más tejido (los de menor suma de píxeles, es decir,
menos blanco) y se pegan en un mosaico 4×4 de 1024 px. Si la imagen tiene menos de 16 tiles se
completa con tiles blancos.

```bash
uv run python scripts/prepare_panda.py --root /media/alejandro/SSD/datasets/panda --workers 8 --make-splits
```

- Sale `<root>/mosaics_l1_16x256/<image_id>.jpg` (JPEG calidad 95 sin submuestreo de color: ocupa
  unos 0,7 MB por mosaico frente a 2,0 MB en PNG, ~8 GB en total frente a ~21 GB).
- Es idempotente (salta los mosaicos ya hechos), paralelo (`--workers`) y tolerante a fallos: los
  TIFF ilegibles o ausentes se anotan en `<out>/errors.csv` y el proceso sigue.
- Parámetros: `--level 1 --tile 256 --n-tiles 16` (`n-tiles` debe ser un cuadrado perfecto).
- Medido sobre dos biopsias de muestra: 0,2 s por biopsia, con un 79-90 % de tejido en el mosaico.
- La lógica (relleno, tiles, selección por tejido, montaje) está en `src/data/wsi.py`, sin torch.

## Partición

`--make-splits` escribe `<root>/splits.csv` (`image_id`, `split`): **70/15/15 estratificada por
(centro, grado ISUP)**, semilla 42, determinista y independiente del orden de las filas. Deja
fuera los ids de `errors.csv`. `train.csv` **no trae identificador de paciente**, así que la
partición es **por biopsia**: si un paciente aportó varias, pueden caer en train y test a la vez
y el resultado saldría algo inflado. Es una limitación conocida del dataset público.

## Lector y configuración

`src/data/panda.py` registra `panda`. El Dataset carga el mosaico y lo **reescala a
`data.image_size`** (512 por defecto) y devuelve la imagen normalizada y un objetivo *one-hot*
`(6,)`. En entrenamiento aplica giros horizontales y verticales y rotaciones de 90° (la
histología no tiene orientación). Claves de `data:`: `root` (obligatoria, con `train.csv`),
y opcionales `mosaics`, `splits` e `image_size`. Config de ejemplo: `configs/train_panda.yaml`.

Con `image_size: 512` falta que el ViT acepte esa entrada (`model.img_size` de timm); ese soporte
va aparte. Con 224 funciona ya.

**Subgrupos por centro:** el Dataset expone `data_provider` (y `groups()`) por muestra. El monitor
aún no escribe `groups.npz` ([inspeccion.md](../inspeccion.md)); cuando lo haga, la vista
*Subgroups* podrá medir si el modelo aprende el centro.

## Riesgos

- **Sesgo de centro:** Karolinska concentra los grados bajos y Radboud los altos, y tiñen
  distinto. Un modelo puede acertar por el color y no por el patrón; hay que revisarlo con
  *Subgroups*.
- **Etiquetas ruidosas:** el grado ISUP de entrenamiento tiene discrepancias conocidas entre
  patólogos; ni el techo ni la comparación con la literatura son exactos.
- **Métrica:** la oficial es el kappa cuadrático ponderado (QWK); aún no está implementada.
