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
- Dataset completo (02/11/2026, 16 núcleos, `--workers 8`): 10.616 mosaicos, 0 errores, 8 GB,
  0,38 s por biopsia y 77 % de tejido de media. `splits.csv`: 7.433 / 1.593 / 1.590.
- La lógica (relleno, tiles, selección por tejido, montaje) está en `src/data/wsi.py`, sin torch.

## Partición

`--make-splits` escribe `<root>/splits.csv` (`image_id`, `split`): **70/15/15 estratificada por
(centro, grado ISUP)**, semilla 42, determinista y independiente del orden de las filas. Deja
fuera los ids de `errors.csv`. Dentro de cada estrato las biopsias se ordenan por una huella
(`sha256` de la semilla y el id) y se corta: el sitio de cada biopsia no depende de las demás, así
que quitar unas pocas (TIFF ilegibles) solo mueve las que quedan junto a un corte, en vez de
rebarajar todo el estrato. Aun así, **un modelo se evalúa con el mismo `splits.csv` con el que se
entrenó**. `train.csv` **no trae identificador de paciente**, así que la
partición es **por biopsia**: si un paciente aportó varias, pueden caer en train y test a la vez
y el resultado saldría algo inflado. Es una limitación conocida del dataset público.

## Lector y configuración

`src/data/panda.py` registra `panda`. El Dataset carga el mosaico y lo **reescala a
`data.image_size`** (512 por defecto) y devuelve la imagen normalizada y un objetivo *one-hot*
`(6,)`. En entrenamiento aplica giros horizontales y verticales y rotaciones de 90° (la
histología no tiene orientación). Claves de `data:`: `root` (obligatoria, con `train.csv`),
y opcionales `mosaics`, `splits` e `image_size`. Config de ejemplo: `configs/train_panda.yaml`.

La ficha declara las clases **ordenadas** (`ordinal=True`): las métricas incluyen el kappa
cuadrático ponderado (columna `val_qwk`) y el config elige el checkpoint con `select_by: qwk`.
`model.img_size` debe coincidir con `data.image_size` (512 en el config); si no, timm falla en el
primer lote.

**Subgrupos por centro:** el Dataset expone `data_provider` (y `groups()`) por muestra. El monitor
aún no escribe `groups.npz` ([inspeccion.md](../inspeccion.md)); cuando lo haga, la vista
*Subgroups* podrá medir si el modelo aprende el centro.

## Línea base (02/11/2026)

Primer entrenamiento, sin ajustes: `configs/train_panda.yaml` (ViT-S/16 preentrenado, 512 px,
batch 16, lr 0,0001, semilla 42, `select_by: qwk`) con `--precision amp --monitor` en la RTX 3060 Ti
local. 82 s por época; parada temprana en la época 14 (paciencia 5). Run:
`logs/local/panda/single/vit_small_patch16_224/*_02112026_214728`.

| Partición | QWK | Exactitud | F1 macro | AUC macro | ±1 grado |
|---|---|---|---|---|---|
| Validación (mejor época, la 9) | 0,814 | 0,613 | 0,585 | 0,888 | — |
| Test (checkpoint de la época 9) | **0,781** | 0,609 | 0,578 | 0,887 | 0,880 |
| Test, solo Karolinska (816) | 0,745 | 0,623 | — | — | — |
| Test, solo Radboud (774) | 0,760 | 0,594 | — | — | — |

- Los fallos se concentran en el grado vecino; el grado 5 se confunde con el 4 (31 %).
- El QWK de cada centro es menor que el global: parte del global puede venir de que los centros
  tienen distribuciones de grado distintas (sesgo de centro, ver Riesgos).
- La pérdida de validación sube desde la época 5 mientras el QWK se aplana en ~0,80: sobreajuste.
- Referencias: patólogos en hospitales externos 0,86-0,87 y ganadores de Kaggle 0,94
  ([research/panda.md](../research/panda.md)).
- El QWK por centro se calculó con un script temporal sobre el checkpoint; cuando el monitor
  escriba `groups.npz`, saldrá en la vista *Subgroups*.

## Riesgos

- **Sesgo de centro:** Karolinska concentra los grados bajos y Radboud los altos, y tiñen
  distinto. Un modelo puede acertar por el color y no por el patrón; hay que revisarlo con
  *Subgroups*.
- **Etiquetas ruidosas:** el grado ISUP de entrenamiento tiene discrepancias conocidas entre
  patólogos; ni el techo ni la comparación con la literatura son exactos.
- **Métrica:** la oficial es el kappa cuadrático ponderado (QWK). De momento se entrena como
  multiclase (entropía cruzada) y se mide el QWK; una pérdida ordinal queda como mejora.
