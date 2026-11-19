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
y opcionales `mosaics`, `splits`, `image_size`, `color_jitter` y `providers` (abajo). Config de ejemplo: `configs/train_panda.yaml`.

La ficha declara las clases **ordenadas** (`ordinal=True`): las métricas incluyen el kappa
cuadrático ponderado (columna `val_qwk`) y el config elige el checkpoint con `select_by: qwk`.
`model.img_size` debe coincidir con `data.image_size` (512 en el config); si no, timm falla en el
primer lote.

**Subgrupos por centro:** el Dataset expone `data_provider` (y `groups()`) por muestra. Con
`--monitor`, el monitor escribe `groups.npz` ([inspeccion.md](../inspeccion.md)) alineado con el
subconjunto monitorizado, y la vista *Subgroups* mide si el modelo rinde distinto en cada centro.

**Aumentos de color (`data.color_jitter`):** la tinción H&E varía entre hospitales y escáneres
(Tellez et al., 2019). Con `color_jitter: s` (número ≥ 0; 0 = desactivado, por defecto), el split
`train` aplica `ColorJitter(brillo=s, contraste=s, saturación=s, matiz=s/4)` sobre el tensor en
[0, 1], antes de normalizar; val y test no cambian. Con 0 el resultado es idéntico al anterior.
Config: `configs/train_panda_color.yaml` (`color_jitter: 0.2`). Como la fábrica de transformaciones
solo recibe el split, `_build` reconstruye ahí la cadena de train con la intensidad del config.

**Generalización entre centros (`data.providers`):** diccionario opcional con claves
`train`/`val`/`test` y, de valor, una lista de centros (`karolinska`, `radboud`). Filtra la
partición de `splits.csv` (no la rehace); un split sin clave usa todos los centros, y
`class_counts` (y por tanto `pos_weight: auto`) respeta el filtro. Un nombre de centro o de split
desconocido da un error claro. Experimento: entrenar en un hospital y evaluar en el otro, con
`configs/train_panda_radboud.yaml` (train/val Radboud, test Karolinska) y
`configs/train_panda_karolinska.yaml` (al revés); la caída de QWK respecto al test mixto mide la
sensibilidad al centro.

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
- El QWK por centro se calculó con un script temporal sobre el checkpoint; ahora lo da
  `eval.py` (tabla por grupo) y la vista *Subgroups* (`groups.npz`).

## Mejoras y generalización entre hospitales (18-19/11/2026)

Mismas condiciones que la línea base (ViT-S/16, 512 px, `--precision amp --monitor`, RTX 3060 Ti).
Cada run se evalúa en test con el checkpoint de su mejor época de validación y
`eval --save-predictions`; las tablas salen de esas predicciones (`src/analysis/clinical.py`).

**Conclusión (con tres semillas): ni la pérdida ordinal ni el color mejoran el QWK de forma
clara a 512 px.** Con una sola semilla la ordinal parecía mejorar +0,025 con un intervalo
*bootstrap* del test que excluía el 0. Ese intervalo solo recoge la variación del test, no la de
volver a entrenar, y al repetir la ventaja desapareció. Desde ahora cada comparación se hace con
tres semillas.

**Base frente a ordinal, tres semillas** (`training.seed` 42, 43 y 44; el resto, igual):

| Variante | Runs (semillas 42, 43, 44) | QWK test por semilla | Media ± desv. | Exactitud media | ±1 grado medio |
|---|---|---|---|---|---|
| Base (`train_panda.yaml`) | `02112026_214728`, `19112026_172712`, `19112026_181517` | 0,781 · 0,804 · 0,797 | 0,794 ± 0,012 | 0,636 | 0,884 |
| Ordinal (`train_panda_ordinal.yaml`) | `18112026_203820`, `19112026_175114`, `19112026_183927` | 0,807 · 0,809 · 0,792 | 0,802 ± 0,009 | 0,633 | 0,887 |

- Ordinal − base por semilla: +0,025, +0,005 y −0,005 (media +0,008). *Bootstrap* emparejado del
  test por semilla: [+0,003; +0,048], [−0,013; +0,022] y [−0,027; +0,017].
- La desviación entre semillas de la base (0,012) es del orden de la diferencia: a esta
  resolución el límite parece estar en la entrada (16 tiles reducidos a la mitad), no en la pérdida.

**Primera ronda, una semilla (42)** (la última columna es el *bootstrap* emparejado del test,
2000 remuestreos de biopsias):

| Modelo (config) | Run | Mejor época | QWK test | Exactitud | ±1 grado | AUC | Karolinska | Radboud | ΔQWK frente a la base [IC 95 %] |
|---|---|---|---|---|---|---|---|---|---|
| Base (`train_panda.yaml`) | `02112026_214728` | 9 | 0,781 | 0,609 | 0,880 | 0,887 | 0,745 | 0,760 | — |
| Ordinal (`train_panda_ordinal.yaml`) | `18112026_203820` | 13 | 0,807 | 0,644 | 0,886 | 0,892 | 0,774 | 0,784 | +0,025 [+0,003; +0,048] |
| Color (`train_panda_color.yaml`) | `18112026_215618` | 7 | 0,787 | 0,637 | 0,884 | 0,889 | 0,764 | 0,755 | +0,005 [−0,018; +0,031] |
| Ordinal + color (`train_panda_ordinal_color.yaml`) | `19112026_145034` | 15 | **0,812** | 0,654 | 0,889 | 0,896 | 0,764 | 0,802 | +0,031 [+0,007; +0,056] |

- Con una semilla, la ordinal y la ordinal + color parecían mejorar (intervalo sin el 0); las
  semillas 43 y 44 no lo confirman (tabla anterior).
- **El color no aporta nada medible**: frente a la ordinal sola, ordinal + color da +0,005
  [−0,017; +0,028]. Además, el `ColorJitter` (sobre todo el tono) va en CPU, a ~0,2 s por imagen
  de 512² y núcleo, y alarga la época de 82 s a ~200 s, aun con `num_workers: 14` (la máquina tiene
  8 núcleos físicos). Los dos runs con color se lanzaron con `num_workers: 14`; las configs
  se quedan en 4, como la base, para que cada variante cambie una sola cosa.
- En ordinal + color la mejor época es la última (15): podría ganar algo con más épocas.

**Cortes clínicos en test** (`eval.py`, tareas ordinales; sensibilidad y especificidad con la
clase predicha, AUC con la probabilidad acumulada de grado ≥ k). Media de las tres semillas:

| Variante (3 semillas) | ISUP ≥ 1 sens. | ISUP ≥ 1 espec. | ISUP ≥ 1 AUC | ISUP ≥ 2 sens. | ISUP ≥ 2 espec. | ISUP ≥ 2 AUC |
|---|---|---|---|---|---|---|
| Base | 0,913 | 0,781 | 0,941 | 0,824 | 0,889 | 0,930 |
| Ordinal | 0,902 | 0,828 | 0,945 | 0,810 | 0,904 | 0,931 |

La ordinal sube la especificidad de ISUP ≥ 1 (0,78 → 0,83) con casi la misma sensibilidad. Por
run, semilla 42:

| Modelo | ISUP ≥ 1 sens. | ISUP ≥ 1 espec. | ISUP ≥ 1 AUC | ISUP ≥ 2 sens. | ISUP ≥ 2 espec. | ISUP ≥ 2 AUC |
|---|---|---|---|---|---|---|
| Base | 0,924 | 0,718 | 0,938 | 0,849 | 0,856 | 0,927 |
| Ordinal | 0,908 | 0,841 | 0,950 | 0,847 | 0,875 | 0,935 |
| Color | 0,899 | 0,838 | 0,944 | 0,801 | 0,914 | 0,929 |
| Ordinal + color | 0,899 | 0,871 | 0,952 | 0,838 | 0,909 | 0,933 |

Referencia: Bulten et al. (2020), a resolución completa, AUC 0,99 (ISUP ≥ 1) y 0,98 (ISUP ≥ 2) en
su test ([research/medicion_clinica_prostata.md](../research/medicion_clinica_prostata.md)).

**Entrenar en un hospital y medir en el otro** (`data.providers`, una semilla, la 42; el test del
propio hospital se evalúa con un YAML que solo cambia `providers` de test; las variantes ordinal y
color son `train_panda_{radboud,karolinska}.yaml` con `loss: ce_emd` o `data.color_jitter: 0.2`):

| Entrenado con | Variante (run) | Train / val | QWK en su hospital | QWK en el otro |
|---|---|---|---|---|
| Radboud | Base (`19112026_154325`) | 3612 / 774 | 0,810 | 0,547 (Karolinska) |
| Radboud | Ordinal (`19112026_190023`) | 3612 / 774 | 0,813 | 0,479 |
| Radboud | Color (`19112026_192633`) | 3612 / 774 | 0,809 | 0,510 |
| Karolinska | Base (`19112026_155555`) | 3821 / 819 | 0,754 | 0,374 (Radboud) |
| Karolinska | Ordinal (`19112026_191255`) | 3821 / 819 | 0,740 | 0,381 |
| Karolinska | Color (`19112026_195324`) | 3821 / 819 | 0,759 | 0,451 |

- En su hospital rinden como el modelo base entrenado con los dos (0,760 y 0,745), con la mitad
  de datos. En el otro, el QWK cae 0,26 y 0,38 puntos.
- Ni la ordinal ni el color evitan la caída. El color ayuda de Karolinska a Radboud (+0,08) y
  empeora en el sentido contrario (−0,04); con una semilla no se separa del azar.
- La literatura ve lo mismo: Radboud → Karolinska 0,247 ± 0,138 (Gustafsson y Rantalainen, UNI +
  ABMIL) y 0,240-0,364 (Nishio et al. 2023) ([research/panda.md](../research/panda.md)).
- La caída mezcla tres causas que este experimento no separa: tinción y escáner, distribución de
  grados (Karolinska, bajos; Radboud, altos) y criterio de etiquetado de cada centro.
- Consecuencia para un dataset clínico nuevo: no basta con el modelo entrenado en PANDA; hay que
  ajustarlo con datos del hospital y medirlo por separado en ellos.

## Riesgos

- **Sesgo de centro:** Karolinska concentra los grados bajos y Radboud los altos, y tiñen
  distinto. Un modelo puede acertar por el color y no por el patrón; hay que revisarlo con
  *Subgroups*. Confirmado (19/11/2026): entrenado en un solo hospital, el QWK cae 0,26-0,38 en
  el otro (ver la sección anterior).
- **Etiquetas ruidosas:** el grado ISUP de entrenamiento tiene discrepancias conocidas entre
  patólogos; ni el techo ni la comparación con la literatura son exactos.
- **Métrica:** la oficial es el kappa cuadrático ponderado (QWK). De momento se entrena como
  multiclase (entropía cruzada) y se mide el QWK; la pérdida ordinal `ce_emd`
  (`configs/train_panda_ordinal.yaml`) tiene en cuenta el orden de los grados.
