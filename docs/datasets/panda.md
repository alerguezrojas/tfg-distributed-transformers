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
- Promediar las probabilidades de las tres semillas sube el QWK de la base de 0,794 (media) a 0,819.
  Se reproduce (0,8193, comprobado el 21/11/2026) con `paravit.py eval --config configs/train_panda.yaml --checkpoint <run 42>/best.pt
  --checkpoint <run 43>/best.pt --checkpoint <run 44>/best.pt --save-predictions RUTA.npz`
  (ver [`guia_de_uso.md`](../guia_de_uso.md)).

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

## Resolución 1024 px en Kaggle (20/11/2026)

**Pregunta:** ¿el límite a 512 px está en la entrada? Los mosaicos se generan a 1024 px y se
reducían a la mitad. A 1024 px ViT-S/16 ve 4096 *tokens* por biopsia y no cabe en la RTX 3060 Ti
con lote 16, así que se entrena en Kaggle (2 × T4, DDP, 8 por GPU = lote global 16, `amp`,
`--monitor`) con los kernels de [`kaggle/`](../../kaggle/README.md). Los mosaicos se regeneran en
Kaggle con la misma receta, y el kernel comprueba que el md5 de `splits.csv` coincide con el
local: misma partición. La evaluación en test se hizo en la GPU local con el `best.pt` de cada run
(en Kaggle falló por el prefijo `module.` de los checkpoints DDP, corregido en el PR #214).

| Run (`logs/kaggle/panda/ddp/`) | Variante | Mejor val | QWK test | Exact. | ±1 grado | Karolinska | Radboud | ISUP ≥ 1 sens. / espec. / AUC | ISUP ≥ 2 sens. / espec. / AUC | min/época |
|---|---|---|---|---|---|---|---|---|---|---|
| `20112026_163243` | Base, 512 px (control) | 0,805 | 0,801 | 0,636 | 0,886 | 0,755 | 0,786 | 0,895 / 0,836 / 0,942 | 0,761 / 0,944 / 0,926 | 2,4 |
| `20112026_171003` | Base, 1024 px | 0,803 | **0,821** | 0,667 | 0,896 | 0,762 | 0,821 | 0,925 / 0,834 / 0,956 | 0,856 / 0,885 / 0,936 | 11,0 |
| `20112026_200237` | Ordinal, 1024 px | 0,813 | 0,806 | 0,662 | 0,894 | 0,745 | 0,806 | 0,940 / 0,776 / 0,958 | 0,831 / 0,910 / 0,936 | 11,0 |

Semilla 42 en los tres. *Bootstrap* emparejado del test (2000 remuestreos):

- Control: Kaggle 512 − local 512 (semilla 42) = +0,019 [−0,005; +0,044]. El 0,801 cae dentro de
  las tres semillas locales (0,794 ± 0,012), así que **los runs de Kaggle se pueden comparar con
  los locales**.
- Base 1024 − base 512 (Kaggle) = +0,021 [−0,005; +0,048]. Frente a la media local, +0,027 (más de
  dos desviaciones entre semillas). Es prometedor, pero **con una semilla no basta** (la ordinal a
  512 enseñó que una semilla engaña).
- Ordinal 1024 − ordinal 512 (local, semilla 42) = −0,001 [−0,029; +0,027]: a 1024 la ordinal no
  gana nada.
- Por hospital, la base a 1024 sube sobre todo en Radboud (0,786 → 0,821), que tiene los grados
  altos, y la sensibilidad de ISUP ≥ 2 pasa de 0,76 a 0,86 con menos especificidad (0,94 → 0,89).
- El mejor valor de validación no ordena bien el test: la ordinal a 1024 es la mejor en
  validación (0,813) y la peor de las dos a 1024 en test.

Coste: a 1024 px cada época tarda ~11 min y un run de 15 épocas, ~2,9 h (frente a 37 min a 512).
La GPU 0 llega al 98 % de memoria, porque el Grad-CAM del monitor se queda sin memoria: es solo un
aviso, el run sigue y se guardan las predicciones.

### Tres semillas a 1024 px (21/11/2026)

Semillas 43 y 44 de la base a 1024 px (lote `seeds_1024` del kernel), lanzadas tras el PR #216.
Hasta ese arreglo, `DistributedSampler` barajaba siempre con la semilla 0, así que en DDP las
semillas solo cambiaban la inicialización y los aumentos, no el orden de los datos. La semilla 42
de la tabla anterior se entrenó así, lo que equivale a un orden de datos más; los runs a 512 px de
una GPU no se ven afectados. Esta vez la evaluación en test se hizo en Kaggle sin errores.

| Run (`logs/kaggle/panda/ddp/`) | Semilla | Mejor val | QWK test | Exact. | ±1 grado | Karolinska | Radboud | ISUP ≥ 1 sens. / espec. / AUC | ISUP ≥ 2 sens. / espec. / AUC |
|---|---|---|---|---|---|---|---|---|---|
| `20112026_171003` | 42 | 0,803 | 0,821 | 0,667 | 0,896 | 0,762 | 0,821 | 0,925 / 0,834 / 0,956 | 0,856 / 0,885 / 0,936 |
| `20112026_233503` | 43 | 0,804 | 0,787 | 0,665 | 0,889 | 0,732 | 0,785 | 0,916 / 0,820 / 0,956 | 0,834 / 0,886 / 0,927 |
| `21112026_024444` | 44 | 0,814 | 0,805 | 0,669 | 0,892 | 0,748 | 0,802 | 0,924 / 0,822 / 0,954 | 0,852 / 0,869 / 0,930 |

| Resolución (3 semillas) | QWK test | Karolinska | Radboud | Exact. | Conjunto de las 3 (QWK) |
|---|---|---|---|---|---|
| 512 px (local, una GPU) | 0,794 ± 0,012 | 0,743 ± 0,005 | 0,783 ± 0,020 | 0,636 | 0,819 |
| 1024 px (Kaggle, DDP) | 0,804 ± 0,017 | 0,747 ± 0,015 | 0,803 ± 0,018 | 0,667 | 0,815 |

*Bootstrap* emparejado del test (2000 remuestreos, el mismo índice para las seis semillas) sobre
la media de las tres semillas, 1024 − 512:

- **QWK: +0,010 [−0,010; +0,030].** No es significativo. Por semilla, la diferencia va de −0,017
  a +0,040: el 0,821 de la semilla 42 fue una semilla buena, no el efecto de la resolución.
- **Exactitud: +0,030 [+0,013; +0,048]** y **AUC de ISUP ≥ 1 (cáncer sí o no): +0,014
  [+0,006; +0,022].** Las dos son significativas. A 1024 px el modelo acierta más veces el grado
  exacto y separa mejor tejido sano de tumoral, pero el QWK, que pesa poco los errores de un
  grado, apenas lo recoge.
- Promediar las probabilidades de las tres semillas (conjunto) da 0,815-0,819 a las dos
  resoluciones (diferencia −0,004 [−0,029; +0,022]). Las semillas a 512 px se equivocan en casos
  más distintos entre sí y el conjunto compensa la menor resolución.

Las dos semillas nuevas se entrenaron en Kaggle con DDP y las de 512 px en local con una GPU.
Esa diferencia no explica el resultado: el control a 512 px en Kaggle (0,801, exactitud 0,636,
AUC 0,942) coincide con las semillas locales.

**Conclusión:** subir a 1024 px cuadruplica el coste (11 min por época frente a 2,4 en las T4) a
cambio de unos 3 puntos de exactitud y nada de QWK. La resolución no es el cuello de botella del
QWK. Lo siguiente con más margen es el ruido de etiquetas y la diferencia entre hospitales
(Karolinska sigue en 0,75 a las dos resoluciones). La literatura de PANDA apunta a lo mismo: los
mejores equipos limpiaron las etiquetas ruidosas (sobre todo las de Radboud) y usaron 36-64
mosaicos en lugar de 16 ([research/panda.md](../research/panda.md)).

## Etiquetas ruidosas: OUSM (21/11/2026)

**Pregunta:** si el límite está en las etiquetas (el grado ISUP tiene discrepancias conocidas
entre patólogos, sobre todo en Radboud), ¿ayuda ignorar en cada lote las muestras que el modelo no
consigue ajustar? Se prueba OUSM (`training.ousm_k`, Xue et al. 2019; lo usó el 6.º equipo de
PANDA, [research/panda.md](../research/panda.md)): `configs/train_panda_ousm.yaml`, la base a
512 px en la GPU local con k = 2 de cada lote de 16 (12,5 %). Tres semillas, con el mismo test.

| Run (`logs/local/panda/single/`) | Semilla | Mejor val | QWK test | Exact. | ±1 grado | Karolinska | Radboud | ISUP ≥ 1 sens. / espec. / AUC | ISUP ≥ 2 sens. / espec. / AUC |
|---|---|---|---|---|---|---|---|---|---|
| `21112026_061527` | 42 | 0,792 | 0,796 | 0,647 | 0,877 | 0,758 | 0,776 | 0,884 / 0,864 / 0,940 | 0,782 / 0,934 / 0,929 |
| `21112026_064000` | 43 | 0,813 | 0,787 | 0,630 | 0,884 | 0,747 | 0,765 | 0,876 / 0,861 / 0,940 | 0,786 / 0,911 / 0,929 |
| `21112026_070409` | 44 | 0,801 | 0,797 | 0,652 | 0,883 | 0,759 | 0,777 | 0,903 / 0,838 / 0,943 | 0,802 / 0,912 / 0,926 |

| 512 px, tres semillas | QWK test | Karolinska | Radboud | Conjunto de las 3 (QWK) |
|---|---|---|---|---|
| Base | 0,794 ± 0,012 | 0,743 ± 0,005 | 0,783 ± 0,020 | 0,819 |
| OUSM (k = 2) | 0,793 ± 0,006 | 0,755 ± 0,007 | 0,773 ± 0,006 | 0,809 |

*Bootstrap* emparejado (2000 remuestreos) de la media de las tres semillas, OUSM − base:

- **QWK −0,001 [−0,014; +0,013]**, exactitud +0,006 [−0,006; +0,019] y AUC de ISUP ≥ 1
  −0,000 [−0,005; +0,005]. Con k = 2, **OUSM no cambia el resultado**.
- Cambia el reparto. Karolinska sube un poco y Radboud baja, justo el hospital con más ruido y
  con los grados altos. Además, la especificidad de ISUP ≥ 1 sube (0,78 → 0,85) y la
  sensibilidad baja (0,91 → 0,89).
- Lectura (no comprobada): con mosaicos de 16 tiles, las biopsias de mayor pérdida no son solo
  las mal etiquetadas. También lo son las difíciles de verdad (grados altos, poco tumor en el
  mosaico), y descartarlas quita señal útil. Las semillas varían menos (± 0,006), pero con tres
  no se puede afirmar.

**Conclusión:** OUSM en su forma simple no ayuda aquí. Los equipos que ganaron con limpieza de
etiquetas usaron predicciones **fuera de pliegue** (validación cruzada): una muestra se marca como
ruidosa por un modelo que no la ha visto, no por la pérdida del lote en curso. Promediar tres
semillas sigue siendo la mejora más barata y fiable (0,819 frente a 0,794 de media).

## Limpieza de etiquetas fuera de pliegue (21/11/2026)

Es la receta del 1.er puesto ([research/panda.md](../research/panda.md)): cada biopsia de train la
juzga un modelo que no la ha visto. Con OUSM, en cambio, la decide la pérdida del lote en curso.
La lógica está en `src/data/label_noise.py` y el procedimiento, en `scripts/panda_oof.py`:

1. `folds` reparte train en 5 pliegues, estratificados por hospital y grado y fijos con una
   semilla. Por cada pliegue escribe una partición (`splits_fold<k>.csv`: el pliegue es `val`, el
   resto de train es `train`, y val y test originales pasan a `excluded`) y su config
   (`data.splits`).
2. Se entrena un modelo por pliegue (la base a 512 px, semilla 42) y se guardan sus predicciones
   sobre el pliegue apartado (`eval --split val --save-predictions`).
3. `clean` junta las predicciones fuera de pliegue (`oof.csv`) y marca las biopsias cuyo grado
   esperado (`Σ k·p_k`) se aleja de la etiqueta más de 1,5 grados. Escribe `splits_clean.csv`, con
   las marcadas como `excluded`: **validación y test no se tocan**, así que el resultado se compara
   con la base sobre el mismo test.

El umbral de 1,5 se fijó antes de ver ningún resultado. Hay dos sesgos conocidos, y los dos son pequeños:
- El checkpoint de cada pliegue se elige por su QWK en ese mismo pliegue, lo que acerca un poco
  las predicciones a las etiquetas. Las etiquetas no entran en los pesos, así que el efecto es
  conservador: se marcan algo menos biopsias, no más.
- El grado esperado tiende al centro de la escala, de modo que las etiquetas 0 y 5 pueden
  marcarse más. Por eso hay que mirar la tasa de marcadas por grado y por hospital que imprime
  `clean`.

Los ficheros viven en `/media/alejandro/SSD/paravit-oof/` y la cola, en
`~/experimentos_paravit/cola_oof.sh`. Después se entrenan tres semillas con `train_clean.yaml` y
se comparan con las tres semillas base sobre el mismo test.

**Qué se marcó.** Los cinco modelos de pliegue tienen un QWK de 0,773-0,804 en su pliegue apartado
(0,781 sobre las 7.433 biopsias de train juntas, con una exactitud de 0,617): son tan buenos como
el modelo final, ni más ni menos. Con el umbral de 1,5 se marcan **873 biopsias (11,7 %)**, entre
el 11 % y el 12,5 % en cada pliegue. El reparto no es uniforme:

| Grado ISUP | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Karolinska | 3,3 % | 2,8 % | 10,9 % | 20,3 % | **33,8 %** | 26,1 % |
| Radboud | 15,4 % | 10,9 % | 9,7 % | 19,1 % | 14,5 % | 17,8 % |

En total se marca el 8,8 % de Karolinska y el 14,9 % de Radboud, el hospital con fama de ruidoso.
Pero se va **uno de cada cinco ejemplos de los grados 3-5** (de 870, 875 y 857 quedan 701, 683 y
690), frente a un 5-7 % de los grados 0-1. En el 42 % de las marcadas el modelo predice un grado
mayor que la etiqueta y en el 58 %, uno menor.

**Resultado en test** (QWK, mismo test que la base):

| Run | QWK | Exactitud | Karolinska | Radboud | AUC ISUP ≥ 1 |
|---|---|---|---|---|---|
| Limpio, semilla 42 | 0,787 | 0,630 | 0,765 | 0,750 | 0,940 |
| Limpio, semilla 43 | 0,784 | 0,638 | 0,727 | 0,777 | 0,932 |
| Limpio, semilla 44 | 0,765 | 0,609 | 0,731 | 0,730 | 0,932 |
| **Limpio, media** | **0,779 ± 0,012** | 0,626 | 0,741 ± 0,021 | 0,752 ± 0,024 | 0,935 |
| Base, media | 0,794 ± 0,012 | 0,636 | 0,743 ± 0,005 | 0,783 ± 0,020 | 0,941 |
| Conjunto limpio (3) | 0,782 | 0,645 | 0,729 | 0,766 | 0,943 |
| Conjunto base (3) | 0,819 | 0,674 | 0,772 | 0,810 | 0,949 |

*Bootstrap* emparejado (2.000 remuestreos del test) de limpio − base:

- media de las tres semillas: **−0,015 [−0,032; +0,000]** (el 97 % de los remuestreos dan una
  diferencia negativa);
- conjunto de tres: **−0,038 [−0,060; −0,017]**, significativo;
- por semilla: +0,005, −0,019 y −0,032;
- exactitud: −0,011 [−0,025; +0,004];
- AUC de ISUP ≥ 1: −0,006 [−0,012; −0,001].

**Conclusión: la limpieza empeora el resultado, sobre todo en Radboud (−0,031 de media).**
Karolinska queda igual. La especificidad del corte ISUP ≥ 1 sube (0,79-0,83 frente a
0,72-0,81), pero a costa de la sensibilidad y del QWK. La explicación más probable es que, con un
modelo de QWK ≈ 0,78 (el 8.º puesto de Kaggle llegó a 0,920 en el test privado), la discrepancia fuera de pliegue
mide sobre todo los **errores del modelo** en los grados altos, que son los difíciles, y no el
ruido de las etiquetas. Al quitar el 20 % de esos ejemplos, el modelo aprende peor justo lo que ya
le costaba. No se prueban otros umbrales: elegirlos mirando el test sería hacer trampa, y la señal
es clara. **Con esto se cierra la línea de etiquetas ruidosas:** ni OUSM ni la limpieza fuera de
pliegue ayudan con este modelo. El cuello de botella es el modelo, no las etiquetas. Lo único que
sube el QWK sigue siendo promediar semillas.

Los logs están en `logs/local/panda/single/vit_small_patch16_224/`: los cinco pliegues son
`train_21112026_{093959,095337,101332,103327,104931}` y los tres runs limpios,
`train_21112026_{110930,112826,115012}`. La web los distingue por la etiqueta
`splits splits_fold<k>` o `splits splits_clean`.

## Criba de modelos (22/11/2026)

**Pregunta:** con la receta de la base fija (`configs/train_panda.yaml`: 512 px, 15 épocas, lr
0,0001, LLRD 0,75, batch 16, AMP, `select_by: qwk`), ¿hay otra arquitectura que sume más que la
resolución o la limpieza de etiquetas? Desde el PR #234 el LLRD funciona con cualquier modelo de
timm (reparto normalizado a 12 bloques), así que todos se entrenan igual; solo cambia `--model`.
Primero una semilla (42) por modelo en la RTX 3060 Ti local; los que destacan pasan a tres. La
comparación es con `paravit compare-preds` sobre los `test_preds.npz` (*bootstrap* emparejado de
2.000 remuestreos).

| Modelo (timm) | Pesos | Parám. | Min/run | Mejor val | QWK test | Exact. | Karolinska | Radboud | QWK − base [IC 95 %] |
|---|---|---|---|---|---|---|---|---|---|
| ViT-S/16 (base) | in21k → in1k | 22,0 M | 24 | 0,814 | 0,781 | 0,609 | 0,745 | 0,760 | — |
| `swin_tiny_patch4_window7_224` | in1k | 27,5 M | 38 | 0,826 | **0,830** | 0,701 | 0,762 | 0,837 | **+0,048 [+0,021; +0,076]** |
| `convnext_tiny.fb_in22k` | in22k | 27,8 M | 26 | 0,794 | 0,798 | 0,628 | 0,772 | 0,770 | +0,017 [−0,010; +0,045] |
| `eva02_small_patch14_224.mim_in22k` | MIM in22k | 21,6 M | 39 | 0,826 | 0,787 | 0,654 | 0,758 | 0,757 | +0,006 [−0,022; +0,035] |
| `deit3_small_patch16_224.fb_in22k_ft_in1k` | in22k → in1k | 21,7 M | 20 | 0,790 | 0,787 | 0,614 | 0,755 | 0,765 | +0,005 [−0,019; +0,032] |
| `tf_efficientnetv2_s` | in21k → in1k | 20,2 M | 30 | 0,776 | 0,783 | 0,641 | 0,758 | 0,752 | +0,002 [−0,030; +0,032] |
| `efficientnet_b0` | in1k | 4,0 M | 17 | 0,778 | 0,781 | 0,625 | 0,762 | 0,746 | +0,000 [−0,027; +0,028] |
| `resnet50` | in1k | 23,5 M | 24 | 0,745 | 0,706 | 0,589 | 0,663 | 0,674 | −0,075 [−0,109; −0,042] |

Semilla 42; la diferencia es frente a la base con la misma semilla. Con una semilla el intervalo no
recoge la variación entre semillas (± 0,012 en la base), así que solo Swin-T y ResNet-50 se salen
claramente.

- **Swin-T** es el único que mejora de forma clara, y no por unos pesos mejores: es el único
  *transformer* de la criba preentrenado solo en ImageNet-1k. La ganancia viene casi toda de
  Radboud (0,760 → 0,837), el hospital con los grados altos.
- **EVA-02-S** iguala a Swin-T en validación (0,826) pero no en test (0,787): el criterio de parada
  se ajustó a la validación. Lo mismo, en menor medida, en la base (0,814 → 0,781).
- **Las CNN** no ganan al ViT-S. EfficientNet-B0 iguala a la base con 5 veces menos parámetros y
  en 17 minutos; ResNet-50 se queda claramente por debajo (con la receta de los ViT; en el TFG se
  entrenaba sin LLRD).

Los dos mejores, con tres semillas (42, 43 y 44):

| 512 px, tres semillas | QWK test | Exact. | ±1 grado | AUC ISUP ≥ 1 | Karolinska | Radboud | Conjunto de las 3 (QWK) |
|---|---|---|---|---|---|---|---|
| Base (ViT-S) | 0,794 ± 0,012 | 0,636 ± 0,025 | 0,884 ± 0,008 | 0,941 ± 0,003 | 0,743 ± 0,005 | 0,783 ± 0,020 | 0,819 |
| **Swin-T** | **0,821 ± 0,011** | 0,694 ± 0,006 | 0,903 ± 0,004 | 0,958 ± 0,001 | 0,769 ± 0,016 | 0,817 ± 0,018 | 0,829 |
| ConvNeXt-T | 0,796 ± 0,009 | 0,648 ± 0,017 | 0,881 ± 0,010 | 0,950 ± 0,003 | 0,767 ± 0,007 | 0,773 ± 0,019 | 0,805 |

QWK de test por semilla: Swin-T 0,830, 0,808 y 0,825; ConvNeXt-T 0,798, 0,787 y 0,804.

*Bootstrap* emparejado, Swin-T − base:

- media de las tres semillas: **QWK +0,027 [+0,008; +0,046]**, exactitud +0,058
  [+0,039; +0,077], ±1 grado +0,019 [+0,007; +0,031] y AUC de ISUP ≥ 1 +0,017 [+0,010; +0,024];
  todo significativo;
- conjunto de tres: QWK +0,010 [−0,014; +0,034], no significativo; exactitud +0,031
  [+0,008; +0,055] y AUC +0,011 [+0,004; +0,018], sí.

ConvNeXt-T − base: QWK +0,002 [−0,017; +0,020] de media y −0,014 [−0,038; +0,010] en conjunto;
**no mejora** (la semilla 42 tuvo suerte).

**Conclusión:** **Swin-T es la primera mejora significativa del QWK** de PANDA en este proyecto: un
solo modelo (0,821) rinde como el conjunto de tres ViT-S (0,819), con un 60 % más de tiempo por
run. La resolución (+0,010 a 1.024 px, no significativo, 4,5 veces más cara) y las etiquetas
ruidosas (OUSM, limpieza fuera de pliegue) no lo consiguieron. Hipótesis sin comprobar: la
atención por ventanas locales y la jerarquía de escalas de Swin encajan mejor con un mosaico de
16 *tiles* de 128 px que la atención global de un ViT con parches de 16 px. Swin-T pasa a ser el
modelo de referencia a 512 px para las siguientes pruebas. En curso en Kaggle (2 × T4, DDP) los
modelos que no caben en 8 GB a 512 px: ViT-B, ConvNeXt-S, SwinV2-T, EfficientNet-B3 y MaxViT-T.

Logs: `logs/local/panda/single/<modelo>/train_22112026_*` (Swin-T: `160932`, `182151` y
`190007`; ConvNeXt-T: `152254`, `193826` y `200621`).

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
