# PANDA (Prostate cANcer graDe Assessment): dataset, soluciones y encaje en ParaViT-Lab

> Revisión para decidir cómo incorporar PANDA como dataset médico de trabajo. Fecha: 02/11/2026.
> Cada fuente se ha abierto durante la revisión. Las cifras son **las que dan los autores** de
> cada fuente, con su configuración; la interpretación propia va marcada como tal. Lo que no se ha
> podido abrir o contrastar con una fuente primaria se indica como *no verificado*.

Resumen de dos líneas: PANDA es un problema de **clasificación ordinal (ISUP 0-5) a partir de
imágenes gigantes** (WSI). Los ganadores no usaron la imagen entera: la reducen a un mosaico de
unos pocos tiles de 128-256 px y entrenan una CNN con regresión. Eso encaja bien con un
preprocesado offline y un lector de imágenes pequeñas, sin tocar el bucle de entrenamiento.

## 1. Qué es el dataset

**Origen.** Biopsias de próstata teñidas con hematoxilina-eosina, digitalizadas, de dos centros:
Radboud University Medical Center (Nimega, Países Bajos) y Karolinska Institutet (Estocolmo,
Suecia). El *development set* tiene **10.616 biopsias** de 2.113 pacientes (5.160 de Radboud y
5.456 de Karolinska) y es lo que se publicó para investigación [1]. Las imágenes se obtuvieron con
cuatro modelos de escáner de tres fabricantes (3DHISTECH, Hamamatsu y Leica) [1]; la asignación
escáner-centro no se ha verificado. El reparto por centro y grado de `train.csv` (comprobado
en local) coincide con la Tabla 1 del artículo [1]:

| ISUP | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Karolinska | 1.925 | 1.814 | 668 | 317 | 481 | 251 |
| Radboud | 967 | 852 | 675 | 925 | 768 | 973 |

Lectura propia: Karolinska tiene el 69 % de sus biopsias en los grados 0-1; Radboud reparte casi
por igual. Centro y grado están correlacionados, de modo que un modelo puede acertar parte de la
tarea reconociendo el centro (sección 5).

**Etiquetas y referencia.** El grado ISUP del conjunto de desarrollo sale del informe clínico
rutinario: en Radboud, del informe de patología; en Karolinska, de un único uropatólogo. No hay
consenso de varios patólogos para el entrenamiento [1]. El proceso de generación de las máscaras figura en los
métodos suplementarios del artículo, que *no se han abierto*.

**Formato de la imagen.** Cada WSI es un TIFF multinivel. Según la descripción de Kaggle citada
por una fuente secundaria [9] (la página de datos de Kaggle no se puede abrir de forma automática),
hay **3 niveles cargables con reducción 1, 4 y 16** y los lados miden típicamente entre 5.000 y
40.000 píxeles. Otras fuentes coinciden en el nivel 0: unos 20.000 x 20.000 x 3, a 20x y
0,5 µm/píxel [8][11][12]; el tamaño total sin comprimir es de unos 347 GB según [8]. Los factores
reales pueden diferir ligeramente de 4 y 16 y es mejor leerlos del fichero (`level_downsamples`).
**Comprobado sobre dos TIFF reales** (02/11/2026, `tifffile`): un Radboud ISUP 4
(`0018ae58…`, nivel 0 de 25.344 × 5.888) y un Karolinska ISUP 1 (`003046e2…`, 33.024 × 23.552).
Ambos tienen exactamente 3 niveles con reducción 1, 4 y 16, teselas JPEG y resolución de
~0,49 µm/px (Radboud) y ~0,45 µm/px (Karolinska) según la etiqueta `XResolution`. El tejido
ocupa solo el 5-17 % del nivel más reducido. Las máscaras tienen los mismos tres niveles; el canal
0 de la máscara Radboud toma los valores {0, 1, 2, 4} y el de la Karolinska {0, 1, 2},
coherente con la tabla siguiente.

**Máscaras (`train_label_masks`).** Una por imagen, con sufijo `_mask.tiff`; no todas las imágenes
tienen máscara y pueden contener errores [8]. Los valores dependen del centro (descripción de
Kaggle, vía [9]; coincide con lo que citan otras fuentes secundarias):

| Valor | Radboud (glándulas individuales) | Karolinska (regiones) |
|---|---|---|
| 0 | fondo o desconocido | fondo o desconocido |
| 1 | estroma | tejido benigno (estroma y epitelio juntos) |
| 2 | epitelio sano (benigno) | tejido canceroso (estroma y epitelio juntos) |
| 3 | epitelio canceroso, Gleason 3 | no aparece |
| 4 | epitelio canceroso, Gleason 4 | no aparece |
| 5 | epitelio canceroso, Gleason 5 | no aparece |

**Licencia.** El artículo dice que los datos se publican "under a Creative Commons BY-SA-NC 4.0
license" y que hay que citar el artículo [1]. La página oficial del reto repite "CC BY-SA-NC 4.0"
y pide citar el artículo [2]; el nombre no es el estándar de Creative Commons (la combinación
habitual es CC BY-NC-SA 4.0: atribución, **no comercial**, compartir igual). El artículo habla de
"uso en investigación no comercial". No se han podido abrir las reglas de la competición en Kaggle,
que son otro texto contractual: *no verificado*. Para la publicación basta con uso académico y
atribución, pero conviene que el autor revise las reglas que aceptó al descargar.

**Cómo citarlo.**

> Bulten, W., Kartasalo, K., Chen, P.-H. C., Ström, P., Pinckaers, H., Nagpal, K., ... y el consorcio
> del reto PANDA. "Artificial intelligence for diagnosis and Gleason grading of prostate cancer: the
> PANDA challenge". *Nature Medicine* 28(1), 154-163 (2022). DOI: 10.1038/s41591-021-01620-2.

**¿Hay un test con etiquetas públicas?** No. El test de Kaggle está oculto. El artículo solo dice
que se publicó el *development set* [1]; el *tuning set* (393 biopsias), el *internal validation*
(545) y las dos validaciones externas (741 de EE. UU. y 330 de Europa) no figuran como publicados
(no se ha encontrado un enlace de descarga). Hace falta, por tanto, una **partición propia**
(sección 6).

## 2. Métrica oficial y resultados de referencia

**Métrica.** Kappa de Cohen cuadrático ponderado (QWK) entre el grado predicho y el de referencia
[1]. (La fórmula no se contrasta aquí; es la `cohen_kappa_score(..., weights="quadratic")` de
scikit-learn, interpretación propia.) Penaliza más los errores lejanos que los adyacentes.

**Leaderboard privado de Kaggle.** Cifras de los resúmenes que los propios equipos presentaron en el
taller del reto [3] (QWK): PND (1.º) 0,9409; Save The Prostate (2.º) 0,9377; BarelyBears (6.º)
0,9326; ChienYiChi (8.º) 0,9323; Iafoss (11.º) 0,9301. El repositorio de PND declara público 0,904
y privado 0,940 [4]. Son resultados contra el test oculto de Kaggle, de los mismos centros que el
entrenamiento. El artículo habla de un *tuning set* de 393 biopsias (fase de competición) y de una
validación interna de 545 [1]; [11] los trata como test público y privado, pero [1] no lo
confirma así. El primer clasificado puntuó menos en el público (0,904) que en el privado (0,940).

**Validación independiente y patólogos [1].**

- En la validación interna, los 15 algoritmos seleccionados alcanzaron un QWK medio de **0,931**
  (IC 95 %: 0,918-0,944) frente a los uropatólogos de referencia.
- En las externas: **0,862** (IC 0,840-0,884) en EE. UU. y **0,868** (IC 0,835-0,900) en Europa.
  Los autores atribuyen la caída a un cambio de distribución entre entrenamiento y validación
  externa; el error principal fue sobrediagnosticar casos benignos como ISUP 1.
- Frente a patólogos generales, en un subconjunto de 70 casos de la validación interna los
  algoritmos obtuvieron 0,876 y los patólogos 0,765; en 237 casos de EE. UU., 0,828 y 0,820
  (diferencia no significativa, P = 0,53).
- Acuerdo entre patólogos de referencia: 0,926 y 0,907 (pares) en subconjuntos de las validaciones
  interna y externa de EE. UU. [1].

Lectura propia: el techo del 0,93-0,94 de Kaggle mide acuerdo con **etiquetas del mismo centro y
proceso**; la cifra honesta de generalización es la externa (~0,86-0,87).

## 3. Cómo lo resolvieron las mejores soluciones

**Resumen de los autores del artículo [1].** Casi todos los equipos extraen primero parches de la
WSI, los pasan por una CNN y concatenan las características antes de la clasificación final. Varios
usaron entrenamiento de extremo a extremo con solo la etiqueta de la biopsia, limpieza automática de
etiquetas (eliminar o reetiquetar muestras en las que el patólogo parece haberse equivocado) y
*ensembles* de modelos diversos.

**Soluciones concretas** (de los repositorios de los propios equipos; la lectura es propia):

| Equipo (puesto privado) | Entrada | Modelo y pérdida | Ruido de etiquetas |
|---|---|---|---|
| PND (1.º) [4] | 64 tiles de 192 px del nivel 1 en entrenamiento; 36 tiles de 256 px en uno de los modelos en inferencia | EfficientNet-B0/B1 (`avgpool`, *mixup*) | EfficientNet-B1 en 5 pliegues; con las predicciones fuera de pliegue se eliminan muestras (umbral 16; un segundo modelo quita además el 20 % de las etiquetas de Radboud) |
| 6.º (analokmaus) [5] | 64 tiles de 224 px | SE-ResNeXt50, regresión ordinal con pérdida CORAL | OUSM (excluye las muestras de mayor pérdida del lote); 10 semillas para marcar ruido; mejor: ratio 0,10 y OUSM (k=1) |
| ChienYiChi (8.º) [6] | 16, 20 o 36 tiles de 256 px | EfficientNet-B0/B4, ResNeXt50 con NetVLAD; probaron clasificación, regresión y regresión ordinal | Las mejores corridas fueron de **regresión**; mejor QWK privado 0,920 (B0, 36 tiles) |
| Iafoss (11.º) [7] | *Concatenate tile pooling*: tiles de la imagen reducida, seleccionados por cantidad de tejido | CNN sobre mosaico; preentrenamiento a baja resolución | Autodestilación progresiva de etiquetas y eliminación de ruidosas [3] |

- **Preparación.** Ninguna trabaja con el nivel 0. Se parte de un nivel reducido (el 1, es decir,
  hasta 2 µm/píxel si el nivel 0 es de 0,5 µm; interpretación propia), se descartan los tiles sin
  tejido y se conservan N tiles de 128-256 px (N de 16 a 64) elegidos por cantidad de tejido. Los
  tiles se pasan por la CNN y se agregan (concatenación de mosaico, NetVLAD, atención) o se
  recomponen en una imagen rectangular que va a una CNN normal [3][4][5][6][7]. La variante
  iafoss `16x128x128` y la `36x256x256` son nombres de *notebooks* y conjuntos de Kaggle que no se han
  podido abrir (Kaggle bloquea la lectura); el contenido exacto de la selección *no está verificado*.
- **MIL y atención.** Save The Prostate usó MIL con selección de tiles por atención y con
  *Squeeze-and-Excite*; UCLA, un MIL en dos etapas de baja a alta resolución; ChienYiChi,
  atención + NetVLAD [3].
- **Tratamiento ordinal.** Predominan la **regresión** (MSE/pérdida cercana) con **umbrales
  optimizados** y CORAL; en la tabla de ChienYiChi la regresión superó a la clasificación [6]. La
  optimización de umbrales no aparece en las fuentes abiertas como tal: *no verificada*.
- **Ruido (Radboud).** Hay dos estrategias en las fuentes: (a) limpiar el entrenamiento con
  predicciones fuera de pliegue [4]; (b) pérdidas robustas (OUSM, *mixup*, pseudoetiquetas)
  [3][5]. PND eliminó adicionalmente un 20 % de etiquetas de Radboud en uno de sus modelos [4].
- **Preprocesados públicos en Kaggle.** Existen conjuntos de tiles (p. ej. los de iafoss), pero
  **no se ha podido verificar nombre, tamaño ni licencia**; Kaggle no se deja abrir. Se derivan de
  datos con licencia no comercial, así que heredarían esa restricción (interpretación propia).
  Recomendación: generar el preprocesado propio, que además se controla y se documenta.

## 4. Vision Transformers y modelos de fundación en PANDA

| Trabajo | Qué hace | Cifra de los autores |
|---|---|---|
| Grisi, Litjens y van der Laak, "Hierarchical Vision Transformers for Context-Aware Prostate Cancer Grading in Whole Slide Images", arXiv:2312.12619 (2023; *Medical Imaging meets NeurIPS*) [11] | H-ViT: parches de 256 px a 0,50 µm/px y regiones de 2.048 px, preentrenados con DINO sobre PANDA | Local H-ViT: QWK 0,950 (tuning), 0,904 (público), 0,904 (privado) en su Tabla 1; en el apéndice 0,9149 / 0,9170. Las dos tablas se contradicen en el privado; se citan ambas |
| Gustafsson y Rantalainen, "Evaluating Computational Pathology Foundation Models for Prostate Cancer Grading under Distribution Shifts", arXiv:2410.06723 [12] | 12 extractores congelados + ABMIL (sin TransMIL), parches de 256 px a 20x | UNI + ABMIL, validación cruzada interna: QWK 0,888 ± 0,013; **Radboud a Karolinska: 0,247 ± 0,138** |
| Butt et al., "Benchmarking multiple instance learning architectures from patches to pathology...", *Scientific Reports* 16, 11535 (2026) [13] | 6 MIL x 3 codificadores (ResNet50, CTransPath, UNI2) x 4 configuraciones de parche | Mejor (UNI2 + ILRA-MIL, 256 px con 50 % de solape): QWK 90,12 %, exactitud 78,75 % |

- Ninguno de ellos es un ViT entrenado de extremo a extremo sobre un mosaico de tiles reducido, que
  es lo que haría ParaViT-Lab. No hay resultados verificados de ese esquema. Es una **laguna
  útil**, no una referencia que replicar.
- TransMIL, HIPT y CTransPath: de los tres solo CTransPath aparece evaluado en PANDA en los trabajos
  abiertos aquí ([13], solo como codificador). No se han comprobado resultados de TransMIL ni de
  HIPT sobre PANDA en sus artículos originales: *no verificado*.
- Las cifras no son comparables entre sí: cambian el conjunto de test (Kaggle, interno, validación
  cruzada) y el protocolo.

## 5. Problemas conocidos

- **Etiquetas ruidosas.** Los propios autores indican la subjetividad de la gradación como problema
  clave y que varios equipos limpiaron etiquetas [1]. El proyecto eva excluye ~10 % de las
  biopsias por ruido siguiendo al 6.º puesto de Kaggle, y se queda con **9.555** WSI [8]
  (lista de exclusión concreta: *no verificada*, hay que pedirla a [5] o a [8]).
- **PANDA-PLUS.** Hopson et al., "PANDA-PLUS: Improved dataset of prostate whole slide images from
  PANDA Challenge with pixel-level expert annotations", *J Pathol Inform* 20 (2026), 100540,
  DOI 10.1016/j.jpi.2025.100540 [14]: **546** imágenes de PANDA con anotaciones a nivel de píxel,
  hechas por voluntarios y supervisadas por un patólogo. Los autores encuentran que sus grados son
  sistemáticamente más bajos que los de PANDA, sobre todo en los grados altos, lo que indica
  posibles errores en las etiquetas originales. No se ha comprobado dónde se descarga.
- **Sesgo de centro.** Gustafsson y Rantalainen [12] muestran que, entrenando con Radboud, el QWK
  en Karolinska cae a 0,247 y que el espacio de características separa por centro más que por
  grado (sin normalización de tinción ni aumento de color, para conservar el desplazamiento).
  Nishio et al., *Cancers* 15(5), 1535 (2023), DOI 10.3390/cancers15051535 [15]: entrenando con
  los 5.160 de Radboud y probando con los 5.456 de Karolinska, QWK 0,364 con *label distribution
  learning* y 0,240 sin él. Los autores del reto atribuyen la caída en validación externa al
  desplazamiento de distribución [1]. *Shortcut* por centro es una **interpretación propia**
  compatible con estas cifras, no una afirmación directa de [1].
- **Slides a eliminar.** Aparte del ~10 % ruidoso (arriba), [12] descarta las WSI sin tejido
  extraíble (10.616 a 10.433). No se ha verificado una lista oficial de slides defectuosas.
- **Desbalance.** Muy desigual por grado y centro (tabla de la sección 1).

## 6. Implicaciones para ParaViT-Lab

Todo lo que sigue es **recomendación propia**, apoyada en las fuentes anteriores.

**Encaje técnico (sin tocar `EpochController.fit()`).**

1. **Preprocesado offline** (script aparte, fuera del bucle): leer cada TIFF en el nivel
   reducido, quitar fondo, elegir N tiles por cantidad de tejido y guardar **un mosaico por
   biopsia** (p. ej. 6x6 tiles de 256 px = 1.536 px, o 4x4 de 128 px) redimensionado a 224x224 o
   en un formato listo (PNG/JPEG de calidad alta, o `.npy`). Así el coste de la WSI de 193 GB se
   paga una vez y el entrenamiento ve imágenes pequeñas y regulares, justo el caso de los ViT de
   timm (224x224). Estimación propia: 10.616 mosaicos de 1-2 MB ocuparían unos 10-20 GB.
   Hay que decidir mosaico en una sola imagen (más simple; los ganadores lo hicieron) frente a
   bolsas de tiles con MIL (mejor para ViT pero necesita otro modelo y otra pérdida).
2. **Lector en la `DatasetSpec`**: una ficha `panda` con 3 canales, 6 clases y un lector que
   devuelva el mosaico preprocesado desde disco. Es el mecanismo previsto en
   [`docs/datasets/README.md`](../datasets/README.md); sin cambios en el entrenamiento.
3. **Tarea.** Empezar con **multiclase (softmax + entropía cruzada)** porque reutiliza lo existente
   y permite un primer entrenamiento. En paralelo, una `Task` **ordinal/regresión** (regresión
   sobre ISUP con umbrales optimizados en validación), que es lo que usaron los ganadores. Es una
   decisión de diseño con coste real: la pérdida, la salida (1 neurona frente a 6) y la métrica
   cambian.
4. **Métrica QWK** añadida a las métricas de la tarea (con `weights="quadratic"`), junto a
   exactitud y matriz de confusión. Es la que permite comparar con las cifras de la sección 2.
5. **Partición propia** train/val/test **estratificada por centro y grado** (por ejemplo 70/15/15),
   fijada con semilla y guardada en un CSV versionado. Dos consideraciones: no hay test público
   con etiquetas, y para estimar la generalización haría falta también una partición
   **por centro** (entrenar con uno y probar con el otro) como experimento aparte. Además, hay
   varias biopsias por paciente (2.113 pacientes para 10.616 biopsias [1]) y el artículo separó
   por paciente; **el reparto debería hacerse por paciente** para evitar fugas, pero `train.csv`
   no trae el identificador de paciente (*no verificado* si se puede reconstruir).
6. **Vista Subgroups.** Usar `data_provider` como subgrupo, y opcionalmente `isup_grade`: el
   `groups.npz` lo recoge. Es una buena demostración de sesgo de centro con la herramienta del
   propio proyecto.

**Para el predictor de rendimiento (objetivo del artículo).** PANDA aporta un caso con carga de
datos pesada (lectura de TIFF enormes) y otro ligero (mosaicos ya preparados): sirve para
comprobar qué parte del coste es entrada/salida. Conviene medir ambos y no mezclar el coste del
preprocesado con el del entrenamiento.

**Decisiones a tomar con el tutor.**

1. Mosaico único con ViT supervisado de timm, o MIL sobre bolsas de tiles (más cercano a la
   bibliografía actual, pero otro modelo).
2. Multiclase u ordinal como tarea principal; si es ordinal, política de umbrales.
3. Nivel, tamaño y número de tiles del preprocesado (hay que probar sobre una muestra: no hay
   cifra oficial que lo fije).
4. Qué hacer con las etiquetas ruidosas: usar las 10.616, aplicar la exclusión de ~10 % [8] o
   dejarlo como experimento. Se aconseja empezar con todas y documentarlo.
5. Qué parte del artículo es el dataset y cuál el predictor: PANDA como caso de uso médico frente
   a objeto del estudio de sesgo de centro.
6. Licencia: confirmar el alcance de "no comercial" para publicar resultados y, en su caso, los
   mosaicos derivados.

## 7. Qué no se ha podido verificar

- La página de datos y las reglas de Kaggle, los *notebooks* y las discusiones de las soluciones:
  Kaggle bloquea la lectura automática. Los niveles TIFF (1, 4, 16) y los valores de las máscaras
  se han comprobado sobre dos ficheros reales (sección 1), pero no sobre todo el conjunto; el tamaño típico viene de [9]; el nivel 0 a 20x y 0,5 µm/px de [8], [11] y [12].
- Nombre, tamaño y licencia de los conjuntos de tiles ya preparados en Kaggle.
- Métodos suplementarios del artículo (generación de máscaras, escáner por centro).
- Resultados de TransMIL y HIPT en PANDA, y la optimización de umbrales como técnica de los equipos.
- Lista concreta de ~10 % de etiquetas ruidosas y la descarga de PANDA-PLUS.

## Referencias

1. Bulten, W., Kartasalo, K., Chen, P.-H. C. et al. "Artificial intelligence for diagnosis and
   Gleason grading of prostate cancer: the PANDA challenge". *Nature Medicine* 28(1), 154-163
   (2022). DOI [10.1038/s41591-021-01620-2](https://doi.org/10.1038/s41591-021-01620-2). Acceso
   abierto, CC BY 4.0 (el artículo); PDF leído en
   [escholarship.org](https://escholarship.org/content/qt8px8m0dt/qt8px8m0dt.pdf).
2. Página oficial del reto, sección de datos:
   [panda.grand-challenge.org/data](https://panda.grand-challenge.org/data/).
3. Resúmenes de los equipos en el taller del reto:
   [panda.grand-challenge.org/workshop-abstracts](https://panda.grand-challenge.org/workshop-abstracts/).
4. Equipo PND (1.º), código y modelos:
   [github.com/kentaroy47/Kaggle-PANDA-1st-place-solution](https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution)
   (archivo en Zenodo, DOI [10.5281/zenodo.14279891](https://doi.org/10.5281/zenodo.14279891)).
5. Solución del 6.º puesto (analokmaus):
   [github.com/analokmaus/kaggle-panda-challenge-public](https://github.com/analokmaus/kaggle-panda-challenge-public).
6. Equipo ChienYiChi (8.º):
   [github.com/ChienYiChi/kaggle-panda-challenge](https://github.com/ChienYiChi/kaggle-panda-challenge).
7. Iafoss (11.º): [github.com/iafoss/PANDA](https://github.com/iafoss/PANDA).
8. Documentación de kaiko-ai/eva sobre PANDA (fuente secundaria):
   [kaiko-ai.github.io/eva/0.3/datasets/panda](https://kaiko-ai.github.io/eva/0.3/datasets/panda).
9. Copia de la descripción de datos de Kaggle (fuente secundaria, en chino):
   [gitcode.csdn.net/6a25243210ee7a33f278fa71.html](https://gitcode.csdn.net/6a25243210ee7a33f278fa71.html).
   Descripción original:
   [kaggle.com/c/prostate-cancer-grade-assessment/data](https://www.kaggle.com/c/prostate-cancer-grade-assessment/data)
   (no legible automáticamente).
10. Documento de diseño del reto: Bulten, W. et al. "The PANDA challenge: Prostate cANcer graDe
    Assessment using the Gleason grading system" (Zenodo, 2020),
    [zenodo.org/records/3715938](https://zenodo.org/records/3715938). El registro indica licencia
    CC BY-ND 4.0 **para el documento**, no para las imágenes.
11. Grisi, C., Litjens, G., van der Laak, J. "Hierarchical Vision Transformers for Context-Aware
    Prostate Cancer Grading in Whole Slide Images". arXiv:2312.12619 (2023),
    [arxiv.org/abs/2312.12619](https://arxiv.org/abs/2312.12619).
12. Gustafsson, F. K., Rantalainen, M. "Evaluating Computational Pathology Foundation Models for
    Prostate Cancer Grading under Distribution Shifts". arXiv:2410.06723 (v2, 2026),
    [arxiv.org/abs/2410.06723](https://arxiv.org/abs/2410.06723).
13. Butt, N. A. et al. "Benchmarking multiple instance learning architectures from patches to
    pathology for prostate cancer detection and grading using attention-based weak supervision".
    *Scientific Reports* 16, 11535 (2026). DOI
    [10.1038/s41598-026-39196-x](https://doi.org/10.1038/s41598-026-39196-x) (resumen leído vía
    Europe PMC y repositorio institucional; texto completo no abierto).
14. Hopson, S. et al. "PANDA-PLUS: Improved dataset of prostate whole slide images from PANDA
    Challenge with pixel-level expert annotations". *J Pathol Inform* 20, 100540 (2026). DOI
    [10.1016/j.jpi.2025.100540](https://doi.org/10.1016/j.jpi.2025.100540) (solo resumen).
15. Nishio, M., Matsuo, H., Kurata, Y., Sugiyama, O., Fujimoto, K. "Label Distribution Learning
    for Automatic Cancer Grading of Histopathological Images of Prostate Cancer". *Cancers* 15(5),
    1535 (2023). DOI [10.3390/cancers15051535](https://doi.org/10.3390/cancers15051535) (solo
    resumen, vía Europe PMC).
