# Roadmap posterior al TFG

> Plan de trabajo desde octubre de 2026, acordado tras la reunión con el tutor. Se avanza sin
> prisa, una fase cada vez. Objetivo a medio plazo: una **publicación** sobre la predicción del
> rendimiento del entrenamiento distribuido.

## Estado actual

> Actualizado el 23/11/2026. Se reescribe al cerrar cada bloque de trabajo (ver
> `CLAUDE.md`, "Continuidad entre sesiones"); el hook de inicio de sesión de Claude Code lo muestra.

**Fase 0 de la auditoría (23/11/2026): errores corregidos** (PR a `develop`). DDP con el índice del
dispositivo y no con el rank global (multinodo), `test_<sello>.csv` atribuido a su run y no a toda
la carpeta, `paravit runs` con la métrica principal del run, QWK y semilla en los resúmenes de
semillas (la línea `Configuración:` de DDP lleva ya `seed=` y `loss=`), `gpu_spec()` sin coincidencia
por subcadena, test del muestreador independiente de `CUDA_VISIBLE_DEVICES` y reglas de `.gitignore`
para `events_*.jsonl` y `test_preds_*.npz`. Siguiente: fase 1 (entorno y perfiles de máquina).

**Criba de modelos en PANDA (22/11/2026): Swin-T mejora a la base.** Con la misma receta que el
ViT-S a 512 px (el LLRD ya funciona con cualquier modelo de timm, PR #234) se probaron siete
modelos con una semilla en la GPU local. Solo Swin-T (`swin_tiny_patch4_window7_224`) se sale:
con tres semillas, **QWK de test 0,821 ± 0,011 frente a 0,794 ± 0,012, +0,027 [+0,008; +0,046]**,
la primera mejora significativa del QWK en el proyecto; también la exactitud (+0,058) y el AUC de
cáncer sí o no (+0,017). Un Swin-T rinde como el conjunto de tres ViT-S (0,819), con un 60 % más
de tiempo por run (38 min). ConvNeXt-T, que parecía mejorar con la semilla 42, queda igual con tres
(+0,002); DeiT-III-S, EVA-02-S, EfficientNet-B0 y EfficientNetV2-S empatan con la base y ResNet-50
queda por debajo (−0,075). Swin-T pasa a ser el modelo de referencia. Tablas en
[datasets/panda.md](datasets/panda.md#criba-de-modelos-22112026).

**Modelos grandes y conjuntos de arquitecturas (23/11/2026): QWK de test 0,858.** En Kaggle
(2 × T4), SwinV2-T (0,835), ViT-B (0,823) y ConvNeXt-S (0,811) superan a la base con la semilla 42,
pero ninguno a Swin-T; tampoco las variantes de Swin-T (pesos in22k, 25 épocas). Lo que sí sube es
**mezclar arquitecturas**: con los miembros elegidos solo en validación (el mejor run de cada
arquitectura, los 8 primeros), el conjunto da **0,858 en test**, +0,029 [+0,013; +0,047] sobre el
conjunto de tres semillas de Swin-T (0,829), y dos arquitecturas (Swin-T + SwinV2-T) ya dan 0,852.
Para elegir sin mirar el test se guardaron las predicciones de validación de los 21 runs.

**PANDA a 1.024 px en Kaggle (20-21/11/2026), pregunta cerrada:** con tres semillas por
resolución (1.024 px en 2 × T4 con DDP; 512 px en local), el QWK de test pasa de 0,794 ± 0,012 a
0,804 ± 0,017, una diferencia de **+0,010 [−0,010; +0,030] que no es significativa**. El 0,821 de
la primera semilla fue suerte. Sí mejoran de forma significativa la exactitud (+0,030
[+0,013; +0,048]) y el AUC de cáncer sí o no (+0,014 [+0,006; +0,022]). El conjunto de las tres
semillas da 0,815-0,819 a las dos resoluciones. A 1.024 px cada época cuesta 4,5 veces más, así
que **el QWK no está limitado por la resolución** y se sigue a 512 px. El control a 512 px en
Kaggle (0,801) confirma que Kaggle y la máquina local son comparables. El predictor subestima el
tiempo en T4 (−40 % a 512, −16 % a 1.024), porque las T4 trabajan limitadas por potencia (reloj
mediano 960 MHz frente a 1.590). Tablas en
[datasets/panda.md](datasets/panda.md#tres-semillas-a-1024-px-21112026) y
[performance_model.md](performance_model.md). Por el camino se corrigieron dos fallos de DDP: los
checkpoints guardaban el prefijo `module.` (PR #214) y `--seed` no cambiaba el orden de los datos
(PR #216).

**Web multi-dataset (20/11/2026):** el panel conoce el dataset y la métrica de cada run (QWK en
PANDA, AUC en MedMNIST, F1 en BigEarthNet), con etiquetas que distinguen semilla, pérdida, hospital,
color y resolución, paneles de dataset solo para el dataset en curso y *Compare* con agregación por
semilla (media ± desviación y banda). Cada run escribe un manifiesto `run_<ts>.json` (se versiona,
a diferencia de los eventos) y `scripts/backfill_run_manifests.py` lo genera para los viejos
([esquema](eventos.md#manifiesto-del-run-run_ddmmyyyy_hhmmssjson)). Los manifiestos de los runs
anteriores están en `logs/`; los de PANDA salen de sus eventos y los demás, de la línea
`Configuración:`.

**Reunión con el tutor (20/10/2026):** se le presentaron la etiqueta `v1.0-tfg`, el diseño
multi-dataset, MedMNIST y el trabajo con agentes de IA. **El tutor enviará el dataset médico
concreto** en el que centrar el trabajo; hasta entonces, MedMNIST sirve de banco de pruebas y el
diseño debe seguir admitiendo datasets nuevos sin tocar el entrenamiento. Pidió además **analizar
con más detalle la inferencia en las trazas**, para ver en tiempo real las decisiones del modelo
durante el entrenamiento, posiblemente desde la web. Auditoría y plan en "Auditoría de
observabilidad (25/10/2026)" y en la fase 4.

**Dataset del tutor (02/11/2026): PANDA** (*Prostate cANcer graDe Assessment*, Kaggle 2020;
Radboud y Karolinska): 10.616 biopsias de próstata (WSI `.tiff` de tres niveles, 1×/4×/16×,
~25-33 k px de alto a ~0,45-0,49 µm/px) con grado ISUP 0-5 y el hospital de origen. Licencia
CC BY-NC-SA 4.0. Karolinska concentra los grados bajos y Radboud los altos, y tiñen distinto:
riesgo de que el modelo aprenda el centro (comprobarlo con *Subgroups*). Descarga en
`/media/alejandro/SSD/datasets/panda/` (zip de 193 GB; `uvx --with six kaggle competitions download
-c prostate-cancer-grade-assessment`, token en `~/.kaggle/access_token`). Primer prototipo de
preparación: mosaico de los 16 tiles de 256 px con más tejido del nivel 4× (1024 px, 80-90 % de
tejido, <1 s por biopsia). Se presenta al tutor el 03/11/2026.

**Decisión (02/11/2026, el autor la delega):** primero **mosaico + multiclase con QWK** a 512 px
(no toca el entrenamiento; es el enfoque de los ganadores de Kaggle). Después, pérdida ordinal; MIL
queda como mejora si el mosaico se queda corto; los modelos de base de patología (UNI,
CTransPath) se descartan porque apenas entrenan el ViT. Se usan las 10.616 biopsias; la limpieza
de etiquetas ruidosas, más adelante. Detalle en [research/panda.md](research/panda.md).

**Reunión con el tutor del 20/11/2026:** informe con las mejoras de PANDA, medidas clínicas,
generalización entre hospitales, Grad-CAM y lista de lo que pedir del dataset del médico (fuera
del repo, en la carpeta de reuniones del autor).

**Hecho:**
- **Versión 1.3.0 en `main` (21/11/2026):** OUSM, limpieza de etiquetas fuera de pliegue,
  conjunto de checkpoints en `eval` y 17 dependencias al día (numpy 2.5, pandas 3.1, plotly 7,
  timm 1.0.31…; PR #213). Las dependencias se validaron con la suite completa (también con los
  avisos de deprecación como errores), una época de PANDA en GPU con el entorno viejo y el nuevo
  (Val QWK 0,7177 frente a 0,7176, mismo tiempo) y un recorrido de la web con `AppTest` (6 páginas
  y 13 vistas del Inspector, sin excepciones e idénticas en los dos entornos).
- **Versiones 1.1.0 y 1.2.0 en `main` (20/11/2026)**, con etiqueta y release. La 1.1.0 llevó
  PANDA, la medición clínica, la inspección del entrenamiento y la seguridad del repositorio
  (secret scanning, Dependabot, CodeQL, gitleaks, ramas protegidas). La 1.2.0 lleva las
  dependencias sin vulnerabilidades y torch 2.15, los checkpoints por run (`best.pt`/`last.pt`,
  escritura atómica; los DDP y de paralelismo de modelo ya cargan en `eval.py`), el registro de
  backbones no timm (`hf:`, I-JEPA, sonda lineal), la guía para colaboradores, la web
  multi-dataset, `--seed`/`--data-root` en todos los scripts y los kernels de Kaggle con los
  primeros runs de PANDA a 1.024 px.
- **El Inspector lee las predicciones de test (20/11/2026):** `eval --save-predictions` sin ruta
  guarda `test_preds_<ts>.npz` en la carpeta de logs del run (sello deducido del checkpoint) y la
  web ofrece la fuente «Test (eval)» en Curves, Clinical (con cortes «grado ≥ k»), Calibration y
  Subgroups. Solo se abren ficheros de `logs/`.
- **PANDA, mejoras y generalización (18-19/11/2026, PRs #186-#189):**
  - **Con tres semillas, ni la pérdida ordinal `ce_emd` ni los aumentos de color mejoran el QWK
    de forma clara a 512 px.** Base 0,794 ± 0,012 y ordinal 0,802 ± 0,009 en test (ordinal −
    base por semilla: +0,025, +0,005 y −0,005). Con una sola semilla la ordinal parecía mejorar
    (intervalo *bootstrap* del test sin el 0): ese intervalo no recoge la variación del
    entrenamiento, así que desde ahora se compara con tres semillas. El color (una semilla) no
    mejora y multiplica por 2,5 el tiempo por época (el `ColorJitter` va en CPU).
  - Cortes clínicos (media de 3 semillas, ordinal): ISUP ≥ 1, sensibilidad 0,90 / especificidad
    0,83 / AUC 0,95; ISUP ≥ 2, 0,81 / 0,90 / 0,93. La ordinal sube la especificidad de ISUP ≥ 1
    (0,78 → 0,83).
  - Entrenado en un solo hospital, el QWK cae en el otro: Radboud 0,81 → Karolinska 0,55;
    Karolinska 0,75 → Radboud 0,37. Ni la ordinal ni el color lo evitan (una semilla). La
    literatura ve la misma caída. Tablas en [datasets/panda.md](datasets/panda.md).
  - Código: `data.color_jitter` y `data.providers` en PANDA, `groups.npz` del monitor (vista
    *Subgroups*), `eval --save-predictions` con tablas por grupo y cortes «grado ≥ k»
    (`src/analysis/clinical.py`), y la revisión de medición clínica con la lista para un dataset
    clínico ([research/medicion_clinica_prostata.md](research/medicion_clinica_prostata.md)).
  - Infraestructura local: `/home` (59 GB) se llenó con los checkpoints. `checkpoints/local` y
    `checkpoints/runs` son ahora enlaces al SSD (ignorados por git), y los entrenamientos largos se
    lanzan como servicio de usuario con `nice`.
- **PANDA, línea base (02/11/2026):** dataset descargado y descomprimido (412 GB en el SSD), mosaicos
  de las 10.616 biopsias (0 errores, 8 GB) y partición 7.433/1.593/1.590. ViT-S a 512 px en la
  RTX 3060 Ti local (82 s/época, parada en la época 14): **QWK 0,781 en test** (0,814 en
  validación), exactitud 0,609, 88 % a un grado o menos; por centro, 0,745 (Karolinska) y 0,760
  (Radboud). Sobreajusta desde la época 5. Detalle en [datasets/panda.md](datasets/panda.md).
  PR #184: la línea `Configuración:` dice la pérdida real de la tarea y el log muestra `qwk`/`auc`
  por época.
- PANDA (PRs #179-#182, revisados por el agente revisor): estado del arte
  (`docs/research/panda.md`), script de mosaicos, partición fija 70/15/15 estable por huella,
  ficha `panda` ordinal, config a 512 px con `select_by: qwk`. Núcleo: `DatasetSpec.ordinal` + QWK
  (`val_qwk`) y `model.img_size` (cuatro estrategias, `eval.py`, benchmark).
- Fases 1 (higiene) y 2 (abstracción de dataset y tarea, verificada en GPU con las cuatro
  estrategias).
- Fase 3, primer bloque: MedMNIST (cuatro datasets médicos) con **3 semillas**. DermaMNIST y
  PathMNIST superan a todas las referencias oficiales; BreastMNIST y ChestMNIST, dentro del rango.
  AUC en validación y test, selección por AUC, umbral elegido en validación, `--seed` y
  `scripts/summarize_runs.py`.
- Predictor: perfilado automático de cualquier modelo de timm (fase 5, primer punto), revisado
  por el agente revisor y corregido.
- Web: AUC de validación y filtro por dataset. Estado del arte sobre predicción del rendimiento
  y sobre observabilidad (`docs/research/`).
- Fase 4, primer bloque (observabilidad): **flujo de eventos** `events_*.jsonl` en cada run
  ([esquema](eventos.md); observadores del bucle, *heartbeat*, nunca detiene el entrenamiento) y
  pestaña **Live** de la web, que sigue el run en tiempo real. Verificado con un entrenamiento real
  en GPU seguido desde el navegador. PRs #160-#163.
- Fase 4, segundo bloque: **inspección profesional del entrenamiento** (PRs #164-#175), con el
  [inventario del sector](research/inventario_herramientas_observabilidad.md) como referencia:
  - Terminal: monitor de predicciones (`--monitor`: subconjunto fijo de validación por época con
    probabilidades, pérdida por imagen, *embeddings* y Grad-CAM), estadísticas por capa
    (gradientes, pesos, ratio de actualización, histogramas), métricas de sistema (GPU/CPU/RAM),
    rendimiento por batch (espera del DataLoader, muestras/s) y **alertas** (NaN, divergencia,
    sobreajuste, gradientes, cuello de I/O, GPU al límite). Nada de esto cambia el entrenamiento
    (tests de pesos idénticos) y todo queda fuera del tiempo medido.
  - Análisis (`src/analysis/`, NumPy): curvas ROC/PR, calibración completa (ECE, pendiente,
    intercepto), IC *bootstrap*, punto de operación clínico, subgrupos, cartografía y proyecciones.
  - Web: página **Inspector** con 13 vistas (System, Performance, Predictions, Errors, Curves,
    Clinical, Calibration, Subgroups, Dynamics, Embeddings, Explain, Layers, Alerts), en vivo y a
    posteriori. `paravit report` genera un **informe HTML** autocontenido con la lista TRIPOD+AI.
  - Verificado con entrenamientos reales en GPU (DermaMNIST) recorriendo cada vista en el navegador.
- Arnés de trabajo con IA (agentes, skills, hooks, permisos, lint en CI y pre-commit).
- Corregidos los fallos de `model.dropout` (todos los runs del TFG usaron 0.1) y del heterogéneo
  con `pos_weight`.
- Limpieza de trazas (deuda de la auditoría): docstrings de `plot`/`confusion` al día,
  `LayerHooksDecorator` al logger, tests de los metric reporters, `psutil` declarado y
  `--trace deep` con `batch-table` ya alimenta a `batch-monitor`.

**OUSM (21/11/2026):** con tres semillas a 512 px (k = 2 de 16), el QWK de test queda en
0,793 ± 0,006 frente a 0,794 ± 0,012 de la base (−0,001 [−0,014; +0,013]): **no ayuda**. Sube algo
Karolinska y baja Radboud, el hospital ruidoso, lo que sugiere que descarta biopsias difíciles de
verdad y no solo mal etiquetadas. Detalle en
[datasets/panda.md](datasets/panda.md#etiquetas-ruidosas-ousm-21112026). Balance de PANDA hasta
ahora: ni la resolución, ni la pérdida ordinal, ni los aumentos de color, ni OUSM mueven el QWK
medio (≈ 0,79-0,80). Lo único que lo sube es **promediar tres semillas (0,819)**.

**Limpieza de etiquetas fuera de pliegue (21/11/2026), pregunta cerrada:** cinco modelos en
validación cruzada sobre train marcan el 11,7 % de las biopsias (umbral de 1,5 grados fijado de
antemano), pero uno de cada cinco ejemplos de los grados 3-5. Con tres semillas sobre el train
limpio, el QWK de test baja a 0,779 ± 0,012 (−0,015 [−0,032; +0,000]; el conjunto de tres,
−0,038 [−0,060; −0,017]) y cae sobre todo Radboud. Con un modelo de QWK ≈ 0,78, la discrepancia
fuera de pliegue mide los errores del modelo en los grados altos más que el ruido de las
etiquetas. **Se cierra la línea de etiquetas ruidosas** (ni OUSM ni la limpieza ayudan). Detalle
en [datasets/panda.md](datasets/panda.md#limpieza-de-etiquetas-fuera-de-pliegue-21112026).

**Conjunto de checkpoints en `eval` (21/11/2026):** `--checkpoint` repetido promedia las
probabilidades de varios modelos (cargados de uno en uno). Con las tres semillas base reproduce el
QWK de test de 0,819 (0,8193, exactitud 0,674).

**Predictor y arquitecturas (23/11/2026).** El tiempo de entrenamiento de arquitecturas que no
son ViT no sigue a los FLOPs: con 13 arquitecturas medidas con lotes sintéticos a 512 px (RTX 3060
Ti y una T4 de Kaggle), un modelo de dos términos (FLOPs + GB de activaciones que guarda autograd,
medidos en CPU) baja el error en validación cruzada de 193 % a 10 % y predice el coste relativo de
las arquitecturas en la T4, sin reajustar, con un 11 % (82 % con solo FLOPs). En la T4 queda un
factor común de 1,49 de causa sin identificar (sobre todo en el término de memoria) y la familia
Swin se escapa (+24-42 %). El −40 % de ViT-S a 512 px en Kaggle no es la GPU; la hipótesis es
contención de CPU, pendiente de un control con `num_workers=2`. Es estudio, no predictor todavía:
[performance_model.md, sección 8](performance_model.md#8-arquitecturas-distintas-de-vit-flops-frente-a-flops--memoria-23112026).
El benchmark ya usa los tamaños de partición del dataset de la config (PR #246).

**En curso (23/11/2026):** en Kaggle, el lote `seeds_512` (versión 5 del kernel): semillas 43 y 44
de SwinV2-T y ViT-B, para confirmarlos y repetir el conjunto con runs que no intervinieron en la
elección, y Swin-T con DDP en 2 × T4, el mismo run que el local en una GPU (comparación de
estrategias). Predicciones de validación y test en `checkpoints/runs/panda_m_*/` (local) y en
`/media/alejandro/SSD/kaggle-out/panda_models_512/results/` (Kaggle). Las colas locales están en
`~/experimentos_paravit/cola_*.sh`. VERODE no es accesible sin la VPN de la ULL.

**Siguiente paso recomendado:** (1) **Adaptación a otro hospital:** ajustar con pocas biopsias
del otro centro y medir cuántas hacen falta (lo que habrá que hacer con el dataset del médico);
normalización de tinción (Macenko). (2) Con Swin-T como referencia: ver si los modelos grandes de
Kaggle (SwinV2-T sobre todo) suben más, y llevar Swin-T a las estrategias de distribución (DDP en
2 × T4 frente a una GPU, y el tiempo medido frente al del predictor). Después, MIL con atención
sobre los tiles si el mosaico se queda corto. (3) Grad-CAM frente a las máscaras de PANDA. (4) Predictor: el tiempo en T4 (limitación
por potencia, lotes pequeños por GPU).

Técnicos: aumentos de color en GPU y el Grad-CAM del monitor sin memoria a 1.024 px en la T4 (es
un aviso, no detiene el run). Pendientes de la fase 4: `paravit explain` con la prueba
de cordura de Grad-CAM.

**Decisiones pendientes del autor o del tutor:**
- Dataset del contacto médico del tutor: qué se puede pedir y cuándo (lista en
  [research/medicion_clinica_prostata.md](research/medicion_clinica_prostata.md)).
- Pedir la preparación y los resultados del TFG previo que usó PANDA (referencia para compararse).
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
- [x] Checkpoints por run (20/11/2026): carpeta con sello de tiempo, `best.pt` + `last.pt` (`checkpoint.keep`), escritura atómica.
- [ ] (Opcional) Integración con Zenodo para tener un DOI de cada release.
- [x] Seguridad del repo: `SECURITY.md`, Dependabot, CodeQL, auditoría de dependencias en CI y gitleaks en pre-commit (PR pendiente de merge: el lock tiene ~250 avisos en dependencias, a resolver con Dependabot).

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
- [x] El benchmark ya no toma de BigEarthNet los tamaños de split (salen del dataset de la config
      o de `--train-size/--val-size`), ni el perfil de I/O `*.tif` ni el prior de F1 (solo con
      BigEarthNet). Pendiente: un perfil de I/O propio de cada dataset, junto con el modelo de
      I/O de la fase 5.

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
- [ ] **PANDA** (dataset del tutor, 02/11/2026): preparación offline a mosaicos, partición propia,
      ficha, config, métrica QWK y subgrupos por centro.
  - [x] Preparación a mosaicos, partición 70/15/15, ficha y config (`docs/datasets/panda.md`).
  - [x] `model.img_size` para timm (entrada de 512) y métrica QWK (`ordinal=True`, `select_by: qwk`).
  - [x] Mosaicos y `splits.csv` del dataset completo y línea base a 512 px (QWK 0,781 en test).
  - [x] `groups.npz` con `data_provider` (vista *Subgroups*) y pérdida ordinal (sin mejora clara con 3 semillas).
  - [x] Aumentos de color y entrenamiento por hospital (generalización entre centros).
  - [x] Tres semillas de base y ordinal; ordinal y color entrenando en un solo hospital.
  - [x] Herramienta de comparación estadística de grupos de runs (`paravit compare-preds`, `docs/guia_de_uso.md`): media ± std, conjunto y *bootstrap* emparejado desde `test_preds.npz`.
  - [x] Selección de conjuntos en validación (`paravit select-ensemble`, `docs/guia_de_uso.md`): `per-group` y `greedy`, una sola evaluación en test.
  - [x] Más resolución: 1.024 px en Kaggle con tres semillas (21/11/2026). No mejora el QWK de
    forma significativa.
  - [x] Etiquetas ruidosas con OUSM (21/11/2026): no mejora el QWK.
  - [x] Limpieza de etiquetas fuera de pliegue (21/11/2026): empeora el QWK (−0,015).
  - [x] Criba de modelos a 512 px (22/11/2026): Swin-T mejora el QWK de forma significativa
    (0,821 ± 0,011, tres semillas); ConvNeXt-T, DeiT-III, EVA-02 y EfficientNet empatan.
  - [x] Modelos grandes en Kaggle (ViT-B, ConvNeXt-S, SwinV2-T, EfficientNet-B3, MaxViT-T):
    ninguno supera a Swin-T (23/11/2026).
  - [x] Conjunto de arquitecturas elegido en validación: QWK de test 0,858 (23/11/2026).
  - [ ] Semillas 43 y 44 de SwinV2-T y ViT-B; conjunto repetido con ellas.
  - [ ] Adaptación a otro hospital y normalización de tinción.
- [ ] Siguiente dataset médico grande: NIH ChestX-ray14 a resolución completa o PatchCamelyon.
- [ ] NIH ChestX-ray14 (multi-etiqueta, 14 clases, PNG grandes: I/O dominado por la decodificación).
- [ ] Un dataset de cáncer: PatchCamelyon (binaria, HDF5, parches de 96²) o ISIC (multiclase).
- [ ] Una ficha por dataset en `docs/datasets/`.

### Modelos no timm (JEPA)
- [x] Registro de backbones (`BackboneSpec`), adaptador `hf:` de `transformers`, I-JEPA ViT-H/14 y
      sonda lineal (`model.freeze_backbone`); guía en [modelos/README.md](modelos/README.md).
- [ ] Entrenar y comparar JEPA frente a un ViT supervisado en PANDA (alumno nuevo; protocolo en la guía).
- [ ] Normalización propia del backbone (hoy solo se avisa si difiere de ImageNet).

### Fase 4 — Instrumentación, observabilidad e inferencia
- [x] Flujo de eventos `events_*.jsonl` (esquema v1 en [eventos.md](eventos.md)): inicio con
      configuración y huella del entorno, épocas, batches, parada temprana y fin (también si falla).
- [x] Web: pestaña *Live* que sigue un run en curso con lectura incremental del flujo.
- [x] Limpieza de la deuda de las trazas (docstrings, `LayerHooksDecorator`, *batch hooks* con
      `batch-table`, dependencias).
- [x] Muestreo de sistema (evento `system`: GPU por NVML, CPU y RAM) y rendimiento por batch
      (`data_time_s`, `step_time_s`, `samples_per_s`, `data_wait_frac`) en el flujo de eventos
      ([inspeccion.md](inspeccion.md)).
- [x] Panel fijo de validación: probabilidades, pérdida, *embeddings* y Grad-CAM de un
      subconjunto fijo de validación en cada época (`--monitor`, `inspect_<ts>/`).
- [x] Análisis de la inspección en CPU: paquete puro `src/analysis/` (curvas, umbrales,
      calibración, confusión, cartografía, flips, peores errores, proyección 2D y lector de
      `inspect_<ts>/`); la web los muestra en la página *Inspector*.
- [x] Predicciones por muestra en test (`eval.py --save-predictions`) con métricas por grupo y
      cortes clínicos; calibración, curvas ROC/PR y peores errores ya los da el Inspector.
- [x] Que la web lea las predicciones de test (`test_preds_<ts>.npz`) además del panel de validación:
      selector «Validation / Test» en el Inspector (Curves, Clinical con cortes «grado ≥ k»,
      Calibration y Subgroups).
- [x] `paravit predict`: clasificar imágenes sueltas o una carpeta con un checkpoint (o un conjunto),
      con la transformación de evaluación de la ficha del dataset (`DatasetSpec.image_reader`).
- [ ] `paravit explain`: mapas Grad-CAM del panel fijo fuera del bucle medido; la web los muestra.
- Segunda ola, según el [inventario del sector](research/inventario_herramientas_observabilidad.md)
  (lo que ninguna herramienta general trae y la imagen médica exige):
  - [x] Intervalos de confianza *bootstrap*, calibración completa (pendiente, intercepto, ECE por
        clase), punto de operación clínico (sensibilidad objetivo, Youden, VPP/VPN) y métricas
        por subgrupo: cálculo en `src/analysis/` y vistas *Curves*, *Clinical*, *Calibration* y
        *Subgroups* del Inspector (lee el `groups.npz` que escribe el monitor; en PANDA, el
        hospital).
  - [x] Alertas en el flujo de eventos (pérdida NaN/infinita, divergencia, sobreajuste, gradientes
        fuera de rango, cuello de I/O, GPU al límite, caída de throughput): evento `alert`
        (`src/training/alerts.py`); lado web hecho (banner en Live e Inspector, vista
        *Alerts*).
  - [x] Histogramas de gradientes (evento `histograms` con `kind`; la web los muestra).
  - [ ] Prueba de cordura de Grad-CAM (aleatorizar pesos, Adebayo et al. 2018).
  - [x] Informe exportable de un run (HTML) con la lista TRIPOD+AI: `paravit report`;
        falta CLAIM y los subgrupos (dependen de los metadatos del dataset del tutor).
  - [x] Agregación por semilla en la comparación de runs (media y banda): *Compare* agrupa los
        runs que solo difieren en la semilla (`src/web/seed_agg.py`).
  - [ ] Métricas por *rank* en DDP (rezagados; cómputo frente a comunicación).
  - [ ] Duplicados y fuga entre particiones (hash/embeddings; por paciente).
  - [ ] Perfil puntual con `torch.profiler` (traza Perfetto) y exportador opcional a TensorBoard.
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
  - [x] Estudio de arquitecturas: FLOPs frente a FLOPs + memoria de activaciones, validado en otra
    GPU (T4) y con modelos fuera de la calibración (23/11/2026; `performance_model.md`, sección 8).
  - [ ] Integrar el término de memoria en `performance_model.py`: ancho de banda en `GPU_TABLE`,
    reloj sostenido por GPU, MFU recalibrada y una tercera GPU para validar.
  - [ ] ¿Contención de CPU en Kaggle? ViT-S a 512 px va a 122 s/época donde la GPU haría ~77 s:
    control con `num_workers=2` o mosaicos ya decodificados; si se confirma, término de CPU.

### Fase 6 — Campaña en máquinas nuevas y publicación
- [ ] Barridos con ≥3 repeticiones por configuración.
- [ ] Revisión del trabajo relacionado (Paleo, Daydream, Habitat…).
- [ ] Dataset de medidas y herramienta publicados con DOI.

### Ideas aparcadas
FSDP/ZeRO · pipeline con micro-batches (GPipe/1F1B) · gradient checkpointing · GPUs heterogéneas en
el predictor · formato de lectura rápida (LMDB/WebDataset) · focal loss a escala en VERODE.
