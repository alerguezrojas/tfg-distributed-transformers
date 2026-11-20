# JEPA y modelos de base para PANDA

> Revisión para preparar el trabajo de un alumno nuevo que probará otros backbones (en concreto
> JEPA) sobre PANDA con ParaViT-Lab. Fecha: 20/11/2026. Cada referencia se ha consultado en su
> página oficial (arXiv, Hugging Face, GitHub). Las cifras son **las que dan los propios autores**,
> con su configuración. Lo que no se ha podido comprobar se marca como *no verificado*. Las frases
> que empiezan por "Interpretación" son nuestras, no de los autores.

## 1. Qué es JEPA y en qué se diferencia

### 1.1 Idea básica

JEPA (*Joint-Embedding Predictive Architecture*) aprende a partir de imágenes sin etiquetas
prediciendo **representaciones** (vectores) de zonas ocultas de la imagen a partir de una zona
visible, en lugar de predecir píxeles. I-JEPA toma un bloque de contexto y predice las
representaciones de varios bloques objetivo de la misma imagen; el codificador objetivo se usa para
evaluar y su salida se promedia para obtener la representación global [1]. Los autores subrayan
que no usan aumentos de datos hechos a mano (recortes, color, etc.) para fabricar las vistas [1].

Comparación breve (resumen de lo que dicen las fuentes):

| Método | Qué predice o compara | Aumentos de datos | Token CLS |
|---|---|---|---|
| Supervisado | Etiqueta de clase | Sí, habituales | Sí (ViT estándar) |
| MAE [2] | Reconstruye píxeles de parches ocultos | No esenciales | Sí |
| DINO / iBOT / DINOv2 [3][4] | Autodestilación entre vistas distintas (iBOT y DINOv2 añaden modelado de imagen enmascarada sobre tokens) | Sí, recortes múltiples y color | Sí |
| I-JEPA [1] | Representaciones de bloques ocultos, en el espacio latente | No hechos a mano | **No** |

Dos matices tomados del propio artículo de I-JEPA [1]:

- El preentrenamiento se hace **sin token `[cls]`** y, para evaluar, se promedia la salida del
  codificador objetivo. Para MAE, iBOT y DINO sí usan `[cls]`.
- Los autores dicen que preentrenar un ViT-H/14 en ImageNet cuesta menos de 1200 horas de GPU, más
  de 2,5 veces menos que un ViT-S/16 con iBOT y más de 10 veces menos que un ViT-H/14 con MAE.

Resultados de I-JEPA en ImageNet-1k con sonda lineal (Tabla 1 de [1]): ViT-L/16 77,5 %, ViT-H/14
79,3 %, ViT-H/16 a 448 px 81,1 %. La comparación completa con los métodos que sí usan aumentos de datos
está en esa misma tabla [1]; aquí no se reproduce.

Interpretación: como I-JEPA no usa vistas aumentadas, no se le han "enseñado" las invarianzas de
color o de escala que la histopatología necesita (tinciones distintas entre centros). Eso es una
hipótesis a contrastar, no un hecho publicado. El estudio de [9] (ver 1.3) apunta en esa
dirección, pero sin cifras verificadas aquí.

### 1.2 Variantes de Meta

- **I-JEPA** (imagen). Assran et al., CVPR 2023, pp. 15619-15629 [1].
- **V-JEPA** (vídeo). Bardes et al., arXiv 2404.08471 (2024) [5]. Los autores indican que su
  ViT-H/16 entrenado solo con vídeo llega al 77,9 % en ImageNet-1k (sonda con el codificador
  congelado) [5].
- **V-JEPA 2** (vídeo, y modelo de mundo con acciones). Assran et al., arXiv 2506.09985 (2025) [6].
- **V-JEPA 2.1**. Mur-Labadia et al., arXiv 2603.14482 (marzo 2026) [7]. Los autores lo presentan
  como familia que aprende rasgos densos "para imágenes y vídeos", con tokenizadores multimodales
  [7]. Es lo más reciente y el único con tamaños pequeños (ver sección 2).

### 1.3 Derivados en imagen médica o patología (verificado solo el resumen)

- **RadJEPA**: adaptación de I-JEPA a radiografías de tórax, unas 840 000 imágenes sin etiqueta
  (Khan et al., arXiv 2601.15891, aceptado en EMNLP 2026) [8]. La página no enlaza pesos ni
  código; **no verificado** que estén publicados.
- **"Pretext Matters"** (Ivezić et al., arXiv 2603.22649, marzo 2026) compara métodos JEA, JEPA y de
  reconstrucción en ecografía hepática e histopatología [9]. Los autores concluyen que los JEA
  (métodos tipo DINO) funcionan mejor cuando la señal útil es **local**, como en histopatología, y
  los JEPA cuando la información es de estructura global [9]. La página consultada no da cifras ni
  el nombre del dataset de histopatología: **no verificado** el detalle.
- Mencionados por un buscador y **no abiertos por mí**: un estudio de MAE frente a JEPA en RM
  cerebral 3D (arXiv 2606.13315) y COJEPA (arXiv 2607.11962). Se dejan como pistas, no como
  referencias.
- **Ningún modelo de base de patología publicado usa JEPA** entre los revisados en la sección 3.
  No he encontrado un modelo de base de patología con JEPA; ausencia de evidencia, no prueba.

## 2. Pesos publicados de la familia JEPA

### 2.1 Tabla de checkpoints

| Checkpoint | Arquitectura / resolución | Dónde | Licencia | Carga |
|---|---|---|---|---|
| `facebook/ijepa_vith14_1k` | ViT-H/14, 224 px, ImageNet-1k, 300 ép. [11][12] | HF y `dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar` | **CC BY-NC 4.0** (etiqueta HF [13]; `LICENSE` del repo [11]) | `transformers` |
| `facebook/ijepa_vith16_1k` | ViT-H/16, 448 px, ImageNet-1k [11] | HF [14] | **CC BY-NC 4.0** [14] | `transformers` |
| `facebook/ijepa_vith14_22k` | ViT-H/14, 224 px, ImageNet-22k [11] | HF [15] | **CC BY-NC 4.0** [15] | `transformers` |
| `facebook/ijepa_vitg16_22k` | ViT-g/16, 224 px, ImageNet-22k, ~1B parámetros [11][16] | HF [16] | **CC BY-NC 4.0** [16] | `transformers` |
| V-JEPA (ViT-L/16, ViT-H/16 a 224, ViT-H/16 a 384) | Vídeo, VideoMix2M [17] | `dl.fbaipublicfiles.com/jepa/...` | Repo: **CC BY-NC 4.0** [17][18] | Solo código del repo |
| `facebook/vjepa2-vitl-fpc64-256` | ViT-L/16, ~300M, 256 px, 64 fotogramas [19][20] | HF [20] | **MIT** (etiqueta HF) [20] | `transformers` |
| `facebook/vjepa2-vith-fpc64-256` | ViT-H/16, ~600M, 256 px [19] | HF | no comprobada individualmente | `transformers` |
| `facebook/vjepa2-vitg-fpc64-256` y `-384` | ViT-g/16, ~1B [19] | HF | `-384`: **Apache-2.0** (etiqueta HF) [21]; `-256` no comprobada | `transformers` |
| V-JEPA 2.1 (ViT-B/16 80M, ViT-L/16 300M, ViT-g 1B, ViT-G 2B; 384 px) | Imagen y vídeo [19][7] | URLs `dl.fbaipublicfiles.com/vjepa2/vjepa2_1_*.pt` [19] | Repo V-JEPA 2: MIT [19][22]; licencia de estos pesos **no verificada** | `torch.hub` del repo [19] |

Notas:

- Los identificadores y las URLs de la tabla salen del README de I-JEPA [11], del de V-JEPA 2 [19],
  del listado de `facebook` en HF y de la colección V-JEPA 2 en HF [12][14][15][16][23].
- La colección HF de V-JEPA 2 incluye además variantes ajustadas en vídeo (SSv2, Diving48) [23].
- **Hay una incoherencia a vigilar**: el repo V-JEPA 2 declara MIT [22], la ficha de ViT-L dice MIT
  [20] y la de ViT-g/384 dice Apache-2.0 [21]. Antes de usar cualquiera, leer la ficha concreta.
- El código de I-JEPA está bajo CC BY-NC 4.0 según su `LICENSE` [11] (un espejo lo daba como MIT;
  prevalece el fichero oficial). El de V-JEPA (el primero) también es CC BY-NC 4.0 [18].
- **No hay I-JEPA pequeño publicado.** Los únicos checkpoints son ViT-H y ViT-g [11][12]. Un
  ViT-S o ViT-B con I-JEPA habría que preentrenarlo con el repo (el artículo describe un ViT-B/16
  en sus ablaciones [1], pero no se ha encontrado un checkpoint público).
- Existen también `facebook/sparsh-ijepa-small` y `-base` en el listado de HF [12]; por el nombre
  no son de imágenes naturales y **no se han revisado**.

### 2.2 Carga

- **I-JEPA con `transformers`**: clases `IJepaConfig`, `IJepaModel`, `IJepaForImageClassification`
  [24]. Se añadió en la versión **4.47.0** (documentación: añadido el 05/12/2024) [24][25]. Admite
  `interpolate_pos_encoding=True` para resoluciones distintas a las de preentrenamiento [24]. Los
  ejemplos de la documentación obtienen el embedding con `outputs.last_hidden_state.mean(dim=1)`
  [24], coherente con que el modelo no tiene CLS [1].
- **V-JEPA 2 con `transformers`**: `VJEPA2Model` y `VJEPA2ForVideoClassification`, con
  `AutoVideoProcessor`; llegó con la **4.53.0** (hubo una etiqueta previa `v4.52.4-VJEPA-2-preview`)
  [26][27]. `VJEPA2Model` permite `skip_predictor=True` para usar solo el codificador [26]. La ficha
  HF explica que, para imágenes, se repite el fotograma 16 veces [20].
- **`timm`**: no he encontrado ningún checkpoint JEPA en el Hub de `timm` ni una receta oficial
  para pasar las claves del `state_dict` de Meta a `timm`. Una arquitectura de `timm` con
  `class_token=False` y `global_pool="avg"` existe, pero **no está verificado** que los pesos de
  I-JEPA carguen con `strict=True`. Comprobarlo con un script corto es lo primero que debería
  hacer el alumno.
- **Solo código del repo original**: V-JEPA (el primero) [17] y V-JEPA 2.1 (vía `torch.hub`) [19].
  El README de V-JEPA 2 solo menciona `timm` como dependencia de instalación [19].

## 3. Modelos de base de patología relevantes para PANDA

### 3.1 Tabla

Todos reciben tiles de 224 x 224 y son ViT con parche 14 o 16. Datos de las fichas oficiales.

| Modelo | Arquitectura | Método de preentrenamiento | Acceso | Licencia |
|---|---|---|---|---|
| UNI [28] | ViT-L/16, ~0,3B, emb. 1024; Mass-100K (100 000 WSI) | DINOv2 (DINO + iBOT + KoLeo) | Con solicitud (HF, correo institucional) | CC BY-NC-ND 4.0 |
| UNI2-h [29] | ViT-H/14, 681M, emb. 1536, 8 tokens de registro | DINOv2 | Con solicitud | CC BY-NC-ND 4.0 |
| CONCH [30] | ViT-B/16 (~90M visión) + texto | CoCa (contrastivo + descripción), 1,17 M pares imagen-texto | Con solicitud | CC BY-NC-ND 4.0 |
| Virchow [31] | ViT-H/14, ~632M, ~1,5 M WSI | DINOv2 | Con solicitud | Apache-2.0 con términos adicionales y "no para diagnóstico" |
| Virchow2 [32][33] | ViT-H/14, 632M, 3,1 M WSI, emb. 1280 (CLS + media = 2560) | DINOv2 modificado (KDE en vez de KoLeo; traslación en vez de recorte) | Con solicitud | CC BY-NC-ND 4.0 |
| H-optimus-0 [34] | ViT, ~1,1B, emb. 1536, >500 000 WSI propias | "Autosupervisado" (método no detallado en la ficha) | Con solicitud | Apache-2.0 |
| H-optimus-1 [35] | ViT, ~1,1B, emb. 1536, >1 M de WSI propias | No detallado en la ficha | Con solicitud | CC BY-NC-ND 4.0 |
| Phikon-v2 [36][37] | ViT-L/16, ~0,3B, 456 M de tiles (~60 000 WSI públicas) | DINOv2 (la versión 1 usaba iBOT) | Sin solicitud visible | Licencia no comercial de Owkin |
| CTransPath [38] | Transformador (detalle de arquitectura no verificado) | Contrastivo (código basado en BYOL y MoCo v2, según el README) | Google Drive | GPL-3.0, "solo uso académico no comercial" |
| Prov-GigaPath [39] | Codificador de tiles (ViT-g, ~1B según [37]) + codificador de WSI | DINOv2 (tiles); MAE (codificador de WSI) *según agradecimientos y código; no confirmado en la ficha* | Con solicitud | Apache-2.0, solo investigación |

**Ninguno de estos modelos usa JEPA.** Casi todos son DINOv2/iBOT, lo que casa con la conclusión
de [9] (los métodos con vistas aumentadas van mejor con señal local).

Detalles útiles:

- Normalización: UNI y UNI2-h usan media y desviación de ImageNet [28][29]; **H-optimus** usa
  media (0,707223; 0,578729; 0,703617) y desviación (0,211883; 0,230117; 0,177517) propias, con
  tiles a 0,5 micras por píxel [34][35]. Mezclar normalizaciones es un error fácil.
- Virchow2: 261 tokens de salida (1 CLS, 4 de registro, 256 de parche); la ficha recomienda
  concatenar CLS y media de parches [32].
- Carga con `timm`: UNI (`hf-hub:MahmoodLab/uni`, con `init_values=1e-5`, `dynamic_img_size=True`)
  [28]; UNI2-h (`hf-hub:MahmoodLab/UNI2-h`, con `SwiGLUPacked`) [29]; Virchow2 (`timm >= 0.9.11`,
  `mlp_layer=SwiGLUPacked`) [32]; H-optimus-1 [35]; Prov-GigaPath [39]. Phikon-v2 se carga con
  `transformers` (`AutoModel`) [36]. CONCH usa su propio paquete `conch`, con opción `timm`
  también anunciada [30].
- Todos requieren, según las fichas, aceptar términos en HF; en varios, correo institucional.

### 3.2 Resultados publicados en PANDA

- **Gustafsson y Rantalainen** (arXiv 2410.06723, v2 abril 2026) evalúan modelos de base
  congelados con agregación sobre PANDA (Resnet-IN, Phikon-v2, UNI, UNI2-h, H-optimus-0/1,
  Prov-GigaPath, Virchow, Virchow2, H0-mini, CONCH, CONCHv1.5) con tiles de 256 px a 20x y 10
  particiones [10]. Métrica: **QWK**. Con UNI y ABMIL: **0,888 ± 0,013** en PANDA completo; al
  entrenar en Radboud y probar en Karolinska cae a **0,247 ± 0,138** [10]. Los autores
  concluyen que los modelos grandes dan mejoras "modestas o inconsistentes" en robustez entre
  centros [10]. (Cifras leídas del resumen de la versión HTML; los valores de los demás
  modelos no se han transcrito.)
- **Tabla del repositorio de UNI** (PANDA, 5 clases, tarea a nivel de lámina): UNI2-g-preview
  0,953; H-optimus 0,953; UNI2-h 0,946; Virchow 0,946; UNI 0,944; Prov-GigaPath 0,944; CONCH 0,934;
  Virchow2 0,931; MUSK 0,923 [40]. **La página no nombra la métrica** y es de los autores de UNI
  (parte interesada); no son comparables con el QWK de arriba ni con el nuestro.
- **PANDA-PLUS-Bench** (arXiv 2512.14922) usa PANDA para medir si los modelos aprenden artefactos de
  la lámina; da cifras de generalización entre láminas (Virchow2 47,2 %; HistoEncoder 59,7 %) [41].
  No es un resultado de grado ISUP.

Interpretación: con modelos de patología congelados, el QWK dentro de distribución (~0,89 con UNI
y ABMIL, tiles a 256 px y 20x) queda por encima de nuestro ≈ 0,80, pero **no es comparable
directamente**: ellos agregan cientos de tiles por lámina con ABMIL; nosotros usamos un mosaico de
16 tiles. La caída entre centros (0,247) avisa de que el sesgo de centro de la sección 5 de
[panda.md](panda.md) no desaparece con modelos de base.

## 4. Qué hace falta para meter un backbone que no es de `timm`

(Interpretación de ingeniería, apoyada en las fichas citadas; AGENTS.md describe `TimmClassifier`
como backbone `timm` más cabeza.)

1. **Cargar el codificador.** Para I-JEPA, `IJepaModel.from_pretrained(...)` [24]; para V-JEPA 2,
   `VJEPA2Model` con `skip_predictor=True` [26]; para V-JEPA 2.1, `torch.hub.load` del repo [19].
   Los modelos de patología se cargan con `timm.create_model("hf-hub:...")` [28][29][32].
2. **Extraer el embedding.** I-JEPA no tiene CLS: usar la **media de tokens** de
   `last_hidden_state` [1][24]. Los modelos DINOv2/iBOT sí tienen CLS (UNI usa CLS [28]); para
   Virchow2 se recomienda CLS + media [32]. Hay que fijar esta regla por backbone.
3. **Resolución y parche.** ViT-S/16 a 224 px da 196 tokens; ViT-H/14 a 224 px da 256; ViT-H/16 a
   448 px da 784 (cuenta nuestra). Cambiar de resolución exige interpolar los embeddings de
   posición (`interpolate_pos_encoding=True` en HF [24]; `dynamic_img_size=True` en `timm` [28]).
4. **Normalización.** Media y desviación de ImageNet para UNI y UNI2-h [28][29]; propias para
   H-optimus [34][35]; el procesador HF de I-JEPA trae la suya [24]. Debe venir del backbone, no
   estar fija en el framework.
5. **Ajuste fino completo o sonda lineal.** El artículo de I-JEPA solo evalúa sonda lineal y un
   caso semisupervisado con el 1 % de etiquetas [1]; **no hay receta publicada de ajuste fino
   completo en patología** que haya encontrado. Con modelos de este tamaño, empezar con
   codificador congelado es lo razonable (interpretación).
6. **Memoria y tiempo frente a ViT-S/16.** Cifras de las fichas de `timm` (inferencia, 224 px):

   | Modelo | Parámetros | GMACs | Activaciones |
   |---|---|---|---|
   | ViT-S/16 [42] | 22,1 M | 4,3 | 8,2 M |
   | ViT-B/16 [43] | 86,6 M | 16,9 | 16,5 M |
   | ViT-L/16 [44] | 304,3 M | 59,7 | 43,8 M |
   | ViT-H/14 [45] | 630,8 M | 162,0 | 95,1 M |

   Cuentas nuestras: ViT-H/14 tiene ~29 veces los parámetros de ViT-S/16 y ~38 veces los GMACs.
   Un ViT-H/14 de I-JEPA tiene un tamaño equivalente (0,6B según HF [13]) pero no se ha medido
   aquí su GMACs exacto. Para entrenar completo con AdamW en fp32, solo pesos, gradientes y estados
   del optimizador son unos 16 bytes por parámetro, es decir, unos 10 GB para 630 M de parámetros
   (cuenta nuestra), más de lo que cabe en una RTX 3060 Ti de 8 GB; con el codificador congelado
   solo hace falta el paso hacia delante. Con 16 tiles por muestra, todo se multiplica por 16
   frente a una sola imagen.

## 5. Recomendación para el alumno

**Por dónde empezar**

1. **Checkpoint JEPA: `facebook/ijepa_vith14_1k`** (ViT-H/14, 224 px, `transformers >= 4.47.0`)
   [11][13][24][25], con **codificador congelado + cabeza** como primer experimento. Es el
   checkpoint I-JEPA más pequeño disponible, y el que tiene la carga más estándar.
2. **Segunda opción de tamaño asumible: V-JEPA 2.1 ViT-B/16 (80M)** [19], para tener un JEPA de
   tamaño comparable. Faltan licencia verificada y carga solo por `torch.hub`.
3. **Referencia de patología: UNI (ViT-L/16, `timm`)** [28] y, si el acceso se retrasa, Phikon-v2
   [36]. Sirven para saber cuánto se gana con preentrenamiento específico de dominio.
4. **Control**: ViT-S/16 de la línea base (ImageNet supervisado) con el mismo protocolo congelado.
   Sin él, no se sabe si la diferencia viene del tamaño o del método.

**Protocolo de comparación** (propuesta): mismas particiones que la línea base, **3 semillas**,
QWK sobre ISUP 0-5 y los cortes clínicos ISUP ≥ 1 y ISUP ≥ 2 como en el resto del proyecto, y
la misma resolución de mosaico. Tabular también parámetros, GMACs, tiempo por época y memoria
máxima (el propio predictor del proyecto puede validarse con estos nuevos modelos). Informar de la
caída entre centros (Radboud a Karolinska) además de la media, dada la cifra de [10].

**Riesgos**

- **Licencias no comerciales**: todos los pesos de I-JEPA y los de UNI, CONCH, Virchow2 y
  H-optimus-1 son CC BY-NC(-ND) [13][14][15][16][28][29][30][32][35]; nada de redistribuir
  pesos ni mezclarlos en un artefacto público. Los ND impiden publicar versiones derivadas.
  Para publicar resultados académicos es viable; para un uso comercial, no.
- **Acceso con solicitud** y plazos (correo institucional) [28][29][32].
- **Tamaño**: ViT-H es ~29 veces mayor en parámetros que ViT-S/16; el ajuste fino completo no
  cabe en 8 GB.
- **Resolución y dominio**: I-JEPA se entrenó con ImageNet a 224 px [11]; la histología es otro
  dominio. Es posible que rinda por debajo de los modelos de patología, sobre todo por [9], pero es
  una hipótesis.
- **Incoherencias de licencia** en las fichas de V-JEPA 2 (MIT frente a Apache-2.0) [20][21].

**Qué debería ofrecer el framework (sin tocar `EpochController.fit()`)**

1. Un **adaptador de backbone** con un contrato mínimo: `build()`, `embed_dim`, `forward_features`
   que devuelva el vector global, `img_size`, `patch_size`, media y desviación, y la regla de
   pooling (CLS, media o CLS + media). Origen configurable en el YAML: `timm`, `transformers`,
   `torch.hub`.
2. **Modo congelado** (sonda lineal) y descongelado con LLRD generalizado a cualquier lista de
   bloques, más caché opcional de embeddings para repetir semillas sin repetir el paso hacia delante.
3. **Metadatos de licencia y acceso** por backbone y un aviso al usarlo.
4. Manejo del **mosaico de 16 tiles** compatible con el nuevo backbone y la interpolación de
   posiciones; que el predictor reciba parámetros y GMACs del adaptador.

## 6. Qué no se ha podido verificar

- Licencia de los pesos V-JEPA 2.1, de `vitg-fpc64-256` y de `vith`; coherencia entre las etiquetas
  HF y el repo.
- Que los pesos de I-JEPA carguen en `timm` (no hay receta); mapeo de claves.
- Existencia de pesos de RadJEPA; datos y cifras de [9]; los trabajos 2606.13315, 2607.11962 y el
  preprint de Liu, Lu y Balestriero sobre objetivos JEPA/LeJEPA en lámina completa (solo vistos en
  un buscador).
- Método de preentrenamiento de H-optimus (la ficha no lo detalla) y el de Prov-GigaPath para
  tiles (la ficha no lo describe).
- Metadatos de las publicaciones en *Nature* y *Nature Medicine* de UNI, CONCH y Prov-GigaPath: el
  sitio de la revista bloqueó el acceso; se han tomado de los README oficiales [28][30][39].
- GMACs exactos del ViT-H/14 de I-JEPA (se usa como aproximación el de `timm`, misma arquitectura
  nominal).
- Resultados de V-JEPA 2 en imagen y cifras de coste por época: no se ha medido nada en máquina.

## Referencias

1. Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., Ballas, N.
   "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture". CVPR 2023,
   pp. 15619-15629. arXiv:2301.08243,
   [arxiv.org/abs/2301.08243](https://arxiv.org/abs/2301.08243). (Texto completo leído:
   apéndice A.1 y Tablas 1-2.)
2. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R. "Masked Autoencoders Are Scalable
   Vision Learners". arXiv:2111.06377 (2021),
   [arxiv.org/abs/2111.06377](https://arxiv.org/abs/2111.06377).
3. Oquab, M. et al. "DINOv2: Learning Robust Visual Features without Supervision". arXiv:2304.07193
   (2023), [arxiv.org/abs/2304.07193](https://arxiv.org/abs/2304.07193).
4. Zhou, J. et al. "iBOT: Image BERT Pre-Training with Online Tokenizer". arXiv:2111.07832 (2021),
   [arxiv.org/abs/2111.07832](https://arxiv.org/abs/2111.07832).
5. Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran, M., Ballas, N.
   "Revisiting Feature Prediction for Learning Visual Representations from Video". arXiv:2404.08471
   (2024), [arxiv.org/abs/2404.08471](https://arxiv.org/abs/2404.08471).
6. Assran, M. et al. "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and
   Planning". arXiv:2506.09985 (2025), [arxiv.org/abs/2506.09985](https://arxiv.org/abs/2506.09985).
7. Mur-Labadia, L. et al. "V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning".
   arXiv:2603.14482 (2026), [arxiv.org/abs/2603.14482](https://arxiv.org/abs/2603.14482).
8. Khan, A. A. H., Husain, M., Jalan, P., Jadhav, K. "RadJEPA: Radiology Encoder for Chest X-Rays via
   Joint Embedding Predictive Architecture". arXiv:2601.15891 (2026),
   [arxiv.org/abs/2601.15891](https://arxiv.org/abs/2601.15891) (solo resumen).
9. Ivezić, V. et al. "Pretext Matters: An Empirical Study of SSL Methods in Medical Imaging".
   arXiv:2603.22649 (2026), [arxiv.org/abs/2603.22649](https://arxiv.org/abs/2603.22649) (solo
   resumen).
10. Gustafsson, F. K., Rantalainen, M. "Evaluating Computational Pathology Foundation Models for
    Prostate Cancer Grading under Distribution Shifts". arXiv:2410.06723 (v2, 2026),
    [arxiv.org/abs/2410.06723](https://arxiv.org/abs/2410.06723) (cifras de la versión
    [HTML](https://arxiv.org/html/2410.06723)). Es la referencia 12 de [panda.md](panda.md).
11. Repositorio oficial I-JEPA (README y `LICENSE`, CC BY-NC 4.0),
    [github.com/facebookresearch/ijepa](https://github.com/facebookresearch/ijepa); licencia:
    [LICENSE](https://raw.githubusercontent.com/facebookresearch/ijepa/main/LICENSE).
12. Listado `facebook` con "ijepa" en Hugging Face,
    [huggingface.co/facebook/models?search=ijepa](https://huggingface.co/facebook/models?search=ijepa).
13. [huggingface.co/facebook/ijepa_vith14_1k](https://huggingface.co/facebook/ijepa_vith14_1k):
    cc-by-nc-4.0, 0,6B parámetros.
14. [huggingface.co/facebook/ijepa_vith16_1k](https://huggingface.co/facebook/ijepa_vith16_1k):
    cc-by-nc-4.0, 0,6B.
15. [huggingface.co/facebook/ijepa_vith14_22k](https://huggingface.co/facebook/ijepa_vith14_22k):
    cc-by-nc-4.0, 0,6B.
16. [huggingface.co/facebook/ijepa_vitg16_22k](https://huggingface.co/facebook/ijepa_vitg16_22k):
    cc-by-nc-4.0, 1B.
17. Repositorio V-JEPA (checkpoints y arXiv),
    [github.com/facebookresearch/jepa](https://github.com/facebookresearch/jepa).
18. Licencia de V-JEPA (CC BY-NC 4.0),
    [LICENSE](https://raw.githubusercontent.com/facebookresearch/jepa/main/LICENSE).
19. Repositorio V-JEPA 2 (checkpoints, V-JEPA 2.1, `torch.hub`, licencia),
    [github.com/facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2).
20. [huggingface.co/facebook/vjepa2-vitl-fpc64-256](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256):
    MIT, 0,3B, ejemplo con `AutoModel`.
21. [huggingface.co/facebook/vjepa2-vitg-fpc64-384](https://huggingface.co/facebook/vjepa2-vitg-fpc64-384):
    apache-2.0, 1B.
22. Licencia del repositorio V-JEPA 2 (MIT),
    [LICENSE](https://raw.githubusercontent.com/facebookresearch/vjepa2/main/LICENSE).
23. Colección V-JEPA 2 en Hugging Face,
    [huggingface.co/collections/facebook/v-jepa-2-...](https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6).
24. Documentación de `transformers` sobre I-JEPA,
    [huggingface.co/docs/transformers/model_doc/ijepa](https://huggingface.co/docs/transformers/model_doc/ijepa).
25. Notas de la versión 4.47.0 de `transformers` (título: "PaliGemma-2, I-JEPA, OLMo-2…"),
    [newreleases.io](https://newreleases.io/project/pypi/transformers/release/4.47.0) (espejo; no
    se abrió la página oficial de GitHub).
26. Documentación de `transformers` sobre V-JEPA 2 (v4.53.0),
    [huggingface.co/docs/transformers/v4.53.0/en/model_doc/vjepa2](https://huggingface.co/docs/transformers/v4.53.0/en/model_doc/vjepa2).
27. Etiqueta previa y versión 4.53.0 (espejos),
    [newreleases.io/.../v4.52.4-VJEPA-2-preview](https://newreleases.io/project/github/huggingface/transformers/release/v4.52.4-VJEPA-2-preview),
    [newreleases.io/.../4.53.0](https://newreleases.io/project/pypi/transformers/release/4.53.0).
28. UNI (Chen, R. J. et al. "Towards a General-Purpose Foundation Model for Computational
    Pathology". *Nature Medicine* 2024, DOI
    [10.1038/s41591-024-02857-3](https://doi.org/10.1038/s41591-024-02857-3), citado desde el
    README); ficha [huggingface.co/MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI).
29. [huggingface.co/MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h).
30. CONCH (Lu, M. Y. et al. "A visual-language foundation model for computational pathology".
    *Nature Medicine* 30, 863-874, 2024, según el README);
    [huggingface.co/MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH),
    [github.com/mahmoodlab/CONCH](https://github.com/mahmoodlab/CONCH).
31. [huggingface.co/paige-ai/Virchow](https://huggingface.co/paige-ai/Virchow).
32. [huggingface.co/paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2).
33. Zimmermann, E. et al. "Virchow2: Scaling Self-Supervised Mixed Magnification Models in
    Pathology". arXiv:2408.00738 (2024), [arxiv.org/abs/2408.00738](https://arxiv.org/abs/2408.00738).
34. [huggingface.co/bioptimus/H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0).
35. [huggingface.co/bioptimus/H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1).
36. [huggingface.co/owkin/phikon-v2](https://huggingface.co/owkin/phikon-v2).
37. Filiot, A., Jacob, P., Mac Kain, A., Saillard, C. "Phikon-v2, A large and public feature
    extractor for biomarker prediction". arXiv:2409.09173 (2024),
    [arxiv.org/abs/2409.09173](https://arxiv.org/abs/2409.09173). Repositorio de Phikon:
    [github.com/owkin/HistoSSLscaling](https://github.com/owkin/HistoSSLscaling) (iBOT; no
    menciona PANDA).
38. Wang, X. et al. "Transformer-based Unsupervised Contrastive Learning for Histopathological Image
    Classification". *Medical Image Analysis* 2022 (según el README),
    [github.com/Xiyue-Wang/TransPath](https://github.com/Xiyue-Wang/TransPath).
39. Xu, H. et al. "A whole-slide foundation model for digital pathology from real-world data".
    *Nature* 2024 (según el README);
    [github.com/prov-gigapath/prov-gigapath](https://github.com/prov-gigapath/prov-gigapath),
    [huggingface.co/prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath).
40. Tabla de tareas a nivel de lámina del repositorio de UNI,
    [github.com/mahmoodlab/UNI](https://github.com/mahmoodlab/UNI).
41. Ebbert, J. L., Della Corte, D. "PANDA-PLUS-Bench: A Clinical Benchmark for Evaluating Robustness
    of AI Foundation Models in Prostate Cancer Diagnosis". arXiv:2512.14922 (2025),
    [arxiv.org/abs/2512.14922](https://arxiv.org/abs/2512.14922) (solo resumen).
42. [huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k](https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k).
43. [huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k).
44. [huggingface.co/timm/vit_large_patch16_224.augreg_in21k_ft_in1k](https://huggingface.co/timm/vit_large_patch16_224.augreg_in21k_ft_in1k).
45. [huggingface.co/timm/vit_huge_patch14_224.orig_in21k](https://huggingface.co/timm/vit_huge_patch14_224.orig_in21k).
