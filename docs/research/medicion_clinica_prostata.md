# Medición clínica del cáncer de próstata y evaluación de modelos de IA

> Revisión para entender cómo se mide el cáncer de próstata en la clínica, cómo se miden los
> modelos de IA sobre ese problema y qué pedir si un médico cede un dataset propio. Fecha:
> 18/11/2026. Cada fuente se ha consultado durante la revisión. Las cifras son **las que dan los
> autores**, con su configuración; la interpretación propia va marcada como tal. Lo que no se ha
> podido contrastar con la fuente primaria se indica como *no verificado*. El apartado legal
> recoge qué suele hacer falta y **no es asesoría jurídica**.

Resumen en cuatro líneas: el grado clínico es el **grupo de grado ISUP 1-5** (derivado de la
puntuación Gleason), y el corte que más pesa en la práctica es entre el grupo 1 y el resto. Los
patólogos no coinciden del todo entre sí (kappa entre 0,4 y 0,9 según quiénes y cómo se midan), de
modo que la etiqueta tiene ruido. Los trabajos de referencia reportan **kappa cuadrático, AUC y
sensibilidad/especificidad con validación externa**. Un dataset clínico nuevo vale por tres
cosas: etiquetas fiables, identificador de paciente y datos de centro y escáner.

## 1. Cómo se gradúa el cáncer de próstata en la clínica

### 1.1 Gleason y grupos de grado

El patólogo mira la biopsia con el microscopio (o en pantalla) y asigna **patrones de Gleason**
según lo que parecen las glándulas tumorales [1]:

| Patrón | Qué se ve (según la revisión [1]) |
|---|---|
| 3 | Glándulas bien formadas, individuales y separadas |
| 4 | Glándulas mal formadas, fusionadas o cribiformes (también las glomeruloides) |
| 5 | Láminas de células, células sueltas o cordones; a veces comedonecrosis |

La **puntuación Gleason** suma el patrón predominante y el segundo más frecuente (por ejemplo,
3+4 = 7; descripción general, no cifrada en las fuentes abiertas aquí). El problema es que el 7
mezcla dos tumores de pronóstico distinto: 3+4 y 4+3. La conferencia de consenso de la ISUP de
2014 propuso cinco **grupos de grado** que lo resuelven [2], aceptados por la OMS en su edición
de 2016 [1][2]:

| Grupo de grado (ISUP) | Puntuación Gleason |
|---|---|
| 1 | 6 o menos |
| 2 | 3+4 = 7 |
| 3 | 4+3 = 7 |
| 4 | 8 |
| 5 | 9-10 |

(Tabla según [1]; la tabla de las guías de la EAU [4] es equivalente.) Una conferencia posterior
(ISUP 2019) trató cómo informar los patrones 4 y 5, el carcinoma cribiforme y las biopsias
dirigidas por resonancia [3]; no se ha revisado su detalle.

### 1.2 Relación con el pronóstico

Los grupos se validaron con grandes cohortes tratadas con cirugía o radioterapia. El artículo de
validación [5] (20.845 hombres operados en cinco centros y 5.501 con radioterapia en dos) mide la
recurrencia bioquímica (subida del PSA tras el tratamiento), no la muerte por cáncer. Los autores
dan, frente a Gleason 6 en la cohorte quirúrgica, cocientes de riesgo de unos 1,9 (3+4), 5,1 (4+3),
8,0 (Gleason 8) y 11,7 (Gleason 9-10) [5]. La revisión [1] resume el mismo resultado como
supervivencia libre de recurrencia bioquímica a cinco años del **96 %, 88 %, 63 %, 48 % y 26 %**
para los grupos 1 a 5. Lectura propia: el pronóstico empeora de forma escalonada, pero el gran
salto está entre los grupos 2 y 3, y la diferencia entre el 1 y el 2 es pequeña.

### 1.3 "Cáncer clínicamente significativo" y vigilancia activa

**No existe una definición única.** Las dos más citadas en estas fuentes:

- **Por grado:** ISUP ≥ 2. Es la convención habitual en la literatura de IA; por ejemplo, Bulten et
  al. (2020) reportan el AUC para "grupo ≥ 2" y "grupo ≥ 3" [8].
- **Por estudio clínico:** el ensayo PROMIS (resonancia frente a biopsia) la define como
  Gleason ≥ 4+3 **o** longitud máxima de cáncer en un cilindro ≥ 6 mm [6]. Es una definición más
  estricta que "ISUP ≥ 2" y mezcla grado con tamaño.

Por qué el corte importa: las guías europeas (EAU) señalan que el grupo 1 en la pieza quirúrgica
"puede considerarse clínicamente insignificante" y, junto con PSA < 10 ng/ml y estadio cT1-2,
forma el grupo de **riesgo bajo**, para el que se recomienda **vigilancia activa** (seguimiento
sin tratar) [4]. Los grupos 2 en adelante son, en general, riesgo intermedio o alto y se plantea
tratamiento, con vigilancia activa solo en casos muy seleccionados dentro del grupo 2 [4]. Las
mismas guías advierten de que la biopsia puede infravalorar el grado y de que el grupo 1 progresa
a un ritmo de en torno al 1 % anual, por lo que no se decide solo con el grado [4]. Lectura
propia: de ahí que el corte **ISUP ≥ 2** sea el que más importa clínicamente (¿vigilar o tratar?)
y el corte **ISUP ≥ 1** responda a otra pregunta (¿hay cáncer?).

### 1.4 Variabilidad entre patólogos

La etiqueta de referencia no es una verdad exacta. Cifras publicadas (cada una con su diseño, por
lo que **no son comparables entre sí**):

| Estudio | Qué se midió | Resultado de los autores |
|---|---|---|
| Allsbrook et al. 2001 [9] | 41 patólogos generales, 38 biopsias, contra el consenso de uropatólogos | kappa 0,435 (rango por caso 0,00-0,88); tienden a infragraduar |
| Allsbrook et al. 2001 [10] | 10 uropatólogos, 46 biopsias | kappa ponderado por pares de unos 0,56-0,70 |
| Egevad et al. 2013 [11] | 337 patólogos europeos, 85 microfotografías de 15 biopsias, frente al consenso de 15 expertos | acuerdo con el consenso del 71,4 % de media en Gleason 6 y del 56,4 % en Gleason 7 |
| Ozkan et al. 2016 [12] | 2 patólogos, 407 cilindros de 34 pacientes | kappa 0,39 en grupos de grado (acuerdo del 52 %); estudio pequeño |
| Ström et al. 2020 [7] | 23 uropatólogos, 87 biopsias | kappa por pares de 0,60-0,73 |
| Bulten et al. 2022 [13] | QWK entre uropatólogos (consenso interno y externo EE. UU.) | 0,926 y 0,907; patólogos generales contra el consenso: 0,765 (70 casos) y 0,820 (237 casos) |
| Bulten et al. 2020 [8] | Observador con 100 biopsias, 15 patólogos | kappa mediano del panel 0,819 |

Lectura propia: entre expertos el acuerdo es alto (QWK ≈ 0,9), pero baja con patólogos generales y
en cuanto se mide por grupo de grado exacto (kappa no ponderado 0,4-0,6). Un modelo que llegue a
QWK ≈ 0,85-0,9 contra una etiqueta de un solo patólogo ya está cerca del **techo de ruido** de esa
etiqueta.

## 2. Cómo se miden los modelos de IA en este problema

### 2.1 Qué responde cada métrica

| Métrica | Qué responde, en lenguaje clínico | Cifras de referencia publicadas |
|---|---|---|
| **Kappa cuadrático ponderado (QWK)** | ¿Cuánto coincide el modelo con la referencia en los 6 niveles (0-5), penalizando más los errores grandes? Confundir 1 con 5 cuesta mucho más que 1 con 2 | PANDA [13]: 0,931 (validación interna), 0,862 (EE. UU.), 0,868 (Europa). Bulten 2020 [8]: 0,918 (test), 0,723 y 0,707 (externo, dos patólogos) |
| **AUC, cáncer sí/no (ISUP ≥ 1)** | Con un umbral libre, ¿separa bien las biopsias con cáncer de las benignas? | Bulten 2020 [8]: 0,990. Ström [7]: 0,997 (test) y 0,986 (externo). Pantanowitz [14]: ≈0,997 y ≈0,991 (externo, otro escáner) |
| **AUC, ISUP ≥ 2 (y ≥ 3)** | ¿Separa lo que obliga a decidir tratamiento de lo vigilable? | Bulten 2020 [8]: 0,978 (≥ 2) y 0,974 (≥ 3) |
| **Sensibilidad, ISUP ≥ 1** | De las biopsias con cáncer, ¿a cuántas avisa? (falsos negativos = cáncer perdido) | PANDA [13], por extracción automática de las tablas, no contrastada: 98,6 % (EE. UU.) y 97,7 % (Europa) |
| **Especificidad, ISUP ≥ 1** | De las benignas, ¿a cuántas deja en paz? (falsos positivos = trabajo extra, ansiedad) | PANDA [13], mismo aviso: 75,2 % (EE. UU.) y 84,3 % (Europa); patólogos generales de EE. UU.: 95,0 %. Los autores atribuyen el sobrediagnóstico de benignas como grupo 1 en los externos a un cambio de distribución |
| **Sensibilidad y especificidad, ISUP ≥ 2** | De los tumores que justifican tratar, ¿a cuántos detecta? Y de los vigilables, ¿a cuántos trata de más? | PANDA [13] **no** las reporta en este corte; no hay cifra de referencia en las fuentes revisadas |
| **Kappa/QWK contra patólogos** | ¿Está dentro del desacuerdo normal entre humanos? | Bulten 2020 [8]: modelo 0,854 frente a 0,819 del panel; Ström [7]: 0,62 dentro del rango 0,60-0,73; PANDA [13]: algoritmos 0,876 frente a 0,765 de patólogos internacionales y 0,828 frente a 0,820 en EE. UU. |

**Por qué el kappa cuadrático.** Los grados son ordinales (0 < 1 < ... < 5), de modo que la
exactitud (acierto/fallo) trata igual un error de un escalón que uno de cinco. El kappa ponderado
[15] da crédito parcial al desacuerdo según la distancia, y con pesos cuadráticos el error crece
con el cuadrado de la distancia; además descuenta el acuerdo esperable por azar, lo que importa
con clases desiguales. Es la métrica oficial del reto PANDA [13] y de la mayoría de trabajos del
área. Limitaciones (interpretación propia): un solo número no distingue *dónde* falla (cáncer sí/no
frente a grado) ni dice nada de falsos negativos clínicamente graves, y depende de la
distribución de clases, así que se compara solo entre conjuntos similares.

### 2.2 Validación externa

Un modelo que funciona en el hospital donde se entrenó puede caer en otro (distinta tinción,
escáner, población). Por eso los trabajos de referencia validan con **centros distintos**:

- **PANDA (Bulten 2022)** [13]: 10.616 biopsias de desarrollo (Radboud y Karolinska; cuatro modelos
  de escáner de tres fabricantes), validación interna de 545 biopsias, y externas de **741
  biopsias de EE. UU.** (referencia: panel de seis uropatólogos) y **330 de Karolinska** (un
  uropatólogo). Los algoritmos se aplicaron a la cohorte de EE. UU. sin reentrenar. Limitaciones
  que reconocen: evaluación por biopsia (no por paciente), solo adenocarcinoma acinar,
  referencias de uno o pocos patólogos, datos retrospectivos, población mayoritariamente blanca y
  sin enlace con resultados clínicos. Solo se validaron 15 de 1.010 equipos [13].
- **Ström 2020** [7]: 6.682 láminas de 976 hombres del estudio STHLM3 para entrenar; test interno
  de 1.631 biopsias y **externo de 330 biopsias de 73 hombres**.
- **Bulten 2020** [8]: test de 550 biopsias con consenso de tres uropatólogos y externo de 886
  cilindros (245 de otro centro). El kappa cayó de 0,918 a 0,71-0,72, que los autores sitúan
  dentro de la variabilidad entre observadores.
- **Pantanowitz 2020** [14]: validación ciega con 100 casos consecutivos escaneados en otro
  escáner, y uso posterior como segunda lectura en 11.429 láminas de 941 casos.

Cuando el centro se puede reconocer en la imagen, hay otro riesgo: que el modelo aprenda el centro
y no el tumor (ver [panda.md](panda.md), sección 5).

## 3. Qué pedirle a un dataset clínico nuevo

Lista pensada para que el tutor se la pase al médico. Cada punto indica por qué importa.

**Imagen**
- Formato original de la lámina digitalizada (WSI): SVS, NDPI, MRXS, TIFF multinivel o DICOM. Sin
  recomprimir ni convertir a JPEG. DICOM para patología digital es un estándar real [16].
- Aumento (20x o 40x) y **micras por píxel** del nivel base; fabricante y modelo de escáner.
  Como referencia, PANDA va a ~0,5 µm/px (20x) según las fuentes citadas en [panda.md](panda.md).
- Tipo de tinción (hematoxilina-eosina) y si hay láminas con marcadores.
- Qué es una unidad: biopsia, cilindro o lámina; cuántos cilindros por biopsia.

**Etiquetas**
- Grado ISUP y/o Gleason (patrones primario y secundario) **por biopsia** y, si es posible, por
  paciente (la biopsia más alta). Distinguir si la etiqueta es del informe de rutina o de una
  revisión.
- **Quién etiquetó y cuántos patólogos**, su experiencia (uropatólogo o patólogo general), si
  vieron la clínica y si hubo consenso. Con la variabilidad de la sección 1.4, una etiqueta de un
  solo patólogo tiene ruido propio.
- Año de la biopsia y versión del criterio (Gleason 2005, ISUP 2014 o 2019); cambia el grado.
- Anotaciones de región (máscaras de patrones, límites del tumor), si existen, y cómo se hicieron.
- Si hay seguimiento: prostatectomía posterior, recurrencia o supervivencia. Es lo único que
  permite hablar de pronóstico y no solo de acuerdo con un patólogo [13].

**Identificación y reparto**
- Un **identificador de paciente** (seudónimo) en cada biopsia, para repartir **por paciente**:
  un paciente con varias biopsias no puede estar en entrenamiento y test a la vez. PANDA publicó
  10.616 biopsias de 2.113 pacientes sin ese identificador, y el proyecto reparte por biopsia
  (limitación en [datasets/panda.md](../datasets/panda.md)).
- Centro, fecha y escáner de cada lámina, para medir por subgrupo y para validar en un centro
  distinto del de entrenamiento.

**Datos clínicos opcionales y sesgos**
- Edad y PSA al diagnóstico, estadio clínico, número de cilindros positivos y porcentaje de
  afectación: son los factores que las guías europeas usan junto al grado [4].
- Cómo se seleccionó el conjunto: casos consecutivos o elegidos, lugar de procedencia, periodo.
  Un conjunto "limpio" (solo casos claros) infla las métricas; uno de un único centro y escáner
  no dice nada del resto.
- **Número de casos por grado** (con el reparto por centro). Con pocos casos en un grado, las
  métricas por grado tienen intervalos muy anchos (el proyecto calcula intervalos bootstrap, pero
  no puede crear datos). Conviene pedir esta tabla antes de aceptar el conjunto.

**Anonimización y base legal (qué suele hacer falta, sin asesoría legal)**
- Es **dato de salud**, categoría especial del RGPD: el tratamiento está prohibido por defecto, y
  se admite para investigación científica con base en el Derecho de la UE o del Estado miembro y
  con garantías (art. 9.1 y 9.2.j) [17]. El tratamiento a gran escala de estas categorías exige
  evaluación de impacto (art. 35.3.b) [17]. Los datos verdaderamente anónimos quedan fuera del
  RGPD, pero los seudonimizados siguen siendo datos personales (considerando 26) [17].
- En España, la Ley Orgánica 3/2018 (LOPDGDD), disposición adicional decimoséptima, regula los
  datos de salud en investigación [18]. Según el texto consultado: el interesado puede consentir
  usos amplios dentro de un área de investigación; el uso de **datos seudonimizados** exige separar
  técnica y funcionalmente al equipo de investigación de quien seudonimiza, y un **informe previo
  del comité de ética de la investigación**; la reidentificación solo cabe en supuestos tasados.
  El extracto del texto estaba parcialmente cortado, de modo que los detalles (por ejemplo, los
  plazos de la figura del delegado de protección de datos) deben leerse en el BOE.
- La Ley 14/2007 de Investigación biomédica [19] exige informe favorable de un **Comité de Ética de
  la Investigación** acreditado para proyectos con personas o muestras biológicas (arts. 2 y 12) y
  consentimiento expreso, específico y escrito como regla general (arts. 4 y 13). Define dato
  anonimizado (no asociable a persona, art. 3.i) y codificado (reversible, art. 3.k). Nota: esta
  ley remite a la LO 15/1999, hoy sustituida por la normativa de [17] y [18].
- La AEPD publicó una guía de orientaciones y garantías para la anonimización [20]; indica que
  anonimizar no garantiza al 100 % la no reidentificación y recomienda complementar las medidas
  técnicas con compromisos jurídicos (licencias o acuerdos de confidencialidad).
- En la práctica, el centro cedente suele pedir un **acuerdo de cesión de datos** entre
  instituciones (qué se cede, para qué, quién accede, cuánto tiempo, publicación). No se ha
  encontrado una plantilla oficial verificable; debe canalizarlo la Universidad con el servicio
  jurídico y el delegado de protección de datos de ambas partes.
- Las imágenes WSI pueden llevar datos personales en los metadatos o en la etiqueta física de la
  lámina (nombre, historia clínica): pedir que se eliminen y se compruebe (interpretación propia).

## 4. Implicaciones para ParaViT-Lab

**Qué conviene reportar** (interpretación propia, apoyada en la sección 2):
1. **QWK** con intervalo de confianza, en validación y en test, nunca solo exactitud.
2. **AUC, sensibilidad y especificidad en ISUP ≥ 1 y ISUP ≥ 2**, con un umbral elegido en
   validación (no en test) y con intervalo. Con un clasificador softmax de seis clases, la
   probabilidad de "≥ k" es la suma de las clases k a 5.
3. **Por centro y por escáner**, y al menos un resultado de **validación externa**: entrenar sin un
   centro y probar en él, que es lo que de verdad dice si generaliza.
4. **Matriz de confusión** (errores de un escalón frente a errores grandes) y **calibración**, para
   leer las probabilidades como riesgo.
5. Referencia humana: qué acuerdo hay entre patólogos en ese conjunto (sección 1.4).

**Qué ya tiene el proyecto** (según el código y `docs/`): QWK en `eval.py` y como `select_by: qwk`;
AUC macro; intervalos bootstrap (`src/analysis/bootstrap.py`); calibración (ECE, Brier,
`src/analysis/calibration.py`); la vista *Subgroups* por centro (`src/analysis/subgroups.py`), que
espera un `groups.npz` que el entrenamiento aún no escribe; una lista TRIPOD+AI [21]
(`src/reporting/tripod.py`); y una línea base en PANDA (QWK 0,781 en test, 0,745/0,760 por centro).
Otro agente está añadiendo a `eval.py` los cortes clínicos ISUP ≥ 1 e ISUP ≥ 2.

**Qué le falta** (a revisar, no es una decisión tomada):
- Escribir `groups.npz` (centro, escáner) para que *Subgroups* funcione con datos reales.
- Partición **por paciente**, que PANDA no permite y un dataset clínico sí debería.
- Sensibilidad y especificidad con intervalo en los dos cortes y elección del umbral en
  validación.
- Un protocolo de validación **por centro** (dejar un centro fuera), no solo por subgrupo.
- Comparar con la variabilidad entre patólogos de ese dataset, si trae varias lecturas.
- Una **cifra de referencia en ISUP ≥ 2** propia: PANDA no la publica en sensibilidad/especificidad
  [13], así que la comparación será con el AUC de [8] y con lo que mida el proyecto.

## 5. Qué no se ha podido verificar

- **Sensibilidades y especificidades de PANDA** [13] (sección 2.1): leídas del texto con una
  herramienta automática; algunos intervalos de confianza no contienen la estimación puntual, y
  son rangos entre algoritmos. Úsense solo como orden de magnitud hasta contrastarlas con la
  tabla original.
- **PubMed** y la página de *Nature Medicine* no se pudieron abrir (exigen cookies o autenticación);
  los datos bibliográficos se contrastaron en Europe PMC, Crossref y PubMed Central. Los resúmenes
  de [7], [8], [10] y [11] se leyeron en Europe PMC y no el texto completo.
- Los **resultados de Gordetsky y Epstein [1]** sobre la supervivencia a cinco años proceden de
  una revisión; el valor por grupo no aparece en el resumen del artículo original [5], que solo
  da cocientes de riesgo.
- **Texto completo de la LOPDGDD y del RGPD**: lectura parcial; el art. 89 del RGPD no se pudo
  revisar. El texto sobre el acuerdo de cesión de datos no tiene fuente oficial.
- Contenido de las conferencias ISUP 2019 [3] más allá del resumen.
- No se buscó un ensayo que fije "ISUP ≥ 2" como definición de cáncer significativo; se afirma
  solo como convención de la literatura de IA, y la definición del ensayo PROMIS [6] es distinta.

## Referencias

1. Gordetsky J, Epstein J. "Grading of prostatic adenocarcinoma: current state and prognostic
   implications". *Diagnostic Pathology* 11:25 (2016). DOI:
   [10.1186/s13000-016-0478-2](https://doi.org/10.1186/s13000-016-0478-2) (texto en
   [PMC4784293](https://pmc.ncbi.nlm.nih.gov/articles/PMC4784293)).
2. Epstein JI, Egevad L, Amin MB, Delahunt B, Srigley JR, Humphrey PA; Grading Committee. "The 2014
   International Society of Urological Pathology (ISUP) Consensus Conference on Gleason Grading of
   Prostatic Carcinoma: Definition of Grading Patterns and Proposal for a New Grading System".
   *Am J Surg Pathol* 40(2):244-252 (2016). DOI:
   [10.1097/PAS.0000000000000530](https://doi.org/10.1097/PAS.0000000000000530).
3. van Leenders GJLH, van der Kwast TH, Grignon DJ, et al. "The 2019 International Society of
   Urological Pathology (ISUP) Consensus Conference on Grading of Prostatic Carcinoma". *Am J Surg
   Pathol* 44(8):e87-e99 (2020). DOI:
   [10.1097/PAS.0000000000001497](https://doi.org/10.1097/PAS.0000000000001497) (PMC7382533).
4. EAU-EANM-ESTRO-ESUR-ISUP-SIOG Guidelines on Prostate Cancer, edición 2026, cap. 4
   "Classification and staging systems":
   [uroweb.org/guidelines/prostate-cancer/chapter/classification-and-staging-systems](https://uroweb.org/guidelines/prostate-cancer/chapter/classification-and-staging-systems).
   Las páginas de la guía están en revisión continua; contrástese la edición vigente.
5. Epstein JI, Zelefsky MJ, Sjoberg DD, et al. "A Contemporary Prostate Cancer Grading System: A
   Validated Alternative to the Gleason Score". *Eur Urol* 69(3):428-435 (2016). DOI:
   [10.1016/j.eururo.2015.06.046](https://doi.org/10.1016/j.eururo.2015.06.046).
6. Ahmed HU, El-Shater Bosaily A, Brown LC, et al. "Diagnostic accuracy of multi-parametric MRI and
   TRUS biopsy in prostate cancer (PROMIS): a paired validating confirmatory study". *Lancet*
   389(10071):815-822 (2017). Definición de cáncer significativo: Gleason ≥ 4+3 o longitud
   máxima ≥ 6 mm. DOI: [10.1016/S0140-6736(16)32401-1](https://doi.org/10.1016/S0140-6736(16)32401-1)
   (registro en Europe PMC; la definición se leyó en el resumen de
   [eprints.whiterose.ac.uk/128690](https://eprints.whiterose.ac.uk/128690/)).
7. Ström P, Kartasalo K, Olsson H, et al. "Artificial intelligence for diagnosis and grading of
   prostate cancer in biopsies: a population-based, diagnostic study". *Lancet Oncol*
   21(2):222-232 (2020). DOI:
   [10.1016/S1470-2045(19)30738-7](https://doi.org/10.1016/S1470-2045(19)30738-7).
8. Bulten W, Pinckaers H, van Boven H, et al. "Automated deep-learning system for Gleason grading
   of prostate cancer using biopsies: a diagnostic study". *Lancet Oncol* 21(2):233-241 (2020).
   DOI: [10.1016/S1470-2045(19)30739-9](https://doi.org/10.1016/S1470-2045(19)30739-9). Preprint:
   [arXiv:1907.07980](https://arxiv.org/abs/1907.07980) (las cifras del preprint difieren
   ligeramente).
9. Allsbrook WC Jr, Mangold KA, Johnson MH, Lane RB, Lane CG, Epstein JI. "Interobserver
   reproducibility of Gleason grading of prostatic carcinoma: General pathologist". *Hum Pathol*
   32(1):81-88 (2001). DOI: [10.1053/hupa.2001.21135](https://doi.org/10.1053/hupa.2001.21135).
10. Allsbrook WC Jr, et al. "Interobserver reproducibility of Gleason grading of prostatic
    carcinoma: Urologic pathologists". *Hum Pathol* 32(1):74-80 (2001). DOI:
    [10.1053/hupa.2001.21134](https://doi.org/10.1053/hupa.2001.21134).
11. Egevad L, Ahmad AS, Algaba F, et al. "Standardization of Gleason grading among 337 European
    pathologists". *Histopathology* 62(2):247-256 (2013). DOI:
    [10.1111/his.12008](https://doi.org/10.1111/his.12008).
12. Ozkan TA, Eruyar AT, Cebeci OO, Memik O, Ozcan L, Kuskonmaz I. "Interobserver variability in
    Gleason histological grading of prostate cancer". *Scand J Urol* 50(6):420-424 (2016). DOI:
    [10.1080/21681805.2016.1206619](https://doi.org/10.1080/21681805.2016.1206619).
13. Bulten W, Kartasalo K, Chen PC, Ström P, Pinckaers H, Nagpal K, et al.; PANDA challenge
    consortium. "Artificial intelligence for diagnosis and Gleason grading of prostate cancer: the
    PANDA challenge". *Nat Med* 28(1):154-163 (2022). DOI:
    [10.1038/s41591-021-01620-2](https://doi.org/10.1038/s41591-021-01620-2) (texto abierto en
    [PMC8799467](https://pmc.ncbi.nlm.nih.gov/articles/PMC8799467/)).
14. Pantanowitz L, Quiroga-Garza GM, Bien L, et al. "An artificial intelligence algorithm for
    prostate cancer diagnosis in whole slide images of core needle biopsies: a blinded clinical
    validation and deployment study". *Lancet Digit Health* 2(8):e407-e416 (2020). DOI:
    [10.1016/S2589-7500(20)30159-X](https://doi.org/10.1016/S2589-7500(20)30159-X).
15. Cohen J. "Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial
    credit". *Psychological Bulletin* 70(4):213-220 (1968). DOI:
    [10.1037/h0026256](https://doi.org/10.1037/h0026256).
16. Herrmann MD, Clunie DA, Fedorov A, et al. "Implementing the DICOM Standard for Digital
    Pathology". *J Pathol Inform* 9:37 (2018). DOI:
    [10.4103/jpi.jpi_42_18](https://doi.org/10.4103/jpi.jpi_42_18).
17. Reglamento (UE) 2016/679 (RGPD), arts. 9, 35 y considerando 26:
    [eur-lex.europa.eu/legal-content/ES/TXT/?uri=CELEX:32016R0679](https://eur-lex.europa.eu/legal-content/ES/TXT/?uri=CELEX:32016R0679).
18. Ley Orgánica 3/2018, de 5 de diciembre, de Protección de Datos Personales y garantía de los
    derechos digitales (BOE núm. 294), disposición adicional decimoséptima:
    [boe.es/buscar/act.php?id=BOE-A-2018-16673](https://www.boe.es/buscar/act.php?id=BOE-A-2018-16673).
19. Ley 14/2007, de 3 de julio, de Investigación biomédica (BOE núm. 159):
    [boe.es/buscar/act.php?id=BOE-A-2007-12945](https://www.boe.es/buscar/act.php?id=BOE-A-2007-12945).
20. AEPD. "Orientaciones y garantías en los procedimientos de anonimización de datos personales"
    (2016): [aepd.es/guias/guia-orientaciones-procedimientos-anonimizacion.pdf](https://www.aepd.es/guias/guia-orientaciones-procedimientos-anonimizacion.pdf).
21. Collins GS, Moons KGM, Dhiman P, et al. "TRIPOD+AI statement: updated guidance for reporting
    clinical prediction models that use regression or machine learning methods". *BMJ* 385:e078378
    (2024). DOI: [10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378).
