# Estado del arte: predicción del rendimiento del entrenamiento

> Revisión del trabajo relacionado para la fase 6 del [roadmap](../roadmap.md) (publicación sobre
> el predictor). Fecha de la revisión: 17/10/2026. Cada referencia se ha comprobado contra la
> página oficial, el DOI o arXiv; las cifras de precisión son **las que dan los propios autores**,
> con su configuración. Lo que no se ha podido comprobar se indica como *no verificado*.

## 1. Alcance

Se revisan trabajos que predicen, **antes o al principio** de un entrenamiento de redes profundas,
alguna de estas magnitudes:

- tiempo por iteración o por época (o throughput);
- memoria de GPU y riesgo de OOM;
- energía o potencia;
- escalado en entrenamiento distribuido (sobre todo paralelismo de datos).

Para cada trabajo se indica qué predice, el método (analítico, basado en *profiling*, aprendido o
simulación), qué necesita de la máquina destino, la precisión que reportan sus autores y si
contempla multi-GPU, hardware heterogéneo, carga de datos (I/O) y energía.

## 2. Trabajos revisados

### 2.1 Modelos analíticos y de caja gris

**Paleo** (Qi et al., ICLR 2017). Modelo analítico que, a partir de la especificación de la red,
estima el tiempo de cada capa como lectura + cómputo + escritura, con el cómputo igual a
`FLOPs / velocidad del dispositivo`, y añade la comunicación según la estrategia (paralelismo de
datos, de modelo o híbrido). Para acercarse a la realidad ajusta un **"platform percent of peak"
(PPP)**: una constante que recoge la ineficiencia media del framework frente al pico de FLOPS,
obtenida con una ejecución corta de cargas representativas en una GPU. En TITAN X, sus pasadas
completas de AlexNet y VGG-16 quedan cerca de TensorFlow (p. ej. 45,96 frente a 44,00 ms en el
forward de AlexNet), aunque el backward de AlexNet se queda corto (118,44 frente a 155,10 ms, en
torno a −24 %). También reproduce resultados de escalabilidad publicados (NiN hasta 128 GPUs,
Inception v3 hasta 100). Deja fuera lo que no es cómputo de la red (p. ej. la copia de datos), así
que no modela la carga de datos, y tampoco la energía.

**Optimus** (Peng et al., EuroSys 2018). Planificador de clúster que incluye un **modelo de
velocidad de entrenamiento** con forma derivada del funcionamiento de un servidor de parámetros,
`f(p, w)`, cuyos coeficientes se ajustan por mínimos cuadrados no negativos (NNLS) con unos pocos
pasos de entrenamiento reales en varias combinaciones de servidores y workers, y se recalibran
durante el entrenamiento. Los autores obtienen menos del 10 % de error con solo 10 pares `(p, w)`.
También ajusta en línea la curva de pérdida para estimar las épocas hasta converger.

**Pollux** (Qiao et al., OSDI 2021). Planificador que modela el tiempo por iteración del paralelismo
de datos como `T_iter = (T_grad^γ + T_sync^γ)^(1/γ)`, con `T_grad` lineal en el batch por GPU y
`T_sync` lineal en el número de GPUs (con parámetros distintos según estén en el mismo nodo o no).
El exponente `γ` interpola entre suma sin solapamiento (`γ = 1`) y solapamiento perfecto
(`max`, `γ → ∞`). Los parámetros se ajustan en línea; el error medio del modelo ajustado fue como
máximo del 10 % en un clúster de 64 GPUs. Los autores indican que su modelo **no contempla
heterogeneidad** de aceleradores.

**DS-Analyzer** (Mohan et al., PVLDB 2021). Herramienta para medir y predecir los **data stalls**:
el tiempo en que la GPU espera a la lectura o al preprocesado. Mide las tasas de ingestión de la
GPU (`G`), de preprocesado en CPU (`P`), de lectura desde caché (`C`) y desde almacenamiento (`S`),
y predice la velocidad como `min(F, P, G)`, donde `F` depende de la fracción del dataset en caché.
Al predecir el tamaño de caché óptimo para AlexNet en una V100 con SSD, la velocidad predicha se
desvía como máximo un 4 % de la medida. Es el trabajo más cercano al término de I/O de nuestro
predictor.

**Plumber** (Kuchnik et al., MLSys 2022). Traza el pipeline de entrada (tf.data) y usa un modelo
analítico de análisis operacional para encontrar el cuello y ajustar paralelismo, prefetch y caché.
Reporta aceleraciones de hasta 47× en pipelines mal configurados; no publica una cifra de error de
su estimación de throughput.

**Roofline** (Williams et al., CACM 2009). Modelo visual que acota el rendimiento por el mínimo entre
el pico de cómputo y `ancho de banda × intensidad aritmética`. No es específico de deep learning,
pero es la cota de referencia de casi todos los predictores analíticos y una baseline natural.

### 2.2 Predicción basada en profiling y simulación de trazas

**Daydream** (Zhu et al., USENIX ATC 2020). Construye un grafo de dependencias de tareas (CPU, GPU,
comunicación y carga de datos) a partir de trazas CUPTI de una ejecución real y lo simula tras
transformarlo para responder preguntas *what-if* sobre optimizaciones (precisión mixta, fusión de
kernels, etc.). Errores por debajo del 13 % en la mayoría de casos; para entrenamiento distribuido
predice a partir del perfil de **una sola GPU** y, con P3 sobre cuatro máquinas, el error es como
máximo del 16,2 %. Incluye las tareas de carga de datos "por completitud", pero los autores asumen
que no suelen ser el cuello.

**Habitat** (Yu et al., USENIX ATC 2021). Predice el tiempo por iteración en **otra GPU** a partir de
un perfil en la GPU que el usuario ya tiene: escala cada operación con *wave scaling* (basado en el
modelo de ejecución de la GPU) o con MLPs preentrenados para las operaciones cuyo kernel cambia.
Error medio del 11,8 % entre pares de seis GPUs con ResNet-50, Inception v3, Transformer, GNMT y
DCGAN. Un resultado clave para nosotros: **escalar por el cociente de FLOPS de pico da errores del
42,5 al 64,9 %** en su ejemplo con DCGAN. El entrenamiento distribuido queda como trabajo futuro
(proponen combinar su predicción de cómputo con modelos existentes de comunicación).

**dPRO** (Hu et al., MLSys 2022). Perfila trazas finas de una ejecución distribuida (incluida la
comunicación), construye un grafo global de flujo de datos y lo reproduce para diagnosticar cuellos
y buscar optimizaciones. Soporta PyTorch, TensorFlow y MXNet con AllReduce o servidor de
parámetros. Predice el rendimiento con menos del 5 % de error en la mayoría de los casos.

**ASTRA-sim 2.0** (Won et al., ISPASS 2023). Simulador de entrenamiento distribuido en tres capas
(carga, sistema y red). Los tiempos de cómputo vienen de modelos externos o de medidas reales; la
red se simula. Su backend analítico de red se validó con All-Reduce de 64 MB a 1,5 GB en sistemas
reales de 4 y 16 V100 con un error medio del 5 %. Orientado al co-diseño a gran escala, no a
predecir un entrenamiento concreto de punta a punta.

**vTrain** (Bang et al., MICRO 2024). Simulador guiado por profiling para entrenamiento de LLMs:
perfila los operadores en la GPU destino (CUPTI) y las colectivas intra-nodo (NCCL), y modela la red
entre nodos de forma analítica. MAPE del 8,37 % en un nodo de 8 A100 y del 14,73 % con 512 A100
(64 nodos).

**Calculon** (Isaev et al., SC 2023). Modelo analítico parametrizado del rendimiento de LLMs tipo
Transformer, pensado para co-diseño hardware–software y para explorar miles de millones de
combinaciones de estrategias (tensor, pipeline y datos) bajo restricciones de memoria. El
repositorio indica una validación frente a ejecuciones de Megatron en Selene (A100); **la cifra de
error no se ha verificado** en esta revisión.

### 2.3 Predictores aprendidos

**Justus et al.** (IEEE Big Data 2018). Entrena una red que predice el tiempo de cada capa o parte
del modelo a partir de rasgos de la capa y del hardware (núcleos, reloj, ancho de banda, FLOPS de
pico) y los suma para el batch o la época. Argumentan que la relación lineal entre FLOPs y tiempo no
se cumple. Evalúan en seis GPUs (V100 a K40) y reportan RMSE y correlación, no un error porcentual
único.

**Driple** (Yang et al., SIGMETRICS 2022). Usa redes neuronales de grafos para predecir el consumo
de recursos de entrenamiento distribuido y *transfer learning* para adaptarse a otros entornos
(reduce hasta 7,3× el tiempo de adaptación). **El error de predicción no se ha verificado** (el
resumen no lo da).

**NeuSight** (Lee et al., ASPLOS 2025). Divide cada kernel en *tiles*, predice su latencia con
aprendizaje automático acotado por leyes de rendimiento (roofline) y agrega. Predice GPUs **no
vistas** sin ejecutar en ellas: error del 2,3 % en el entrenamiento de GPT-3 en H100 (frente al
121,4 % de trabajos previos) y del 5,4 % en entrenamiento distribuido en un servidor de 4 GPUs.

### 2.4 Memoria

**DNNMem** (Gao et al., ESEC/FSE 2020). Estimador analítico de la memoria de GPU que recorre el grafo
de cómputo y modela el runtime del framework (asignador, workspaces), evaluado en cinco modelos con
TensorFlow, PyTorch y MXNet. **La cifra de error de sus autores no se ha verificado**; trabajos
posteriores lo usan como baseline con errores mayores.

**VeritasEst** (Shi y Elkhatib, arXiv 2025). Predice el pico de memoria **solo con CPU**, sin acceder
a la GPU destino, mediante análisis dinámico; reduce el error relativo un 84 % frente a los
estimadores previos en miles de ejecuciones con CNNs (preprint).

**Fórmulas cerradas para Transformers.** ZeRO (Rajbhandari et al., SC 2020) fija la cuenta estándar
de los *model states* con Adam en precisión mixta en `2Ψ + 2Ψ + KΨ = 16Ψ` bytes, y Korthikanti et
al. (MLSys 2023) dan la memoria de activaciones por capa, `sbh(34 + 5as/h)` bytes.

### 2.5 Energía

**Carbontracker** (Anthony et al., ICML 2020 Workshop on Challenges in Deploying and Monitoring ML
Systems; arXiv 2007.03051). Mide GPU (NVML) y CPU/DRAM (RAPL) durante las primeras épocas (una por
defecto) y extrapola linealmente al total. Con una época, el error es del 4,9–19,1 % en energía y
del 0,8–4,6 % en duración.

**Zeus** (You et al., NSDI 2023). Más que un predictor, un optimizador: explora en línea el batch y
el límite de potencia de la GPU con *profiling* de energía *just-in-time*, y mejora la eficiencia
energética un 15,3–75,8 %.

**del Rey et al.** (arXiv 2307.05520, v5 2025). Miden la energía de entrenar modelos de visión en
distintos entornos y muestran que las estimaciones basadas en FLOPs o en el TDP de la GPU fallan.
Proponen STEP (proyección desde épocas estables) y PRE (regresión antes de entrenar). La página de
arXiv cita un DOI en *Computer Standards & Interfaces* que **no se ha podido confirmar**.

**Huber et al.** (arXiv 2508.07706, 2025). Estudio de escalado del paralelismo de datos (ResNet50 y
FourCastNet): la energía escala **aproximadamente lineal con las GPU-horas**, con un factor que
varía según modelo y hardware (preprint).

### 2.6 Heterogeneidad

**Gavel** (Narayanan et al., OSDI 2020) planifica en clústeres con varios tipos de acelerador
mediante el *throughput efectivo* de cada trabajo en cada tipo, que necesita medido (o estimado) por
tipo; no predice desde especificaciones. Ninguno de los trabajos revisados predice un entrenamiento
**síncrono con GPU y CPU mezcladas** como el heterogéneo de ParaViT-Lab.

## 3. Tabla comparativa

*No* = el método no lo contempla según el propio artículo; *—* = no consta o no se ha comprobado.
"Entre tipos de GPU" significa predecir para una GPU distinta, no entrenar con GPUs mezcladas.

| Trabajo | Predice | Método | Necesita de la máquina destino | Precisión (autores) | Multi-GPU | Heterog. | I/O | Energía |
|---|---|---|---|---|---|---|---|---|
| Paleo (2017) | Tiempo, escalado | Analítico + PPP | Ejecución corta para PPP | ~4–24 % por pasada (TITAN X) | Sí | No | No | No |
| Optimus (2018) | Velocidad, convergencia | Caja gris (NNLS) | Pasos reales en varias config. | <10 % (velocidad) | Sí (PS) | No | No | No |
| Daydream (2020) | Tiempo con optimizaciones | Profiling + simulación | Traza CUPTI de una ejecución | <13 %; ≤16,2 % distr. | Sí | No | Parcial | No |
| DS-Analyzer (2021) | Data stalls, velocidad | Analítico + medidas | Medir G, P, C, S | ≤4 % (AlexNet, V100) | Parcial | No | **Sí** | No |
| Habitat (2021) | Tiempo/iter. en otra GPU | Profiling + wave scaling + MLP | Una GPU de origen | 11,8 % medio | No | Entre tipos de GPU | No | No |
| Pollux (2021) | Throughput DDP | Caja gris, ajuste en línea | Observar el propio trabajo | ≤10 % medio | Sí | No | No | No |
| dPRO (2022) | Tiempo distribuido | Profiling + reproducción | Traza del run distribuido | <5 % mayoría | Sí | No | No | No |
| Plumber (2022) | Throughput del pipeline | Análisis operacional | Traza del pipeline | No publicada | No | No | **Sí** | No |
| Justus et al. (2018) | Tiempo por capa/época | Aprendido | Datos de entrenamiento por GPU | RMSE (no %) | No | Entre tipos de GPU | No | No |
| Driple (2022) | Recursos distribuidos | Aprendido (GNN + TL) | Datos para adaptar | No verificada | Sí | Entre entornos (TL) | No | No |
| NeuSight (2025) | Latencia en GPU no vista | Aprendido + cotas | Nada (specs) | 2,3 % (GPT-3, H100) | Sí (1 servidor) | Entre tipos de GPU | No | No |
| ASTRA-sim 2.0 (2023) | Tiempo a gran escala | Simulación | Tiempos de cómputo externos | 5 % (All-Reduce) | Sí | Topologías | No | No |
| Calculon (2023) | Tiempo y memoria LLM | Analítico | Nada (specs) | No verificada | Sí | No | No | No |
| vTrain (2024) | Tiempo y coste LLM | Profiling + simulación | Profiling de operadores en GPU | 8,37 % / 14,73 % MAPE | Sí | No | No | Coste |
| DNNMem (2020) | Memoria pico | Analítico | Nada | No verificada | No | No | No | No |
| VeritasEst (2025) | Memoria pico | Análisis dinámico en CPU | Nada (CPU) | −84 % error vs. baselines | No | No | No | No |
| Carbontracker (2020) | Energía, duración | Medir + extrapolar | Primera(s) época(s) | 4,9–19,1 % energía | — | No | Implícito | **Sí** |
| Zeus (2023) | Optimiza energía | Profiling en línea | El propio trabajo | Ahorro 15,3–75,8 % | — | No | No | **Sí** |
| del Rey et al. (2025) | Energía | Proyección / regresión | Épocas estables o nada | No verificada | — | No | No | **Sí** |
| **ParaViT-Lab** | Tiempo, speedup, memoria, energía, coste | Analítico, constantes calibradas con pocos runs | Nada (o una medida opcional) | <10 % compute-bound (2×T4) | Sí | GPU+CPU | **Sí** | **Sí** |

## 4. Dónde encaja ParaViT-Lab

El predictor actual (`src/performance_model.py`, derivación en
[`performance_model.md`](../performance_model.md)) es una fórmula cerrada:

```
T(N, n) = φ · [ max( N/(π·r_c·n), N/r_io ) + (8P/β)·⌈N/(b·n)⌉ ]
r_c = MFU · TFLOPS_fp32 / (3·FLOPs_fwd),   E = potencia efectiva × T
```

con memoria `16P (+2P en AMP) + activaciones por imagen × batch + 0,6 GB`, cuatro estrategias
(single, DDP, paralelismo de modelo ingenuo, DDP heterogéneo GPU+CPU) y calibración opcional con
`rc_measured`/`rio_measured`. En vit_base sobre Kaggle 2×T4 acierta el speedup de DDP (1,95× frente
a 1,96×), el de AMP (3,80×) y el OOM de vit_large; el tiempo absoluto cae a 1–6 % en régimen de
cómputo.

**Lo que no es nuevo** (y hay que decirlo así en un artículo):

- `FLOPs / (fracción del pico)` es exactamente el **PPP de Paleo**; nuestra `MFU = 0,17` es la misma
  idea con otro nombre. Habitat muestra que escalar por FLOPS de pico puede fallar entre un 42 y un
  65 %, lo que explica nuestros −24 % en ResNet50 y +18 % en vit_large con una sola MFU.
- `max(cómputo, I/O)` es el `min(F, P, G)` de **DS-Analyzer**, que además separa lectura,
  preprocesado y caché (nosotros no, limitación 3 del roadmap).
- `cómputo + sincronización` en DDP está en Paleo, Optimus y **Pollux**; Pollux modela además el
  solapamiento (`γ`) y la localidad de nodo, que nosotros no tenemos (limitación 4).
- La memoria `16P` es la cuenta de **ZeRO**; existen fórmulas de activaciones más finas
  (Korthikanti et al.) y estimadores mejor validados (DNNMem, VeritasEst).
- `energía = potencia × tiempo` es la base de **Carbontracker**, y el escalado lineal con GPU-horas
  está documentado por **Huber et al.**
- La precisión es modesta frente a los predictores con profiling o aprendidos (dPRO <5 %, NeuSight
  2,3 %), y nuestra validación es estrecha: una pareja de GPUs, un dataset, una repetición.

**Lo que sí es menos habitual** es la combinación: un único modelo cerrado, sin ejecutar nada en la
máquina destino, que da a la vez tiempo, speedup, régimen (cómputo, I/O o sincronización), memoria
y OOM, **energía** y coste, para varias estrategias, incluido el caso síncrono GPU+CPU. Los
predictores precisos necesitan la GPU destino o una traza (Daydream, dPRO, vTrain, Habitat) o
ignoran el I/O y la energía (NeuSight, Calculon). Los de energía (Carbontracker, STEP) extrapolan
desde una ejecución y no predicen otra configuración.

## 5. Posibles contribuciones y huecos

1. **Predicción conjunta de tiempo y energía con régimen de I/O.** La tesis defendible es que la
   energía total del paralelismo de datos se conserva solo cuando el entrenamiento está limitado por
   cómputo y crece hacia `n×` cuando está limitado por I/O, y que un modelo cerrado lo anticipa
   antes de entrenar. *Evidencia necesaria:* barridos de 1 a n GPUs con potencia medida en todas,
   en datasets con perfiles de I/O muy distintos (MedMNIST en RAM, BigEarthNet en TIFF por banda,
   ChestX-ray14 en PNG), ≥3 repeticiones, y comparación con Carbontracker (extrapolar una época) y
   con el escalado lineal en GPU-horas de Huber et al.
2. **Calibración barata y transferible.** Sustituir la MFU única por una MFU por (GPU, familia de
   modelo) obtenida con un microbenchmark de segundos, y mostrar cuánto error se recupera frente a
   Paleo/PPP puro, Habitat y la cota roofline. *Evidencia:* protocolo de validación que deje fuera
   GPU, modelo o dataset (fase 5), con intervalos de incertidumbre y el coste de calibrar
   (segundos de GPU) frente a la ganancia de precisión.
3. **Entrenamiento síncrono con hardware desbalanceado (GPU+CPU).** Ningún trabajo revisado predice
   este caso; ParaViT-Lab tiene la cota `S ≤ 1 + r_CPU/r_GPU`, el sampler proporcional y un run
   real (0,12×). *Evidencia:* más pares de hardware (otras CPUs, GPU+GPU distintas), sustituir las
   constantes ad hoc (`/50`, `×2,0`) por términos medidos y validar fuera de muestra. Es un ángulo
   estrecho; quizá mejor como sección que como artículo.
4. **Modelo de I/O separado** (almacenamiento, decodificación × workers, caché), en la línea de
   DS-Analyzer pero sin necesidad de medir en la máquina destino. *Evidencia:* medir la espera del
   DataLoader (fase 4) y corregir el +130 % de vit_tiny.
5. **Dataset abierto de medidas** (huella de hardware y software, throughput estable, VRAM pico,
   potencia, espera de datos) con DOI. Útil por sí mismo para comparar predictores; no hay uno
   equivalente para entrenamiento de ViTs con energía y I/O.

**Baselines que un revisor pedirá:** Paleo/PPP (o roofline), extrapolación lineal desde un paso,
Amdahl, Habitat (código abierto) para el cómputo en otra GPU y Carbontracker para la energía.

## 6. Referencias

1. H. Qi, E. R. Sparks, A. Talwalkar. *Paleo: A Performance Model for Deep Neural Networks.* ICLR 2017. <https://openreview.net/forum?id=SyVVJ85lg>
2. Y. Peng et al. *Optimus: An Efficient Dynamic Resource Scheduler for Deep Learning Clusters.* EuroSys 2018. <https://doi.org/10.1145/3190508.3190517>
3. H. Zhu, A. Phanishayee, G. Pekhimenko. *Daydream: Accurately Estimating the Efficacy of Optimizations for DNN Training.* USENIX ATC 2020. <https://www.usenix.org/conference/atc20/presentation/zhu-hongyu> · <https://arxiv.org/abs/2006.03318>
4. G. X. Yu, Y. Gao, P. Golikov, G. Pekhimenko. *Habitat: A Runtime-Based Computational Performance Predictor for Deep Neural Network Training.* USENIX ATC 2021. <https://www.usenix.org/conference/atc21/presentation/yu> · <https://arxiv.org/abs/2102.00527>
5. A. Qiao et al. *Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning.* OSDI 2021. <https://www.usenix.org/conference/osdi21/presentation/qiao>
6. J. Mohan, A. Phanishayee, A. Raniwala, V. Chidambaram. *Analyzing and Mitigating Data Stalls in DNN Training.* PVLDB 14(5), 2021. <https://doi.org/10.14778/3446095.3446100> · <https://arxiv.org/abs/2007.06775>
7. M. Kuchnik et al. *Plumber: Diagnosing and Removing Performance Bottlenecks in Machine Learning Data Pipelines.* MLSys 2022. <https://arxiv.org/abs/2111.04131>
8. H. Hu et al. *dPRO: A Generic Performance Diagnosis and Optimization Toolkit for Expediting Distributed DNN Training.* MLSys 2022. <https://proceedings.mlsys.org/paper_files/paper/2022/hash/b422680f3db0986ddd7f8f126baaf0fa-Abstract.html> · <https://arxiv.org/abs/2205.02473>
9. S. Williams, A. Waterman, D. Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures.* Communications of the ACM 52(4), 2009. <https://doi.org/10.1145/1498765.1498785>
10. W. Won et al. *ASTRA-sim2.0: Modeling Hierarchical Networks and Disaggregated Systems for Large-model Training at Scale.* ISPASS 2023. <https://doi.org/10.1109/ISPASS57527.2023.00035> · <https://arxiv.org/abs/2303.14006>
11. M. Isaev, N. McDonald, L. Dennison, R. Vuduc. *Calculon: A Methodology and Tool for High-Level Co-Design of Systems and Large Language Models.* SC 2023. <https://doi.org/10.1145/3581784.3607102>
12. J. Bang et al. *vTrain: A Simulation Framework for Evaluating Cost-effective and Compute-optimal Large Language Model Training.* MICRO 2024. <https://doi.org/10.1109/MICRO61859.2024.00021> · <https://arxiv.org/abs/2312.12391>
13. D. Justus, J. Brennan, S. Bonner, A. S. McGough. *Predicting the Computational Cost of Deep Learning Models.* IEEE Big Data 2018. <https://arxiv.org/abs/1811.11880>
14. G. Yang et al. *Prediction of the Resource Consumption of Distributed Deep Learning Systems* (Driple). Proc. ACM Meas. Anal. Comput. Syst. 6(2), SIGMETRICS 2022. <https://doi.org/10.1145/3530895>
15. S. Lee, A. Phanishayee, D. Mahajan. *Forecasting GPU Performance for Deep Learning Training and Inference* (NeuSight). ASPLOS 2025. <https://arxiv.org/abs/2407.13853>
16. Y. Gao et al. *Estimating GPU Memory Consumption of Deep Learning Models* (DNNMem). ESEC/FSE 2020. <https://doi.org/10.1145/3368089.3417050>
17. J. Shi, Y. Elkhatib. *Accurate GPU Memory Prediction for Deep Learning Jobs through Dynamic Analysis* (VeritasEst). arXiv, 2025. <https://arxiv.org/abs/2504.03887>
18. S. Rajbhandari, J. Rasley, O. Ruwase, Y. He. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* SC 2020. <https://sc20.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap379.html> · <https://arxiv.org/abs/1910.02054>
19. V. Korthikanti et al. *Reducing Activation Recomputation in Large Transformer Models.* MLSys 2023. <https://proceedings.mlsys.org/paper_files/paper/2023/hash/80083951326cf5b35e5100260d64ed81-Abstract-mlsys2023.html> · <https://arxiv.org/abs/2205.05198>
20. L. F. W. Anthony, B. Kanding, R. Selvan. *Carbontracker: Tracking and Predicting the Carbon Footprint of Training Deep Learning Models.* ICML 2020 Workshop. <https://arxiv.org/abs/2007.03051>
21. J. You, J.-W. Chung, M. Chowdhury. *Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training.* NSDI 2023. <https://www.usenix.org/conference/nsdi23/presentation/you>
22. S. del Rey, L. Cruz, X. Franch, S. Martínez-Fernández. *Estimating Deep Learning energy consumption based on model architecture and training environment.* arXiv, v5 2025 (versión en revista no verificada). <https://arxiv.org/abs/2307.05520>
23. P. Huber et al. *Energy Consumption in Parallel Neural Network Training.* arXiv, 2025. <https://arxiv.org/abs/2508.07706>
24. D. Narayanan et al. *Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads* (Gavel). OSDI 2020. <https://www.usenix.org/conference/osdi20/presentation/narayanan-deepak>
