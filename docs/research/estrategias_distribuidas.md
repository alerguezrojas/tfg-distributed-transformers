# Estrategias de entrenamiento distribuido y cómo modelarlas en el predictor

> Estudio para decidir qué estrategias "consolidadas" añadir a ParaViT-Lab y qué términos nuevos
> necesita el predictor. Complementa el [estado del arte](estado_del_arte_prediccion.md), que ya
> cubre Paleo, Daydream, Habitat, dPRO, Pollux, DS-Analyzer y compañía: aquí solo se añade lo que
> afecta a las estrategias. **Fecha de consulta de todas las fuentes en línea: 23/11/2026.**
>
> Convenciones. **Dicen los autores** = cifra o afirmación de la fuente. **Interpretación** =
> derivación o juicio nuestro, no comprobado en el hardware del proyecto. **No verificado** = no se
> ha podido abrir o contrastar la fuente. La documentación de PyTorch citada es la de la versión
> 2.15, a la que redirige hoy `docs.pytorch.org/docs/stable`.

## 1. Notación

`P` = parámetros del modelo; `N` = GPU en el grupo; `b`, `s`, `h`, `a`, `L` = lote, longitud de
secuencia (tokens), dimensión oculta, cabezas y capas; `p` = etapas de pipeline; `m` = micro-lotes;
`t` = grado de tensor-paralelo. Los **bytes por GPU** se dan para un anillo (la implementación
habitual de NCCL): un *all-reduce* de `S` bytes mueve `2(N−1)/N · S` por GPU y un *all-gather* o
*reduce-scatter*, `(N−1)/N · S` (factores de bus de
[nccl-tests](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md), leídos el
23/11/2026; coinciden con el modelo α-β de la sección 4).

Memoria de los estados del modelo con Adam, según ZeRO: `2Ψ + 2Ψ + KΨ = 16Ψ` bytes (`K = 12`)
**(dicen los autores)**. *Interpretación:* el proyecto entrena con *autocast* (pesos y gradientes en
fp32, Adam con 8 bytes), lo que también suma `16P` pero con otra composición (4+4+8); la cifra de
`performance_model.py` coincide, pero al repartir pesos con FSDP hay que decidir si se reúnen en
fp32 o en fp16 (`MixedPrecision`/`MixedPrecisionPolicy`), y eso cambia los bytes de comunicación
(véase 2.3).

## 2. Catálogo

### 2.1 Tabla resumen

"Estado" = lo que dice la página de documentación de PyTorch 2.15 (ninguna etiqueta significa que
la página no declara nivel de madurez; no es lo mismo que "estable").

| Estrategia | Qué reparte | Comunicación por paso y GPU (anillo) | Memoria de estados por GPU | Cuándo compensa | Estado / dónde vive |
|---|---|---|---|---|---|
| **DDP** | Datos (réplica completa) | all-reduce de gradientes: `2(N−1)/N · P·d_g` (fp32: `8P(N−1)/N`; `N=2`: `4P`) | `16P` | El modelo cabe y el cómputo domina | PyTorch, `DistributedDataParallel`; sin etiqueta |
| **Acumulación de gradientes** | Nada (más pasos por actualización) | La sincronización baja a 1 de cada `k` micro-pasos (`no_sync`) | Igual que DDP; activaciones del micro-lote | Lote efectivo mayor sin más memoria; menos sincronización | PyTorch (`no_sync`); sin etiqueta |
| **ZeRO-1** (`ZeroRedundancyOptimizer`, DeepSpeed stage 1) | Estado del optimizador | ZeRO: `2Ψ` (igual que DDP). Con `ZeroRedundancyOptimizer` sobre DDP: all-reduce completo + difusión de los parámetros actualizados (*interpretación*: ≈`3Ψ`, a medir) | `4P + 12P/N` | Optimizador grande (Adam) y red buena | `ZeroRedundancyOptimizer`: **experimental** en la doc; DeepSpeed: librería |
| **ZeRO-2 / `SHARD_GRAD_OP`** | Optimizador + gradientes | ZeRO: `2Ψ` (reduce-scatter + all-gather) | `2P + 14P/N` | Cuando los gradientes también pesan | FSDP1 `ShardingStrategy.SHARD_GRAD_OP`; FSDP2 con `reshard_after_forward=False`; DeepSpeed stage 2 |
| **ZeRO-3 / FSDP `FULL_SHARD` / `fully_shard`** | Optimizador + gradientes + pesos | ZeRO: `3Ψ` (**1,5× DDP**): 2 all-gather (adelante y atrás) + reduce-scatter | `16P/N` + la mayor unidad reunida (FSDP) | El modelo no cabe; la red aguanta 1,5× de tráfico | FSDP1 (`FullyShardedDataParallel`) y FSDP2 (`fully_shard`, la doc recomienda migrar); DeepSpeed stage 3 |
| **FSDP `HYBRID_SHARD` / HSDP** | Como FULL_SHARD dentro de un nodo; réplica entre nodos | Tráfico entre nodos reducido; en 1 nodo equivale a FULL_SHARD | `16P/F` (`F` = tamaño del grupo de reparto) | Varios nodos con red lenta entre ellos | FSDP1 `HYBRID_SHARD`; FSDP2 con malla 2D |
| **Offload a CPU** | Estados (y a veces pesos) a la RAM | Transferencias PCIe de tamaño `O(P)` por paso, y optimizador en CPU | Baja en GPU lo que se descarga | Modelo muy grande y poca VRAM; coste en tiempo alto | FSDP `CPUOffload`/`CPUOffloadPolicy`; DeepSpeed ZeRO-Offload/Infinity |
| **Tensor paralelo** | Pesos de cada capa y parte de las activaciones | `8·b·s·h·(t−1)/t` elementos por capa y micro-lote (4 all-reduce) | `16P/t` en las capas particionadas | NVLink; capas muy anchas | `torch.distributed.tensor.parallel`: **experimental**; `DTensor`: **alpha**; Megatron: librería |
| **Pipeline** | Capas (etapas) | Solo activaciones de frontera: `~2·b_µ·s·h·d_a` por frontera y micro-lote (adelante+atrás) | `16P/p`; activaciones de `m` micro-lotes (GPipe) o `≤p` (1F1B) | El modelo no cabe y hay microlotes de sobra | `torch.distributed.pipelining`: **alpha** |
| **Activation checkpointing** | Nada: recalcula activaciones | Ninguna | Activaciones: de `sbhL(34+…)` a `2sbhL` con recálculo completo | Las activaciones limitan (ViT a 512 px) | `torch.utils.checkpoint`; sin etiqueta |
| **`torch.compile`** | Nada: fusiona kernels | Ninguna | Puede bajar activaciones; no medido aquí | Cuando el tiempo de cómputo manda | PyTorch; sin etiqueta global, con avisos puntuales |
| **Precisión mixta** | Nada: cambia el tipo de dato | Reduce bytes si se comunica en fp16 | Activaciones a la mitad (aprox.) | Siempre que haya Tensor Cores | `torch.autocast` + `GradScaler`; sin etiqueta |

Fuentes de las cifras de la tabla: DDP, ZeRO-1/2/3: Rajbhandari et al. (SC 2020), memoria
`2Ψ+14Ψ/Nd`, `4Ψ+12Ψ/Nd`, `16Ψ/Nd` y volumen `2Ψ`/`2Ψ`/`3Ψ` (apartados 5 y 7 del artículo, leídos
el 23/11/2026). FSDP `FULL_SHARD`: Zhao et al. (2023): "1,5× de sobrecarga y volumen sobre DDP si
se usa el anillo óptimo en ancho de banda", y memoria "proporcional al modelo repartido más la
mayor unidad FSDP materializada". Tensor paralelo: Narayanan et al. (SC 2021), "`8bsh((t−1)/t)` por
capa y dispositivo y micro-lote" (leído en la versión ar5iv del artículo; el artículo habla de
"comunicación total" sin fijar bytes; aquí se interpreta como elementos y se multiplica por el
tamaño del dato). Pipeline: Huang et al. (GPipe) y Narayanan et al.; checkpointing y activaciones:
Korthikanti et al.

### 2.2 Notas por estrategia

**DDP.** Los gradientes se agrupan en *buckets* (25 MiB por defecto, `bucket_cap_mb`) y el
*all-reduce* de cada uno puede solaparse con el *backward* **(dice la documentación)**. Con
`static_graph=True` se pueden repetir activaciones recalculadas. Es lo que ya implementa
`ddp_trainer.py`. `no_sync()` da la acumulación de gradientes, pero el *forward* debe ir dentro del
contexto, o se sincroniza igualmente.

**FSDP1 frente a FSDP2.** Ambos reparten parámetros, gradientes y estado del optimizador entre los
procesos de datos-paralelo. FSDP2 (`fully_shard`) representa los parámetros como `DTensor` partidos
en la dimensión 0 y la propia documentación indica "si usan FSDP1, planteen migrar a FSDP2". Según
la doc, FSDP2 sustituye el aplanado y concatenado de FSDP1 por reparto por parámetro, lo que hace
que los *state dicts* repartidos no necesiten comunicación, y no admite directamente *state dicts*
completos (hay que usar las API de DTensor o *Distributed Checkpoint*). `reshard_after_forward=True`
libera los pesos tras el *forward* y los vuelve a reunir en el *backward*; `False` los conserva
(evita el segundo *all-gather*, más memoria); con malla 2D hay HSDP. La *interpretación* es que
`reshard_after_forward=False` en FSDP2 hace el papel de `SHARD_GRAD_OP`/ZeRO-2. Un detalle útil
para el predictor: desde 2.15, `MixedPrecisionPolicy(reduce_dtype=None)` sigue el `grad_dtype` de
cada parámetro (antes, el `param_dtype`); los bytes del *reduce-scatter* dependen de ello.
`FSDP1` conserva `limit_all_gathers=True` por defecto (un limitador de tasa que sincroniza la CPU
para acotar la memoria).

**FSDP1 frente a DeepSpeed.** Hugging Face Accelerate documenta la correspondencia:
`FULL_SHARD` = ZeRO-3, `SHARD_GRAD_OP` = ZeRO-2, `NO_SHARD` = ZeRO-0, `HYBRID_SHARD` = ZeRO++ con
`zero_hpz_partition_size` igual al número de GPU por nodo. DeepSpeed ofrece además
`offload_optimizer`/`offload_param` (CPU o NVMe); ZeRO-Offload es "offloading de gradientes y
estado del optimizador a CPU dentro de ZeRO-2" y ZeRO-Infinity, del modelo completo, con ZeRO-3.
DeepSpeed es una dependencia externa; para un solo nodo de 2 GPU, FSDP2 cubre lo mismo sin nueva
librería (*interpretación*).

**`ZeroRedundancyOptimizer`.** Reparte el estado del optimizador (cada parámetro vive entero en un
rango, asignación voraz ordenada) y, tras el paso, "cada rango difunde sus parámetros al resto". Es
la vía más barata para probar ZeRO-1 sobre el DDP actual. Advertencias de la doc: es experimental,
exige un único tipo denso de parámetros, y varias llamadas deben hacerse en todos los rangos o se
bloquea. Con `overlap_with_ddp=True` las primeras iteraciones no actualizan parámetros.

**Tensor paralelo.** `parallelize_module` con `ColwiseParallel`/`RowwiseParallel` (y
`SequenceParallel` para normalizaciones) sobre una malla 1D; la doc declara las API
experimentales. Pensado para NVLink (véase 3.2).

**Pipeline.** `torch.distributed.pipelining` ofrece GPipe y 1F1B (una etapa por rango) y, con varias
etapas por rango, *Interleaved 1F1B*, *Looped BFS*, *Interleaved Zero Bubble*, *ZBV Zero Bubble* y
*DualPipeV*. La doc lo declara **alpha** ("API changes may be possible"). El corte puede ser
manual (cada etapa es un `nn.Module` envuelto en `PipelineStage`) o automático con `pipeline()` y
`split_spec`, que **exige que el modelo sea trazable con `torch.export`**; las formas de entrada y
salida de cada etapa deben ser estáticas. *Interpretación:* en el diseño actual, `ModelParallelViT`
ya parte el modelo a mano y vive en un solo proceso; `pipelining` asume un rango por etapa
(multiproceso, como DDP), lo que cambia el modo de lanzamiento y de recoger métricas.

**Activation checkpointing.** `torch.utils.checkpoint` "cambia cómputo por memoria"; la doc
recomienda `use_reentrant=False` y ofrece *selective activation checkpointing* (`context_fn`,
`create_selective_checkpoint_contexts`, política `SAVE`/`RECOMPUTE`/`CPU_OFFLOAD`). Korthikanti et
al. dan la memoria de activaciones por capa: `sbh(34 + 5as/h)` sin paralelismo; `sbh(10 + 24/t + 5as/(ht))`
con tensor paralelo; `sbh(34/t)` con tensor + secuencia + recálculo selectivo; `sbh·2` con recálculo
completo **(dicen los autores)**. *Interpretación del coste:* el recálculo completo repite un
*forward* por paso, es decir, ~+33 % de cómputo sobre `3×FLOPs_fwd`.

**Offload a CPU.** `CPUOffloadPolicy` (FSDP2): parámetros, gradientes y estado del optimizador a
CPU; los *shards* se copian a la GPU antes del *all-gather*, los gradientes vuelven en el *backward*
y el paso del optimizador corre en CPU. ZeRO-Offload (Ren et al., arXiv 2021): modelos de más de
13 B parámetros en una V100 con 40 TFLOPS **(dicen los autores)**. Con T4 por PCIe y 4 vCPU en
Kaggle, el cuello sería la CPU y el PCIe (*interpretación*).

**`torch.compile`.** Modos `default`, `reduce-overhead` (grafos CUDA, más memoria, no siempre se
aplica) y `max-autotune`; recompila ante cambios de forma. No añade comunicación: en el predictor
entra como un factor sobre la eficiencia de cómputo (MFU), medido, no derivado. No se ha comprobado
qué compatibilidad tiene con cada estrategia en 2.15.

**Precisión mixta.** `autocast` en CUDA usa fp16 por defecto, mantiene en fp32 `softmax`,
`layer_norm` y pérdidas, y para fp16 hace falta `GradScaler`. Las T4 no tienen bf16 (dato del
enunciado de la tarea). En el predictor ya está (`π`, la memoria de AMP).

## 3. Viabilidad en 2×T4 por PCIe

### 3.1 Qué dice el cálculo de memoria con modelos de 22 a 87 M

Con las fórmulas de ZeRO (`N = 2`), solo estados del modelo, sin activaciones (*interpretación,
cálculo propio, no medido*):

| Modelo | `P` | DDP `16P` | ZeRO-1 | ZeRO-2 | ZeRO-3 / FULL_SHARD |
|---|---|---|---|---|---|
| ViT-S/16 | 22,05 M (perfil) | 0,35 GB | 0,22 GB | 0,20 GB | 0,18 GB |
| Swin-T | ≈28,3 M (*valor de timm de memoria, a confirmar con el perfil*) | 0,45 GB | 0,28 GB | 0,25 GB | 0,23 GB |
| ViT-B/16 | 86,57 M (perfil) | 1,39 GB | 0,87 GB | 0,78 GB | 0,69 GB |
| ViT-L/16 (referencia) | 304,3 M | 4,87 GB | 3,04 GB | 2,74 GB | 2,43 GB |
| ViT-H (referencia) | ≈632 M (*a confirmar*) | 10,1 GB | 6,3 GB | 5,7 GB | 5,1 GB |

**Conclusión de memoria (interpretación).** En el rango de 22 a 87 M los estados ocupan 0,35 a
1,4 GB de unos 15 GB por T4; repartirlos ahorra como mucho 0,7 GB (ViT-B). Lo que limita a 512 px
son las **activaciones**, que ningún reparto de estados toca (solo checkpointing, lote menor o
tensor paralelo). La cuenta de Korthikanti sin el término de atención (la atención *flash* no guarda
la matriz `s×s`) da para ViT-B a 512 px (`s=1024`, `h=768`, `L=12`) `34·s·b·h·L ≈ 0,32 GB` por
imagen en 16 bits y `0,16 GB` para ViT-S, justo la mitad de lo que el estudio del 23/11/2026
(`performance_model.md`, apartado 8) mide con el autograd (0,615 y 0,309 GB por imagen). Es una
coincidencia que apoya el supuesto "en AMP se guarda la mitad", **no una prueba**. Con ese orden de
magnitud, ViT-B a lote 16 por GPU necesita unos 5 GB de activaciones: cabe en una T4 con DDP. FSDP
**no hace posible** nada nuevo en 22-87 M; solo lo haría con ViT-L o mayores (arriba, filas de
referencia), fuera del rango pedido.

### 3.2 Qué cabe esperar de cada estrategia en el hardware

**FSDP/ZeRO-3 frente a DDP en 2×T4.** Para `N=2`, DDP en fp32 mueve `4P` bytes por GPU; FSDP con
reunión en fp16 y *reduce-scatter* en fp32 mueve `(1/2)(2·2P + 4P) = 4P`; con reunión en fp32,
`6P` (*interpretación*). Con 22 y 87 M: 88 y 346 MB por paso con DDP (y con el FSDP de reunión en
fp16). A 4,05 GB/s (medida de un tercero, abajo) son 22 y 86 ms por paso. Con el *throughput*
sintético de una T4 (ViT-S 48 img/s con lote 8, ViT-B ≈18,6 img/s), un paso de ViT-B a lote 16 por
GPU dura ≈0,9 s, así que la comunicación es ~10 % y se solapa en parte: **FSDP debería quedar cerca
de DDP en ViT-B y algo peor en ViT-S**, donde el paso es corto. Es una predicción por derivar,
no un resultado.

Evidencia independiente, débil: el repositorio [`lgoyal6/ddp-crossover`](https://github.com/lgoyal6/ddp-crossover)
(sin fecha, sin revisión por pares) mide en 2×T4 de Kaggle con un MLP: ancho de banda de bus de
*all-reduce* de pico 4,05 GB/s y de *all-gather* 1,73 GB/s, y FSDP más lento que DDP en todos los
tamaños. Su texto habla de una brecha "constante del 33-35 %", pero su propia tabla da ~19 %, 22 %,
22 % y 35 % (solo el último caso es 35 %); no indica si P2P estaba activo y usa un MLP en fp32, no
un transformer. Sirve como **motivo para medir**, no como cifra citable. Habría que medir `β`
(*all-reduce* y *all-gather*) en las T4 de Kaggle y en el clúster antes de nada.

*Observación sobre el modelo actual:* el predictor cobra `8P/β` por paso con `β = 16 GB/s` para PCIe.
`8P` es el límite `N→∞` del factor `2(N−1)/N·4P`; para `N=2` el tráfico exacto es `4P`, y 16 GB/s es el
pico teórico de PCIe 3.0 x16, no el ancho medido por NCCL entre dos T4 (4,05 GB/s en el tercero
citado). Los dos errores se compensan en parte (×2 de más en tráfico, ×4 de menos en ancho), lo que
puede explicar por qué DDP en 2×T4 sale bien (1,95× frente a 1,96×) sin que el término de red esté
bien. *Interpretación; hay que medirlo.*

**Pipeline con micro-lotes frente al model-parallel ingenuo.** El ingenuo es GPipe con `m=1`: una
sola GPU activa a la vez, `S≈1`, que es lo que mide el proyecto (1,02×). Con `p` etapas equilibradas
y `m` micro-lotes, la fracción de burbuja es `(p−1)/m` **(Narayanan et al. 2021; Huang et al. dan
`O((K−1)/(M+K−1))`)** y el tiempo `T ≈ T₁ · (m+p−1)/(p·m)` si cada micro-lote conserva la eficiencia
(*interpretación*). Para `p=2`:

| `m` | burbuja `(p−1)/m` | aceleración ideal frente a una GPU `p·m/(m+p−1)` |
|---|---|---|
| 1 (ingenuo) | 100 % | 1,00× |
| 2 | 50 % | 1,33× |
| 4 | 25 % | 1,60× |
| 8 | 12,5 % | 1,78× |
| 16 | 6,3 % | 1,88× |

La frontera cuesta poco: a 512 px, ViT-B transfiere `s·h·2 B = 1,6 MB` por imagen y sentido, unos
25 MB por lote de 16, ~6 ms a 4 GB/s frente a ~0,9 s de cómputo (*cálculo propio*). **Dicen los
autores (GPipe):** la burbuja es despreciable con `M ≥ 4K`; en el Transformer de la tabla 2 (TPU)
con `K=2,4,8` y `M=1,4,32`, el rendimiento normalizado a `K=2, M=1` es 1,7/3,2/4,8 con `M=4` y
1,8/3,4/6,3 con `M=32` en TPU; en GPU P100 **sin NVLink** (tabla 3, `M=32`) el paso de 2 a 8 particiones da
3,3× en el Transformer de 24 capas y 2,7× en AmoebaNet, porque solo se transfieren las
activaciones de frontera. Es decir: la literatura respalda que el pipeline sí funciona por PCIe, y
que su límite es el equilibrio de carga (AmoebaNet da menos que el Transformer por etapas
desiguales). Con `m=4`, `K=2` el 1,7× del Transformer es muy cercano al 1,6× de la fórmula.
*Interpretación para Swin-T:* las etapas de Swin tienen tamaños de activación distintos (fusión de
parches), así que el corte hay que elegirlo por FLOPs y bytes, y la aceleración real quedará por
debajo del ideal. 1F1B no cambia la burbuja de GPipe pero baja la memoria (activaciones de `≤p`
micro-lotes en vez de `m`, **dicen los autores**); con 2 etapas y una VRAM que ya cabe, su ventaja
es menor.

**Tensor paralelo en PCIe.** Con la fórmula `8bsh(t−1)/t` (`t=2`: `4bsh` elementos por capa), ViT-B a
512 px, lote 16 en fp16: `4·16·1024·768·2 B ≈ 100 MB` por capa, `~1,2 GB` por paso en 12 capas; a
4 GB/s, ~0,3 s frente a ~0,4 s de cómputo por GPU, **sin solape fácil** (*cálculo propio*). Predice
poca o ninguna ganancia sobre DDP; el tensor paralelo se diseñó para NVLink (Megatron-LM, intra-nodo).
No compensa en este hardware.

**Offload.** Tiene sentido en la 3060 Ti de 8 GB si lo que falta es memoria de estados, pero a 512 px
lo que falta es de activaciones (véase 3.1); no es prioritario.

**Acumulación de gradientes y checkpointing.** Sí tienen valor en este hardware: en la 3060 Ti de
8 GB hay modelos que "no caben" a 512 px (`performance_model.md`, apartado 8), y el recálculo cambia
~33 % de tiempo por activaciones mucho menores (*interpretación*).

## 4. Modelos analíticos establecidos

**Modelo α-β.** Enviar un mensaje cuesta `α + nβ` (latencia más ancho de banda inverso) **(Thakur y
Gropp, 2003; Thakur, Rabenseifner y Gropp, 2005)**. Fórmulas leídas en la versión de 2003 (la de
2005 no se ha abierto): anillo de *all-gather*, `T = (p−1)α + (p−1)/p · nβ`; reducción de
Rabenseifner (*reduce-scatter* más reunión), `T = 2 lg p α + 2(p−1)/p · nβ + (p−1)/p · nγ`
(`γ` = coste de cómputo por byte reducido). Patarasuk y Yuan (2009) demuestran una cota inferior de
datos comunicados para el *all-reduce* y que el algoritmo en anillo es óptimo en ancho de banda
(en árboles; no es óptimo en latencia). *Interpretación:* para los mensajes de decenas o cientos de
MB de nuestros gradientes el término `α` es despreciable y queda `2(N−1)/N · S/β`, que es el
factor de bus de NCCL.

**ZeRO** (Rajbhandari et al.): memoria y volumen de la tabla 2.1; por paso, DP `2Ψ`, ZeRO-1/2 `2Ψ`,
ZeRO-3 `3Ψ`. **FSDP** (Zhao et al.): 1,5× de volumen de DDP en `FULL_SHARD` y memoria
`reparto + mayor unidad reunida`. **GPipe** (Huang et al.): burbuja `O((K−1)/(M+K−1))`; memoria
pico con recálculo `O(N + (L/K)·(N/M))`. **1F1B** (PipeDream, Harlap et al. 2018 y Narayanan et
al. 2019; PipeDream-Flush en Narayanan et al. 2021): activaciones de `≤p` micro-lotes; entrelazado
con `v` trozos por dispositivo: burbuja `(1/v)(p−1)/m` y comunicación `×v`. **Activaciones**
(Korthikanti et al.): ecuaciones del apartado 2.2. **Roofline** (Williams et al.): ya en el estado
del arte.

**Predictores que modelan varias estrategias** (cifras de los autores, de sus resúmenes; salvo
Paleo, Daydream, dPRO, ASTRA-sim y vTrain, que ya constan en el estado del arte):

| Trabajo | Estrategias que modela | Cómo valida | Error que reportan |
|---|---|---|---|
| Paleo (2017) | Datos, modelo, híbrido (analítico) | Reproduce escalados publicados | véase estado del arte |
| Calculon (SC 2023) | Tensor, pipeline, datos, memoria (LLM, analítico) | Megatron en Selene (A100) según el repositorio | **No verificado** (se vio `±10 %` solo en un artículo posterior, no en el original) |
| vTrain (MICRO 2024) | 3D (profiling + simulación) | 8 y 512 A100 | MAPE 8,37 % y 14,73 % (según el estado del arte) |
| Proteus (arXiv 2023) | Estrategias complejas con un árbol de estrategias; solapa cómputo y comunicación | Varias DNN en 3 configuraciones de hardware | 3,0 % de error medio (resumen) |
| Echo (arXiv 2024) | 3D en Megatron-LM, un solo dispositivo para trazar | GPT-175B en 96 H800 | 8 % de error medio en el paso |

*Interpretación para situar el proyecto.* Calculon, vTrain y Echo apuntan a LLM en clústeres de
A100/H800 con NVLink e InfiniBand; Proteus declara "varias DNN" sin que el resumen diga cuáles ni qué
GPU. Ninguno, según sus resúmenes, modela hardware de consumo por PCIe, ViT/Swin con atención por
ventanas, ni energía. El hueco defendible no es "otro simulador" sino un **predictor cerrado, sin
ejecutar en la máquina destino, que cubre estrategias y energía en hardware modesto y se valida
fuera de muestra**. Antes de afirmarlo en un artículo hay que leer los artículos completos, no
solo los resúmenes.

## 5. Recomendación

Orden propuesto, con el criterio de valor experimental en 2×T4/3060 Ti/V100, esfuerzo y encaje en
el diseño (subclase de `Trainer` + `with_*` en `src/training/builder.py`).

**1. Pipeline con micro-lotes (GPipe).** Extiende `ModelParallelTrainer`; ya existe la medida del
ingenuo (1,02×) y la fórmula predice 1,6× con `m=4`, así que es una predicción **falsable** y la
más vistosa. *Encaje:* un `with_pipeline(devices, micro_batches)` y un `PipelineTrainer`. La opción
barata es un bucle en un solo proceso que trocea el lote y lanza los micro-lotes de forma
asíncrona; **hipótesis sin probar**: que las copias entre GPU y el *autograd* multihilo den el
solape (`torchgpipe`, Kim et al., arXiv 2020, se diseñó para esto; no se ha verificado cómo).
`torch.distributed.pipelining` es la alternativa oficial, pero está en alpha y es multiproceso.
*Términos nuevos del predictor:* burbuja `(p−1)/m`, transferencia de frontera `2·b_µ·s·h·d_a`
a `β` medido, equilibrio de etapas (factor ≥1 sobre la etapa más lenta) y pérdida de eficiencia del
micro-lote pequeño (el *benchmark* sintético da solo ~5 % entre lote 8 y 16).

**2. Acumulación de gradientes y activation checkpointing sobre DDP/single.** Son opciones del
entrenamiento, no estrategias; encajan como decoradores o parámetros de `with_*`, esfuerzo bajo, y
son lo único que **cambia el techo de memoria** a 512 px. Validables en la 3060 Ti de 8 GB.
*Términos:* sincronizaciones `/k` (`no_sync`), activaciones `×(factor de recálculo)`, cómputo `+1`
*forward* por paso. Obligan a recalibrar `act_gb_per_img` con medidas.

**3. FSDP2 (`fully_shard`) con `FULL_SHARD` y `reshard_after_forward=False`.** Es lo que la
documentación recomienda hoy y evita DeepSpeed. Su valor en 22-87 M es **mostrar qué predice el
modelo y qué ocurre** (de empatar a perder frente a DDP en PCIe) y abrir la puerta a ViT-L o mayores
en el clúster. *Encaje:* `FSDPTrainer` + `with_fsdp(sharding, mixed_precision)`; cuidado con
checkpoints (hay que usar las API de DTensor/DCP para guardar) y con el *golden test*. *Términos:*
memoria `16P/N` más la mayor unidad reunida, comunicación `(N−1)/N·P(2d_p+d_g)` (o `(d_p+d_g)` sin
`reshard`), solape de *all-gather* con cómputo (el `γ` de Pollux) y `β` de *all-gather* medido por
separado del de *all-reduce*.

**4. ZeRO-1 con `ZeroRedundancyOptimizer`.** Casi gratis tras 3 y completa el escalón `DDP → ZeRO-1 →
ZeRO-2/3` del predictor con una sola fila más. API experimental; la comunicación real puede ser
mayor que la de ZeRO-1 teórico (difusión tras el DDP); es **lo primero que hay que medir** para
cerrar esa fila.

**Dejar para después, con motivo:** el **tensor paralelo** (el cálculo del apartado 3.2 predice que
en PCIe no gana, y es experimental; una fila de "predicho negativo" bastaría) y el **offload**
(poco interés si el límite son las activaciones). `torch.compile` se mide aparte: modifica la MFU,
no es estrategia.

**Antes de cualquier estrategia:** medir `β` (*all-reduce*, *all-gather*, *reduce-scatter* y copia
P2P) en 2×T4 de Kaggle, en la V100 del clúster y en la 3060 Ti, y sustituir el `16e9` fijo de
`INTERCONNECT_BW`. Sin eso, ninguna estrategia nueva puede validarse (regla de AGENTS.md: no predecir lo
que no se puede contrastar; separar calibración y validación).

## 6. Lo que no se ha podido verificar

- Error de Calculon del artículo original (SC 2023): la página de SC23 devolvió 403; la cifra de ±10 %
  aparece solo en un trabajo posterior.
- Versión en VLDB del artículo de FSDP (arXiv no la indica) y DOI de Narayanan et al. 2021 (SC 2021
  consta en el campo de comentarios de arXiv).
- Fórmulas del artículo de 2005 de Thakur et al.: se leyó la versión de 2003 (Thakur y Gropp); el
  all-reduce de Rabenseifner del artículo de 2005 no se abrió.
- Si el *autograd* multihilo de PyTorch solapa de verdad micro-lotes en un solo proceso con dos GPU
  (hay que probarlo); compatibilidad de `torch.compile` con cada estrategia en 2.15; parámetros de
  Swin-T y ViT-H (valores de memoria, a confirmar con `model_profile`).
- Especificaciones de la T4 (16 GB, 70 W, 8,1 TFLOPS fp32, PCIe 3.0 x16; ancho de memoria "320+ GB/s"
  en la web y 300 GB/s en una ficha): vistas en resultados de búsqueda de NVIDIA, no abiertas; el
  predictor ya las usa.
- Que Kaggle exponga P2P entre las dos T4: no consta; la cifra de 4,05 GB/s es de un tercero.

## 7. Bibliografía

Documentación consultada el 23/11/2026 (versión 2.15 de PyTorch):

1. PyTorch, `fully_shard` (FSDP2): <https://docs.pytorch.org/docs/2.15/distributed.fsdp.fully_shard.html>
2. PyTorch, `FullyShardedDataParallel` (FSDP1): <https://docs.pytorch.org/docs/2.15/fsdp.html>
3. PyTorch, `torch.distributed.pipelining`: <https://docs.pytorch.org/docs/2.15/distributed.pipelining.html>
4. PyTorch, tensor paralelo: <https://docs.pytorch.org/docs/2.15/distributed.tensor.parallel.html> · `DTensor`: <https://docs.pytorch.org/docs/2.15/distributed.tensor.html>
5. PyTorch, `ZeroRedundancyOptimizer`: <https://docs.pytorch.org/docs/2.15/distributed.optim.html>
6. PyTorch, `DistributedDataParallel`: <https://docs.pytorch.org/docs/2.15/generated/torch.nn.parallel.DistributedDataParallel.html>
7. PyTorch, `torch.utils.checkpoint`: <https://docs.pytorch.org/docs/2.15/checkpoint.html> · AMP: <https://docs.pytorch.org/docs/2.15/amp.html> · `torch.compile`: <https://docs.pytorch.org/docs/2.15/generated/torch.compile.html>
8. DeepSpeed, tutorial de ZeRO: <https://www.deepspeed.ai/tutorials/zero/>
9. Hugging Face Accelerate, guía de FSDP: <https://huggingface.co/docs/accelerate/usage_guides/fsdp>
10. NVIDIA, nccl-tests, rendimiento (`algbw`, `busbw`): <https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md>
11. NVIDIA, T4: <https://www.nvidia.com/en-us/data-center/tesla-t4/> (visto en resultados de búsqueda)

Artículos:

12. S. Rajbhandari, J. Rasley, O. Ruwase, Y. He. *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.* SC 2020. <https://arxiv.org/abs/1910.02054>
13. Y. Zhao et al. *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel.* arXiv 2304.11277, v2 2023. <https://arxiv.org/abs/2304.11277>
14. Y. Huang et al. *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism.* NeurIPS 2019. <https://papers.nips.cc/paper/2019/file/093f65e080a295f8076b1c5722a46aa2-Paper.pdf> · <https://arxiv.org/abs/1811.06965>
15. A. Harlap et al. *PipeDream: Fast and Efficient Pipeline Parallel DNN Training.* arXiv 1806.03377, 2018. <https://arxiv.org/abs/1806.03377>
16. D. Narayanan et al. *PipeDream: Generalized Pipeline Parallelism for DNN Training.* SOSP 2019. <https://doi.org/10.1145/3341301.3359646>
17. D. Narayanan et al. *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.* SC 2021 (según arXiv). <https://arxiv.org/abs/2104.04473>
18. M. Shoeybi et al. *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.* arXiv 1909.08053. <https://arxiv.org/abs/1909.08053>
19. V. Korthikanti et al. *Reducing Activation Recomputation in Large Transformer Models.* MLSys 2023. <https://arxiv.org/abs/2205.05198>
20. J. Ren et al. *ZeRO-Offload: Democratizing Billion-Scale Model Training.* arXiv 2101.06840, 2021. <https://arxiv.org/abs/2101.06840>
21. W. Liang et al. *TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training.* arXiv 2410.06511, 2024. <https://arxiv.org/abs/2410.06511>
22. C. Kim et al. *torchgpipe: On-the-fly Pipeline Parallelism for Training Giant Models.* arXiv 2004.09910, 2020. <https://arxiv.org/abs/2004.09910>
23. R. Thakur, R. Rabenseifner, W. Gropp. *Optimization of Collective Communication Operations in MPICH.* IJHPCA 19(1), 2005. <https://doi.org/10.1177/1094342005051521>
24. R. Thakur, W. Gropp. *Improving the Performance of Collective Operations in MPICH.* EuroPVM/MPI 2003 (texto leído). <https://web.cels.anl.gov/~thakur/papers/mpi-coll.pdf>
25. P. Patarasuk, X. Yuan. *Bandwidth optimal all-reduce algorithms for clusters of workstations.* JPDC 69(2), 2009. <https://doi.org/10.1016/j.jpdc.2008.09.002>
26. J. Duan et al. *Proteus: Simulating the Performance of Distributed DNN Training.* arXiv 2306.02267, 2023. <https://arxiv.org/abs/2306.02267>
27. Y. Feng et al. *Echo: Simulating Distributed Training At Scale.* arXiv 2412.12487, 2024. <https://arxiv.org/abs/2412.12487>
28. M. Isaev et al. *Calculon: A Methodology and Tool for High-Level Co-Design of Systems and Large Language Models.* SC 2023. <https://doi.org/10.1145/3581784.3607102>
29. J. Bang et al. *vTrain.* MICRO 2024. <https://arxiv.org/abs/2312.12391>
30. S. Williams, A. Waterman, D. Patterson. *Roofline.* CACM 52(4), 2009. <https://doi.org/10.1145/1498765.1498785>

Fuente no académica (solo como motivo para medir): 31. `lgoyal6/ddp-crossover`, GitHub. <https://github.com/lgoyal6/ddp-crossover>
