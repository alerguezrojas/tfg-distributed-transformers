# Entrenamiento elástico y su encaje en ParaViT-Lab

> Revisión pedida en la reunión con el tutor del 24/11/2026. Fecha de la revisión: 24/11/2026.
> Las referencias se han abierto una a una (página oficial, DOI o arXiv). Las cifras son **las que
> dan los propios autores**; lo que es interpretación nuestra va marcado como tal y lo que no se
> pudo comprobar, como *no verificado* (resumen en el apartado 6).

## 1. Qué es el entrenamiento elástico

Un entrenamiento distribuido clásico fija al lanzarlo cuántas GPU usa. Si una falla, o si el
planificador de un clúster quiere quitarle una, el trabajo muere. Un entrenamiento **elástico**
admite que ese número cambie mientras el trabajo está vivo. Tres situaciones lo motivan:

- **Fallos**: un nodo se cae y se prefiere seguir con los que quedan antes que repetir todo.
- **Máquinas interrumpibles** (spot, preemptible): son más baratas, pero el proveedor puede
  reclamarlas en cualquier momento (Varuna, Bamboo).
- **Clústeres compartidos**: un planificador reparte las GPU libres entre trabajos y las va
  moviendo según la carga (Pollux, AFS, Sia).

Los mecanismos comunes son pocos:

1. **Rendezvous**: los procesos se registran en un punto de encuentro, acuerdan quién forma parte
   del grupo y reciben su `rank` y el `world_size`. Si el grupo cambia, se repite.
2. **Reinicio desde estado guardado**: el grupo nuevo recupera el modelo, el optimizador y el
   punto del bucle (época, paso) de un checkpoint, o de la memoria de un compañero vivo.
3. **Reescalado de la receta**: con otro número de GPU cambia el lote global, y con él el *learning
   rate* y, si hace falta, el calentamiento (apartado 2.5).

## 2. Implementaciones de referencia

### 2.1 PyTorch Elastic / `torchrun`

Está integrado en PyTorch (`torch.distributed.elastic`) y `torchrun` es su lanzador. Según la
[documentación oficial](https://docs.pytorch.org/docs/2.15/elastic/run.html) (versión 2.15):

- `--nnodes=MIN:MAX` da el rango de nodos; el ejemplo del manual es `--nnodes=1:4` (mínimo 1,
  máximo 4). Con un número fijo (`--nnodes=N`) se obtiene solo tolerancia a fallos, sin elasticidad.
- `--max-restarts` limita los reinicios. Cuando fallan *k* de *n* procesos, **se detienen todos y
  se reinician** (hasta ese máximo); el contador se expone en `TORCHELASTIC_RESTART_COUNT`.
- Ante un cambio de miembros (entra o sale un nodo), «todos los procesos supervivientes se matan
  de inmediato» y se forma un grupo nuevo con nuevos `RANK` y `WORLD_SIZE`.
- Rendezvous: `c10d` (el recomendado), `etcd-v2` y `etcd` (heredado). La documentación avisa de
  que etcd está en mantenimiento y se eliminará. En multinodo se indican `--rdzv-id`,
  `--rdzv-backend` y `--rdzv-endpoint`.
- El programa debe **guardar checkpoints con regularidad** y cargarlos al empezar `main()`, y no
  debe suponer que `rank` ni `world_size` son estables.
- La página **no dice nada** sobre el lote ni el LR al cambiar el número de procesos: es
  responsabilidad del script.

Consecuencia (interpretación nuestra): `torchrun` es elástico *por reinicio*. Cada cambio cuesta lo
que tarde en arrancar el grupo, cargar el checkpoint y repetir lo perdido desde el último guardado.

### 2.2 Horovod Elastic

Según su [documentación](https://horovod.readthedocs.io/en/stable/elastic_include.html):

- El bucle se envuelve en `hvd.elastic.run` y el estado (pesos, optimizador, contadores de época y
  lote) vive en un objeto `State` que se sincroniza entre procesos.
- `state.commit()` guarda una copia **en memoria**; si un proceso falla, se restaura el último
  `commit` y se repite lo que haga falta. `state.check_host_updates()` detecta altas y bajas
  anunciadas y permite salir sin retroceder.
- Los nodos se descubren con `--host-discovery-script`; hay `--min-np`, `--max-np` y una lista
  negra de nodos que fallan. Tras cada cambio, el estado se difunde desde el nuevo proceso 0.
- **Lote y LR**: con promedio de gradientes, el LR debe reescalarse en un *callback* de
  reinicio (p. ej. `base_lr * hvd.size()`). Para el lote, recomienda fijar las particiones de
  datos al máximo (`hvd.max_size()`) y usar agregación local de gradientes para mantener **constante
  el lote global**.

### 2.3 DeepSpeed

- **Universal Checkpointing** (Lian et al., arXiv 2024; ver referencia 7). Guarda el checkpoint
  distribuido y lo convierte a un formato consolidado por parámetro, de modo que se puede
  reanudar con **otro paralelismo y otro número de GPU**. Los autores lo presentan como base para
  seguir entrenando con el hardware sano y aprovechar capacidad elástica.
- En la [configuración oficial](https://www.deepspeed.ai/docs/config-json/) la opción
  `elastic_checkpoint` de ZeRO figura como obsoleta, con Universal Checkpointing como sustituto
  para ZeRO-3.
- DeepSpeed también tiene un módulo de elasticidad con un bloque `elasticity` en el JSON
  (`max_train_batch_size`, `micro_batch_sizes`, `min_gpus`, `max_gpus`). **No he podido
  verificarlo en la documentación oficial** (la página de configuración no lo recoge y la ruta de
  su tutorial devuelve 404); solo lo describen fuentes de terceros. No se afirma nada más de él.

### 2.4 TorchFT y otros

- **TorchFT** ([repositorio](https://github.com/meta-pytorch/torchft)): tolerancia a fallos *por
  paso*, sin parar el trabajo entero. Un `Lighthouse` (servidor) sabe qué grupos están vivos
  mediante latidos por paso, y cada grupo de réplicas tiene un `Manager`. Ofrece DDP e HSDP
  tolerantes a fallos y, de forma experimental, LocalSGD y DiLoCo. El
  [blog de PyTorch](https://pytorch.org/blog/fault-tolerant-llama-training-with-2000-synthetic-failures-every-15-seconds-and-no-checkpoints-on-crusoe-l40s/)
  describe que un grupo que vuelve recupera los pesos de un compañero sano en lugar de un
  checkpoint. El estado «alfa» lo dan buscadores y el paquete de PyPI, **no la página del
  repositorio**; tómese como no confirmado en la fuente oficial.
- **Varuna** (EuroSys 2022) y **Bamboo** (NSDI 2023): entrenamiento de modelos grandes en
  instancias spot. Varuna reconfigura el trabajo dinámicamente conservando la semántica del
  entrenamiento; Bamboo mete cómputo redundante en las burbujas del pipeline para resistir
  expulsiones (los autores dan 3,7 veces más rendimiento que el checkpointing tradicional y 2,4
  veces menos coste que las instancias bajo demanda).
- **EasyScale** (SC'23): elasticidad que busca **resultados idénticos** (misma precisión) con
  cualquier número de GPU, separando el entrenamiento de la asignación de recursos.

### 2.5 Qué pasa con el lote y el LR

Hay dos políticas posibles, y la elección es metodológica:

- **Lote global fijo** (Horovod con `max_size`, EasyScale): al cambiar de GPU se cambia el lote por
  GPU o los pasos de acumulación. La receta no se toca y los resultados son comparables, pero hay
  un tope: el lote por GPU debe caber en memoria y dividirse bien.
- **Lote global variable**: al cambiar de GPU, cambia el lote. Hay que reescalar el LR. La regla
  lineal de Goyal et al. (referencia 8) propone LR proporcional al lote, con calentamiento
  (*warmup*), y los autores la presentan sin hiperparámetros adicionales. Pollux ajusta el LR con
  AdaScale para SGD y con raíz cuadrada para Adam/AdamW (§5.1 del artículo).

La eficiencia de un lote grande tiene un límite que depende de la **escala de ruido del gradiente**
(McCandlish et al., referencia 9); más allá, añadir GPU casi no reduce los pasos necesarios.

## 3. Trabajo académico que modela rendimiento para asignar recursos

Es la parte más cercana a una publicación, porque estos trabajos **necesitan predecir** lo mismo
que nuestro predictor, pero lo hacen en línea y para un planificador.

| Trabajo | Qué modela | Cómo se obtiene el modelo |
|---|---|---|
| Optimus (EuroSys'18) | Velocidad `f(p, w)` según servidores de parámetros y workers; curva de pérdida `l = 1/(β0·k+β1)+β2` para estimar los pasos que faltan | NNLS con ~10 configuraciones de pasos reales, recalibrado en marcha; asigna recursos por ganancia marginal |
| AFS / CoDDL (NSDI'21) | Política de reparto elástico (Apathetic Future Share) y marco con paralelización automática y reajuste frecuente | No es un predictor fino; su aportación es la política |
| Pollux (OSDI'21) | *Goodput* = rendimiento del sistema × eficiencia estadística; `T_iter = (T_grad^γ + T_sync^γ)^(1/γ)`; eficiencia `(φ+M0)/(φ+M)` con la escala de ruido φ | Ajuste en línea de parámetros por trabajo; penalización por reasignación |
| Sia (SOSP'23) | Modelos de rendimiento por tipo de GPU, para clústeres heterogéneos y trabajos con paralelismo híbrido | Modelo inicial con muy poco perfilado, afinado después |

Detalles que dan los autores y conviene conservar:

- **Optimus**: con menos de 10 configuraciones `(p, w)` obtienen un error menor del 10 % en la
  velocidad (cifra recogida en [`estado_del_arte_prediccion.md`](estado_del_arte_prediccion.md)).
- **Pollux**: mide entre 15 y 120 s de coste de reasignación por *checkpoint-restart* según el
  modelo, usa 30 s en sus experimentos y reasigna un trabajo cada ~7 min, con ~8 % de sobrecoste
  (§5.1-5.3). Esa penalización se descuenta de la aceleración de cada trabajo: **el coste del
  reinicio entra en el modelo de decisión**. Los autores reportan un 37-50 % menos de tiempo medio
  de finalización que otros planificadores.
- **Sia**: según los autores, es el primer planificador con escalado elástico de trabajos de
  paralelismo híbrido y heterogeneidad de GPU.

**Relación con nuestro predictor** (interpretación nuestra):

- Nuestra `T_epoch` es de la misma familia que `T_iter` de Pollux: cómputo, sincronización
  (`allreduce_time`) y un término de entrada/salida, con constantes calibradas por GPU. La
  diferencia es que la calibramos **fuera de línea con medidas de cada máquina** y la usamos
  **antes** de entrenar, y además predecimos memoria y energía, que estos planificadores no.
- Lo que no tenemos es la mitad estadística: Pollux y la ley de McCandlish dicen cuántos pasos
  hacen falta con un lote dado. Nuestro predictor da el tiempo por época con el número de épocas
  fijado por el usuario; con lote global fijo no hace falta (las épocas no cambian).
- Un predictor «tiempo/energía con *n* GPU» es justo el ingrediente que Optimus y Pollux estiman
  en línea. Que el nuestro sea analítico y puro permite usarlo **sin ejecutar nada**.

## 4. Encaje en ParaViT-Lab

### 4.1 Lo que ya hay

- `src/slurm.py` (`slurm_torchrun`) ya lanza `torch.distributed.run` con rendezvous `c10d`, el
  mismo comando en todos los nodos y `--rdzv-id` igual al id del trabajo. Hoy usa un `--nnodes=N`
  **fijo** y no pasa `--max-restarts`.
- `src/training/checkpoints.py`: `last.pt` (última época, escritura atómica) y
  `resolve_checkpoint`; `launch.py` tiene `--resume`; `Trainer.load_checkpoint` restaura modelo,
  optimizador, planificador, *scaler*, época, mejor métrica y paciencia.
- `DistributedSampler` se baraja con `semilla + época`, así que una época entera es reproducible al
  reanudar con el mismo número de procesos.
- `kaggle/`: las sesiones se cortan a las 12 h. Hoy se evita partiendo el trabajo en lotes que
  caben en la sesión; es un caso de «reinicio forzoso» ya vivido.
- `performance_model.predict_epoch` ya calcula el tiempo con *n* GPU (cómputo, `allreduce_time`,
  I/O) y la eficiencia de escalado `speedup / n_gpus`.

### 4.2 Lo que faltaría

1. **Reanudar a mitad de época.** Solo se guarda al final de cada época, de modo que un fallo
   pierde hasta una época entera de trabajo. Habría que guardar cada
   *k* pasos con el índice de lote consumido y saltar esos lotes con el mismo barajado.
2. **Cambiar el número de procesos al reanudar.** Con otro `world_size` el reparto del sampler
   cambia (`HeterogeneousDistributedSampler` también), y con el mismo lote por GPU cambia el lote
   global. Hay que decidir la política de 2.5 (recomendación: lote global fijo, ver más abajo).
3. **Tolerancia a fallos en el lanzador.** `--max-restarts` y un `--nnodes=MIN:MAX` en el lanzador
   de Slurm, y que el script cargue `last.pt` solo al detectar un reinicio
   (`TORCHELASTIC_RESTART_COUNT`).
4. **Estado de los decoradores.** El flujo `events_*.jsonl` y los CSV deben anotar los reinicios
   sin romper el contrato con la web (`Configuración:` y demás claves).
5. **Pruebas.** Solo hay CPU en la CI, y `gloo` no admite `ReduceOp.AVG`; un reinicio real se
   comprobaría con dos procesos `gloo` en CPU y matando uno.

### 4.3 Cómo puede aportar el predictor

- **Cuántas GPU usar**: con `predict_epoch` para *n* = 1…N se obtiene tiempo, energía y
  eficiencia; con una restricción (plazo o presupuesto de energía) se elige *n*. El roadmap
  recoge que, frente a los entrenamientos reales en 2 × T4, el error del predictor es del 15 %.
- **Coste de un reinicio**: tiempo de arranque + carga de `last.pt` + trabajo perdido (media de
  medio intervalo entre guardados). Pollux da una referencia de 15-120 s de reinicio por
  *checkpoint-restart*; en nuestro caso habría que medirlo (los logs ya registran los tiempos).
- **Intervalo de guardado óptimo**: con la tasa de fallos o de expulsión y el tiempo por paso
  que predice el modelo, equilibrar coste de guardar frente a trabajo perdido (planteamiento
  clásico; *no verificado* contra una fuente concreta en esta revisión).
- **Decidir si merece la pena reescalar** a mitad de trabajo: comparar el tiempo restante con *n*
  actual y con *n'* más el coste del reinicio, como hace Pollux con su penalización.

## 5. Pasos propuestos (por valor/esfuerzo, sin implementar)

1. **Reanudación robusta con el mismo número de GPU** (esfuerzo bajo, valor alto). Guardar
   `last.pt` cada *k* pasos con el estado del sampler (época y lote consumido), `--max-restarts`
   en `slurm_torchrun` y reanudar solo si hay `last.pt`. Resuelve Kaggle (12 h) y spot, sin tocar
   la receta. Medir el coste real de reinicio y registrarlo en el log.
2. **Predicción del coste de reinicio y del reparto de GPU en el predictor** (esfuerzo bajo-medio,
   valor alto para la publicación). Añadir a `predict` un campo de «tiempo y energía esperados
   con fallos» (parámetros: tasa de fallos, intervalo de guardado, tiempo de reinicio) y un modo
   «`--n-gpus` mejor» que barra *n*. Contrastarlo con reinicios provocados en Kaggle o local, en
   línea con la regla del proyecto de no predecir lo que no se puede contrastar.
3. **Elasticidad real con lote global fijo** (esfuerzo medio-alto, valor medio). `--nnodes=MIN:MAX`
   en Slurm, rederivar lote por GPU y acumulación al reanudar con otro `world_size`, y un test
   con `gloo` que mata un proceso. Se deja para después de 1 y 2; el lote global variable
   (estilo Pollux) exigiría además un modelo de eficiencia estadística que no tenemos.

## 6. Qué no se ha podido verificar

- **DeepSpeed `elasticity`** (`micro_batch_sizes`, `min_gpus`, `max_gpus`): no está en la
  documentación oficial que pude abrir; no se describe aquí.
- **Estado «alfa» de TorchFT**: aparece en terceros; la página del repositorio no lo dice.
- **Optimus en ACM**: la página del DOI devolvió 403. Título, autores, congreso y DOI se confirmaron
  con una búsqueda (registro de la Universidad de Hong Kong) y la fórmula de la curva y de la
  velocidad con un resumen de terceros (Adrian Colyer); no he podido leer el artículo en sí.
- **Fórmula de AdaScale** y valores de la escala de ruido: el artículo de Pollux no los da en el
  fragmento leído; no se reproducen.
- **Intervalo óptimo de guardado** (apartado 4.3): sin fuente.
- Del texto de Pollux se leyó el 90 % (el resto de la página HTML quedó sin leer).

## 7. Referencias

1. PyTorch. *torchrun (Elastic Launch)*, documentación 2.15. <https://docs.pytorch.org/docs/2.15/elastic/run.html>
2. Horovod. *Elastic Horovod*. <https://horovod.readthedocs.io/en/stable/elastic_include.html>
3. PyTorch. *torchft*, repositorio. <https://github.com/meta-pytorch/torchft> · Entrada de blog:
   <https://pytorch.org/blog/fault-tolerant-llama-training-with-2000-synthetic-failures-every-15-seconds-and-no-checkpoints-on-crusoe-l40s/>
4. Y. Peng, Y. Bao, Y. Chen, C. Wu, C. Guo. *Optimus: An Efficient Dynamic Resource Scheduler for Deep
   Learning Clusters.* EuroSys 2018. <https://doi.org/10.1145/3190508.3190517>
5. A. Qiao, S. K. Choe, S. J. Subramanya, W. Neiswanger, Q. Ho, H. Zhang, G. R. Ganger, E. P. Xing.
   *Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning.* OSDI 2021.
   <https://www.usenix.org/conference/osdi21/presentation/qiao> · arXiv:2008.12260 <https://arxiv.org/abs/2008.12260>
6. C. Hwang, T. Kim, S. Kim, J. Shin, K. Park. *Elastic Resource Sharing for Distributed Deep
   Learning.* NSDI 2021, pp. 721-739. <https://www.usenix.org/conference/nsdi21/presentation/hwang>
7. X. Lian, S. A. Jacobs, L. Kurilenko, M. Tanaka, S. Bekman, O. Ruwase, M. Zhang. *Universal
   Checkpointing: A Flexible and Efficient Distributed Checkpointing System for Large-Scale DNN
   Training with Reconfigurable Parallelism.* arXiv:2406.18820 (2024; v3, 2025), sin venue indicado
   en la página. <https://arxiv.org/abs/2406.18820>
8. P. Goyal, P. Dollár, R. Girshick, P. Noordhuis, L. Wesolowski, A. Kyrola, A. Tulloch, Y. Jia,
   K. He. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* arXiv:1706.02677 (2017).
   <https://arxiv.org/abs/1706.02677>
9. S. McCandlish, J. Kaplan, D. Amodei, OpenAI Dota Team. *An Empirical Model of Large-Batch
   Training.* arXiv:1812.06162 (2018). <https://arxiv.org/abs/1812.06162>
10. S. Jayaram Subramanya, D. Arfeen, S. Lin, A. Qiao, Z. Jia, G. R. Ganger. *Sia:
    Heterogeneity-aware, Goodput-optimized ML-cluster Scheduling.* SOSP 2023.
    <https://www.pdl.cmu.edu/PDL-FTP/BigLearning/sia_sosp23-final_abs.shtml> (DOI 10.1145/3600006.3613175
    según el registro de ACM en la búsqueda; la página de ACM devolvió 403)
11. S. Athlur, N. Saran, M. Sivathanu, R. Ramjee, N. Kwatra. *Varuna: Scalable, Low-cost Training of
    Massive Deep Learning Models.* EuroSys 2022; arXiv:2111.04007. <https://arxiv.org/abs/2111.04007>
    (el congreso y el premio constan en resultados de búsqueda y en el PDL de CMU; la ficha de arXiv
    no trae referencia de revista)
12. J. Thorpe, P. Zhao, J. Eyolfson, Y. Qiao, Z. Jia, M. Zhang, R. Netravali, G. H. Xu. *Bamboo: Making
    Preemptible Instances Resilient for Affordable Training of Large DNNs.* NSDI 2023, pp. 497-513.
    <https://www.usenix.org/conference/nsdi23/presentation/thorpe>
13. M. Li, W. Xiao, B. Sun, H. Zhao, H. Yang, S. Ren, Z. Luan, X. Jia, Y. Liu, Y. Li, W. Lin, D. Qian.
    *EasyScale: Accuracy-consistent Elastic Training for Deep Learning.* arXiv:2208.14228 (la ficha
    indica «SC'23»). <https://arxiv.org/abs/2208.14228>
