# Analytic performance model

`src/performance_model.py` predicts training **time, speedup, memory and the
bottleneck** for any combination of *(strategy, model, GPU, number of GPUs,
dataset size, batch, precision)* **without running a benchmark**. It generalizes
the feasibility checker — which until now needed a real measurement on the target
machine — into a closed-form predictor, as requested by the tutor.

It is a pure module (no torch, no GPU, no I/O), so it is fully unit-testable and
runs instantly. If a real measured throughput is available it can be passed in to
**calibrate** the prediction (it overrides the estimate).

---

## 1. Parameters

| Symbol | Meaning | How it is obtained |
|--------|---------|--------------------|
| `N`    | train images per epoch | input |
| `n`    | number of GPUs (ranks) | input |
| `b`    | batch size per GPU | input (global batch ÷ n for DDP) |
| `P`    | model parameters | `MODEL_TABLE`, or an automatic profile (§7) |
| `r_c`  | pure compute throughput (img/s, 1 GPU) | **estimated from specs** |
| `r_io` | disk read throughput (img/s) | `DISK_RIO` table or a real probe |
| `π`    | precision / Tensor-core speedup | `GPU_TABLE` (fp32 → 1) |
| `β`    | interconnect bandwidth (B/s) | NVLink / PCIe / Ethernet |
| `φ`    | NFS penalty | 1.3 on the cluster, 1.0 local |

### Estimating `r_c` without a benchmark (the key piece)

```
r_c(fp32) = MFU · TFLOPS_fp32(gpu) / FLOPs_train(model)
r_c(prec) = π(gpu, prec) · r_c(fp32)
FLOPs_train = 3 · FLOPs_forward      (forward + backward ≈ 3× forward)
```

`MFU` (model-FLOPs utilization) is the single calibration constant. It is fitted
so the model reproduces the real measurements: vit_base on a T4 (fp32) measured
**≈ 26 img/s**, and with `FLOPs_train = 3 × 17.6 GFLOP` and the T4's `8.1` fp32
TFLOPS that pins **MFU ≈ 0.17**.

`π` is the *measured* Tensor-core speedup per GPU (T4 = 3.8, RTX 3060 Ti = 2.54),
not the raw fp16/fp32 TFLOPS ratio (which is ~8× and never reached in practice).

**Mixed precision (since 23/11/2026).** In `amp`, `r_c` comes from the three-term model of
section 8 instead, when the model has a saved-activation figure (`bf16`, not measured, keeps
MFU × π):

```
1 / r_c(amp) = a·F + b·A + d/B      (F: GFLOPs fwd, A: GB saved by autograd, B: images per step)
```

That adds the memory traffic the FLOPs-only form misses (Swin, CNNs, EfficientNet). fp32 keeps the
MFU form above, and a measured `r_c` always wins.

---

## 2. Master formula — DDP train epoch

```
T(N, n) = φ · [ max( N/(π·r_c·n) , N/r_io )  +  (8P/β)·⌈N/(b·n)⌉ ]
               \________ compute ∥ disk ________/   \____ gradient sync ____/
```

* compute and disk run **at the same time** → the **slower** one counts (`max`);
* the gradient all-reduce (`8P` bytes = 2·P fp32 grads, per batch) runs **after**
  → it is **added**.

### Speedup and the three regimes

```
S(n) = T(N,1)/T(N,n)
```

| Regime | Condition | Limit | Example |
|--------|-----------|-------|---------|
| **Compute-bound** | `r_c ≪ r_io`, sync ~ 0 | `S(n) → n` (≈linear) | vit_base |
| **I/O-bound** | `r_io ≪ r_c` | `S(n) → 1` (disk dominates) | vit_tiny |
| **Sync-bound** | big model / low β | sync dominates → S flattens | vit_large multi-node |

Defining the serial fraction `s = (non-scaling part)/(T(1)/φ)`, this **is** Amdahl
`S(n) = 1/(s + (1−s)/n)` — which is why the dashboard overlays Linear / Amdahl /
Gustafson on the predicted curve.

### Variants per strategy

* **single** (`n=1`): `T = φ · max(N/(π·r_c), N/r_io)` — no sync.
* **ddp**: the master formula.
* **precision/AMP**: `π` scales **only** compute. So DDP+AMP does **not** multiply
  (measured 5.97× < 1.96 × 3.80 = 7.4×): shrinking compute makes sync weigh more,
  so DDP efficiency drops 98% → 78%.
* **model-parallel** (naive pipeline): stages serialize → `T ≈ N/r_c + transfer ≥
  T_single` → `S ≈ 1`. It does not accelerate; it exists to **fit models too big
  for one GPU** (vit_large OOMs on one T4, trains split across two).
* **heterogeneous** (GPU + CPU, synchronous DDP): the slowest rank sets the pace.
  Ceiling `S ≤ 1 + r_CPU/r_GPU ≈ 1.02×`; in practice 0.12× from the synchronous
  barrier + gloo/Ethernet overhead. Imbalanced hardware penalizes.

---

## 3. Memory model (OOM prediction)

```
VRAM ≈ weights+grad+optimizer(P, precision) + activations(P, batch) + overhead
     = 16·P  (+ 2·P fp16 copy for AMP)  +  k · P · batch  +  0.6 GB
```

`k = 0.86e-9 GB / (param · image)` is fitted to the precisely measured
`vit_large = 13.78 GB at batch 32 on a T4`. It predicts the largest batch that
fits and flags OOM for any (model, GPU).

---

## 4. Calibration vs. validation (be honest about which is which)

The fp32 model has **two** fitted constants, each pinned by **one** measurement:

* `MFU = 0.17` — from the single-GPU fp32 throughput of vit_base on a T4.
* `k = 0.86e-9` — from the measured vit_large OOM point (13.78 GB @ batch 32, T4).

Mixed precision (AMP) adds the three coefficients per GPU of `THROUGHPUT_FITS` (section 8),
fitted on synthetic benchmarks of 10-13 architectures; the rows below that use AMP are marked ¹.

Reproducing those two points is therefore **interpolation, not prediction** — they
are *calibration targets*, marked **(in-sample)** below.

The genuinely predictive results are the **speedups and regimes**, and they are
**out-of-sample because they do not depend on the fitted constants**. A speedup is
a ratio `S(n)=T(N,1)/T(N,n)`: in the compute-bound regime `MFU` cancels, so `S→n`
regardless of its value; the FP32→AMP factor comes from the three-term AMP model, not from
`MFU` (see ¹); the I/O-bound and heterogeneous regimes are set by
`r_io`/β, again independent of the calibration. These are the rows that actually
test the model.

| Quantity | Predicted | Real | Error | Kind |
|----------|-----------|------|-------|------|
| Single fp32, train/epoch | 192 s | 194 s | +1% | calibration (in-sample) |
| vit_large @ batch 32, T4 | 13.8 GB, fits | 13.78 GB, fits | <1% | calibration (in-sample) |
| **DDP 2×T4 speedup** | **1.95×** | **1.96×** | **<1%** | **validation (out-of-sample)** |
| **FP32 → AMP speedup** | **3.91×** | **3.80×** | **+3%** | **validation (out-of-sample)** ¹ |
| **Model-parallel speedup** | **1.00×** | **1.02×** | **−2%** | **validation (out-of-sample)** |
| **vit_tiny DDP regime** | I/O-bound, <1.4× | 1.27× | ✓ | **validation (out-of-sample)** |
| **vit_large @ batch 48, T4** | OOM | OOM | ✓ | **validation (out-of-sample)** |
| **Heterogeneous V100+CPU** | < 1× | 0.12× | ✓ (penalizes) | **validation (out-of-sample)** |
| Energy single fp32, train/epoch | 3.4 Wh | 3.2 Wh | +6% | calibration (power in-sample) |
| **Energy single AMP, train/epoch** | **0.87 Wh** | **0.84 Wh** | **+4%** | **validation (time-driven)** ¹ |
| **Energy model-parallel/epoch** | **5.7 Wh** | **5.4 Wh** | **+5%** | validation (MP factor in-sample) |

All rows are reproduced by the unit tests in `tests/unit/test_performance_model.py`.

¹ Since 23/11/2026 mixed precision uses the three-term throughput model of section 8 (T4 fit on
synthetic 512 px batches of 13 architectures, none of them this BigEarthNet run at 224 px).
Before, AMP was the fp32 MFU estimate × π, and π for the T4 (3.8) had been measured from this
very ratio, so the old 3.80× row was not a real test; the energy row moved from 0.90 to 0.87 Wh.

### Out-of-sample check: PANDA on 2×T4 (Kaggle, 20/11/2026)

A new dataset, a new resolution and a profiled model (ViT-S/16 at 512 and 1024 px, not a table
entry), so nothing here was fitted. DDP on 2 × T4, global batch 16 (8 per GPU), AMP, 7,433 train
images, `--monitor`. "Real" is the median train time per epoch of the run's `epoch_end` events
(`logs/kaggle/panda/ddp/vit_small_patch16_224/`); "Predicted" is `time_per_epoch_train_s`.

| Run | Predicted train/epoch | Real train/epoch | Error | Predicted regime | Measured |
|---|---|---|---|---|---|
| 512 px (`20112026_163243`) | 73 s | 122 s | −40 % | I/O-bound | GPU util. median 71 %, data wait 1.9 % |
| 1024 px (`20112026_171003`) | 523 s | 620 s | −16 % | compute-bound | GPU util. 100 %, data wait 0.4 % |
| VRAM per GPU, 1024 px | 4.9 GB | 7.0 GB median (nvidia-smi) | — | fits | fits (15.0 GB peak with the monitor's Grad-CAM) |

Both times are **under-estimated**, and the regime is only half right:

* At 1024 px the regime is right (compute-bound), but the T4s run **power-capped**: the median SM
  clock is 960 MHz against a 1,590 MHz boost, at ~65 W of their 70 W. The `MFU = 0.17`
  calibration comes from short vit_base runs at 224 px. The monitor's per-batch work also counts
  in the measured time.
* At 512 px the GPU waits on its input for only 1.9 % of the time, so the epoch is not I/O-bound
  as predicted. 71 % utilisation with 8 images per GPU suggests the small per-GPU batch, not the
  disk, limits throughput. The model has no batch-size term in `MFU`.
* The memory model gives the PyTorch peak of the training step; nvidia-smi adds the CUDA context
  and the allocator's cache, so the two are not the same quantity. A peak from
  `torch.cuda.max_memory_allocated` is needed to settle it.

These rows are **validation, not calibration**: no constant was changed after seeing them. They
point at two missing terms (sustained clock under a power cap, efficiency at small per-GPU
batches) that need measurements on more machines before they are added.

**Revisión (23/11/2026).** El *benchmark* sintético en una T4 (sin datos ni I/O, kernel
`kaggle/benchmark_t4`) da para ViT-S a 512 px 48 img/s con lote 8 y 51 con lote 16: el lote pequeño
por GPU apenas cuesta un 5 %, así que no explica el −40 %. Las muestras del sistema del run de
512 px dicen otra cosa: **CPU al 100 %** (mediana), GPU al 71 % y reloj a 1.275 MHz (no limitado
por potencia). Las 4 vCPU de Kaggle no dan abasto entre la decodificación de los mosaicos JPEG de
512 px y los dos procesos de entrenamiento; la contención de CPU frena el paso aunque el bucle
apenas espere al *loader* (1,9 %). Con 2 × 48 img/s la GPU haría la época en ~77 s (el predictor
dice 73), así que los 122 s medidos no son la GPU. La hipótesis más probable es la **contención
de CPU**, pero no está probada: EfficientNet-B3 se comporta igual (GPU al 76 %), y en cambio
ConvNeXt-S y SwinV2-T tienen la CPU casi igual de saturada (97 %) con la GPU al 95-97 %, de modo
que "CPU al 100 %" no basta para distinguir unos runs de otros (los procesos esperan activamente a
la GPU). Lo que sí la apoya es que el proceso de entrenamiento de ViT-S usa el 44 % de un núcleo,
frente a ~100 % en los modelos pesados. El experimento que la cierra: el mismo run en Kaggle con
`num_workers=2`, o con los mosaicos ya decodificados. En los modelos pesados (ViT-B, ConvNeXt-S,
SwinV2-T, MaxViT-T: GPU al 94-100 %, reloj a 817-967 MHz) sí manda la GPU.

---

## 5. Energy model

Energy is just **power × the time the model already predicts**, so the only missing
term is the average power the GPU draws while training. Instead of the nameplate TDP
(a ceiling the GPU rarely reaches), the **effective** average power is calibrated from
the project's own `--fn energy` logs (the `potencia media XX W` lines):

```
energy_train_wh = power_total · time_train_s / 3600
power_total     = train_power_w(gpu) · n_working_gpus            (× 0.83 for model-parallel)
energy_eval_wh  = 0.9 · train_power_w · n · time_eval_s / 3600   (eval ≈ 0.9× train power)
```

Two measured facts shape it:

1. **Power is almost precision-independent** (T4 fp32 ≈ 64 W ≈ AMP ≈ 63 W). AMP saves
   energy by finishing ~4× faster — through **time** — not by drawing less power. So
   precision is not a parameter of the power, only of the time.
2. **n_working_gpus**: `single`/`heterogeneous` = 1 (the heterogeneous second worker is
   a CPU, not a GPU), `ddp`/`model_parallel` = n. Naive model-parallel keeps both GPUs
   powered but idling between stages, so each draws ~0.83× (calibrated from the 2×T4 MP
   run: 106 W ≈ 0.83 × 2 × 64 W).

Calibrated per GPU: **T4 64 W, V100 103 W, RTX 3060 Ti 180 W** (measured); the rest fall
back to `0.65 × nameplate TDP` and are flagged `power_calibrated=False`.

**Consequence for DDP:** the total energy is **conserved across GPUs only when
compute-bound** (n GPUs each do 1/n of the work in 1/n the time, at n× the power). When
the run is **I/O-bound**, the wall-clock does not drop while n GPUs draw full power, so
total energy rises toward ~n×.

**Known over-estimates (documented, not bugs):** the analytic energy reads **high** when
the GPU is under-used, because it inherits the time model's I/O floor and uses a
compute-bound power: (a) light/I/O-bound models on a RAM-cached subset (vit_tiny);
(b) **DDP+AMP on a RAM-cached subset** — AMP shrinks compute below the disk floor, the
epoch goes I/O-bound in the model and energy ~doubles, while the real RAM-cached run
stays compute-bound (`predict()` flags this with an "I/O floor active under this
precision" note; pass `rio_measured` to correct it); (c) heterogeneous (the GPU idles
near ~45 W waiting on the slow CPU, below its compute-bound power).

---

## 6. Unified API

```python
from src.performance_model import predict

p = predict(strategy="ddp", model_name="vit_base_patch16_224",
            gpu_name="Tesla T4", n_gpus=2, dataset_size=5000, batch=96,
            precision="amp", epochs=15, disk_type="ssd", nfs=False, val_size=1500)
# time:   p.time_per_epoch_train_s, p.time_per_epoch_eval_s, p.time_per_epoch_total_s
# scaling: p.speedup, p.efficiency, p.bottleneck
# memory: p.vram_per_gpu_gb, p.fits_in_memory, p.recommended_batch
# energy: p.avg_power_w, p.power_total_w, p.energy_train_wh, p.energy_eval_wh,
#         p.energy_per_epoch_wh, p.energy_total_wh, p.power_calibrated
# p.notes
```

`strategy ∈ {single, ddp, model_parallel, heterogeneous}` (an unknown strategy returns
`None`). Pass `model=<ModelSpec>` to predict a model outside
`MODEL_TABLE` (see §7). Pass `rc_measured` / `rio_measured` to calibrate against a real benchmark.
Exposed in the web at **Estimate → Predict** and in the CLI as `paravit estimate`
(predict for hardware you don't have in front of you).

---

## 7. Modelos fuera de la tabla: perfilado automático

`MODEL_TABLE` solo recoge seis modelos. Para cualquier otro modelo de timm,
`src/model_profile.py` mide del propio modelo lo que el predictor necesita, sin
entrenar y sin GPU:

* **Parámetros** (millones), con el `in_chans` y el `num_classes` indicados.
* **FLOPs de la pasada hacia delante por imagen**, con
  `torch.utils.flop_counter.FlopCounterMode`. El modelo se instancia en el dispositivo
  `meta` (solo formas, sin reservar pesos: vit_large tarda ~0,3 s); si un modelo no se
  puede construir o ejecutar ahí, se repite en CPU. La atención se fuerza por el kernel
  SDPA `math` para que el contador vea sus multiplicaciones de matrices (el kernel
  fusionado de CPU no se cuenta e infravalora los ViT un ~4 %).

**Unidad.** `FlopCounterMode` cuenta una multiplicación-suma como 2 FLOPs (vit_base =
35,1 G), pero la columna `gflops_fwd` de `MODEL_TABLE` —como las tablas de timm— la
cuenta como 1 (vit_base = 17,6, es decir, GMACs). La MFU = 0,17 se calibró con esos
números, así que el perfil divide su cuenta entre 2 para quedarse en la unidad de la
tabla.

**Comprobación contra la tabla** (cabeza de 1000 clases, como en timm):

| Modelo | Parámetros (perfil / tabla) | GFLOPs (perfil / tabla) |
|---|---|---|
| vit_tiny_patch16_224 | 5,72 / 5,7 (+0,3 %) | 1,25 / 1,3 (−3,6 %) |
| vit_small_patch16_224 | 22,05 / 22,0 (+0,2 %) | 4,60 / 4,6 (0,0 %) |
| vit_base_patch16_224 | 86,57 / 85,8 (+0,9 %) | 17,56 / 17,6 (−0,2 %) |
| vit_large_patch16_224 | 304,33 / 303,0 (+0,4 %) | 61,55 / 61,6 (−0,1 %) |
| resnet50 | 25,56 / 25,6 (−0,2 %) | 4,09 / 4,1 (−0,3 %) |
| efficientnet_b0 | 5,29 / 5,3 (−0,2 %) | 0,39 / 0,4 (−3,5 %) |

(La tabla mezcla convenios de cabeza: vit_base y vit_large aparecen con la cabeza de 19
clases de BigEarthNet —85,81 y 303,32 M en el perfil—; el resto, con la de 1000.)

**Memoria de activaciones: estimada, no calibrada.** Un contador de FLOPs no da la
memoria por imagen (`act_gb_per_img`), así que para estos modelos se estima con una
regla anclada en los dos puntos ViT calibrados (vit_base 0,093 GB a 17,6 GFLOPs y
vit_large 0,260 GB a 61,6 GFLOPs):

```
act(G) = 0,093 · (G / 17,6)^e        e = 0,82 si G > 17,6  (ley de potencia por base y large)
                                     e = 0,5  si G ≤ 17,6
act    = act(G) · 2,23               si la red es convolucional
```

* Por debajo de vit_base el exponente es 0,5: con la profundidad fija, las activaciones
  de un ViT crecen con la anchura `d` y los FLOPs con `d²` (es el escalado por
  hidden_dim × capas que ya usa la tabla para vit_tiny/vit_small; la regla los
  reproduce a ±3 %). Es la opción prudente: una regla proporcional a los FLOPs daría a
  vit_tiny 0,007 GB en vez de 0,024 y predeciría que cabe cuando no.
* Las CNN guardan mapas de activación a resolución completa (y BatchNorm), con mucha más
  memoria por FLOP. Se considera convolucional un modelo con convoluciones además de la
  del *patch embedding*, y se multiplica por el factor que lleva la regla a la entrada de
  resnet50 (2,23). Esa entrada ya sobrestima los picos medidos de resnet50 en
  `logs/*/benchmark` (×1,2–1,5), así que el factor queda del lado seguro.
* Limitación conocida: efficientnet_b0 sale con 0,031 GB frente a los 0,060 de la tabla
  (que también es una estimación, sin medir). Las redes con convoluciones *depthwise*
  y bloques SE pueden necesitar más de lo que da la regla.

**Qué se considera "de la tabla".** La búsqueda en `MODEL_TABLE` es **exacta** (solo se
descarta una etiqueta de pesos del tipo `nombre.tag`). `model_spec()` casa por subcadena,
lo cual vale para nombres de GPU pero habría tratado `wide_resnet50_2` como `resnet50`
(26 M de parámetros en vez de 68,9 M) y lo habría dado por calibrado. Un modelo de la
tabla al que se pide otro `in_chans` o `img_size` también se perfila (su entrada solo vale
para 3 canales a 224²): `vit_base_patch16_224` a 384² sale con 55,5 GFLOPs frente a los
17,6 de la tabla (×3,15). `num_classes` por sí solo no invalida la entrada (solo cambia la
cabeza).

**Heurística "convolucional".** Una red se considera convolucional si tiene más de una
capa de convolución (la del *patch embedding* no cuenta). Eso también clasifica como CNN a
los híbridos (p. ej. LeViT, con un tronco convolucional): su memoria de activaciones sale
multiplicada por 2,23, lo que es una estimación conservadora (del lado de predecir más
VRAM de la necesaria).

**Qué no está calibrado en estos modelos.** Además de las activaciones, el **tiempo y la
energía** descansan en la MFU = 0,17, calibrada con vit_base; en otras arquitecturas se
sabe que se desvía (resnet50 −24 %, vit_large +18 % en la sesión de Kaggle del 24/06/2026). La **calidad** esperada
(`predict_quality`) usa el prior de la familia más cercana (vit_base si ninguna casa), que
no es una ejecución de esa arquitectura: con `calibrated=False` la confianza se **fuerza a
"low"** (banda ±0,050) y se añade una nota. Es la opción menos sorprendente: se sigue
mostrando un orden de magnitud, pero sin aparentar la confianza de un modelo medido.

Estos modelos se marcan `calibrated=False`; `predict()` añade la nota *"Model profiled
automatically …; activation memory estimated, not calibrated … Compute throughput (MFU)
is calibrated on vit_base and not on this architecture"*, que muestran el CLI y la web.
Para los seis modelos de la tabla no cambia nada: se sigue usando su entrada
(comprobado con valores fijados de la versión anterior del predictor).

```bash
uv run paravit.py estimate --model convnext_tiny --gpu "Tesla T4"
uv run paravit.py estimate --model deit_small_patch16_224 --in-chans 1 --num-classes 9
```

En la web, el desplegable *Model* de **Estimate** ofrece *Other timm model…*. Como
cualquier capacidad nueva del predictor, la regla de activaciones necesita medidas reales
de modelos fuera de la tabla para validarse (fase 4 del roadmap).

---

## 8. Arquitecturas distintas de ViT: FLOPs frente a FLOPs + memoria (23/11/2026)

**Pregunta.** La MFU se calibró con vit_base y el predictor supone que el tiempo de cómputo sigue
a los FLOPs. En la criba de modelos de PANDA (512 px, AMP, lote 16, RTX 3060 Ti) eso falla fuera
de los ViT: el tiempo por época predicho se queda corto un 44 % en Swin-T, un 30 % en ConvNeXt-T,
un 26 % en ResNet-50 y un 50 % en EfficientNetV2-S, mientras que en ViT-S, DeiT-III y EVA-02 el
error está entre −8 % y +12 %. (Predicho: `paravit estimate --model M --gpu "RTX 3060 Ti" --batch 16
--precision amp --img-size 512 --dataset-size 7433`; medido: mediana de `epoch_time_s`, sin la
primera época, de `logs/local/panda/single/<modelo>/epoch_metrics_22112026_*.csv`, que incluye
también la validación.)

**Medidas.** Throughput de entrenamiento con lotes sintéticos (`paravit benchmark`, sin I/O) de
13 arquitecturas a 512 px con AMP: 10 en la RTX 3060 Ti local (las otras tres no caben en 8 GB) y
las 13 en una T4 de Kaggle (`kaggle/benchmark_t4`). Datos en
[`research/datos/throughput_arquitecturas_amp.csv`](research/datos/throughput_arquitecturas_amp.csv)
(logs originales en `logs/local/benchmark/benchmark_23112026_10*` y
`logs/kaggle/benchmark/throughput_t4_23112026.csv`).

**Modelo.** Segundos por imagen en una GPU:

```
t = a · F + b · A
```

`F` son los GFLOPs del *forward* por imagen (la unidad de `MODEL_TABLE`) y `A` los GB que autograd
guarda para el *backward* de una imagen, sin contar los parámetros
(`src/model_profile.py: saved_activation_gb`, medido en CPU en menos de un segundo por modelo, con
el núcleo de atención *flash*, que como los de la GPU no guarda la matriz N×N; que en AMP se guarde
más o menos la mitad es un **supuesto**: LayerNorm y softmax siguen en fp32). `a` es la inversa de un ritmo de cómputo efectivo y `b`
la de un ritmo de memoria efectivo. La forma actual del predictor es `t = a · F`. Ajuste y
validación en `src/analysis/throughput_model.py` (puro, con tests) y el estudio completo, en un
comando:

```bash
uv run python scripts/predictor_architecture_study.py --markdown
```

**Calibración (RTX 3060 Ti, 10 modelos).** Error del throughput predicho; *LOO* = cada modelo
predicho con un ajuste que no lo incluye:

| Modelo | GFLOPs | Activ. GB | GFLOPs/GB | img/s | Solo FLOPs (LOO) | 2 términos (LOO) |
|---|---|---|---|---|---|---|
| ViT-S/16 | 31,7 | 0,309 | 103 | 92,8 | −1 % | −2 % |
| DeiT-III-S | 31,7 | 0,347 | 92 | 86,1 | +7 % | +1 % |
| EVA-02-S | 43,3 | 0,522 | 83 | 54,6 | +27 % | +14 % |
| ViT-B/16 | 107,0 | 0,615 | 174 | 33,9 | −42 % | −17 % |
| Swin-T | 25,5 | 0,591 | 43 | 57,1 | +109 % | +34 % |
| ConvNeXt-T | 23,3 | 0,570 | 41 | 78,0 | +64 % | +0 % |
| ResNet-50 | 21,4 | 0,449 | 48 | 91,9 | +51 % | +2 % |
| EfficientNet-B0 | 2,0 | 0,461 | 4 | 152,8 | +855 % | −5 % |
| EfficientNetV2-S | 14,9 | 0,805 | 18 | 79,2 | +153 % | −15 % |
| EfficientNet-B3 | 5,0 | 0,890 | 6 | 81,4 | +623 % | −13 % |

Error medio (LOO): **solo FLOPs 193 %, dos términos 10 %** (máximo 34 %, Swin-T). Un error positivo
es un modelo que el ajuste cree más rápido de lo que es.

**Validación en otra GPU (T4, 13 modelos, tres que no estaban en la calibración).** Sin reajustar:
`a` se escala con los TFLOPS·π de `GPU_TABLE` (×1,34) y `b` con el ancho de banda de las hojas de
NVIDIA (448 → 320 GB/s, ×1,40). "Forma" es el error una vez quitado el factor común a todos los
modelos, es decir, el coste **relativo** de las arquitecturas:

| Modelo | img/s | Solo FLOPs | Solo FLOPs, forma | 2 términos | 2 términos, forma | En la calibración |
|---|---|---|---|---|---|---|
| ViT-S/16 | 50,6 | +36 % | −55 % | +33 % | −11 % | sí |
| DeiT-III-S | 47,9 | +44 % | −52 % | +33 % | −11 % | sí |
| EVA-02-S | 28,1 | +80 % | −40 % | +60 % | +7 % | sí |
| ViT-B/16 | 18,6 | +10 % | −63 % | +28 % | −14 % | sí |
| Swin-T | 28,9 | +197 % | −1 % | +85 % | +24 % | sí |
| SwinV2-T | 22,7 | +306 % | +35 % | +112 % | +42 % | **no** |
| MaxViT-T | 14,2 | +363 % | +54 % | +53 % | +3 % | **no** |
| ConvNeXt-T | 38,8 | +142 % | −19 % | +46 % | −2 % | sí |
| ConvNeXt-S | 22,6 | +113 % | −29 % | +45 % | −3 % | **no** |
| ResNet-50 | 46,5 | +120 % | −27 % | +46 % | −2 % | sí |
| EfficientNet-B0 | 75,9 | +1.336 % | +378 % | +38 % | −7 % | sí |
| EfficientNet-B3 | 38,4 | +1.035 % | +278 % | +39 % | −7 % | sí |
| EfficientNetV2-S | 35,7 | +312 % | +37 % | +41 % | −6 % | sí |

- **Forma:** con dos términos el coste relativo de las arquitecturas se predice con un **11 %** de
  error medio en una GPU que no se usó para ajustar (frente a 82 % con solo FLOPs): 9 % en los
  diez modelos de la calibración y 16 % en los tres que no estaban (ConvNeXt-S −3 %, MaxViT-T
  +3 %, SwinV2-T +42 %).
- **Absoluto:** casi todo el error es **un factor común de 1,49** que la transferencia por hoja
  técnica no recoge. Su causa **no está identificada**: la T4 trabaja limitada a 70 W y con el reloj
  por debajo del nominal (817-967 MHz en los entrenamientos de Kaggle, frente a 1.590 de *boost*),
  pero el reloj solo escala el término de cómputo, y ese término es un 6-7 % del tiempo de las
  EfficientNet en la T4. Un ajuste directo en la T4 (solo como diagnóstico; lo imprime el script)
  da `b` 1,65 veces mayor que el transferido y `a` solo 1,24 veces: la mayor parte del factor está
  en el término de memoria (ancho de banda efectivo de la GDDR6 a 70 W, u otra causa).
- **Lo que no recoge:** la familia Swin (+24 % Swin-T, +42 % SwinV2-T, no calibrado). La atención
  por ventanas (particiones, desplazamientos, la MLP de sesgos de SwinV2) lanza muchas operaciones
  pequeñas que no son ni FLOPs ni bytes guardados. EVA-02 (+7 %) va en la misma dirección, más
  suave.

**Otras resoluciones (misma GPU, 23/11/2026).** Las mismas 10 arquitecturas a 224 y 384 px en la
RTX 3060 Ti (`logs/local/benchmark/benchmark_23112026_16*`), predichas con el ajuste de 512 px sin
reajustar. A 384 px los dos términos bastan (10 % de error medio, frente a 218 % con solo FLOPs). A
224 px los dos términos creen todos los modelos un ~40 % más rápidos de lo que son: con imágenes
pequeñas pesa un **coste fijo por paso** que no depende de la imagen (lanzamiento de *kernels*,
paso del optimizador, Python). Con un tercer término, `t = a·F + b·A + d/B` (`fit_with_overhead`;
`B` = 16 imágenes por paso), ajustado también solo a 512 px (d ≈ 30 ms por paso):

| Resolución | Solo FLOPs | 2 términos | 2 términos + d/B |
|---|---|---|---|
| 384 px | 218 % | 10 % | 10 % (máx. 22 %) |
| 224 px | 324 % | 43 % | **15 %** (máx. 31 %) |

Calibrando con dos resoluciones y prediciendo la tercera, el error medio queda en 8-11 % y `d` en
19-24 ms por paso. Un término proporcional a los parámetros (el paso del optimizador) no se
sostiene: su coeficiente cambia de signo según la calibración. El coste fijo es otro candidato para
integrar en el predictor, que hoy no tiene ningún término por paso.

**Conclusión del estudio.** El tiempo de entrenamiento de arquitecturas distintas no sigue a
los FLOPs: hace falta un segundo término de tráfico de memoria, medible en CPU sin GPU, y con él
el coste relativo se transfiere entre GPUs a partir de las hojas técnicas.

### Integración en el predictor (23/11/2026)

`src/performance_model.py` usa desde el 23/11/2026 el modelo de tres términos para el throughput
de entrenamiento **en precisión mixta AMP** (`amp`), que es donde está medido (`bf16` no se ha
medido y sigue con la MFU × π):

- **Coeficientes por GPU** (`THROUGHPUT_FITS`, `fit_with_overhead`):
  - RTX 3060 Ti: ajuste conjunto con 45 medidas (10 arquitecturas a 224, 384 y 512 px con lote 16,
    y 5 de ellas a 224 px con lotes 8, 32 y 64). Da a = 2,05·10⁻⁴ s/GFLOP, b = 1,25·10⁻² s/GB y
    d = 18 ms por paso.
  - T4: 13 arquitecturas, solo a 512 px y con lote 16 (3,35·10⁻⁴, 3,10·10⁻² y 24 ms).

  Ambos son calibración: cada GPU reproduce sus propias medidas a 512 px con un error medio del
  8,3 % (3060 Ti) y del 11,6 % (T4).
- **El coste fijo por paso (d/B) está medido.** Sin variar el lote, d no se puede separar de los
  términos por imagen: el primer ajuste, solo a 512 px y lote 16, daba 30 ms y con lote 8 se
  quedaba un 30 % corto. El barrido de lotes de la 3060 Ti (`logs/local/benchmark/
  benchmark_23112026_2232*`) da, para cada modelo, un tiempo por paso que crece en línea recta
  con el lote. La ordenada en el origen es de 13-19 ms según el modelo, y el ajuste conjunto
  da 18 ms. Error por lote: 13 % (8), 8 % (16), 9 % (32) y 8 % (64). Dejando fuera cada modelo
  entero, el error es del 11 %. En la T4 no se ha variado el lote: su d (24 ms) es un supuesto.
- **GPU sin medidas** (V100, A100…): los coeficientes de la 3060 Ti se escalan por TFLOPS·π y
  por el ancho de banda (`GpuSpec.mem_bw_gbs`, de las hojas de NVIDIA). La predicción lleva una
  nota: el coste relativo entre arquitecturas se sostiene (11 % en la T4 con los tres términos),
  pero el absoluto puede ser optimista (×1,5 en la T4, mediana de los 13 modelos).
- **Rasgos del modelo**: `ModelSpec.saved_gb_per_img`, medido con `saved_activation_gb` (en
  `MODEL_TABLE`, a 224 px; para los modelos perfilados, a la resolución pedida, o a 512 px y escalado por los píxeles si es mayor, para no retener gigas
  de activaciones al perfilar). Sin ese dato, o en fp32 o bf16, el predictor sigue con la MFU, igual que antes.
- `Prediction` informa de `train_ips_per_gpu` y del origen del throughput (`throughput_model`:
  `three-term`, `mfu` o `measured`); la CLI y la web lo muestran en la línea de supuestos.

**Validación con entrenamientos reales.** Es el tiempo por época (con la validación) de
entrenamientos de PANDA que no intervinieron en ningún ajuste: el ajuste se hizo con lotes
sintéticos sin I/O. Los modelos y las GPU son los mismos que en la calibración, pero la magnitud
es otra: entrenamiento real con lectura de datos, validación, DDP y, en Kaggle, 8 imágenes por
GPU. Se compara con la forma anterior del predictor (la misma ficha sin el dato de activaciones).
La última sección de `scripts/predictor_architecture_study.py` lo reproduce:

| Entrenamientos | Modelos | Solo FLOPs | 3 términos | Peor caso (3 términos) |
|---|---|---|---|---|
| 1 × RTX 3060 Ti, lote 16 | 8 | 20 % | **8 %** | Swin-T −22 % |
| 2 × T4 (Kaggle, DDP), 8 por GPU | 7 | 47 % | **17 %** | SwinV2-T −31 % |

Con 2 × T4, MaxViT-T pasa de −69 % a +1 % y ConvNeXt-S de −42 % a −6 %. Siguen quedándose cortos
la familia Swin, por el mismo motivo que en el estudio (operaciones pequeñas que no son ni FLOPs
ni bytes), y ViT-S y EfficientNet-B3 en Kaggle (−28 % y −20 %): son los dos runs en los que se
sospecha que manda la CPU de Kaggle (sección 4, "Revisión"), algo que el predictor no modela.

**Pendiente.** Validar en una tercera GPU con medidas propias (V100 de VERODE), sin ajustarla, y
entender el factor común de la T4 para no depender de una medida por GPU.
