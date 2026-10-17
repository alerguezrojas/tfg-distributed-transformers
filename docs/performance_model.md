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

The model has **two** fitted constants, each pinned by **one** measurement:

* `MFU = 0.17` — from the single-GPU fp32 throughput of vit_base on a T4.
* `k = 0.86e-9` — from the measured vit_large OOM point (13.78 GB @ batch 32, T4).

Reproducing those two points is therefore **interpolation, not prediction** — they
are *calibration targets*, marked **(in-sample)** below.

The genuinely predictive results are the **speedups and regimes**, and they are
**out-of-sample because they do not depend on the fitted constants**. A speedup is
a ratio `S(n)=T(N,1)/T(N,n)`: in the compute-bound regime `MFU` cancels, so `S→n`
regardless of its value; the FP32→AMP factor is the separately-measured `π` from
`GPU_TABLE` (not `MFU`); the I/O-bound and heterogeneous regimes are set by
`r_io`/β, again independent of the calibration. These are the rows that actually
test the model.

| Quantity | Predicted | Real | Error | Kind |
|----------|-----------|------|-------|------|
| Single fp32, train/epoch | 192 s | 194 s | +1% | calibration (in-sample) |
| vit_large @ batch 32, T4 | 13.8 GB, fits | 13.78 GB, fits | <1% | calibration (in-sample) |
| **DDP 2×T4 speedup** | **1.95×** | **1.96×** | **<1%** | **validation (out-of-sample)** |
| **FP32 → AMP speedup** | **3.80×** | **3.80×** | **~0%** | **validation (out-of-sample)** |
| **Model-parallel speedup** | **1.00×** | **1.02×** | **−2%** | **validation (out-of-sample)** |
| **vit_tiny DDP regime** | I/O-bound, <1.4× | 1.27× | ✓ | **validation (out-of-sample)** |
| **vit_large @ batch 48, T4** | OOM | OOM | ✓ | **validation (out-of-sample)** |
| **Heterogeneous V100+CPU** | < 1× | 0.12× | ✓ (penalizes) | **validation (out-of-sample)** |
| Energy single fp32, train/epoch | 3.4 Wh | 3.2 Wh | +6% | calibration (power in-sample) |
| **Energy single AMP, train/epoch** | **0.90 Wh** | **0.84 Wh** | **+7%** | **validation (time-driven)** |
| **Energy model-parallel/epoch** | **5.7 Wh** | **5.4 Wh** | **+5%** | validation (MP factor in-sample) |

All rows are reproduced by the unit tests in `tests/unit/test_performance_model.py`.

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

Estos modelos se marcan `calibrated=False`; `predict()` añade la nota *"Model profiled
automatically …; activation memory estimated, not calibrated"*, que muestran el CLI y la
web. Para los seis modelos de la tabla no cambia nada: se sigue usando su entrada
(comprobado con 2 880 combinaciones idénticas a la versión anterior).

```bash
uv run paravit.py estimate --model convnext_tiny --gpu "Tesla T4"
uv run paravit.py estimate --model deit_small_patch16_224 --in-chans 1 --num-classes 9
```

En la web, el desplegable *Model* de **Estimate** ofrece *Other timm model…*. Como
cualquier capacidad nueva del predictor, la regla de activaciones necesita medidas reales
de modelos fuera de la tabla para validarse (fase 4 del roadmap).
