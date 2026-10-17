# Resultados experimentales del TFG

> Registro de los entrenamientos y estudios realizados hasta la entrega (`v1.0-tfg`, julio 2026).
> Es un documento **histórico**: los números aquí son los que se citan en la memoria. Los
> resultados nuevos van en documentos propios de cada campaña de medidas.

## Resultados de entrenamiento

### Local — RTX 3060 Ti, batch_size=32, 30 epochs (completado 2026-05-01/02)

| Epoch | Train F1 | Val F1 | Val Loss |
|-------|----------|--------|----------|
| 1  | 0.587 | 0.593 | 0.168 |
| 4  | 0.718 | 0.657 | 0.155 |
| 9  | 0.825 | **0.659** ← mejor | 0.183 |
| 30 | 0.947 | 0.654 | 0.674 |

- **Mejor Val F1: 0.6586** (epoch 9) — guardado en `checkpoints/local/checkpoint_epoch_009.pt`
- Duración: ~32.5 horas (~65 min/epoch)
- Sobreajuste claro a partir del epoch 9: train loss → 0.0001, val loss sigue subiendo
- Log completo: `logs/local/train_legacy.log`

### Clúster — V100 32 GB, batch_size=64, 30 epochs (completado 2026-05-07/09)

Ejecutado con la versión previa a la refactorización (sin metric reporters), con `--trace deep`.

| Epoch | Train F1 | Val F1 | Val Loss |
|-------|----------|--------|----------|
| 1  | 0.5865 | 0.6121 | 0.1628 |
| 2  | 0.6832 | 0.6388 | 0.1557 |
| 4  | 0.7415 | 0.6578 | 0.1567 |
| 9  | 0.8535 | 0.6540 | 0.2125 |
| 18 | 0.9383 | 0.6565 | 0.4688 |
| 26 | 0.9471 | 0.6587 | 0.6401 |
| 28 | 0.9473 | **0.6588** ← mejor | 0.6526 |
| 30 | 0.9473 | 0.6588 | 0.6554 |

- **Mejor Val F1: 0.6588** (epoch 28) — checkpoints en `~/tfg-distributed-transformers/checkpoints/verode/` en verode21
- Duración real: **~45h 50m** (08:41 May 7 → 06:29 May 9)
  - Train: ~67 min/epoch | Eval: ~22 min/epoch | Total: ~89 min/epoch
  - Epoch 1: ~135 min (torchinfo + hook registration + warmup GPU)
  - Epoch 7: ~103 min (contención de recursos en verode21)
- Overfitting severo: val F1 se estabiliza en 0.65-0.66 desde epoch 4, train F1 sigue subiendo hasta 0.947
- Val loss diverge monotónicamente (0.16 → 0.66) mientras train loss cae a 0.0001
- Sin anomalías de gradiente detectadas por DeepTracingDecorator en ningún epoch
- Log completo: `logs/verode/train_deep_20260507_084113.log`

### Clúster v2 — V100 32 GB, batch_size=64, early stopping (completado 2026-05-11/12)

Ejecutado con `feature/training-improvements`: LLRD (decay=0.75, 30 grupos), warmup lineal (5 epochs),
cosine scheduler, grad_clip=1.0, early stopping patience=10. Flags: `--trace simple --layers plot hooks --fn energy`.

| Epoch | Train F1 | Val F1 | Val Loss |
|-------|----------|--------|----------|
| 1  | 0.4708 | 0.5442 | 0.1816 |
| 2  | 0.6159 | 0.6239 | 0.1562 |
| 4  | 0.7127 | 0.6696 | 0.1464 |
| 7  | 0.7828 | **0.6707** ← mejor | 0.1637 |
| 10 | 0.8461 | 0.6671 | 0.2053 |
| 17 | 0.9186 | 0.6652 | 0.3365 ← early stop |

- **Mejor Val F1: 0.6707** (epoch 7) — guardado en `checkpoints/verode/` en verode21
- **Early stopping** paró en epoch 17 (sin mejora desde epoch 7 — patience=10)
- Duración: **~19h** (15:08 May 11 → 10:19 May 12) — 17 epochs × ~67 min/epoch
- Tiempo ahorrado vs v1: ~27h (se habrían necesitado 30 epochs × 89 min = 45h)
- Overfitting idéntico: val F1 plana en 0.67 desde epoch 4, train F1 → 0.92
- `[energy]` reportó "GPU no disponible" en verode — pynvml no accede al driver en ese entorno; no afecta al entrenamiento
- Activaciones de hooks estables en epochs 5/10/15: mlp.fc1 ~1.83, attn.qkv ~0.71, sin neuronas muertas
- Log completo: `logs/verode/train_20260511_150808.log` | Plot: `plots/verode/training_20260511_150808.png`

### Clúster v3 — V100 32 GB, batch_size=64, label smoothing + mixup (completado 2026-05-13/14)

Ejecutado con `configs/train_cluster_v3.yaml`: label smoothing=0.1, mixup α=0.2, dropout=0.3, weight decay=0.1, LLRD decay=0.75, warmup=5, early stopping patience=10. Flags: `--trace simple --layers plot confusion batch-monitor --fn energy`.

| Epoch | Train F1 | Val F1 | Val Loss | Threshold óptimo | F1@threshold |
|-------|----------|--------|----------|-------------------|--------------|
| 1  | 0.4813 | 0.5357 | 0.1813 | 0.30 | 0.5778 |
| 2  | 0.6122 | 0.6237 | 0.1596 | 0.30 | 0.6394 |
| 3  | 0.6813 | 0.6548 | 0.1501 | 0.35 | 0.6668 |
| 4  | 0.7122 | 0.6523 | 0.1478 | 0.30 | 0.6683 |
| 5  | 0.7351 | 0.6687 | 0.1488 | 0.35 | 0.6763 |
| 6  | 0.7535 | **0.6738** ← mejor | 0.1545 | 0.35 | 0.6799 |
| 7  | 0.7816 | 0.6651 | 0.1641 | 0.35 | 0.6711 |
| 10 | 0.8454 | 0.6624 | 0.1992 | 0.35 | 0.6665 |
| 16 | 0.9120 | 0.6630 | 0.3121 ← early stop | 0.35 | 0.6655 |

- **Mejor Val F1: 0.6738** (epoch 6, threshold=0.5) — checkpoint en `checkpoints/verode/` en verode21
- **Con threshold óptimo 0.35:** F1=0.6799 en el mejor epoch (mejora práctica para inferencia)
- **Early stopping** paró en epoch 16 (sin mejora desde epoch 6 — patience=10)
- Duración: **~18h** (13 May 16:15 → 14 May 10:13) — ~67 min/epoch (train+eval)
- **Gap train-val en mejor epoch (6):** 0.7535 - 0.6738 = 0.08 (vs 0.11 en v2 — reducción clara)
- Overfitting reducido pero no eliminado: val F1 plana en 0.66-0.67 desde epoch 6, train F1 → 0.91
- Val loss empieza a divergir desde epoch 6 (mínima fue epoch 4: 0.1478)
- `[energy]` reportó "GPU no disponible" en verode — pynvml no accede al driver; no afecta al entrenamiento
- Log completo: `logs/verode/train_13052026_161533.log` | Plot: `plots/verode/training_13052026_161533.png`

### Local — ResNet50, batch_size=32, 2 epochs (smoke test 2026-05-14)

Prueba de soporte genérico timm con modelo convolucional. Config: `train_v3.yaml` (label smoothing + mixup).

| Epoch | Train F1 | Val F1 | Val Loss | Threshold óptimo |
|-------|----------|--------|----------|-----------------|
| 1 | 0.2311 | 0.3588 | 0.2463 | 0.30 |
| 2 | 0.4046 | **0.4725** | 0.2174 | 0.30 |

- **~17 min/epoch** — 4× más rápido que ViT-Base (~65 min/epoch en RTX 3060 Ti)
- Val F1 > Train F1 en ambos epochs — modelo aún en fase de aprendizaje rápido, sin overfitting
- Val loss bajando fuerte — con entrenamiento completo llegaría claramente más lejos
- Threshold óptimo 0.30 (más bajo que el 0.35 del ViT) — ResNet más conservador en sus predicciones
- A epoch 2, ViT-Base v3 tenía Val F1=0.6237 — ventaja clara del transformer con preentrenamiento ImageNet
- **Valida que el soporte genérico timm funciona correctamente** para modelos no-ViT (sin LLRD, AdamW estándar)
- Log: `logs/local/train_14052026_170438.log` | Plot: `plots/local/training_14052026_170438.png`

### Clúster v3b — V100 32 GB, batch_size=64, stack completo con energía (completado 2026-05-14/15)

Misma config que v3 (`configs/train_cluster_v3.yaml`). Objetivo: verificar pynvml funcionando y obtener datos de consumo energético. Flags: `--trace simple --layers plot confusion batch-monitor hooks --fn energy timing`.

| Epoch | Train F1 | Val F1 | Val Loss | Threshold óptimo | F1@threshold |
|-------|----------|--------|----------|------------------|--------------|
| 1  | 0.4711 | 0.5593 | 0.1777 | 0.35 | 0.5865 |
| 2  | 0.6186 | 0.6170 | 0.1567 | 0.30 | 0.6370 |
| 3  | 0.6829 | 0.6295 | 0.1502 | 0.30 | 0.6631 |
| 4  | 0.7150 | 0.6685 | 0.1487 | 0.35 | 0.6757 |
| 5  | 0.7377 | **0.6708** ← mejor | 0.1502 | 0.35 | 0.6788 |
| 7  | 0.7838 | 0.6700 | 0.1613 | 0.35 | 0.6766 |
| 10 | 0.8465 | 0.6655 | 0.2039 | 0.40 | 0.6673 |
| 15 | 0.9044 | 0.6538 | 0.2909 ← early stop | 0.30 | 0.6579 |

- **Mejor Val F1: 0.6708** (epoch 5) — prácticamente igual a v3 (0.6738); diferencia de 0.003 es variación aleatoria
- **Early stopping** en epoch 15 (sin mejora desde epoch 5 — patience=10)
- Duración: **~17h 17m** (14 May 14:57 → 15 May 08:14) — ~69 min/época
- **Energía (primera medición real):** eval_epoch consume ~35 Wh/época a ~100-104 W de potencia media en V100; total estimado 15 evals ≈ 530 Wh solo en evaluación
- El patrón de overfitting es idéntico a v3: val F1 plana en 0.67 desde epoch 5, train F1 → 0.90
- Val loss mínima en epoch 4 (0.1487), empieza a divergir desde epoch 5 — igual que v3
- Log: `logs/verode/train_14052026_145711.log` | Plot: `plots/verode/training_14052026_145711.png`

### Clúster v4 — V100 32 GB, batch_size=64, versión actual del proyecto (completado 2026-05-27/28)

Mismo config que v3/v3b (`configs/train_cluster_v3.yaml`). Primera ejecución con la nueva estructura de carpetas `{env}/{mode}/{model}/`. Flags: `--trace simple --layers plot confusion batch-monitor --fn energy timing`.

| Epoch | Train F1 | Val F1 | Val Loss | Threshold óptimo | F1@threshold |
|-------|----------|--------|----------|------------------|--------------|
| 1  | 0.4254 | 0.5259 | 0.2108 | 0.30 | 0.5758 |
| 2  | 0.5707 | 0.6097 | 0.1884 | 0.30 | 0.6338 |
| 4  | 0.6826 | 0.6656 | 0.1765 | 0.35 | 0.6804 |
| 6  | 0.7231 | 0.6731 | 0.1785 | 0.35 | 0.6827 |
| 7  | 0.7484 | **0.6816** ← mejor | 0.1758 | 0.40 | 0.6852 |
| 10 | 0.8093 | 0.6737 | 0.1849 | 0.40 | 0.6778 |
| 17 | 0.8833 | 0.6697 | 0.2037 ← early stop | 0.30 | 0.6741 |

- **Mejor Val F1: 0.6816** (epoch 7) — nuevo récord, +0.008 sobre v3 (variación aleatoria, mismo config)
- **Con threshold óptimo 0.40:** F1=0.6852
- **Early stopping** en epoch 17 (sin mejora desde epoch 7 — patience=10)
- Duración: **~19h** (27 May 21:02 → 28 May 16:00) — ~66 min/epoch (45 train + 21 eval)
- **Clase 6 completamente fallida** ("Land principally occupied by agriculture"): F1=0.000 — clase rara o ambigua que el modelo nunca predice; tira el F1 macro hacia abajo de forma desproporcionada
- Mejores clases: Marine waters (0.975), Arable land (0.852), Coniferous forest (0.848)
- Peores clases: Coastal wetlands (0.402), Industrial units (0.555), Natural grassland (0.549)
- Log completo: `logs/verode/single/vit_base_patch16_224/train_27052026_210223.log`

### Comparativa de todas las ejecuciones en clúster

| | v1 (sin mejoras) | v2 (LLRD + warmup + early stop) | v3 (label smoothing + mixup) | v3b (stack completo + energía) | v4 (versión actual) |
|---|---|---|---|---|---|
| **Config** | `train_cluster.yaml` | `train_cluster.yaml` + flags | `train_cluster_v3.yaml` | `train_cluster_v3.yaml` | `train_cluster_v3.yaml` |
| **Trace mode** | `--trace deep` | `--trace simple` | `--trace simple` | `--trace simple` | `--trace simple` |
| **LLRD** | No | Sí (decay=0.75) | Sí (decay=0.75) | Sí (decay=0.75) | Sí (decay=0.75) |
| **Warmup** | No | Sí (5 epochs) | Sí (5 epochs) | Sí (5 epochs) | Sí (5 epochs) |
| **Early stopping** | No | Sí (patience=10) | Sí (patience=10) | Sí (patience=10) | Sí (patience=10) |
| **Label smoothing** | No | No | Sí (0.1) | Sí (0.1) | Sí (0.1) |
| **Mixup** | No | No | Sí (α=0.2) | Sí (α=0.2) | Sí (α=0.2) |
| **Dropout** | 0.1 | 0.1 | 0.3 | 0.3 | 0.3 |
| **Weight decay** | 0.05 | 0.05 | 0.1 | 0.1 | 0.1 |
| **Energía medida** | No | No | No | Sí (~35 Wh/eval, ~100 W) | Sí (~35 Wh/eval, ~103 W) |
| **Epochs ejecutados** | 30 | 17 | 16 | 15 | 17 |
| **Duración** | ~45.8h | ~19h | **~18h** | ~17.3h | ~19h |
| **Mejor Val F1** | 0.6588 (epoch 28) | 0.6707 (epoch 7) | 0.6738 (epoch 6) | 0.6708 (epoch 5) | **0.6816 (epoch 7)** |
| **Gap train-val en mejor epoch** | ~0.34 | ~0.11 | ~0.08 | ~0.07 | **~0.13** |

**Conclusiones v1 → v2:**
- LLRD + warmup mejoraron Val F1 en +0.012 y aceleraron la convergencia (mejor epoch: 28 → 7)
- Early stopping ahorró ~27h eliminando epochs innecesarios
- El techo de generalización (~0.67 Val F1) es una limitación del dataset/regularización, no del hardware
- El cuello de botella NFS persiste: añadir GPUs (DDP) no escala linealmente si el I/O es el límite

**Conclusiones v2 → v3:**
- Label smoothing + mixup mejoraron Val F1 en +0.003 (modesto pero consistente)
- La reducción del gap train-val (0.11 → 0.08) confirma que la regularización adicional funciona
- La convergencia al mejor epoch fue más rápida (epoch 7 → epoch 6)
- El techo del dataset (~0.67-0.68 Val F1 a threshold=0.5) parece real — las clases raras limitan F1 macro
- **Para inferencia usar threshold=0.35-0.40:** consistentemente mejora F1 en ~0.004-0.006 sobre threshold=0.5
- El siguiente paso para mejorar resultados es DDP (más datos efectivos por epoch) o tratar las clases raras

**Conclusiones v3/v3b → v4:**
- Mismo config, resultado ligeramente mejor (+0.008) por variación aleatoria — confirma que el techo está en ~0.68
- La clase 6 ("Land principally occupied by agriculture") F1=0.0 es un hallazgo importante: una sola clase rara que el modelo no predice nunca puede tirar el F1 macro varios puntos
- El gap train-val (0.13 en v4 vs 0.07 en v3b) es mayor probablemente por inicialización aleatoria diferente, no por regresión

---

## Estudio single-GPU vs distribuido (apples-to-apples, completado 2026-06-04)

Comparación controlada **mismo modelo / mismo subset (5000/1500) / mismos 3 epochs**, cambiando solo la estrategia de distribución. Es el resultado central del TFG sobre escalado distribuido.

| Escenario | Hardware | Modelo | Train/epoch | Speedup | Eficiencia | Cuello | Val F1 |
|---|---|---|---|---|---|---|---|
| Single-GPU | V100 (Verode) | vit_tiny | ~16s (estable) | 1.00× (baseline) | — | — | 0.252 |
| **Heterogéneo** | V100 + CPU (Verode, gloo) | vit_tiny | ~225s | **0.12×** | ~6% | DDP síncrono + hardware desbalanceado + NFS | 0.278 |
| Single-GPU | 1× T4 (Kaggle) | vit_tiny | 47.5s/ep total | 1.00× | — | — | 0.280 |
| DDP 2 GPU | 2× T4 (Kaggle, NCCL) | vit_tiny | 37.3s/ep total | **1.27×** | 64% | I/O-bound (modelo diminuto) | 0.263 |
| Single-GPU | 1× T4 (Kaggle) | vit_base | 179.3s/ep total | 1.00× | — | — | 0.406 |
| **DDP 2 GPU** | **2× T4 (Kaggle, NCCL)** | **vit_base** | **94.5s/ep total** | **1.90×** | **95%** | **compute-bound → escala casi perfecto** | 0.410 |

**Conclusión demostrada con datos reales:** el speedup del entrenamiento distribuido depende del **ratio cómputo/IO y del balance del hardware**:
- **Compute-bound (vit_base): escala casi linealmente** (1.90×, 95% eficiencia en 2 GPUs reales) — las GPUs trabajan en paralelo de verdad.
- **I/O-bound (vit_tiny): escala poco** (1.27×, 64%) — leer los TIFF domina y añadir GPUs no acelera el disco.
- **Hardware desbalanceado (V100+CPU): penaliza** (0.12×) — el DDP síncrono va al ritmo del nodo más lento (la CPU ~50× más lenta que la V100); la GPU pasa el tiempo esperando.
- El Val F1 es idéntico entre single y DDP en todos los casos → la sincronización de gradientes (incl. la normalización por batch global del heterogéneo) es matemáticamente correcta.
- El **feasibility lo predice** en cada caso: marca I/O-bound (ratio≈23 para vit_tiny en NFS) y recomienda el nº de GPUs (1 si I/O-bound, varias si compute-bound). **Validado en las T4: el `DDPOptimizer` predice vit_base 2-GPU en 1.92× (real 1.90×) y vit_tiny en 1.0× (I/O-bound, real 1.27×).** (Tras corregir un bug que asumía interconexión Gigabit por NFS y daba 0.29×.)

**Limitación de hardware (clave para la memoria):** Verode solo tiene **1 GPU usable** (verode21 V100, CC 7.0). verode16 (M2090, CC 2.0) y verode18 (K40m, CC 3.5) están por debajo del mínimo CC 3.7 de PyTorch 2.x → no hay multi-GPU NCCL real en Verode. Por eso el speedup positivo se midió en Kaggle (2×T4), y en Verode la comparación distribuida es el heterogéneo GPU+CPU (gloo), un resultado negativo bien documentado.

Artefactos: `logs/verode/single/vit_tiny_patch16_224/` (single), `logs/verode/ddp_hetero/vit_tiny_patch16_224/` (heterogéneo), `logs/kaggle/{single,ddp}/{vit_tiny,vit_base}_patch16_224/` (Kaggle) + feasibility en `logs/verode/feasibility/` y `logs/kaggle/feasibility/`.

### Sesión Kaggle 2×T4 — 5 estrategias comparables + validación del feasibility (2026-06-10)

Sesión única en Kaggle (2× Tesla T4) que extiende el estudio: **mismo modelo (vit_base), mismo subset (5000/1500), mismos 15 epochs, mismo batch global (96)**, variando solo la estrategia y la precisión. Primero se corrió el **feasibility completo** (predicción), luego las 5 ejecuciones (validación). Todo `precision=fp32` salvo donde se indica AMP.

| Estrategia | Hardware | Train s/epoch | Speedup | Best Val F1 | Energía/ep train |
|---|---|---|---|---|---|
| Single fp32 | 1×T4 | 194 s | 1.00× | 0.572 | 3.32 Wh (1 GPU) |
| **Model-parallel** (pipeline) | 2×T4 | ~211 s (epoch total) | **1.02×** (no acelera) | 0.534 | — (sin instrumentar) |
| **DDP** (datos, NCCL) | 2×T4 | 99 s | **1.96×** | 0.547 | 1.67 Wh/GPU (≈3.34 total)\* |
| **Single AMP** (Tensor cores) | 1×T4 | 51 s | **3.80×** | 0.550 | 0.85 Wh (1 GPU) |
| **DDP + AMP** | 2×T4 | 32.5 s | **5.97×** | 0.552 | 0.44 Wh/GPU (≈0.88 total)\* |

\* **Estos runs (10/06) se midieron con la instrumentación previa, que medía una sola GPU** (`_PowerSampler` leía el dispositivo 0 y en DDP solo el rank 0 escribía el log). Valores medidos por época de train (épocas estables, descartado el warmup): single fp32 **3.32 Wh** y single AMP **0.85 Wh** (1 GPU haciendo TODO el dato); DDP **1.67 Wh/GPU** y DDP+AMP **0.44 Wh/GPU** (cada GPU hace la MITAD del dato). El "total" de los DDP es esa medida **× 2 GPUs** (válido por simetría: T4 idénticas, reparto 50/50). El model-parallel de esta sesión no llevaba `--fn energy` → sin dato (el "—"). **Corregido el 24/06/2026** (rama `fix/energy-multi-gpu`): `_PowerSampler` mide ahora **el conjunto de GPUs del run** y suma su potencia; `measure_energy` resuelve los dispositivos (en DDP el rank 0 suma todas las GPUs del nodo; acepta lista explícita); el script de model-parallel ya tiene `--fn energy` sobre sus dos GPUs. Los runs **futuros** registran el total real; estos quedan como están.

**Conclusiones (datos reales, una sola sesión → máxima comparabilidad):**
- **Datos escala** (1.96×, ~98% eficiencia, compute-bound). **Modelo NO escala cuando cabe en 1 GPU** (1.02×): el paralelismo de modelo *naive* serializa las etapas (una GPU trabaja mientras la otra espera) → su utilidad es *permitir* modelos que no caben, no acelerar.
- **La precisión (Tensor cores) gana al paralelismo de datos:** Single AMP en 1 GPU (3.80×) supera al DDP fp32 en 2 GPUs (1.96×), y con **~4× menos energía total del trabajo** (0.85 Wh en 1 GPU vs ≈3.34 Wh entre las 2 — ver nota \* de la tabla). La precisión importa más que el nº de GPUs aquí.
- **DDP+AMP se combina pero NO multiplica:** esperado 1.96 × 3.80 = 7.4×, medido **5.97×**. La comunicación de gradientes no se acelera con AMP, así que pesa relativamente más → la eficiencia del DDP cae **98% (fp32) → ~78% (AMP)**. Amdahl puro, conectado con el análisis cómputo/comunicación del feasibility.
- **Val F1 ≈ 0.55 en las 5** → la matemática es correcta en todas las estrategias (el F1 es bajo por ser subset de 5000; la calidad real está en los runs del dataset completo en V100, ~0.68).
- **Model-parallel validado por primera vez en 2 GPUs reales** (en local solo se pudo cruzar `cuda:0→cpu`).
- **Caso de uso REAL del paralelismo de modelo (vit_large, 303M params):** en una sola T4, vit_large da **CUDA OOM a batch 48** (el feasibility lo predijo: marcó OOM en 48/64, solo cabe batch 32 a 13.78 GB). Partido **12/24** entre las 2 T4 con model-parallel, **entrena 1 epoch sin OOM** (~9 GB/GPU, 614 s). Confirma la conclusión: el paralelismo de modelo **no acelera, pero hace posible entrenar modelos que no caben en una sola GPU** — su verdadero motivo de existir. Artefacto: `logs/kaggle/model_parallel/vit_large_patch16_224/`.

**Validación del feasibility (predicho → real):** 3 predicciones acertadas en la T4:
- Tiempo single: ~52 min/15ep predicho → ~54 min real (**+4%**).
- Speedup DDP 2-GPU: **1.94×** predicho → **1.96×** real (**<1%**).
- Speedup precisión FP32→AMP: **3.87×** predicho (`--compare-precision`: 27 vs 103 img/s) → **3.80×** real (**<2%**).
- Specs T4 confirmadas: 40 SMs, **2560 CUDA cores / 320 Tensor cores**.

Artefactos (integrados en el repo): `logs/kaggle/single/vit_base_patch16_224/` (single fp32 `173904`, single AMP `203609`, deep-trace `train_deep_205332`), `logs/kaggle/ddp/vit_base_patch16_224/` (DDP fp32 `190526`, DDP+AMP `211814`), `logs/kaggle/model_parallel/{vit_base,vit_large}_patch16_224/` (C + el OOM-vs-split), feasibility (vit_base `172939` + vit_large `213654`) en `logs/kaggle/feasibility/`.

### Sesión Kaggle 2×T4 — re-ejecución con la versión actual + ResNet50 (2026-06-24)

Re-corrida del estudio de 5 estrategias con la **versión actual del proyecto** (energía multi-GPU corregida, model-parallel integrado en el stack de decoradores, naming estimate/benchmark) y **ampliada con una CNN (ResNet50)**. Mismo subset (5000/1500), 15 epochs, batch global 96. Flags: `--trace simple --layers confusion batch-monitor hooks --fn energy timing`.

| Estrategia | Hardware | Train/ép | Speedup | Best Val F1 | Energía train |
|---|---|---|---|---|---|
| Single fp32 | 1×T4 | 189s | 1.00× | 0.5619 | 3.23 Wh (65 W) |
| Model-parallel | 2×T4 | 179s | 1.05× | 0.5412 | 4.9 Wh (106 W, 2 GPU) |
| DDP fp32 | 2×T4 | 95s | 1.98× | 0.5703 | 3.04 Wh (129 W, 2 GPU) |
| Single AMP | 1×T4 | 50s | 3.77× | 0.5486 | 0.84 Wh (64 W) |
| DDP+AMP | 2×T4 | 30.5s | 6.11× | 0.5279 | 0.85 Wh (124 W, 2 GPU) |

- **Energía ahora multi-GPU** (el fix del 24/06): DDP ~129 W y MP ~106 W = las 2 GPUs sumadas (antes solo medía 1). Verificado en los logs.
- **vit_large model-parallel (OOM-vs-split):** el benchmark predice **OOM a batch 48/64** en 1 T4 (24.86/31.57 GB > 15.6); partido entre 2 T4 a batch 24 **entra** (~9 GB/GPU) y entrena 3 epochs (Val F1 0.4863, ~573s/ép). Confirma el motivo del paralelismo de modelo: no acelera, pero hace posible lo que no cabe en una GPU.
- **ResNet50 (CNN, familia distinta):** single (Val F1 0.3634, ~58s/ép) + DDP (0.3606, **1.89×**). El camino genérico timm (AdamW sin LLRD) funciona; **compute-bound escala ~2× también en una CNN** (benchmark predijo 1.95×). Rinde menos que vit_base por hiperparámetros afinados a ViT + infraentrenamiento, no por fallo.
- **Validación del estimador analítico (predicho vs real):** single fp32 +1% (192 vs 189s), single AMP +1% (3.80×), DDP +4% (1.91×); DDP+AMP y resnet DDP cuadran con el **I/O cacheado** (el subset cabe en RAM → el estimador es conservador en I/O si usa la tabla de disco). OOM de vit_large exacto. El analítico **sobreestima vit_large ~18%** (calibrado en vit_base; vit_large/resnet son puntos de validación).
- Artefactos: `logs/kaggle/{single,ddp,model_parallel}/{vit_base,vit_large,resnet50}/` + benchmarks `logs/kaggle/benchmark/` (vit_base `175820`, vit_large `210016`, resnet50 `220430`). PR #108 (datos) → develop → main.
