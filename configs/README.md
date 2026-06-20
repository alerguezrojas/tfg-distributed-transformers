# Configuraciones de entrenamiento

Cada `.yaml` describe **un escenario** (modelo, dataset, hardware, estrategia). Se pasan con
`--config`, p. ej. `uv run tfg.py train --strategy single --config configs/train_v3.yaml`.

**Dataset:** `metadata.parquet` = BigEarthNet-S2 **completo** (237 871 train) · `metadata_demo.parquet`
= **subconjunto** de 5 000/1 500 (estudios rápidos y comparables).

---

## ¿Cuál uso? (guía rápida)

| Quiero… | Config |
|---|---|
| Entrenar en **mi máquina** (RTX 3060 Ti), baseline | `train.yaml` |
| Entrenar en **mi máquina** con la regularización buena (v3) | `train_v3.yaml` |
| El **mejor resultado** en Verode (V100, dataset completo) | `train_cluster_v3.yaml` |
| Probar **focal vs BCE** rápido en local | `train_local_demo_focal.yaml` / `train_local_demo_bce.yaml` |
| Probar **focal vs BCE** a escala en Verode | `train_cluster_focal.yaml` / `train_cluster_bce.yaml` |
| El estudio **single vs distribuido** (subset, comparable) | `train_demo_single.yaml` · `train_demo_ddp.yaml` · `train_model_parallel_kaggle.yaml` |
| El **heterogéneo GPU+CPU** (demo que termina en ~13 min) | `train_heterogeneous_ddp_demo.yaml` |

---

## Local — RTX 3060 Ti (desarrollo)

| Config | Modelo | Dataset | Para qué |
|---|---|---|---|
| `train.yaml` | vit_base (bs 32) | completo | **Baseline local.** El punto de partida por defecto. |
| `train_v3.yaml` | vit_base (bs 32) | completo | Local con **regularización v3** (label smoothing + mixup + dropout 0.3). |
| `train_local_demo_bce.yaml` | vit_tiny (bs 96) | subset | Demo local rápida (4 ep), pérdida **BCE** — brazo de control del experimento focal. |
| `train_local_demo_focal.yaml` | vit_tiny (bs 96) | subset | Igual pero con **focal loss** — para ver si rescata clases raras en local. |

## Verode — Tesla V100 (entrenamiento serio, dataset completo)

| Config | Modelo | Para qué |
|---|---|---|
| `train_cluster.yaml` | vit_base (bs 64) | Baseline en Verode (sin v3). |
| `train_cluster_v3.yaml` | vit_base (bs 64) | **v3 completo** (label smoothing + mixup + dropout 0.3 + weight decay 0.1). El de los mejores resultados (~0.68 Val F1). |
| `train_cluster_bce.yaml` | vit_base (bs 64) | Brazo **BCE** del experimento focal a escala (apples-to-apples, selección por umbral óptimo). |
| `train_cluster_focal.yaml` | vit_base (bs 64) | Brazo **focal** del mismo experimento. |

## Entrenamiento distribuido

| Config | Modelo | Backend | Para qué |
|---|---|---|---|
| `train_ddp_verode.yaml` | vit_base | NCCL | DDP multi-GPU en Verode. ⚠ Verode solo tiene 1 GPU usable; el speedup real se midió en Kaggle. |
| `train_ddp_cpu_test.yaml` | vit_tiny | gloo | Test de **infraestructura multi-nodo por CPU** (valida la sincronización sin GPU compatible). |
| `train_heterogeneous_ddp.yaml` | vit_base | gloo | Heterogéneo GPU+CPU sobre el **dataset completo** (referencia; inviable en tiempo, ~16 días). |
| `train_heterogeneous_ddp_demo.yaml` | vit_tiny | gloo | **Heterogéneo demo** (subset, 3 ep, ~13 min) — el que se usa para obtener métricas reales. |

## Estudio *single vs distribuido* (mismo modelo / subset / épocas, solo cambia la estrategia)

| Config | Hardware | Para qué |
|---|---|---|
| `train_demo_single.yaml` | 1 GPU | Baseline single-GPU del estudio comparativo. |
| `train_demo_ddp.yaml` | 2 GPU (NCCL) | Paralelismo de **datos** — speedup positivo real (Kaggle 2×T4). |
| `train_model_parallel_kaggle.yaml` | 2 GPU | Paralelismo de **modelo** — no acelera, pero permite modelos que no caben en 1 GPU (Kaggle 2×T4). |

---

> Sugerencia: usa `uv run tfg.py menu` → el menú lista estos configs para elegir sin recordar nombres.
