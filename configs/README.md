# Configuraciones de entrenamiento

Cada `.yaml` describe **un escenario** (modelo, dataset, hardware, estrategia). Se pasan con
`--config`, p. ej. `uv run paravit.py train --strategy single --config configs/train_v3.yaml`.

**Dataset:** `metadata.parquet` = BigEarthNet-S2 **completo** (237 871 train) · `metadata_demo.parquet`
= **subconjunto** de 5 000/1 500 (estudios rápidos y comparables).

**Rutas y máquina:** las rutas de los YAML son las del autor o las de VERODE. Los perfiles de
`machines/` (`local`, `verode`, `kaggle` y `example.yaml`, plantilla para equipos nuevos) las
sustituyen según la máquina: `--machine NOMBRE`, `PARAVIT_MACHINE` o el nombre del host; sin perfil
las configs se usan tal cual. `--set clave.sub=valor` cambia cualquier clave suelta. Ver la
[guía de uso](../docs/guia_de_uso.md#perfiles-de-máquina-y---set).

---

## ¿Qué config elegir? (guía rápida)

| Si se quiere… | Config |
|---|---|
| Entrenar en **la máquina local** (RTX 3060 Ti), baseline | `train.yaml` |
| Entrenar en **la máquina local** con la regularización buena (v3) | `train_v3.yaml` |
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

## Opciones de memoria (cualquier config)

`training.grad_accum_steps: k` (paso del optimizador cada `k` lotes) y `training.grad_checkpointing: true`
(*activation checkpointing*) están desactivadas por defecto; se activan con `--set` y no tienen
config propia. Cuándo usarlas y cuánto cuestan: [guía de uso](../docs/guia_de_uso.md).

## Otros datasets (`data.name`)

| Config | Dataset | Para qué |
|---|---|---|
| `medmnist_*_local.yaml` | MedMNIST (breast, chest, derma, path) | Baselines de MedMNIST con vit_tiny en local ([ficha](../docs/datasets/medmnist.md)). |
| `train_panda.yaml` | PANDA (ISUP 0-5) | Biopsias de próstata como mosaicos de 1024 px reescalados a 512 ([ficha](../docs/datasets/panda.md)). Requiere preparar los mosaicos con `scripts/prepare_panda.py`. |
| `train_panda_36.yaml` | PANDA | Swin-T con mosaicos de 36 tiles (`mosaics_l1_36x256`, 1536 px) reescalados a 768: el mismo tamaño por tile que la base y más tejido. `prepare_panda.py --n-tiles 36`. |
| `train_panda_ordinal.yaml` | PANDA | Igual que `train_panda.yaml` con la pérdida ordinal `ce_emd` (entropía cruzada + EMD²). |
| `train_panda_ousm.yaml` | PANDA | Igual que `train_panda.yaml` con OUSM (`training.ousm_k: 2`): cada lote ignora sus 2 muestras de mayor pérdida, contra las etiquetas ruidosas. |
| `train_panda_ordinal_color.yaml` | PANDA | Pérdida ordinal y aumentos de color a la vez. |
| `train_panda_color.yaml` | PANDA | Igual que `train_panda.yaml` con `data.color_jitter: 0.2` (aumentos de color solo en train). |
| `train_panda_radboud.yaml` | PANDA | Generalización entre centros: train y val solo Radboud, test solo Karolinska (`data.providers`). |
| `train_panda_karolinska.yaml` | PANDA | Lo contrario: train y val solo Karolinska, test solo Radboud. |

---

> Sugerencia: con `uv run paravit.py menu` el menú lista estos configs para elegir sin recordar nombres.
