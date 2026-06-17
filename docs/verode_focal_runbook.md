# Runbook — Focal loss vs BCE en Verode (V100, dataset completo)

Compara **focal loss** contra **BCE** en vit_base sobre BigEarthNet-S2 completo,
para ver si focal rescata las clases raras (p.ej. clase 6 "Land principally
occupied by agriculture", F1=0 en v4) y sube el F1 macro por encima del techo ~0.68.

**Pareja apples-to-apples** (idénticas salvo la pérdida, ambas fp32, 30 epochs con
early stopping patience=10):
- BCE  : `configs/train_cluster_v3.yaml`     (tu baseline documentado, ~0.68)
- Focal: `configs/train_cluster_focal.yaml`  (igual + `loss: focal`, `focal_gamma: 2.0`)

Código necesario en la rama **`feature/critical-fixes`** (focal loss + train-F1
corregido). No hace falta `uv sync` (sin dependencias nuevas).

---

## 0. Conectar y preparar el código

```bash
ssh alu0101317038@verode00.pcg.ull.es
module add slurm/client/20.11.04

cd ~/tfg-distributed-transformers
git fetch origin
git stash          # por si hubiera cambios locales en el checkout del clúster
git checkout feature/critical-fixes
git pull origin feature/critical-fixes
```

## 1. Sesión tmux + nodo de cómputo (sobrevive a desconexiones)

```bash
tmux new-session -s focal
# dentro de tmux, pedir la V100:
/opt/soft/slurm/20.11.04/bin/srun --partition=batch --nodelist=verode21 \
  --gres=gpu:tesla:1 --time=72:00:00 --pty bash

cd ~/tfg-distributed-transformers
.venv/bin/python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.version.cuda)"
# Si CUDA=False: reinstalar cu118 (ver Errores conocidos en CLAUDE.md)
```

## 2. Lanzar los dos entrenamientos (uno detrás de otro)

```bash
# (A) BASELINE BCE — ~18-20h con early stopping
.venv/bin/python scripts/train_single_gpu.py \
  --config configs/train_cluster_v3.yaml \
  --trace simple --layers confusion batch-monitor --fn energy timing

# (B) FOCAL — ~18-20h con early stopping
.venv/bin/python scripts/train_single_gpu.py \
  --config configs/train_cluster_focal.yaml \
  --trace simple --layers confusion batch-monitor --fn energy timing
```

Para soltar la sesión sin matarla: `Ctrl+B`, luego `D`. Reconectar: `tmux attach -t focal`.

> **Atajo (opcional, ~la mitad de tiempo):** añadir `precision: amp` a AMBOS configs
> usa los Tensor cores del V100 (~2× más rápido). La comparación sigue siendo limpia
> si las dos usan AMP. Si solo te interesa focal, puedes **saltarte (A)** y comparar
> contra el BCE ya documentado (v4, Val F1=0.6816) — el entrenamiento no cambia el
> resultado de BCE, así que ese número sigue valiendo como referencia.

## 3. Seguir el progreso

```bash
# en otra ventana tmux o tras reconectar:
tail -f logs/verode/single/vit_base_patch16_224/train_*.log
```

## 4. Cómo leer el resultado (importante con focal)

Focal **empuja las probabilidades hacia abajo**, así que el F1 a **umbral 0.5**
sale artificialmente bajo. La comparación justa es el **F1 al umbral óptimo**, que
el log reporta en cada epoch como `threshold óptimo=… F1=…`. Mira:
- el `threshold óptimo` y su `F1` (no solo el F1 a 0.5),
- el CSV por clase `perclass_metrics_*.csv` → **¿sube la F1 de las clases raras
  (clase 6, etc.) respecto al BCE?** Ese es el resultado clave.

## 5. Subir los resultados (al terminar)

```bash
git add logs/verode/ plots/verode/
git commit -m "feat: focal vs BCE full-dataset runs on Verode V100"
git push origin feature/critical-fixes
```

Luego, en local, `git pull` y abrir el dashboard:
`uv run streamlit run src/web/app.py` → **Compare** (los dos runs) y
**Run results → Per-class** (F1 por clase, clases raras).
