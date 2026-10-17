# Infraestructura: máquinas, entornos y lanzamiento

> Hardware disponible, configuración del clúster VERODE, cómo lanzar en cada sitio y errores
> conocidos con su solución. Para el uso general del CLI, ver [`guia_de_uso.md`](guia_de_uso.md).

## Hardware

### Local (desarrollo)
- **GPU:** NVIDIA RTX 3060 Ti (8 GB VRAM)
- **Driver:** nvidia-driver-580-open, kernel 6.8
- **Dataset:** SSD externo montado en `/media/alejandro/SSD/` (ext4, ~120 GB)
- **Gestión de paquetes:** `uv`

### Clúster VERODE (ULL) — entrenamiento
- **Login:** `ssh alu0101317038@verode00.pcg.ull.es`
- **Nodos de cómputo:** verode[16-21] — hardware heterogéneo, solo verode21 es compatible con PyTorch 2.x
  - **verode16:** Tesla M2090 (2011, CC 2.0, 6 GB) — driver no activo + CC < 3.7 → incompatible
  - **verode18:** Tesla K40m (2013, CC 3.5, 11 GB) — driver 460/CUDA 11.2 + CC < 3.7 → incompatible
  - **verode21:** Tesla V100-PCIE (2017, CC 7.0, 32 GB) — operativo ✓
- **CUDA:** 12.0, Driver 525.147.05 (verode21)
- **CPUs por nodo:** 16
- **RAM por nodo:** ~112 GB
- **Sistema de colas:** Slurm 20.11.04
- **Almacenamiento:** `/home/bejeque/alu0101317038/` (NFS)

#### Configuración del clúster (hacer en cada sesión SSH)
```bash
module add slurm/client/20.11.04   # o añadir al ~/.bashrc
```

#### Entorno en el clúster
- **Miniconda:** instalado en `~/miniconda3`, `auto_activate_base=false` — solo para zstd
- **zstd:** instalado via conda (`~/miniconda3/bin/zstd`) — necesario para extraer el dataset
- **uv:** instalado en `~/.local/bin/uv`
- **Entorno Python:** `~/tfg-distributed-transformers/.venv` creado con `uv sync`
- **PyTorch:** `2.7.1+cu118` — instalado con cu118 para compatibilidad con driver 525 (CUDA 12.0 máx.)
  - ⚠️ `uv sync` instala cu13 por defecto → incompatible. Después de sync, ejecutar:
  ```bash
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
  ```
  - ⚠️ `python -m pip` no está disponible en el venv del clúster → usar siempre `uv pip` en su lugar

#### Problemas conocidos del clúster
- `sbatch` falla con "I/O error writing script/environment to file" — bug de configuración de Slurm, no reparable por el usuario
- Alternativa: usar `tmux` + `srun` para jobs que sobrevivan a desconexiones SSH
- El login node (`verode00`) no soporta instrucciones AVX2 → "Illegal instruction" al ejecutar Python con numpy/torch. Usar siempre nodo de cómputo (`srun`) para ejecutar código
- `srun` es el único mecanismo que funciona; ver sección de comandos

## Gestión de dependencias

### Local (uv)
```bash
uv sync                   # instalar entorno
uv run python ...         # ejecutar
uv add <paquete>          # añadir dependencia
```

### Clúster
El entorno `.venv` ya está creado. Si se reinstala desde cero:
```bash
cd ~/tfg-distributed-transformers
uv sync
# Después, reinstalar PyTorch con cu118 (cu13 por defecto no es compatible con driver 525):
# OJO: usar uv pip, NO python -m pip (el módulo pip no está disponible en el venv del clúster)
uv pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
```

Dependencias principales: `torch`, `timm`, `torchvision`, `torchinfo`, `tqdm`, `rasterio`, `pandas`, `pyarrow`, `pyyaml`, `matplotlib`, `nvidia-ml-py`, `streamlit`, `streamlit-option-menu`, `plotly`

## Lanzamiento en VERODE


```bash
# Smoke test local (1 GPU, usa DDPTrainer real — distributed=True siempre activo en train_ddp.py):
torchrun --nproc_per_node=1 scripts/train_ddp.py \
  --model vit_tiny_patch16_224 --epochs 1 \
  --config configs/train.yaml --trace simple
# → Val F1=0.4353, completado sin errores (verificado 20/05/26)
```

**En Verode — GPU real (NCCL, cuando haya 2 nodos operativos):**
Usar el sistema de colas (Slurm) como recomienda el cotutor. Sin `--gres` si la especificación gres falla.
```bash
# Desde login node en tmux:
/opt/soft/slurm/20.11.04/bin/srun --partition=batch \
  --nodes=2 --nodelist=verode16,verode21 \
  --ntasks-per-node=1 --cpus-per-task=8 --time=72:00:00 \
  bash -c '
    cd ~/tfg-distributed-transformers
    .venv/bin/torchrun --nnodes=2 --nproc_per_node=1 \
      --node_rank=$SLURM_NODEID \
      --master_addr=verode16 --master_port=29500 \
      scripts/train_ddp.py \
      --config configs/train_ddp_verode.yaml --trace simple
  '
```

**En Verode — test funcional CPU (gloo, nodos down en Slurm):**
verode16 y verode18 aparecen como `down*` en Slurm y no son asignables por el sistema de colas,
pero sí son accesibles via SSH directo. Para el test CPU se usa SSH en dos terminales tmux:
```bash
# Terminal 1 — SSH a verode16 (nodo 0, master):
ssh verode16
cd ~/tfg-distributed-transformers && git pull origin main
.venv/bin/torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr=verode16 --master_port=29500 \
  scripts/train_ddp.py --config configs/train_ddp_cpu_test.yaml --trace simple

# Terminal 2 — SSH a verode21 (nodo 1):
ssh verode21
cd ~/tfg-distributed-transformers
.venv/bin/torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
  --master_addr=verode16 --master_port=29500 \
  scripts/train_ddp.py --config configs/train_ddp_cpu_test.yaml --trace simple
```
Extremadamente lento (CPU vs V100 ~100x). Solo para confirmar que la comunicación
multi-nodo y la sincronización de gradientes funcionan antes de tener hardware homogéneo.

### DDP heterogéneo — verode21 (GPU) + verode16 (CPU)

```bash
# Primero hacer git pull en ambos nodos:
# ssh verode21 && cd ~/tfg-distributed-transformers && git pull origin main
# ssh verode16 && cd ~/tfg-distributed-transformers && git pull origin main

# Terminal 1 — verode21 (V100, rank 0):
ssh verode21
cd ~/tfg-distributed-transformers
.venv/bin/torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr=verode21 --master_port=29500 \
  scripts/train_heterogeneous_ddp.py \
  --config configs/train_heterogeneous_ddp.yaml \
  --trace simple --layers confusion batch-monitor --fn energy

# Terminal 2 — verode16 (CPU, rank 1):
ssh verode16
cd ~/tfg-distributed-transformers
.venv/bin/torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
  --master_addr=verode21 --master_port=29500 \
  scripts/train_heterogeneous_ddp.py \
  --config configs/train_heterogeneous_ddp.yaml \
  --trace simple --layers confusion batch-monitor --fn energy
```

**Cómo funciona:**
- `HeterogeneousDistributedSampler` da a rank 0 (GPU) ≈94% del dataset y a rank 1 (CPU) ≈6%, proporcionalmente a sus compute_weights (16:1)
- `HeterogeneousDDPTrainer` normaliza los gradientes por el batch global real: `loss = criterion_sum / global_batch_size` → gradiente matemáticamente correcto aunque los batch sizes sean distintos
- Solo rank 0 (GPU) escribe logs, CSVs y checkpoints
- Los artefactos van a `logs/verode/ddp_hetero/vit_base_patch16_224/`

### Configuración SSH en Verode (hecho una sola vez)
```bash
ssh-keygen -t ed25519 -C "alu0101317038@verode" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub   # añadir en github.com → Settings → SSH keys
git remote set-url origin git@github.com:alerguezrojas/tfg-distributed-transformers.git
```

## Errores conocidos y soluciones

| Error | Causa | Solución |
|-------|-------|----------|
| `lr '<=' not supported between float and str` | `1e-4` en YAML se parsea como string | Usar `0.0001` en el YAML |
| `CUDA not available` en el clúster | `uv sync` instala torch cu13, incompatible con driver 525 | Reinstalar con `--index-url .../cu118` tras el sync |
| `Illegal instruction` en login node | Login node sin AVX2; numpy/torch usan AVX2 | Ejecutar siempre en nodo de cómputo via `srun` |
| `sbatch` I/O error | Bug de configuración de Slurm en VERODE | Usar `tmux` + `srun` |
| `CUDA out of memory` batch_size=64 (local) | ViT-B necesita ~11.5 GB para batch 64 | Usar batch_size=32 en local (4.95 GB) |
| `CUDA out of memory` en hooks | Tensores grandes copiados a RAM | Calcular en GPU con `.detach().float()`, solo `.item()` para el escalar |
| `FileNotFoundError` metadata.parquet | `configs/train.yaml` tiene rutas del SSD local | Usar `configs/train_cluster.yaml` en el clúster |
| `nvidia driver` no funciona con kernel 6.8 | Driver 470 incompatible | Actualizar a `nvidia-driver-580-open` |
| `[energy] GPU no disponible (pynvml no instalado)` en Verode | `uv sync` antes de añadir `nvidia-ml-py` al pyproject.toml | `uv sync` + reinstall torch cu118 |
| `srun --gres=gpu:1` falla en Verode | El recurso GPU se llama `gpu:tesla` en este clúster | Usar `--gres=gpu:tesla:1` |
| `libcudnn.so.9` en srun no-interactivo | `LD_LIBRARY_PATH` vacío en shells no-interactivos; cu13 instalado por `uv sync` | Reinstalar cu118: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --force-reinstall` |
| `ZeroDivisionError` en cosine scheduler | `epochs ≤ warmup_epochs` → `T_max = 0` | `builder.py` usa `max(1, epochs - warmup_epochs)` |
| `Duplicate GPU detected: rank X and rank Y both on CUDA device` (NCCL) | NCCL exige 1 rank = 1 GPU física; no se pueden 2 procesos en la misma GPU | Usar 2 GPUs reales (Kaggle 2×T4) o gloo CPU. `train_ddp.py` mapea `cuda:(local_rank % device_count)` para multi-GPU real |
| `Cannot use ReduceOp.AVG with Gloo` | gloo no soporta AVG en all_reduce (sí NCCL) | `ddp_trainer.py` usa `ReduceOp.SUM` y divide por `world_size` |
| `git clone` pide usuario en Kaggle/CI | El repo es privado | Token de GitHub (fine-grained, Contents:Read) vía Kaggle Secrets: `https://x-access-token:$GH_TOKEN@github.com/...` |
| El benchmark estimaba ~47× de más para un subset | `dataset_train` hardcodeado a 237871 | Corregido: lee el conteo real del metadata del config y escribe bloque `#sizes` |
| Run `--trace deep` invisible en el dashboard (0 epochs) | Regex posicional no casaba el orden de campos de la línea RESUMEN | Corregido: `_parse_deep` extrae cada campo por nombre |

## Comandos útiles

### Local
```bash
# Benchmark empírico (genera .log + .csv en logs/{env}/benchmark/)
uv run python scripts/benchmark.py --batch-sizes 16 32 --epochs 5
uv run python scripts/benchmark.py --model resnet50 --batch-sizes 32 64

# Test rápido single-GPU (1 epoch)
uv run python scripts/train_single_gpu.py --model vit_tiny_patch16_224 --epochs 1 --trace simple

# Entrenamiento completo con config v3
uv run python scripts/train_single_gpu.py --config configs/train_v3.yaml --trace simple --layers plot

# Smoke test DDP local (1 proceso, valida sin 2 GPUs)
torchrun --nproc_per_node=1 scripts/train_ddp.py \
  --model vit_tiny_patch16_224 --epochs 1 --config configs/train.yaml --trace simple

# Dashboard web (5 secciones)
uv run streamlit run src/web/app.py

# Ver log en tiempo real (ajusta la ruta al modelo concreto)
tail -f logs/local/single/vit_base_patch16_224/train_*.log
```

### Clúster VERODE
```bash
# Conectar
ssh alu0101317038@verode00.pcg.ull.es

# Cargar Slurm
module add slurm/client/20.11.04

# Ver estado
sinfo -N
squeue -a

# Abrir sesión tmux (para que el job sobreviva a desconexiones)
tmux new-session -s training   # nueva sesión
tmux attach -t training        # reconectar a sesión existente
# Ctrl+B, D → desconectarse sin matar la sesión

# Entrar al nodo de cómputo (interactivo)
/opt/soft/slurm/20.11.04/bin/srun --partition=batch --nodelist=verode21 --gres=gpu:tesla:1 --time=72:00:00 --pty bash

cd ~/tfg-distributed-transformers

# Verificar CUDA
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# Benchmark previo al entrenamiento (usar --config para que guarde en logs/verode/)
.venv/bin/python scripts/benchmark.py \
  --config configs/train_cluster_v3.yaml \
  --batch-sizes 32 64 128 --epochs 30 --nfs-factor 1.3

# Entrenamiento completo
.venv/bin/python scripts/train_single_gpu.py \
  --config configs/train_cluster_v3.yaml \
  --trace simple \
  --layers plot confusion batch-monitor \
  --fn energy

# Ver log en tiempo real
tail -f logs/verode/train_*.log

# Al terminar: subir resultados (gitignore ya configurado, no necesita -f)
git add logs/verode/
git commit -m "feat: add Verode training results"
git push origin main
```
