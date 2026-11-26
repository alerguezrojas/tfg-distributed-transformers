# Ejecución en clúster (Slurm + Apptainer)

Plantillas para lanzar ParaViT-Lab en un clúster con Slurm, con o sin contenedor. Los valores de
ejemplo son los del clúster VERODE (ULL); ver [`docs/infraestructura.md`](../docs/infraestructura.md).

| Fichero | Para qué |
|---|---|
| `slurm/single.sbatch` | Una GPU en un nodo (`--strategy single`) |
| `slurm/ddp_multinode.sbatch` | DDP en varios nodos con `--launcher slurm` |
| `apptainer/paravit.def` | Imagen con PyTorch cu118 y el proyecto |
| `benchmark/throughput_sweep.py` | Barrido de throughput del predictor en la GPU del nodo (las campañas de la T4) |
| `experiments/v100_panda.sh` | Entrenamientos de PANDA en la V100 para validar el predictor |

## 1. El lanzador `--launcher slurm`

Dentro de un trabajo de Slurm, `paravit train --launcher slurm` construye un `torchrun` con
rendezvous `c10d` en el **primer nodo** de la asignación y **el mismo comando en todos los nodos**
(no hay `--node_rank` ni `--master_addr` que calcular a mano):

```bash
torchrun --nnodes=$SLURM_NNODES --nproc_per_node=<n> --rdzv-backend=c10d \
         --rdzv-endpoint=<primer nodo>:<puerto> --rdzv-id=$SLURM_JOB_ID scripts/train_ddp.py ...
```

- `--n-gpus` son las GPU por nodo; `--master-port` es el puerto del rendezvous (29500 por defecto);
  `--nnodes`, `--node-rank` y `--master-addr` se ignoran (el número de nodos sale de `SLURM_NNODES`).
- El primer nodo sale de `SLURM_JOB_NODELIST` con `scontrol show hostnames` si existe y, si no,
  con un intérprete propio de listas compactas (`verode[16,21]`, `n[01-03],m5`).
- Solo estrategias torchrun (`ddp`, `heterogeneous`). Fuera de un trabajo de Slurm, o con una
  estrategia de un solo proceso (`single`, `model-parallel`), da un error explicando qué falta.
- Sin `--launcher` el comando es el de siempre (`--launcher local`).

Para ver el comando sin ejecutar nada (hace falta el entorno de un trabajo; fuera se simula así):

```bash
SLURM_JOB_ID=1 SLURM_NNODES=2 SLURM_JOB_NODELIST='verode[16,21]' \
  uv run paravit.py train --strategy ddp --launcher slurm --dry-run
```

## 2. Plantillas Slurm

Se copian y se ajustan (partición, `--gres`, `--cpus-per-task`, `--time`). Cada una se lanza con
`sbatch` o, mientras `sbatch` falle en VERODE (*I/O error writing script/environment*), con `srun`:

```bash
# Una GPU
srun --partition=batch --gres=gpu:tesla:1 --cpus-per-task=8 --time=12:00:00 \
     bash cluster/slurm/single.sbatch

# DDP en 2 nodos (una GPU por nodo). NO_SRUN=1: srun ya crea una tarea por nodo.
srun --partition=batch --nodes=2 --ntasks-per-node=1 --gres=gpu:tesla:1 --cpus-per-task=8 \
     --time=12:00:00 env NO_SRUN=1 bash cluster/slurm/ddp_multinode.sbatch --epochs 3
```

Con `sbatch cluster/slurm/ddp_multinode.sbatch` el propio script reparte con `srun`. Los argumentos
extra (`--epochs 3`) pasan a `paravit train`. Variables: `CONFIG`, `MODEL`, `GPUS_PER_NODE`, `PORT`,
`PARAVIT_DIR` y `PARAVIT` (para sustituir el comando, por ejemplo por el contenedor).

En VERODE recuerden: el nodo de login no tiene AVX2 (todo por `srun`), y el entorno virtual debe
llevar torch cu118 (ver la sección siguiente si prefieren no mantenerlo a mano).

## 3. Apptainer

La imagen parte de `pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime` (cu118, compatible con el
driver 525 de VERODE), cambia el Python a 3.12 (lo que pide el proyecto), instala torch/torchvision
cu118 y el resto de dependencias de `pyproject.toml`, y deja `paravit` como punto de entrada.

```bash
# Build (desde la raíz del repositorio; en una máquina con Apptainer y sin límite de disco)
apptainer build paravit.sif cluster/apptainer/paravit.def

# Uso: --nv expone la GPU; se enlazan dataset, logs y checkpoints (la imagen es de solo lectura)
apptainer run --nv \
  --bind /ruta/al/dataset:/datos \
  --bind $PWD/logs:/opt/paravit/logs \
  --bind $PWD/checkpoints:/opt/paravit/checkpoints \
  paravit.sif train --strategy single --config configs/train.yaml --set data.root=/datos
```

Dentro de Slurm, con el contenedor en las plantillas:

```bash
export PARAVIT="apptainer run --nv --bind /ruta/al/dataset:/datos --bind $PWD/logs:/opt/paravit/logs \
 --bind $PWD/checkpoints:/opt/paravit/checkpoints paravit.sif"
srun ... env NO_SRUN=1 bash cluster/slurm/ddp_multinode.sbatch
```

Las rutas del dataset salen del perfil de máquina (`configs/machines/*.yaml`); dentro del
contenedor use `--set data.root=...` (o un perfil propio) con la ruta enlazada.

**Estado:** la definición no se ha construido todavía en VERODE (no hay Apptainer en la máquina de
desarrollo). El primer `build` debe comprobar con `%test` que `paravit --help` y `import torch`
funcionan, y con `apptainer run --nv paravit.sif doctor --require-gpu` en un nodo con GPU que `--nv` ve el
driver.

## 4. Medidas del predictor en VERODE

Las dos mediciones que validan el predictor en la V100 (`docs/performance_model.md`) se lanzan
desde una sesión `tmux` del nodo de login, para que sigan aunque se corte la conexión:

```bash
export PATH=/opt/soft/slurm/20.11.04/bin:$PATH     # por SSH sin terminal no se lee ~/.bashrc
cd ~/tfg-distributed-transformers
# 1) Throughput con lotes sintéticos (sin datos), ~20 min
srun -p batch -w verode21 -n 1 -t 03:00:00 \
     .venv/bin/python cluster/benchmark/throughput_sweep.py --out ~/paravit_runs/v100_sweep
# 2) Entrenamientos reales de PANDA a 512 px (data_roots.panda del perfil verode)
srun -p batch -w verode21 -n 1 --cpus-per-task=8 -t 06:00:00 \
     bash cluster/experiments/v100_panda.sh
```

Dos trampas del Slurm de VERODE (26/11/2026): `--gres=gpu:tesla:1` da *Invalid generic resource
specification* (se pide el nodo con `-w verode21`), y `--cpus-per-task=8` sin `-n 1` lanza
40/8 = 5 copias del mismo comando sobre la misma GPU. Comprueben con `squeue -s` y `ps` que hay un
solo proceso de entrenamiento antes de fiarse de los tiempos.

El barrido escribe `throughput.csv` (campaña, resolución, precisión, modelo, lote, img/s de
entrenamiento y evaluación, VRAM) y `gpu.txt`; se copia al repositorio en `logs/verode/benchmark/`.
Los entrenamientos dejan sus artefactos en `logs/verode/panda/single/<modelo>/`.

