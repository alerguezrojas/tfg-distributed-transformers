#!/bin/bash
# DDP en varios nodos: un torchrun por nodo, todos con EL MISMO comando. `--launcher slurm` lee
# SLURM_JOB_NODELIST y usa el primer nodo como punto de encuentro (rendezvous c10d), sin
# --node_rank ni --master_addr a mano. Valores de ejemplo: clúster VERODE (ULL).
#
#   sbatch cluster/slurm/ddp_multinode.sbatch
#       Este script corre una vez (en el primer nodo) y reparte el trabajo con `srun`.
#
#   srun --partition=batch --nodes=2 --ntasks-per-node=1 --gres=gpu:tesla:1 --cpus-per-task=8 \
#        --time=12:00:00 env NO_SRUN=1 bash cluster/slurm/ddp_multinode.sbatch
#       VERODE (sbatch falla): srun ya crea una tarea por nodo y NO_SRUN=1 evita repartir otra vez.
#
# Con srun las líneas #SBATCH son comentarios: las opciones van en la línea de srun.
#SBATCH --job-name=paravit-ddp
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1        # una tarea (un torchrun) por nodo
#SBATCH --gres=gpu:tesla:1         # GPU por nodo (VERODE: gpu:tesla)
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail
cd "${PARAVIT_DIR:-$HOME/tfg-distributed-transformers}"
mkdir -p logs/slurm

CONFIG="${CONFIG:-configs/train_cluster_v3.yaml}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"       # debe coincidir con --gres
PORT="${PORT:-29500}"
PARAVIT="${PARAVIT:-.venv/bin/python paravit.py}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "Este script debe ejecutarse dentro de un trabajo de Slurm (sbatch o srun)." >&2
  exit 1
fi

# shellcheck disable=SC2086   # $PARAVIT lleva varias palabras a propósito
TRAIN="$PARAVIT train --strategy ddp --launcher slurm --n-gpus $GPUS_PER_NODE --master-port $PORT --config $CONFIG --trace simple"

if [ -n "${NO_SRUN:-}" ]; then
  exec $TRAIN "$@"                                  # ya estamos en una tarea por nodo
fi
exec srun --ntasks-per-node=1 $TRAIN "$@"           # sbatch: una tarea por nodo
