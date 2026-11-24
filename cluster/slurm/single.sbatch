#!/bin/bash
# Entrenamiento en UNA GPU de un nodo (estrategia single). Plantilla comentada: los valores de
# ejemplo son los del clúster VERODE (ULL); ajuste partición, gres, CPUs y tiempo a su clúster.
#
#   sbatch cluster/slurm/single.sbatch            # donde sbatch funcione
#   srun --partition=batch --gres=gpu:tesla:1 --cpus-per-task=8 --time=12:00:00 \
#        bash cluster/slurm/single.sbatch         # en VERODE (sbatch falla): el mismo script con srun
#
# Las líneas #SBATCH solo las lee sbatch; con srun son comentarios y las opciones van en la
# línea de srun.
#SBATCH --job-name=paravit-single
#SBATCH --partition=batch          # VERODE: batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla:1         # VERODE: el recurso se llama gpu:tesla (no gpu:1)
#SBATCH --cpus-per-task=8          # workers del DataLoader
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail
cd "${PARAVIT_DIR:-$HOME/tfg-distributed-transformers}"
mkdir -p logs/slurm

# Lo que cambia casi siempre (se fija al lanzar: CONFIG=... MODEL=... sbatch ...)
CONFIG="${CONFIG:-configs/train.yaml}"
MODEL="${MODEL:-vit_base_patch16_224}"

# Sin contenedor: el entorno virtual del repositorio. Con contenedor (cluster/README.md):
#   PARAVIT="apptainer run --nv --bind /ruta/datos:/datos paravit.sif"
PARAVIT="${PARAVIT:-.venv/bin/python paravit.py}"

# Una sola GPU: no hace falta torchrun, así que no se usa --launcher slurm.
# shellcheck disable=SC2086   # $PARAVIT lleva varias palabras a propósito
$PARAVIT train --strategy single --config "$CONFIG" --model "$MODEL" --trace simple "$@"
