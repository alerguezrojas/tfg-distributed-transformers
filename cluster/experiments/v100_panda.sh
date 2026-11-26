#!/bin/bash
# PANDA a 512 px en la V100 de VERODE (single, lote 16, AMP salvo PRECISION=fp32): tiempo por época y potencia reales
# de varias arquitecturas, para validar el predictor en una tercera GPU con entrenamientos de
# verdad (misma receta que los runs locales de docs/datasets/panda.md, menos épocas: solo se
# mide el tiempo). Se lanza en el nodo de cómputo, desde tmux en el nodo de login:
#
#   srun -p batch -w verode21 --gres=gpu:tesla:1 --cpus-per-task=8 -t 06:00:00 bash cluster/experiments/v100_panda.sh
#   MODELS="resnet50 efficientnet_b0" EPOCHS=3 srun ... bash cluster/experiments/v100_panda.sh
#   PRECISION=fp32 MODELS="vit_small_patch16_224 resnet50" srun ... bash cluster/experiments/v100_panda.sh
#
# Los datos salen del perfil de máquina (configs/machines/verode.yaml, data_roots.panda) y los
# artefactos van a logs/verode/panda/single/<modelo>/. Un modelo que falla no para los demás.
set -uo pipefail
cd "${PARAVIT_DIR:-$HOME/tfg-distributed-transformers}"

EPOCHS="${EPOCHS:-4}"
WORKERS="${WORKERS:-8}"
PRECISION="${PRECISION:-amp}"
# Los ocho modelos de los runs locales (3060 Ti) que no necesitan pesos fuera de la caché, más
# el ViT-B de los runs de Kaggle.
read -r -a MODELS <<< "${MODELS:-vit_small_patch16_224 deit3_small_patch16_224.fb_in22k_ft_in1k \
swin_tiny_patch4_window7_224 convnext_tiny.fb_in22k resnet50 efficientnet_b0 tf_efficientnetv2_s \
vit_base_patch16_224.augreg2_in21k_ft_in1k}"

for model in "${MODELS[@]}"; do
    echo "=== $model $PRECISION ($(date '+%d/%m %H:%M'))"
    .venv/bin/python scripts/train_single_gpu.py --config configs/train_panda.yaml \
        --model "$model" --precision "$PRECISION" --epochs "$EPOCHS" --set "data.num_workers=$WORKERS" \
        --trace simple --fn energy || echo "FALLO $model"
done
