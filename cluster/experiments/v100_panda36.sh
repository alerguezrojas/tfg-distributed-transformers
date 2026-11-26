#!/bin/bash
# PANDA en la V100: ¿ayuda ver más tejido? Swin-T con 36 tiles a 768 px (configs/train_panda_36.yaml)
# frente al mismo modelo con 16 tiles a 512 px (configs/train_panda.yaml, control en esta máquina).
# Cada run entrena, evalúa su mejor checkpoint en test y en validación y deja las predicciones
# (test_preds.npz, val_preds.npz) en ~/paravit_runs/<nombre>/ para paravit compare-preds.
#
#   srun -p batch -w verode21 -n 1 --cpus-per-task=32 -t 24:00:00 bash cluster/experiments/v100_panda36.sh
#   RUNS="swin36_s43 swin36_s44" srun ... bash cluster/experiments/v100_panda36.sh
#
# Con 36 tiles la CPU decodifica 2,25 veces más píxeles por imagen: 8 procesos de carga dan
# ~62 img/s con los mosaicos de 16 (26/11/2026), así que aquí se usan 24 (WORKERS).
set -uo pipefail
cd "${PARAVIT_DIR:-$HOME/tfg-distributed-transformers}"
WORKERS="${WORKERS:-24}"
OUT="${OUT:-$HOME/paravit_runs}"
SWIN=swin_tiny_patch4_window7_224

# nombre -> "config semilla"
declare -A EXP=(
    [swin16_s42]="configs/train_panda.yaml 42"
    [swin36_s42]="configs/train_panda_36.yaml 42"
    [swin36_s43]="configs/train_panda_36.yaml 43"
    [swin36_s44]="configs/train_panda_36.yaml 44"
)
read -r -a RUNS <<< "${RUNS:-swin36_s42 swin16_s42 swin36_s43 swin36_s44}"

for name in "${RUNS[@]}"; do
    read -r cfg seed <<< "${EXP[$name]}"
    dest="$OUT/$name"; ckpt="$dest/ckpt"; mkdir -p "$dest"
    echo "=== $name ($cfg, semilla $seed) $(date '+%d/%m %H:%M')"
    .venv/bin/python scripts/train_single_gpu.py --config "$cfg" --model "$SWIN" --seed "$seed" \
        --precision amp --set "data.num_workers=$WORKERS" --set "checkpoint.dir=$ckpt" \
        --trace simple --fn energy > "$dest/train.out" 2>&1 || { echo "FALLO entrenamiento $name"; continue; }
    best=$(find "$ckpt" -name best.pt -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2)
    [ -n "$best" ] || { echo "FALLO sin best.pt $name"; continue; }
    for split in test val; do
        .venv/bin/python scripts/eval.py --checkpoint "$best" --config "$cfg" --split "$split" \
            --set "data.num_workers=$WORKERS" --save-predictions "$dest/${split}_preds.npz" \
            > "$dest/eval_$split.out" 2>&1 || echo "FALLO eval $split $name"
    done
    grep -h -i -E "qwk" "$dest/eval_test.out" | tail -2
done
