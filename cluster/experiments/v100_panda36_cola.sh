#!/bin/bash
# Cola nocturna del 26-27/11/2026 en verode21, tras cluster/experiments/v100_panda36.sh (copia de
# ~/paravit_runs/queue_next.sh tal como corrió; resultados en docs/datasets/panda.md).
# 1) TTA x8 (test y val) de los cuatro runs de 36/16 tiles.
# 2) Swin-T con 36 tiles a 1024 px (más detalle por tile) y con 64 tiles de 192 px a 768 px
#    (receta del 1.er puesto de Kaggle): semilla 42 primero y después 43 y 44.
# 3) Otras arquitecturas con 36 tiles a 768 px, para el conjunto: ViT-B y SwinV2-T (semilla 42).
# Cada run: entrenamiento, eval normal y TTA x8 (test y val). Un fallo no para la cola.
# Código: worktree ~/paravit_wt (commit 43eb687, develop + --tta) con el .venv del repo principal.
set -uo pipefail
WT=$HOME/paravit_wt
PY=$HOME/tfg-distributed-transformers/.venv/bin/python
OUT=$HOME/paravit_runs
W=24
SWIN=swin_tiny_patch4_window7_224
cd "$WT"

best_of() { find "$OUT/$1/ckpt" -name best.pt -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2; }

evaluate() {   # nombre config tta [--set ...]
    local name=$1 cfg=$2 tta=$3; shift 3
    local best; best=$(best_of "$name")
    [ -n "$best" ] || { echo "FALLO sin best.pt $name"; return; }
    local suf=""; [ "$tta" -gt 1 ] && suf="_tta$tta"
    for split in test val; do
        $PY scripts/eval.py --checkpoint "$best" --config "$cfg" --split "$split" --tta "$tta" \
            --set "data.num_workers=$W" "$@" --save-predictions "$OUT/$name/${split}_preds$suf.npz" \
            > "$OUT/$name/eval_${split}$suf.out" 2>&1 || echo "FALLO eval $split tta$tta $name"
    done
    echo "$name tta$tta: $(grep -h -i 'QWK (cuadr' "$OUT/$name/eval_test$suf.out" | head -1)"
}

train() {      # nombre config semilla [--set ...]
    local name=$1 cfg=$2 seed=$3; shift 3
    mkdir -p "$OUT/$name"
    echo "=== $name ($cfg, semilla $seed, $*) $(date '+%d/%m %H:%M')"
    $PY scripts/train_single_gpu.py --config "$cfg" --model "$SWIN" --seed "$seed" --precision amp \
        --set "data.num_workers=$W" --set "checkpoint.dir=$OUT/$name/ckpt" "$@" \
        --trace simple --fn energy > "$OUT/$name/train.out" 2>&1 || { echo "FALLO entrenamiento $name"; return 1; }
}

echo "=== TTA de los runs de la tarde $(date '+%d/%m %H:%M')"
evaluate swin16_s42 configs/train_panda.yaml 8
for n in swin36_s42 swin36_s43 swin36_s44; do evaluate "$n" configs/train_panda_36.yaml 8; done

R1024=(--set data.image_size=1024 --set model.img_size=1024)
M64=(--set data.mosaics=mosaics_l1_64x192)
run() {        # nombre config semilla modelo [--set ...]
    local name=$1 cfg=$2 seed=$3 model=$4; shift 4
    SWIN=$model
    if train "$name" "$cfg" "$seed" "$@"; then
        evaluate "$name" "$cfg" 1 "$@"
        evaluate "$name" "$cfg" 8 "$@"
    fi
}
C36=configs/train_panda_36.yaml
run swin36_1024_s42 $C36 42 swin_tiny_patch4_window7_224 "${R1024[@]}"
run swin64_s42      $C36 42 swin_tiny_patch4_window7_224 "${M64[@]}"
run swin64_s43      $C36 43 swin_tiny_patch4_window7_224 "${M64[@]}"
run swin64_s44      $C36 44 swin_tiny_patch4_window7_224 "${M64[@]}"
run swin36_1024_s43 $C36 43 swin_tiny_patch4_window7_224 "${R1024[@]}"
run swin36_1024_s44 $C36 44 swin_tiny_patch4_window7_224 "${R1024[@]}"
run vitb36_s42      $C36 42 vit_base_patch16_224.augreg2_in21k_ft_in1k
run swinv2_36_s42   $C36 42 swinv2_tiny_window8_256
echo "=== fin $(date '+%d/%m %H:%M')"
