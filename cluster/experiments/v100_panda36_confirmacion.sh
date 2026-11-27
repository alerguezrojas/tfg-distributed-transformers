#!/bin/bash
# Cola de confirmación del 27-28/11/2026 en verode21: semillas 43 y 44 de ViT-B y SwinV2-T con 36 tiles
# a 768 px (para repetir el conjunto con runs que no intervinieron en su elección) y ViT-B con 36
# tiles a 1024 px (exploración, una semilla). Cada run: entrenamiento, eval normal y TTA x8.
# Código: el repositorio principal en develop (b738b8e).
set -uo pipefail
WT=$HOME/tfg-distributed-transformers
PY=$WT/.venv/bin/python
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
VITB=vit_base_patch16_224.augreg2_in21k_ft_in1k
run vitb36_s43     $C36 43 $VITB
run vitb36_s44     $C36 44 $VITB
run swinv2_36_s43  $C36 43 swinv2_tiny_window8_256
run swinv2_36_s44  $C36 44 swinv2_tiny_window8_256
run vitb36_1024_s42 $C36 42 $VITB "${R1024[@]}"
echo "=== fin $(date '+%d/%m %H:%M')"
