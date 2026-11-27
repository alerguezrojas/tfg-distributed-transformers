#!/bin/bash
# Vigilante de VERODE (26/11/2026): cada 20 minutos escribe ESTADO.md con el progreso de las colas
# de ~/paravit_runs y lo sube a la rama `verode-estado` del repositorio (rama huérfana con un solo
# fichero: no dispara la CI, que solo corre en main y develop). Se ve sin VPN en
# https://github.com/alerguezrojas/tfg-distributed-transformers/blob/verode-estado/ESTADO.md
#
#   bash vigia_verode.sh           # bucle (en tmux); termina solo cuando la cola acaba
#   bash vigia_verode.sh --once    # una pasada, para probar
# Las colas que vigila están en OUTS y FIN: se ajustan a cada cola (en el clúster se usa una copia).
set -u
export PATH=/opt/soft/slurm/20.11.04/bin:$PATH
RUNS=$HOME/paravit_runs
EST=$HOME/paravit_estado
REPO=$HOME/tfg-distributed-transformers
WT=$HOME/paravit_wt
OUTS=("$RUNS/v100_panda36_26112026.out" "$RUNS/queue_next_26112026.out")
FIN="$RUNS/queue_next_26112026.out"
PERIOD=1200
STALL_MIN=120

init_repo() {
    [ -d "$EST/.git" ] && return
    mkdir -p "$EST" && cd "$EST" || exit 1
    git init -q
    git checkout -q --orphan verode-estado
    git remote add origin git@github.com:alerguezrojas/tfg-distributed-transformers.git
    git config user.name "$(git -C "$REPO" config user.name)"
    git config user.email "$(git -C "$REPO" config user.email)"
}

diagnose() {
    local jobs=$1 prev=$2
    if [ -z "$jobs" ] && grep -q "^=== fin" "$FIN" 2>/dev/null; then echo TERMINADO; return; fi
    if [ -z "$jobs" ]; then
        # sin trabajos y la cola sin terminar: entre dos trabajos hay < 1 min, así que solo se
        # declara parada si ya lo estaba en la pasada anterior
        if [ "$prev" = "SIN_TRABAJO" ] || [ "$prev" = "PARADO" ]; then echo PARADO; else echo SIN_TRABAJO; fi
        return
    fi
    if [ -z "$(find "$RUNS" "$WT/logs" "$REPO/logs" -type f -newermt "-$STALL_MIN minutes" 2>/dev/null | head -1)" ]; then
        echo BLOQUEO; return
    fi
    if grep -q "FALLO" "${OUTS[@]}" 2>/dev/null; then echo "EN_MARCHA_CON_FALLOS"; return; fi
    echo EN_MARCHA
}

write_status() {
    local status=$1 jobs=$2
    {
        echo "# Estado de VERODE"
        echo
        echo "**Diagnóstico: $status** — actualizado el $(date '+%d/%m/%Y a las %H:%M') (hora de Canarias)."
        echo
        echo "Se actualiza cada 20 minutos. Si esta hora tiene más de una hora, el vigilante o el"
        echo "nodo de login de VERODE han caído. Diagnósticos: EN_MARCHA, EN_MARCHA_CON_FALLOS (algún"
        echo "run falló y la cola siguió), BLOQUEO (hay trabajo pero nada escribe desde hace"
        echo "$STALL_MIN min), PARADO (no hay trabajo y la cola no ha terminado), TERMINADO."
        echo
        echo "## Trabajos en Slurm"
        echo '```'
        echo "${jobs:-ninguno}"
        echo '```'
        for f in "${OUTS[@]}"; do
            echo "## $(basename "$f")"
            echo '```'
            if [ -f "$f" ]; then grep -E "^=== |FALLO|tta[0-9]+:" "$f" | tail -30; else echo "(aún no ha empezado)"; fi
            echo '```'
        done
        local newest
        newest=$(ls -t "$REPO"/logs/verode/panda/single/*/epoch_metrics_* "$WT"/logs/verode/panda/single/*/epoch_metrics_* 2>/dev/null | head -1)
        echo "## Último entrenamiento"
        if [ -n "$newest" ]; then
            echo "\`$newest\` (modificado: $(date -r "$newest" '+%d/%m %H:%M'))"
            echo '```'
            echo "época, s/época, QWK val"
            tail -n 3 "$newest" | tr -d '\r' | awk -F, '{print $1", "$10", "$NF}'
            echo '```'
        fi
        echo "## QWK de test por run (eval normal y TTA x8)"
        echo '```'
        for d in "$RUNS"/*/; do
            for e in "$d"eval_test.out "$d"eval_test_tta8.out; do
                [ -f "$e" ] && printf '%-18s %-22s %s\n' "$(basename "$d")" "$(basename "$e")" \
                    "$(grep -h 'QWK (cuadr' "$e" | head -1 | sed 's/ *QWK (cuadrático) *: *//')"
            done
        done
        echo '```'
    } > "$EST/ESTADO.md"
}

push_status() {
    cd "$EST" || return
    git add ESTADO.md
    git commit -q -m "estado VERODE $(date '+%d/%m %H:%M')" || return
    timeout 60 git push -q -f origin verode-estado 2>&1 | tail -1
}

init_repo
prev=""
while true; do
    jobs=$(squeue -h -u "$USER" -o "%i %j %T %M %R")
    status=$(diagnose "$jobs" "$prev")
    write_status "$status" "$jobs"
    push_status
    prev=$status
    [ "${1:-}" = "--once" ] && break
    [ "$status" = "TERMINADO" ] && break
    sleep "$PERIOD"
done
