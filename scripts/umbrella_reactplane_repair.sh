#!/bin/bash
# Repair the 7 A3-proper windows that failed all 3 retry attempts due to Blackwell
# CUDA context-startup segfaults. Uses (a) more retry attempts (up to 6), (b) longer
# timeout per attempt (1200 s), (c) alternate RNG seeds per attempt to avoid the
# specific deterministic race that triggered the segfault.
#
# Failed windows (from campaign.log parse):
#   I553A r=2.70
#   I553A r=4.20
#   I553A r=6.20
#   I552A r=3.70
#   I552A r=4.70
#   L754A r=3.20
#   DM    r=3.20

_REPO="${PAULI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

set -uo pipefail

ROOT=${_REPO}
PY="${SLOMD_PYTHON:-/home/liang/anaconda3/envs/slomd/bin/python}"
SCR="$ROOT/scripts/umbrella_reactplane.py"
MD="$ROOT/md/mcpb"
UMB="$ROOT/results/umbrella"
OUT="$ROOT/results/umbrella_reactplane"

LOG="$OUT/campaign_repair.log"
echo "=== A3-proper REPAIR start $(date -Iseconds) ===" | tee -a "$LOG"

declare -A DONOR ACCEPTOR XFERH PRM
PRM[WT]=SLO_sub_solv
for m in I553A I552A L754A V750A I538A L546A DM; do PRM[$m]=SLO_${m}_sub_solv; done
DONOR[WT]=13002    ; ACCEPTOR[WT]=12985    ; XFERH[WT]=13031
DONOR[I553A]=12993 ; ACCEPTOR[I553A]=12976 ; XFERH[I553A]=13022
DONOR[I552A]=12993 ; ACCEPTOR[I552A]=12976 ; XFERH[I552A]=13022
DONOR[L754A]=12993 ; ACCEPTOR[L754A]=12976 ; XFERH[L754A]=13023
DONOR[V750A]=12996 ; ACCEPTOR[V750A]=12979 ; XFERH[V750A]=13025
DONOR[I538A]=12993 ; ACCEPTOR[I538A]=12976 ; XFERH[I538A]=13022
DONOR[L546A]=12993 ; ACCEPTOR[L546A]=12976 ; XFERH[L546A]=13022
DONOR[DM]=12984    ; ACCEPTOR[DM]=12967    ; XFERH[DM]=13013

# (mutant, window) pairs to repair
FAILED=(
    "I553A 2.70"
    "I553A 4.20"
    "I553A 6.20"
    "I552A 3.70"
    "I552A 4.70"
    "L754A 3.20"
    "DM    3.20"
)

for pair in "${FAILED[@]}"; do
    MUT=$(echo "$pair" | awk '{print $1}')
    R0=$(echo "$pair" | awk '{print $2}')
    PRM_FILE="$MD/${PRM[$MUT]}.prmtop"
    SEED="$UMB/${MUT}/win_${R0}_final.rst7"
    OUTP="$OUT/${MUT}_win_${R0}"
    if [[ -f "${OUTP}_reactplane.npz" ]]; then
        echo "  already repaired: $MUT r=$R0" | tee -a "$LOG"
        continue
    fi
    [[ -f "$SEED" ]] || { echo "  no seed for $MUT r=$R0" | tee -a "$LOG"; continue; }
    echo "" | tee -a "$LOG"
    echo "=== REPAIR $MUT r=$R0 ===" | tee -a "$LOG"
    for att in 1 2 3 4 5 6; do
        # Different RNG each attempt to avoid deterministic race
        RNG=$(( 5000 + att * 137 + $(printf "%d" "'${MUT:0:1}") + $(echo "$R0*100/1" | bc) ))
        echo "--- $(date -Iseconds) attempt $att  rng=$RNG ---" | tee -a "$LOG"
        timeout 1200 "$PY" "$SCR" \
            --prmtop "$PRM_FILE" --seed "$SEED" \
            --donor "${DONOR[$MUT]}" --acceptor "${ACCEPTOR[$MUT]}" --xferH "${XFERH[$MUT]}" \
            --r0 "$R0" --k 12.0 --ns-prod 0.2 --temp 300 \
            --out "$OUTP" --rngseed "$RNG" 2>&1 | tee -a "$LOG"
        if [[ -f "${OUTP}_reactplane.npz" ]]; then
            echo "  RECOVERED on attempt $att" | tee -a "$LOG"
            break
        fi
        echo "  attempt $att failed; will retry with new seed" | tee -a "$LOG"
        sleep 5
    done
    if [[ ! -f "${OUTP}_reactplane.npz" ]]; then
        echo "  UNRECOVERED after 6 attempts: $MUT r=$R0" | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "=== A3-proper REPAIR done $(date -Iseconds) ===" | tee -a "$LOG"
ls "$OUT"/*_reactplane.npz | wc -l | xargs -I {} echo "  total npz: {}" | tee -a "$LOG"
