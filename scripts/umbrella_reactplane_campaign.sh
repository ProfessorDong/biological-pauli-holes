#!/bin/bash
# A3-proper: 200 ps re-run per (system, window) dumping per-frame (r_DA, r_HO, theta_CHO)
# for MBAR-reweighted 2D joint volume calculation. Reuses existing win_r0_final.rst7 seeds.
#
# 8 systems x 15 windows x 200 ps = 24 ns MD ~ ~1.2 h GPU (200 ps @ ~3 min).
# We restrict to a single replica (rep 1) for the 2D-volume MBAR pass.

set -uo pipefail

ROOT=/home/liang/Workspace/WritePaper/CatalysisQuamBio
PY=/home/liang/anaconda3/envs/slomd/bin/python
SCR="$ROOT/scripts/umbrella_reactplane.py"
MD="$ROOT/md/mcpb"
UMB="$ROOT/results/umbrella"
OUT="$ROOT/results/umbrella_reactplane"
mkdir -p "$OUT"

LOG="$OUT/campaign.log"
echo "=== A3-proper reactive-plane campaign start $(date -Iseconds) ===" | tee -a "$LOG"

# atom indices per system
# Indices verified by parmed against each prmtop and cross-checked against seed geometry.
# Note: L754A's transferring H is H26 (13023) in the current seed; other systems use H25.
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

WINDOWS="2.70 2.95 3.20 3.45 3.70 3.95 4.20 4.45 4.70 4.95 5.20 5.45 5.70 5.95 6.20"

for MUT in WT I553A I552A L754A V750A I538A L546A DM; do
    PRM_FILE="$MD/${PRM[$MUT]}.prmtop"
    for R0 in $WINDOWS; do
        SEED="$UMB/${MUT}/win_${R0}_final.rst7"
        OUTP="$OUT/${MUT}_win_${R0}"
        [[ -f "${OUTP}_reactplane.npz" ]] && { echo "  skip ${OUTP} (exists)" | tee -a "$LOG"; continue; }
        [[ -f "$SEED" ]] || { echo "  no seed ${SEED}" | tee -a "$LOG"; continue; }
        echo "--- $(date -Iseconds) $MUT r0=$R0 ---" | tee -a "$LOG"
        RNG=$(( 1000 * ( $(echo "$R0*100/1" | bc) ) + $(printf "%d" "'${MUT:0:1}") ))
        for att in 1 2 3; do
            timeout 900 "$PY" "$SCR" \
                --prmtop "$PRM_FILE" --seed "$SEED" \
                --donor "${DONOR[$MUT]}" --acceptor "${ACCEPTOR[$MUT]}" --xferH "${XFERH[$MUT]}" \
                --r0 "$R0" --k 12.0 --ns-prod 0.2 --temp 300 \
                --out "$OUTP" --rngseed "$RNG" 2>&1 | tee -a "$LOG"
            [[ -f "${OUTP}_reactplane.npz" ]] && { echo "  ok" | tee -a "$LOG"; break; }
            echo "  attempt $att failed; retry" | tee -a "$LOG"; sleep 3
        done
    done
done

echo "=== A3-proper done $(date -Iseconds) ===" | tee -a "$LOG"
ls "$OUT"/*_reactplane.npz | wc -l | xargs -I {} echo "  total npz: {}" | tee -a "$LOG"
