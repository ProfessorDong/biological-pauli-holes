#!/bin/bash
# Path-integral MD campaign for the manuscript's decisive test:
# Path multiplicity (Renyi-2 quantum reactive-flux entropy) in I553A vs I552A for H vs D.
#
# Design: 2 mutants x 2 isotopes x 3 replicas at r_DA(bias) = 3.35 Ang, N=8 beads, 15 ps
# production per window (2 ps equilibration + 15 ps production, dt=0.25 fs, no constraints,
# NVT at 300 K, CUDA/Blackwell). 12 runs, ~15 min each => ~3 hours total.
#
# All-bead per-frame positions of the transferring H are saved for downstream
# path-multiplicity analysis (Renyi-2 reactive-flux entropy from the isotope-specific
# quantum marginal density on the reactive plane).

_REPO="${PAULI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

set -uo pipefail

ROOT=${_REPO}
PY="${SLOMD_PYTHON:-/home/liang/anaconda3/envs/slomd/bin/python}"
SCR="$ROOT/scripts/pimd_window.py"
MD="$ROOT/md/mcpb"
UMB="$ROOT/results/umbrella"
OUT="$ROOT/results/pimd_prod"
mkdir -p "$OUT"

LOG="$OUT/campaign.log"
echo "=== PIMD campaign start $(date -Iseconds) ===" | tee -a "$LOG"

# Note: donor=12993 (C14), acceptor=12976 (Fe-OH O), xferH=13022 (H25, pro-S)
# The umbrella seed is win_3.45_final.rst7 (nearest available cached NPT-equilibrated
# near-attack seed for both mutants; the bias at r0=3.35 pulls the r_DA to target during eq).

for MUT in I553A I552A; do
    PRM="$MD/SLO_${MUT}_sub_solv.prmtop"
    SEED="$UMB/${MUT}/win_3.45_final.rst7"
    for MASS in 1.008 2.014; do
        TAG_ISO=H
        [[ "$MASS" == "2.014" ]] && TAG_ISO=D
        for REP in 1 2 3; do
            OUTP="$OUT/${MUT}_${TAG_ISO}_r335_rep${REP}"
            RNG=$(( 1000 * (REP + 1) + ($(printf "%d" "'${MUT:0:1}") * 7) ))
            echo "" | tee -a "$LOG"
            echo "--- $(date -Iseconds) $MUT $TAG_ISO m=$MASS rep=$REP  rng=$RNG ---" | tee -a "$LOG"
            for attempt in 1 2 3; do
                timeout 3600 "$PY" "$SCR" \
                    --prmtop "$PRM" \
                    --seed "$SEED" \
                    --donor 12993 --acceptor 12976 --xferH 13022 \
                    --mass "$MASS" --nbeads 8 --r0 3.35 --k 15.0 \
                    --ps-eq 2.0 --ps-prod 15.0 --dt-fs 0.25 --record-every-fs 50.0 \
                    --out "$OUTP" \
                    --rngseed "$RNG" 2>&1 | tee -a "$LOG"
                if [[ -f "${OUTP}_pimd.npz" ]]; then
                    echo "  ok: ${OUTP}_pimd.npz" | tee -a "$LOG"
                    break
                fi
                echo "  attempt $attempt failed; retrying" | tee -a "$LOG"
                sleep 3
            done
        done
    done
done

echo "" | tee -a "$LOG"
echo "=== PIMD campaign done $(date -Iseconds) ===" | tee -a "$LOG"
ls -la "$OUT"/*_pimd.npz 2>&1 | tee -a "$LOG"
