#!/bin/bash
# B1: RPMD bead-convergence sweep at 300 K.
# 2 systems (I553A, I552A) x 2 isotopes (H, D) x 2 bead counts (P=16, P=32) x
# 2 windows (r_DA=3.35, 3.60) x 3 replicas x 30 ps production
# = 48 windows total; P=16 ~30 min/window, P=32 ~60 min/window -> ~36 hours GPU

_REPO="${PAULI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

set -uo pipefail

ROOT=${_REPO}
PY="${SLOMD_PYTHON:-/home/liang/anaconda3/envs/slomd/bin/python}"
SCR="$ROOT/scripts/pimd_window.py"
MD="$ROOT/md/mcpb"
UMB="$ROOT/results/umbrella"
OUT="$ROOT/results/pimd_convergence"
mkdir -p "$OUT"

LOG="$OUT/campaign.log"
echo "=== B1 RPMD convergence sweep start $(date -Iseconds) ===" | tee -a "$LOG"

for MUT in I553A I552A; do
    PRM="$MD/SLO_${MUT}_sub_solv.prmtop"
    for R0 in 3.35 3.60; do
        SEED_r0=$(if [[ "$R0" == "3.35" ]]; then echo 3.45; else echo 3.70; fi)
        SEED="$UMB/${MUT}/win_${SEED_r0}_final.rst7"
        for MASS in 1.008 2.014; do
            TAG_ISO=H; [[ "$MASS" == "2.014" ]] && TAG_ISO=D
            for NBEADS in 16 32; do
                for REP in 1 2 3; do
                    OUTP="$OUT/${MUT}_${TAG_ISO}_r${R0//./}_P${NBEADS}_rep${REP}"
                    [[ -f "${OUTP}_pimd.npz" ]] && { echo "  skip ${OUTP}" | tee -a "$LOG"; continue; }
                    RNG=$(( 3000 * REP + NBEADS * 100 + $(printf "%d" "'${MUT:0:1}") ))
                    echo "" | tee -a "$LOG"
                    echo "--- $(date -Iseconds) $MUT $TAG_ISO m=$MASS r=$R0 P=$NBEADS rep=$REP rng=$RNG ---" | tee -a "$LOG"
                    # Longer timeout for P=32 which is ~100 min per 30 ps window
                    TO=$(( NBEADS == 32 ? 7200 : 4200 ))
                    # 6 attempts with alternate RNG seeds per attempt to defeat
                    # Blackwell CUDA context-startup race (this is the pattern that
                    # recovered all 7 A3-proper failures on attempt 1-3 of 6).
                    RECOVERED=0
                    for att in 1 2 3 4 5 6; do
                        RNG_ATT=$(( RNG + att * 137 ))
                        timeout $TO "$PY" "$SCR" \
                            --prmtop "$PRM" --seed "$SEED" \
                            --donor 12993 --acceptor 12976 --xferH 13022 \
                            --mass "$MASS" --nbeads "$NBEADS" --r0 "$R0" --k 15.0 \
                            --ps-eq 2.0 --ps-prod 30.0 --dt-fs 0.25 --record-every-fs 50.0 \
                            --out "$OUTP" --rngseed "$RNG_ATT" 2>&1 | tee -a "$LOG"
                        if [[ -f "${OUTP}_pimd.npz" ]]; then
                            echo "  RECOVERED on attempt $att (rng=$RNG_ATT)" | tee -a "$LOG"
                            RECOVERED=1
                            break
                        fi
                        echo "  attempt $att failed; will retry with new rng" | tee -a "$LOG"
                        sleep 5
                    done
                    [[ "$RECOVERED" = "0" ]] && echo "  UNRECOVERED after 6 attempts: $MUT $TAG_ISO r=$R0 P=$NBEADS rep=$REP" | tee -a "$LOG"
                done
            done
        done
    done
done

echo "=== B1 done $(date -Iseconds) ===" | tee -a "$LOG"
ls "$OUT"/*_pimd.npz | wc -l | xargs -I {} echo "  total npz: {}" | tee -a "$LOG"
