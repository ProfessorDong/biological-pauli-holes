#!/bin/bash
# Piece 1 scale-up: bring the I553A/I552A path-multiplicity test to statistical significance.
#
# Adds:
#   (a) 3 more replicas per (system, isotope) at r0=3.35 A (total: 6 reps at that window)
#   (b) 3 replicas per (system, isotope) at a second near-attack window r0=3.60 A
# So 12 new (r0=3.35) + 12 new (r0=3.60) = 24 additional PIMD windows.
# At ~15 min per window: ~6 hours total GPU compute.
#
# All other settings unchanged from pimd_campaign.sh (N=8 beads, 0.25 fs, NVT, no constraints,
# 2 ps eq + 15 ps prod, HBonds relaxed).

_REPO="${PAULI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

set -uo pipefail

ROOT=${_REPO}
PY="${SLOMD_PYTHON:-/home/liang/anaconda3/envs/slomd/bin/python}"
SCR="$ROOT/scripts/pimd_window.py"
MD="$ROOT/md/mcpb"
UMB="$ROOT/results/umbrella"
OUT="$ROOT/results/pimd_prod"
mkdir -p "$OUT"

LOG="$OUT/campaign_scale.log"
echo "=== PIMD scale-up start $(date -Iseconds) ===" | tee -a "$LOG"

# Additional replicas at r0=3.35 A: rep 4, 5, 6 (rep 1-3 already exist)
for MUT in I553A I552A; do
    PRM="$MD/SLO_${MUT}_sub_solv.prmtop"
    SEED="$UMB/${MUT}/win_3.45_final.rst7"
    for MASS in 1.008 2.014; do
        TAG_ISO=H
        [[ "$MASS" == "2.014" ]] && TAG_ISO=D
        for REP in 4 5 6; do
            OUTP="$OUT/${MUT}_${TAG_ISO}_r335_rep${REP}"
            [[ -f "${OUTP}_pimd.npz" ]] && { echo "  skip ${OUTP} (exists)" | tee -a "$LOG"; continue; }
            RNG=$(( 1000 * (REP + 1) + ($(printf "%d" "'${MUT:0:1}") * 7) ))
            echo "" | tee -a "$LOG"
            echo "--- $(date -Iseconds) $MUT $TAG_ISO m=$MASS r=3.35 rep=$REP rng=$RNG ---" | tee -a "$LOG"
            for attempt in 1 2 3; do
                timeout 3600 "$PY" "$SCR" \
                    --prmtop "$PRM" --seed "$SEED" \
                    --donor 12993 --acceptor 12976 --xferH 13022 \
                    --mass "$MASS" --nbeads 8 --r0 3.35 --k 15.0 \
                    --ps-eq 2.0 --ps-prod 15.0 --dt-fs 0.25 --record-every-fs 50.0 \
                    --out "$OUTP" --rngseed "$RNG" 2>&1 | tee -a "$LOG"
                [[ -f "${OUTP}_pimd.npz" ]] && { echo "  ok" | tee -a "$LOG"; break; }
                echo "  attempt $attempt failed; retry" | tee -a "$LOG"; sleep 3
            done
        done
    done
done

# Second near-attack window r0=3.60 A: rep 1, 2, 3 (using the closer win_3.70_final.rst7 seed)
for MUT in I553A I552A; do
    PRM="$MD/SLO_${MUT}_sub_solv.prmtop"
    SEED="$UMB/${MUT}/win_3.70_final.rst7"
    [[ -f "$SEED" ]] || SEED="$UMB/${MUT}/win_3.45_final.rst7"
    for MASS in 1.008 2.014; do
        TAG_ISO=H
        [[ "$MASS" == "2.014" ]] && TAG_ISO=D
        for REP in 1 2 3; do
            OUTP="$OUT/${MUT}_${TAG_ISO}_r360_rep${REP}"
            [[ -f "${OUTP}_pimd.npz" ]] && { echo "  skip ${OUTP} (exists)" | tee -a "$LOG"; continue; }
            RNG=$(( 2000 * (REP + 1) + ($(printf "%d" "'${MUT:0:1}") * 11) ))
            echo "" | tee -a "$LOG"
            echo "--- $(date -Iseconds) $MUT $TAG_ISO m=$MASS r=3.60 rep=$REP rng=$RNG ---" | tee -a "$LOG"
            for attempt in 1 2 3; do
                timeout 3600 "$PY" "$SCR" \
                    --prmtop "$PRM" --seed "$SEED" \
                    --donor 12993 --acceptor 12976 --xferH 13022 \
                    --mass "$MASS" --nbeads 8 --r0 3.60 --k 15.0 \
                    --ps-eq 2.0 --ps-prod 15.0 --dt-fs 0.25 --record-every-fs 50.0 \
                    --out "$OUTP" --rngseed "$RNG" 2>&1 | tee -a "$LOG"
                [[ -f "${OUTP}_pimd.npz" ]] && { echo "  ok" | tee -a "$LOG"; break; }
                echo "  attempt $attempt failed; retry" | tee -a "$LOG"; sleep 3
            done
        done
    done
done

echo "" | tee -a "$LOG"
echo "=== PIMD scale-up done $(date -Iseconds) ===" | tee -a "$LOG"
ls "$OUT"/*_pimd.npz | wc -l | xargs -I {} echo "  total npz: {}" | tee -a "$LOG"
