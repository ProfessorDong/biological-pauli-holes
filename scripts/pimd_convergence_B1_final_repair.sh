#!/bin/bash
# Final repair pass for any B1 windows that the primary campaign marked UNRECOVERED.
# Uses longer inter-attempt sleep (30 s) and randomised RNG offset to defeat any
# quasi-deterministic startup race that survived the primary 6-attempt loop.

set -uo pipefail

ROOT=/home/liang/Workspace/WritePaper/CatalysisQuamBio
PY=/home/liang/anaconda3/envs/slomd/bin/python
SCR="$ROOT/scripts/pimd_window.py"
MD="$ROOT/md/mcpb"
UMB="$ROOT/results/umbrella"
OUT="$ROOT/results/pimd_convergence"
LOG="$OUT/final_repair.log"

# Parse the campaign log for UNRECOVERED entries. Format:
#   UNRECOVERED after 6 attempts: <MUT> <ISO> r=<R0> P=<NB> rep=<REP>
UNREC=$(grep "UNRECOVERED after 6 attempts" "$OUT/campaign.log" 2>/dev/null | \
        sed 's/.*UNRECOVERED after 6 attempts: //')

if [[ -z "$UNREC" ]]; then
    echo "no UNRECOVERED windows in B1 campaign — nothing to repair" | tee -a "$LOG"
    exit 0
fi

echo "=== B1 final repair start $(date -Iseconds) ===" | tee -a "$LOG"
echo "$UNREC" | tee -a "$LOG"

declare -A PRM
for m in I553A I552A; do PRM[$m]=SLO_${m}_sub_solv; done

while IFS= read -r line; do
    # Parse "<MUT> <ISO> r=<R0> P=<NB> rep=<REP>"
    MUT=$(echo "$line" | awk '{print $1}')
    ISO=$(echo "$line" | awk '{print $2}')
    R0=$(echo "$line" | grep -oE 'r=[0-9.]+' | cut -d= -f2)
    NB=$(echo "$line" | grep -oE 'P=[0-9]+' | cut -d= -f2)
    REP=$(echo "$line" | grep -oE 'rep=[0-9]+' | cut -d= -f2)
    OUTP="$OUT/${MUT}_${ISO}_r${R0//./}_P${NB}_rep${REP}"
    if [[ -f "${OUTP}_pimd.npz" ]]; then
        echo "  skip (already exists): $line" | tee -a "$LOG"; continue
    fi
    MASS=1.008; [[ "$ISO" == "D" ]] && MASS=2.014
    SEED_r0=$(if [[ "$R0" == "3.35" ]]; then echo 3.45; else echo 3.70; fi)
    SEED="$UMB/${MUT}/win_${SEED_r0}_final.rst7"
    PRM_FILE="$MD/${PRM[$MUT]}.prmtop"
    echo "" | tee -a "$LOG"
    echo "=== REPAIR $line ===" | tee -a "$LOG"
    TO=$(( NB == 32 ? 7200 : 4200 ))
    RECOVERED=0
    for att in 1 2 3 4 5 6 7 8; do
        # Randomised RNG offset per attempt to sample a fresh region of seed space
        RNG=$(( 20000 + att * 6151 + REP * 137 + NB * 41 ))
        echo "--- attempt $att (rng=$RNG) $(date -Iseconds) ---" | tee -a "$LOG"
        timeout $TO "$PY" "$SCR" \
            --prmtop "$PRM_FILE" --seed "$SEED" \
            --donor 12993 --acceptor 12976 --xferH 13022 \
            --mass "$MASS" --nbeads "$NB" --r0 "$R0" --k 15.0 \
            --ps-eq 2.0 --ps-prod 30.0 --dt-fs 0.25 --record-every-fs 50.0 \
            --out "$OUTP" --rngseed "$RNG" 2>&1 | tail -3 | tee -a "$LOG"
        if [[ -f "${OUTP}_pimd.npz" ]]; then
            echo "  RECOVERED on attempt $att" | tee -a "$LOG"
            RECOVERED=1
            break
        fi
        echo "  attempt $att failed; sleeping 30 s before next attempt" | tee -a "$LOG"
        sleep 30
    done
    if [[ "$RECOVERED" = "0" ]]; then
        echo "  STILL UNRECOVERED after 8 long-sleep attempts: $line" | tee -a "$LOG"
    fi
done <<< "$UNREC"

echo "" | tee -a "$LOG"
echo "=== B1 final repair done $(date -Iseconds) ===" | tee -a "$LOG"
ls "$OUT"/*_pimd.npz | wc -l | xargs -I {} echo "  total B1 npz now: {}" | tee -a "$LOG"
