#!/bin/bash
# B3.4: stratified 5-snapshot pilot QM proton-PES scan on WT SLO.
#
# Uses 5 near-attack rst7 snapshots that span the biased-window r_DA range:
#   WT/win_2.70_final.rst7  (already have this from B3.2)
#   WT/win_3.20_final.rst7
#   WT/win_3.45_final.rst7
#   WT/win_3.70_final.rst7
#   WT/win_3.95_final.rst7
#
# Runs the proton-PES scanner with npts=5, small-mode (73-atom cluster,
# UKS B3LYP-D3BJ/def2-SVP + TRAH), for each snapshot. ~100 min per snapshot.

set -uo pipefail
source /home/liang/Workspace/WritePaper/CatalysisQuamBio/setup/env.sh

ROOT=/home/liang/Workspace/WritePaper/CatalysisQuamBio
PY=/home/liang/anaconda3/envs/slomd/bin/python
SCR="$ROOT/scripts/proton_pes_scanner.py"
PRM="$ROOT/md/mcpb/SLO_sub_solv.prmtop"
UMB="$ROOT/results/umbrella"
OUT="$ROOT/results/pcet_B34"
mkdir -p "$OUT"

LOG="$OUT/campaign.log"
echo "=== B3.4 pilot campaign start $(date -Iseconds) ===" | tee -a "$LOG"

# WT atom indices (verified in B3.2)
DONOR=13002; ACCEPTOR=12985; XFERH=13031

for WIN in 2.70 3.20 3.45 3.70 3.95; do
    SEED="$UMB/WT/win_${WIN}_final.rst7"
    OUTP="$OUT/wt_win${WIN//./}_pt"
    if [[ -f "${OUTP/_pt/}_scan.json" ]]; then
        # Detect existing scan.json
        actual="${OUTP%_pt}_scan.json"
    fi
    JSON="${OUT}/wt_win${WIN//./}_scan.json"
    if [[ -f "$JSON" ]]; then
        echo "  skip win=$WIN (already exists)" | tee -a "$LOG"; continue
    fi
    echo "" | tee -a "$LOG"
    echo "--- $(date -Iseconds) win=$WIN ---" | tee -a "$LOG"
    timeout 14400 "$PY" "$SCR" \
        --prmtop "$PRM" --rst7 "$SEED" \
        --xferH $XFERH --donor $DONOR --acceptor $ACCEPTOR \
        --npts 5 --mode small --charge 2 --mult 6 \
        --out "$OUT/wt_win${WIN//./}" 2>&1 | tee -a "$LOG" | tail -12
    if [[ -f "$JSON" ]]; then
        echo "  ok win=$WIN" | tee -a "$LOG"
    else
        echo "  FAILED win=$WIN" | tee -a "$LOG"
    fi
done

echo "=== B3.4 pilot campaign done $(date -Iseconds) ===" | tee -a "$LOG"
ls "$OUT"/*_scan.json | wc -l | xargs -I {} echo "  scans done: {} / 5" | tee -a "$LOG"
