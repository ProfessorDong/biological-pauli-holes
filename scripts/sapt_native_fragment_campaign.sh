#!/bin/bash
# B2 native-fragment SAPT campaign for all 7 JBC systems.
# Runs Psi4 SAPT0/jun-cc-pVDZ on native isobutane (Leu) or acetamide (Asn) wall
# fragment, 5-point scan. ~10-15 min per system on Psi4 (CPU), ~2 h total for 7 systems.

set -uo pipefail
ROOT=/home/liang/Workspace/WritePaper/CatalysisQuamBio
PY=/home/liang/anaconda3/envs/pauli/bin/python
SCR="$ROOT/scripts/sapt_native_fragment.py"
DIR="$ROOT/results/sapt_bio/native_fragment"
LOG="$DIR/campaign.log"
mkdir -p "$DIR"
echo "=== B2 native-fragment SAPT campaign start $(date -Iseconds) ===" | tee -a "$LOG"

for TAG in WT V750A I552A I538A L754A L546A I553A; do
    if [[ -f "$DIR/${TAG}_native_result.json" ]]; then
        echo "  skip $TAG (already done)" | tee -a "$LOG"; continue
    fi
    echo "" | tee -a "$LOG"
    echo "--- $(date -Iseconds) $TAG ---" | tee -a "$LOG"
    timeout 3600 "$PY" "$SCR" "$TAG" 2>&1 | tee -a "$LOG" | tail -12
    if [[ -f "$DIR/${TAG}_native_result.json" ]]; then
        echo "  ok" | tee -a "$LOG"
    else
        echo "  FAILED $TAG" | tee -a "$LOG"
    fi
done

echo "=== done $(date -Iseconds) ===" | tee -a "$LOG"
ls "$DIR"/*_native_result.json | wc -l | xargs -I {} echo "  total done: {}" | tee -a "$LOG"
