#!/bin/bash
# B3.4-dense: 5 stratified WT snapshots × 15-point QM proton-PES scans.
# Same protocol as B3.4 sparse but 15 points per scan instead of 5,
# to resolve the barrier region and both wells at rate-quality resolution.
# ~15-30 min per point × 15 points × 5 snapshots ~= 30-40 hours compute.

_REPO="${PAULI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

set -uo pipefail
source ${_REPO}/setup/env.sh

ROOT=${_REPO}
PY=/home/liang/anaconda3/envs/slomd/bin/python
SCR="$ROOT/scripts/proton_pes_scanner.py"
PRM="$ROOT/md/mcpb/SLO_sub_solv.prmtop"
UMB="$ROOT/results/umbrella"
OUT="$ROOT/results/pcet_B34_dense"
mkdir -p "$OUT"

LOG="$OUT/campaign.log"
echo "=== B3.4-dense start $(date -Iseconds) ===" | tee -a "$LOG"

# WT atom indices (verified in B3.2)
DONOR=13002; ACCEPTOR=12985; XFERH=13031

for WIN in 2.70 3.20 3.45 3.70 3.95; do
    SEED="$UMB/WT/win_${WIN}_final.rst7"
    JSON="${OUT}/wt_win${WIN//./}_scan.json"
    if [[ -f "$JSON" ]]; then
        echo "  skip win=$WIN (already exists)" | tee -a "$LOG"; continue
    fi
    echo "" | tee -a "$LOG"
    echo "--- $(date -Iseconds) win=$WIN ---" | tee -a "$LOG"
    # 3-attempt retry loop for Python-startup segfault recovery
    for att in 1 2 3; do
        [[ -f "$JSON" ]] && break
        echo "  attempt $att" | tee -a "$LOG"
        timeout 32400 "$PY" "$SCR" \
            --prmtop "$PRM" --rst7 "$SEED" \
            --xferH $XFERH --donor $DONOR --acceptor $ACCEPTOR \
            --npts 15 --mode small --charge 2 --mult 6 \
            --out "$OUT/wt_win${WIN//./}" 2>&1 | tee -a "$LOG" | tail -30
        [[ -f "$JSON" ]] && { echo "  ok win=$WIN (attempt $att)" | tee -a "$LOG"; break; }
        echo "  attempt $att failed; retry" | tee -a "$LOG"; sleep 30
    done
    [[ ! -f "$JSON" ]] && echo "  UNRECOVERED win=$WIN after 3 attempts" | tee -a "$LOG"
done

echo "=== B3.4-dense done $(date -Iseconds) ===" | tee -a "$LOG"
ls "$OUT"/*_scan.json 2>/dev/null | wc -l | xargs -I {} echo "  scans done: {} / 5" | tee -a "$LOG"
