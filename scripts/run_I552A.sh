#!/bin/bash
# Add I552A (the 7th single-site JBC-2019 mutant, KIE=66/60) to the panel.
# minimize + equilibrate + 3 replicas of umbrella sampling.
_REPO="${PAULI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

set -euo pipefail
SL=/home/liang/anaconda3/envs/slomd/bin/python
DRV=${_REPO}/scripts/umbrella_driver.py
MIN=${_REPO}/md/mcpb/minimize_sub.py
EQ=${_REPO}/md/mcpb/equilibrate_sub.py
MCPB=${_REPO}/md/mcpb
LOG=${_REPO}/results/umbrella/pathA_followup.log
cd "$MCPB"
echo "=== I552A pipeline start: $(date -Iseconds) ===" | tee -a "$LOG"

PRE="SLO_I552A_sub"
for attempt in 1 2 3 4 5; do
    [ -f "${PRE}_min_xyz.npy" ] && break
    echo "--- $(date -Iseconds)  minimize I552A (attempt $attempt) ---" | tee -a "$LOG"
    "$SL" "$MIN" "$PRE" 2>&1 | tee -a "$LOG" || true
    sleep 3
done
for attempt in 1 2 3 4 5; do
    [ -f "${PRE}_eq.rst7" ] && break
    echo "--- $(date -Iseconds)  equilibrate I552A (attempt $attempt) ---" | tee -a "$LOG"
    "$SL" "$EQ" 1.0 "$PRE" 2>&1 | tee -a "$LOG" || true
    sleep 3
done
[ -f "${PRE}_eq.rst7" ] || { echo "[FATAL] I552A eq failed 5x" | tee -a "$LOG"; exit 1; }

read -r D A R <<< "$($SL - <<PY
import parmed as p, numpy as np
from openmm.app import AmberInpcrdFile
prm = p.load_file('$MCPB/SLO_I552A_sub_solv.prmtop')
c14 = [a for a in prm.atoms if a.residue.name=='LIG' and a.name=='C14'][0].idx
oO  = [a for a in prm.atoms if a.residue.name=='OH1' and a.name=='O'][0].idx
rst = AmberInpcrdFile('$MCPB/SLO_I552A_sub_eq.rst7')
xyz = np.array([[v.x,v.y,v.z] for v in rst.positions])
r = float(np.linalg.norm(xyz[c14]-xyz[oO]))
print(f'{c14} {oO} {r:.2f}')
PY
)"
echo "--- I552A: donor=$D acceptor=$A r_DA_eq=$R ---" | tee -a "$LOG"

for rep in 1 2 3; do
    echo "--- $(date -Iseconds)  I552A PMF rep=$rep ---" | tee -a "$LOG"
    "$SL" "$DRV" I552A "${PRE}_solv.prmtop" "${PRE}_eq.rst7" "$D" "$A" "$R" "$rep" 2>&1 | tee -a "$LOG"
done

echo "=== I552A pipeline done: $(date -Iseconds) ===" | tee -a "$LOG"
