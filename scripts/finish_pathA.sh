#!/bin/bash
# Finish: equilibrate + PMFs for V750A/I538A/L546A. Skips WT rep-1 retry (14/15 is fine).
# equilibrate_sub.py now writes .rst7 first (before the crashy PDB write) then os._exit(0).
_REPO="${PAULI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

set -euo pipefail
SL="${SLOMD_PYTHON:-/home/liang/anaconda3/envs/slomd/bin/python}"
DRV=${_REPO}/scripts/umbrella_driver.py
MIN=${_REPO}/md/mcpb/minimize_sub.py
EQ=${_REPO}/md/mcpb/equilibrate_sub.py
MCPB=${_REPO}/md/mcpb
LOG=${_REPO}/results/umbrella/pathA_followup.log
cd "$MCPB"
echo "=== PATH A FINISH v2 start: $(date -Iseconds) ===" | tee -a "$LOG"

resolve_da () {
    tag=$1
    "$SL" - <<PY
import parmed as p, numpy as np
from openmm.app import AmberInpcrdFile
prm = p.load_file('$MCPB/SLO_${tag}_sub_solv.prmtop')
c14 = [a for a in prm.atoms if a.residue.name=='LIG' and a.name=='C14'][0].idx
oO  = [a for a in prm.atoms if a.residue.name=='OH1' and a.name=='O'][0].idx
rst = AmberInpcrdFile('$MCPB/SLO_${tag}_sub_eq.rst7')
xyz = np.array([[v.x,v.y,v.z] for v in rst.positions])
r = float(np.linalg.norm(xyz[c14]-xyz[oO]))
print(f"{c14} {oO} {r:.2f}")
PY
}

run_mutant () {
    tag=$1
    PRE="SLO_${tag}_sub"
    # Retry minimize + equilibrate a few times to survive the intermittent OpenMM+CUDA
    # SIGSEGV on subprocess launch.
    for attempt in 1 2 3 4 5; do
        if [ -f "$MCPB/${PRE}_min_xyz.npy" ]; then break; fi
        echo "--- $(date -Iseconds)  minimize $tag (attempt $attempt) ---" | tee -a "$LOG"
        "$SL" "$MIN" "$PRE" 2>&1 | tee -a "$LOG" || true
        sleep 3
    done
    [ -f "$MCPB/${PRE}_min_xyz.npy" ] || { echo "[FATAL] $tag minimize failed 5x" | tee -a "$LOG"; return 1; }
    for attempt in 1 2 3 4 5; do
        if [ -f "$MCPB/${PRE}_eq.rst7" ]; then break; fi
        echo "--- $(date -Iseconds)  equilibrate $tag (1 ns NPT, attempt $attempt) ---" | tee -a "$LOG"
        "$SL" "$EQ" 1.0 "$PRE" 2>&1 | tee -a "$LOG" || true
        sleep 3
    done
    [ -f "$MCPB/${PRE}_eq.rst7" ] || { echo "[FATAL] $tag equilibrate failed 5x" | tee -a "$LOG"; return 1; }
    read -r D A R <<< "$(resolve_da $tag)"
    echo "--- $tag: donor=$D acceptor=$A r_DA_eq=$R ---" | tee -a "$LOG"
    for rep in 1 2 3; do
        echo "--- $(date -Iseconds)  $tag PMF rep=$rep ---" | tee -a "$LOG"
        "$SL" "$DRV" "$tag" "${PRE}_solv.prmtop" "${PRE}_eq.rst7" "$D" "$A" "$R" "$rep" 2>&1 | tee -a "$LOG"
    done
}

run_mutant V750A
run_mutant I538A
run_mutant L546A

echo "=== PATH A FINISH v2 done: $(date -Iseconds) ===" | tee -a "$LOG"
