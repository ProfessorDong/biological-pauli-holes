#!/bin/bash
# Repair the two I538A windows that were left in a partial state by the pre-fix writer bug:
#   * I538A rep 1 window 4.95 (colvar present, rst7 missing; driver would skip it)
#   * I538A rep 2 window 3.20 (both colvar and rst7 missing; driver will try to run it)
# With the current writer (0-d array fix + atomic rename) both should produce complete outputs.
_REPO="${PAULI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

set -euo pipefail
SL="${SLOMD_PYTHON:-/home/liang/anaconda3/envs/slomd/bin/python}"
DRV=${_REPO}/scripts/umbrella_driver.py
MCPB=${_REPO}/md/mcpb
RES=${_REPO}/results/umbrella
LOG=$RES/pathA_followup.log

# --- delete the partial 4.95 outputs so the driver's colvar-existence check triggers a rerun ---
rm -f "$RES/I538A/win_4.95_colvar.dat" "$RES/I538A/win_4.95_final.rst7" "$RES/I538A/win_4.95.rst7.part"
# --- clear FAILED_WINDOWS.txt so the driver's colvar check is the sole gate ---
rm -f "$RES/I538A/FAILED_WINDOWS.txt" "$RES/I538A_rep2/FAILED_WINDOWS.txt"

# resolve donor/acceptor + equilibrated r_DA
read -r D A R <<< "$($SL - <<PY
import parmed as p, numpy as np
from openmm.app import AmberInpcrdFile
prm = p.load_file('$MCPB/SLO_I538A_sub_solv.prmtop')
c14 = [a for a in prm.atoms if a.residue.name=='LIG' and a.name=='C14'][0].idx
oO  = [a for a in prm.atoms if a.residue.name=='OH1' and a.name=='O'][0].idx
rst = AmberInpcrdFile('$MCPB/SLO_I538A_sub_eq.rst7')
xyz = np.array([[v.x,v.y,v.z] for v in rst.positions])
r = float(np.linalg.norm(xyz[c14]-xyz[oO]))
print(f'{c14} {oO} {r:.2f}')
PY
)"

echo "=== REPAIR I538A rep 1 (win 4.95) + rep 2 (win 3.20) start: $(date -Iseconds) ===" | tee -a "$LOG"
cd "$MCPB"
# rep 1: driver will iterate, skip 14 with colvar, and only run 4.95
"$SL" "$DRV" I538A SLO_I538A_sub_solv.prmtop SLO_I538A_sub_eq.rst7 "$D" "$A" "$R" 1 2>&1 | tee -a "$LOG"
# rep 2: driver will iterate, skip 14 with colvar, and only run 3.20
"$SL" "$DRV" I538A SLO_I538A_sub_solv.prmtop SLO_I538A_sub_eq.rst7 "$D" "$A" "$R" 2 2>&1 | tee -a "$LOG"
echo "=== REPAIR done: $(date -Iseconds) ===" | tee -a "$LOG"

# Confirm both
for tag in "I538A/win_4.95" "I538A_rep2/win_3.20"; do
    if [ -f "$RES/$tag"_colvar.dat ] && [ -f "$RES/$tag"_final.rst7 ]; then
        echo "[OK] $tag: colvar + rst7 both present" | tee -a "$LOG"
    else
        echo "[FAIL] $tag: still incomplete" | tee -a "$LOG"
    fi
done
