#!/bin/bash
# CORRECTED reactant proton-PES campaign (supersedes pcet_pilot_B34_dense.sh).
#
# Differences from the superseded campaign:
#   1. Uses scripts/qm_cluster.py::build_cluster -> every severed bond capped with
#      H, formal charge +1 from an explicit fragment table, electron parity
#      asserted, cluster validated against the parent topology (0 dropped bonds).
#      The old builder emitted 35 severed bonds with zero caps.
#   2. mult = 6 (Fe(III) high-spin d5, S = 5/2), <S^2> expected 8.75. The old runs
#      used mult 5 and drifted to <S^2> ~ 11-12.3.
#   3. Records <S^2> and the SCF-converged flag at EVERY point; contamination
#      > 0.5 is flagged loudly. See AUDIT_2026-08-10.md.
#
# 5 stratified WT snapshots x 15 points. Expect ~25-35 min/point => ~35 h total.
_REPO="${PAULI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

set -uo pipefail
source ${_REPO}/setup/env.sh

ROOT=${_REPO}
PY="${SLOMD_PYTHON:-/home/liang/anaconda3/envs/slomd/bin/python}"
SCR="$ROOT/scripts/proton_pes_scanner.py"
PRM="$ROOT/md/mcpb/SLO_sub_solv.prmtop"
UMB="$ROOT/results/umbrella"
OUT="$ROOT/results/pcet_reactant_v2"
mkdir -p "$OUT"

LOG="$OUT/campaign.log"
echo "=== corrected reactant campaign start $(date -Iseconds) ===" | tee -a "$LOG"

DONOR=13002; ACCEPTOR=12985; XFERH=13031

for WIN in 2.70 3.20 3.45 3.70 3.95; do
    SEED="$UMB/WT/win_${WIN}_final.rst7"
    TAG="wt_win${WIN//./}"
    JSON="${OUT}/${TAG}_scan.json"
    if [[ -f "$JSON" ]]; then
        echo "  skip win=$WIN (already complete)" | tee -a "$LOG"; continue
    fi
    echo "" | tee -a "$LOG"
    echo "--- $(date -Iseconds) win=$WIN ---" | tee -a "$LOG"
    for att in 1 2 3; do
        [[ -f "$JSON" ]] && break
        echo "  attempt $att" | tee -a "$LOG"
        timeout 43200 "$PY" "$SCR" \
            --prmtop "$PRM" --rst7 "$SEED" \
            --xferH $XFERH --donor $DONOR --acceptor $ACCEPTOR \
            --npts 15 --mode small \
            --out "$OUT/$TAG" 2>&1 | tee -a "$LOG" | tail -40
        [[ -f "$JSON" ]] && { echo "  ok win=$WIN (attempt $att)" | tee -a "$LOG"; break; }
        echo "  attempt $att failed; retry" | tee -a "$LOG"; sleep 30
    done
    [[ ! -f "$JSON" ]] && echo "  UNRECOVERED win=$WIN after 3 attempts" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== corrected reactant campaign done $(date -Iseconds) ===" | tee -a "$LOG"
ls "$OUT"/*_scan.json 2>/dev/null | wc -l | xargs -I {} echo "  scans complete: {} / 5" | tee -a "$LOG"

# spin-state audit across every point of every scan
"$PY" - <<'EOF' 2>&1 | tee -a "$LOG"
import json, glob
bad = tot = 0
print('\n  spin-state audit (expect <S^2> = 8.75):')
for f in sorted(glob.glob('${_REPO}/results/pcet_reactant_v2/*_scan.json')):
    d = json.load(open(f))
    for p in d['points']:
        tot += 1
        s2 = p.get('S2')
        if s2 is None or abs(s2 - p.get('S2_expected', 8.75)) > 0.5:
            bad += 1
            print(f"    {f.split('/')[-1]} pt{p['point']:02d} alpha={p['alpha']:.3f} "
                  f"<S^2>={s2} conv={p.get('scf_converged')}")
print(f'  {tot - bad}/{tot} points on the target Fe(III) high-spin d5 surface')
EOF
