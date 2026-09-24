#!/bin/bash
# SAPT stage of the reactive-geometry test: k_exch^bio at both clamps.
#
# Reuses the audited main-text 2.8 descriptor without modification. The only
# difference from the published calculation is the donor-acceptor geometry the
# fragments are taken from: here they come from the clamped runs at r_DA = 2.55 A
# (reactive) and 3.40 A (equilibrium reference) rather than from an unrestrained
# umbrella window. Fragments are capped isobutane (Leu wall) or capped acetamide
# (Asn wall), both neutral closed-shell, with a methane donor probe; SAPT0 /
# jun-cc-pVDZ, 5-point scan, k_exch from the quadratic coefficient.
#
# Requires BOTH the colvar and the final rst7 of each clamp (three clamps in the
# first pass streamed a full colvar but died in OpenMM teardown before writing
# the restart file, and a colvar-only completion test scored them as done).
#
# 7 systems x 2 clamps x ~10-15 min => ~2-3 h on CPU.
_REPO="${PAULI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

set -uo pipefail
ROOT=${_REPO}
SL=/home/liang/anaconda3/envs/slomd/bin/python     # parmed + openmm
P4=/home/liang/anaconda3/envs/pauli/bin/python     # psi4
RG="$ROOT/results/reactive_geometry"
LOG="$RG/sapt_campaign.log"
echo "=== clamped SAPT campaign start $(date -Iseconds) ===" | tee -a "$LOG"

for CLAMP in r255 r340; do
  for TAG in WT V750A I552A I538A L754A L546A I553A; do
    RES="$RG/${TAG}_${CLAMP}_native_result.json"
    if [[ -f "$RES" ]]; then echo "  skip $TAG $CLAMP (done)" | tee -a "$LOG"; continue; fi
    if [[ ! -f "$RG/${TAG}_${CLAMP}_final.rst7" ]]; then
      echo "  SKIP $TAG $CLAMP: no final.rst7 (clamp incomplete)" | tee -a "$LOG"; continue
    fi
    echo "" | tee -a "$LOG"
    echo "--- $(date -Iseconds) $TAG $CLAMP ---" | tee -a "$LOG"
    "$SL" "$ROOT/scripts/extract_clamped_fragment.py" "$TAG" "$CLAMP" 2>&1 | tee -a "$LOG"
    [[ -f "$RG/${TAG}_${CLAMP}_geometry.json" ]] || { echo "  extract FAILED" | tee -a "$LOG"; continue; }
    CLAMP="$CLAMP" TAG="$TAG" "$P4" "$ROOT/scripts/sapt_clamped_fragment.py" "$TAG" "$CLAMP" 2>&1 | tee -a "$LOG" | tail -8
  done
done

echo "" | tee -a "$LOG"
echo "=== clamped SAPT campaign done $(date -Iseconds) ===" | tee -a "$LOG"

# collate into the two files the pre-registered analysis expects
"$SL" - <<'PY' 2>&1 | tee -a "$LOG"
import json, glob, os
from pathlib import Path
RG = Path('${_REPO}/results/reactive_geometry')
for clamp in ('r255','r340'):
    out = {}
    for f in sorted(RG.glob(f'*_{clamp}_native_result.json')):
        d = json.loads(f.read_text())
        out[d['TAG']] = d['k_exch_Nm']
    (RG/f'kexch_{clamp}.json').write_text(json.dumps(out, indent=2))
    print(f'  kexch_{clamp}.json: {len(out)}/7 systems')
PY
