#!/bin/bash
# Runs AFTER Path A (rep 2+3 of WT/I553A/L754A/DM) completes.
#   (1) rep 1 rerun of the 4 existing systems (bench check — Blackwell vs old 4060)
#   (2) minimize + equilibrate the 3 fresh mutants (V750A, I538A, L546A)
#   (3) 3-replica DAD PMFs for the 3 fresh mutants
# ~17 GPU-h total on the Blackwell.
_REPO="${PAULI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

set -euo pipefail
SL=/home/liang/anaconda3/envs/slomd/bin/python
DRV=${_REPO}/scripts/umbrella_driver.py
MIN=${_REPO}/md/mcpb/minimize_sub.py
EQ=${_REPO}/md/mcpb/equilibrate_sub.py
MCPB=${_REPO}/md/mcpb
RES=${_REPO}/results/umbrella

cd "$MCPB"
LOG="$RES/pathA_followup.log"
echo "=== PATH A FOLLOWUP start: $(date -Iseconds) ===" | tee -a "$LOG"

# ---------- (1) rep 1 rerun of existing 4 systems into _rerun subdirs (do NOT clobber cached) ----------
# The driver hardcodes the outdir based on tag + suffix. To avoid clobbering the cached rep 1,
# we redirect by symlinking tag -> tag_rerun inside results/umbrella.
rerun_rep1 () {
    tag=$1; prm=$2; rst=$3; d=$4; a=$5; r=$6
    # move existing tag dir aside, run rep 1 fresh, then rename outputs
    tmpname="${tag}_pathAcheck"
    if [ -d "$RES/$tmpname" ]; then
        echo "  [$tag] already have $tmpname, skipping" | tee -a "$LOG"; return
    fi
    # temporarily hide the cached
    if [ -d "$RES/$tag" ]; then mv "$RES/$tag" "$RES/${tag}_CACHED_BACKUP"; fi
    echo "--- $(date -Iseconds)  $tag rep 1 (bench check on Blackwell) ---" | tee -a "$LOG"
    "$SL" "$DRV" "$tag" "$prm" "$rst" "$d" "$a" "$r" 1 2>&1 | tee -a "$LOG"
    # move fresh into _pathAcheck, restore cached
    mv "$RES/$tag" "$RES/$tmpname"
    if [ -d "$RES/${tag}_CACHED_BACKUP" ]; then mv "$RES/${tag}_CACHED_BACKUP" "$RES/$tag"; fi
}
rerun_rep1 WT    SLO_sub_solv.prmtop       SLO_sub_eq.rst7       13002 12985 5.07
rerun_rep1 I553A SLO_I553A_sub_solv.prmtop SLO_I553A_sub_eq.rst7 12993 12976 4.75
rerun_rep1 L754A SLO_L754A_sub_solv.prmtop SLO_L754A_sub_eq.rst7 12993 12976 5.13
rerun_rep1 DM    SLO_DM_sub_solv.prmtop    SLO_DM_sub_eq.rst7    12984 12967 5.53

# ---------- (2) minimize + equilibrate the 3 new mutants ----------
# minimize_sub.py <PRE>       (default SLO_sub)
# equilibrate_sub.py <ns_npt> <PRE>   (default 1.0 SLO_sub)
for tag in V750A I538A L546A; do
    PRE="SLO_${tag}_sub"
    if [ -f "$MCPB/${PRE}_eq.rst7" ]; then
        echo "--- [$tag] already equilibrated, skip ---" | tee -a "$LOG"; continue
    fi
    echo "--- $(date -Iseconds)  minimize $tag (${PRE}) ---" | tee -a "$LOG"
    "$SL" "$MIN" "$PRE" 2>&1 | tee -a "$LOG"
    echo "--- $(date -Iseconds)  equilibrate $tag ---" | tee -a "$LOG"
    "$SL" "$EQ" 1.0 "$PRE" 2>&1 | tee -a "$LOG"
done

# ---------- (3) 3-replica DAD PMFs for the new mutants ----------
# Donor/acceptor indices for the new mutants: identical residue pattern to the existing
# single mutants (I553A/L754A) since side-chain trimming only removes a few atoms, but
# the substrate/Fe indices depend on which residues were trimmed. Determined from the
# fresh prmtop by picking C11(substrate) and the Fe-OH oxygen.
resolve_indices () {
    tag=$1
    "$SL" - <<PY
import parmed as p
s = p.load_file('$MCPB/SLO_${tag}_sub_solv.prmtop')
c14 = [a for a in s.atoms if a.residue.name=='LIG' and a.name=='C14'][0]
o   = [a for a in s.atoms if a.residue.name=='OH1' and a.name=='O'][0]
print(f"{c14.idx} {o.idx}")
PY
}
for tag in V750A I538A L546A; do
    read -r D A <<< "$(resolve_indices $tag)"
    # take equilibrated r_DA from the eq restart
    RDA=$("$SL" - <<PY
import parmed as p, numpy as np
s = p.load_file('$MCPB/SLO_${tag}_sub_solv.prmtop', xyz='$MCPB/SLO_${tag}_sub_eq.rst7')
d,a=$D,$A
r = np.linalg.norm(s.coordinates[d]-s.coordinates[a])
print(f"{r:.2f}")
PY
)
    echo "--- $tag: donor=$D acceptor=$A r_eq=$RDA ---" | tee -a "$LOG"
    for rep in 1 2 3; do
        echo "--- $(date -Iseconds)  $tag rep=$rep ---" | tee -a "$LOG"
        "$SL" "$DRV" "$tag" SLO_${tag}_sub_solv.prmtop SLO_${tag}_sub_eq.rst7 "$D" "$A" "$RDA" "$rep" 2>&1 | tee -a "$LOG"
    done
done

echo "=== PATH A FOLLOWUP done: $(date -Iseconds) ===" | tee -a "$LOG"
