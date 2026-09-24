#!/bin/bash
# Reactive-geometry Pauli-wall campaign.
#
# QUESTION
# The exchange-repulsion curvature k_exch^bio reported in main-text 2.8 was
# evaluated at the equilibrium near-attack geometry (r_DA ~ 3.4 A) and does NOT
# explain the isotope-effect ladder. The Franck-Condon analysis says the reaction
# does not occur there: it requires r_DA ~ 2.55 A, deep in the compressive tail.
# So the Pauli wall may simply have been measured in the wrong place.
#
# DESIGN (deliberately NON-CIRCULAR)
# The per-mutant reactive distances from the Franck-Condon fit (2.55-2.60 A) were
# DERIVED FROM the KIEs. Evaluating each mutant at its own inferred distance and
# then correlating against KIE would be circular and would manufacture a
# correlation. Instead r_DA is CLAMPED AT THE SAME VALUE for every mutant, so the
# only thing that varies between systems is the cavity wall itself. The test is
# then: at identical reactive compression, does the wall's exchange curvature
# track the KIE ladder?
#
# Two clamps per system, so the comparison is apples-to-apples:
#   REACTIVE    r_DA = 2.55 A   (the Franck-Condon reactive geometry)
#   REFERENCE   r_DA = 3.40 A   (the equilibrium near-attack geometry, where the
#                                descriptor is already known to fail)
#
# A stiff restraint (k = 150 kcal/mol/A^2) is required: the standard k = 12 used
# for the PMF leaves <r_DA> ~0.5-0.8 A above r0 under compression, because the
# free energy opposes it steeply. 2.55 A is ~5 sigma into the tail of the most
# compressed PMF window, so it is never reached by ordinary sampling. The stiff
# clamp samples the wall's response at fixed compression, which is precisely the
# Pauli-confinement question; it is not intended to produce a free energy.
#
# 7 JBC systems x 2 clamps x 3 ns. Expect ~10-12 min per window => ~3 h.
_REPO="${PAULI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

set -uo pipefail
SL="${SLOMD_PYTHON:-/home/liang/anaconda3/envs/slomd/bin/python}"
WIN=${_REPO}/scripts/umbrella_window.py
cd ${_REPO}/md/mcpb

OUT=${_REPO}/results/reactive_geometry
mkdir -p "$OUT"
LOG="$OUT/campaign.log"
echo "=== reactive-geometry campaign start $(date -Iseconds) ===" | tee -a "$LOG"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tee -a "$LOG"

KSTIFF=150.0

# tag  prmtop  eq_rst7  donor  acceptor
run_clamp () {
    tag=$1; prm=$2; rst=$3; d=$4; a=$5; r0=$6
    lbl="${tag}_r${r0/./}"
    if [[ -f "$OUT/${lbl}_colvar.dat" && -f "$OUT/${lbl}_final.rst7" ]]; then
        echo "  skip $lbl (exists)" | tee -a "$LOG"; return
    fi
    echo "--- $(date -Iseconds) $tag clamp r_DA=$r0 A ---" | tee -a "$LOG"
    for att in 1 2 3; do
        [[ -f "$OUT/${lbl}_colvar.dat" && -f "$OUT/${lbl}_final.rst7" ]] && break
        timeout 5400 "$SL" "$WIN" --prmtop "$prm" --seed "$rst" \
            --donor "$d" --acceptor "$a" --r0 "$r0" --k "$KSTIFF" \
            --ns-eq 0.3 --ns-prod 3.0 --out "$OUT/$lbl" \
            --rngseed $((RANDOM+1)) 2>&1 | tee -a "$LOG" | tail -3
        [[ -f "$OUT/${lbl}_colvar.dat" && -f "$OUT/${lbl}_final.rst7" ]] && { echo "  ok $lbl" | tee -a "$LOG"; break; }
        echo "  attempt $att failed; retry" | tee -a "$LOG"; sleep 20
    done
    [[ -f "$OUT/${lbl}_colvar.dat" && -f "$OUT/${lbl}_final.rst7" ]] || echo "  UNRECOVERED $lbl" | tee -a "$LOG"
}

# --- topology-resolved atom identities (2026-09-17 correction) -----------------
# V750A is offset by three indices from the other mutants. Hardcoding one index
# across variants silently restrained V750A on LIG:C11 and the Ile817 carboxylate
# instead of C14 and the iron-bound hydroxide; nothing raised and the colvar sat on
# target while the intended coordinate stayed near 5.7 A. Resolve, assert, print.
AMAP=${_REPO}/results/atom_map.json
PY_ENV="${PAULI_PYTHON:-/home/liang/anaconda3/envs/pauli/bin/python}"
[[ -f "$AMAP" ]] || { echo "FATAL: $AMAP missing; run scripts/resolve_atom_map.py"; exit 1; }
amap () {   # amap <TAG> <donor_C|acceptor_O>
    "$PY_ENV" -c "import json,sys;m=json.load(open('$AMAP'))[sys.argv[1]];print(m[sys.argv[2]])" "$1" "$2"
}

# seed each clamp from that system's most compressed equilibrated window
U=${_REPO}/results/umbrella
for r0 in 2.55 3.40; do
  for MUT in WT I553A L754A V750A I538A L546A I552A; do
    case $MUT in WT) PRMF=SLO_sub_solv.prmtop;; *) PRMF=SLO_${MUT}_sub_solv.prmtop;; esac
    D=$(amap $MUT donor_C); A=$(amap $MUT acceptor_O)
    DL=$(amap $MUT donor_C_label); AL=$(amap $MUT acceptor_label)
    echo "  [atoms] $MUT donor=$D ($DL)  acceptor=$A ($AL)" | tee -a "$LOG"
    run_clamp $MUT $PRMF "$U/$MUT/win_2.70_final.rst7" "$D" "$A" $r0
  done
done

echo "" | tee -a "$LOG"
echo "=== reactive-geometry campaign done $(date -Iseconds) ===" | tee -a "$LOG"
for r0 in 2.55 3.40; do
  n=$(ls "$OUT"/*_r${r0/./}_colvar.dat 2>/dev/null | wc -l)
  echo "  r_DA=$r0 : $n / 7 systems" | tee -a "$LOG"
done
echo "  achieved <r_DA> per clamp:" | tee -a "$LOG"
for f in "$OUT"/*_colvar.dat; do
  [ -f "$f" ] || continue
  echo "    $(basename "$f" _colvar.dat)  $(head -1 "$f" | grep -o 'mean=[0-9.]*  *std=[0-9.]*')" | tee -a "$LOG"
done
