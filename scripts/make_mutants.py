#!/usr/bin/env python3
"""
make_mutants.py -- build SLO truncation mutants from WT 1F8N by EXACT side-chain
atom deletion (coordinate-preserving; the most defensible static model for
cavity-creating mutations; no force field, no clashes, no rotamer guessing).

Subset relationships used:
   ILE (N CA C O CB CG1 CG2 CD1):  ->VAL del CD1 ; ->ALA del CG1,CG2,CD1 ; ->GLY del CB,CG1,CG2,CD1
   LEU (N CA C O CB CG  CD1 CD2):  ->ALA del CG,CD1,CD2

Jobs (resid, from, to) -> output file:
   553 ILE V/A/G   (Meyer&Klinman 2005 series; DISTAL ~10 A from transfer H)
   754 LEU A       (CONTACT ~5 A from transfer H -- positive control)
   546 LEU A       (intermediate ~8 A)
Run:  python scripts/make_mutants.py
"""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLO  = os.path.join(ROOT, "pauli_data", "slo")
WT   = os.path.join(SLO, "1F8N.pdb")
CHAIN= "A"

# (resid, from_resname, to_resname, atoms_to_delete, out_stem)
JOBS = [
    (553, "ILE", "VAL", {"CD1"},                    "I553V_model"),
    (553, "ILE", "GLY", {"CB","CG1","CG2","CD1"},   "I553G_model"),
    # JBC 2019 self-consistent single-mutant series (all ->Ala/Val truncations):
    (553, "ILE", "ALA", {"CG1","CG2","CD1"},        "I553A_model"),
    (552, "ILE", "ALA", {"CG1","CG2","CD1"},        "I552A_model"),
    (538, "ILE", "ALA", {"CG1","CG2","CD1"},        "I538A_model"),
    (839, "ILE", "ALA", {"CG1","CG2","CD1"},        "I839A_model"),
    (750, "VAL", "ALA", {"CG1","CG2"},              "V750A_model"),
    (754, "LEU", "ALA", {"CG","CD1","CD2"},         "L754A_model"),
    (546, "LEU", "ALA", {"CG","CD1","CD2"},         "L546A_model"),
]

def build(resid, frm, to, remove, stem):
    out = os.path.join(SLO, stem + ".pdb")
    nd = nr = 0
    with open(WT) as fi, open(out, "w") as fo:
        for ln in fi:
            if ln.startswith(("ATOM  ", "HETATM")):
                atom = ln[12:16].strip(); chain = ln[21]
                rs = ln[22:26].strip(); rn = ln[17:20].strip()
                if chain == CHAIN and rs == str(resid) and rn == frm:
                    if atom in remove:
                        nd += 1; continue
                    ln = ln[:17] + f"{to:>3}" + ln[20:]; nr += 1
            fo.write(ln)
    print(f"  {stem:14s} {frm}{resid}->{to}: renamed {nr}, deleted {nd}")

def build_multi(muts, stem):
    """Apply several truncations in one pass (e.g. the L546A/L754A double mutant).
    muts: list of (resid, from_resname, to_resname, atoms_to_delete)."""
    out = os.path.join(SLO, stem + ".pdb"); nd = nr = 0
    with open(WT) as fi, open(out, "w") as fo:
        for ln in fi:
            if ln.startswith(("ATOM  ", "HETATM")):
                atom = ln[12:16].strip(); chain = ln[21]
                rs = ln[22:26].strip(); rn = ln[17:20].strip()
                drop = False
                for (r, f, t, rem) in muts:
                    if chain == CHAIN and rs == str(r) and rn == f:
                        if atom in rem: nd += 1; drop = True
                        else: ln = ln[:17] + f"{t:>3}" + ln[20:]; nr += 1
                        break
                if drop: continue
            fo.write(ln)
    print(f"  {stem:14s} double mutant: renamed {nr}, deleted {nd}")

def main():
    if not os.path.exists(WT): raise SystemExit(f"missing {WT}")
    print("Building SLO truncation mutants from 1F8N:")
    for j in JOBS: build(*j)
    build_multi([(546, "LEU", "ALA", {"CG","CD1","CD2"}),
                 (754, "LEU", "ALA", {"CG","CD1","CD2"})], "L546A_L754A_model")
    print("done (residue altLocs retained; analysis deduplicates altLocs).")

if __name__ == "__main__":
    main()
