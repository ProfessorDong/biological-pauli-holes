#!/usr/bin/env python3
"""Build a truncation-mutant protein PDB from SLO_mcpbpy_capped.pdb by side-chain trimming.
 Ile/Leu -> Ala: keep backbone + CB only (N,H,CA,HA,CB,C,O + termini), rename to ALA;
 tleap rebuilds HB1/HB2/HB3. Residues are addressed by 1-based file order (matches the
 parmed residue index used to map orig SLO-1 numbering -> the renumbered structure).
 Usage: python make_mutant.py <tag> <resorder1> [<resorder2> ...]
   e.g. python make_mutant.py I553A 532      (ILE532 = orig I553)
        python make_mutant.py L754A 733      (LEU733 = orig L754)
        python make_mutant.py DM  525 733    (L546A/L754A double mutant)"""
import sys
tag=sys.argv[1]; targets={int(x) for x in sys.argv[2:]}
KEEP={'N','H','CA','HA','CB','C','O','OXT','H1','H2','H3'}  # ALA backbone + CB
lines=open('SLO_mcpbpy_capped.pdb').read().splitlines()
out=[]; ridx=0; prev=None; mutated=[]
for l in lines:
    if l[:6] in ('ATOM  ','HETATM'):
        key=(l[17:20],l[21],l[22:26])
        if key!=prev:
            ridx+=1; prev=key
        if ridx in targets:
            an=l[12:16].strip()
            if an not in KEEP:
                continue  # drop side-chain atom
            newl=l[:17]+'ALA'+l[20:]   # rename residue to ALA
            out.append(newl)
            if l[17:20].strip()!='ALA' and (l[17:20],ridx) not in [(m[0],m[1]) for m in mutated]:
                mutated.append((l[17:20].strip(),ridx))
        else:
            out.append(l)
    else:
        out.append(l)
open(f'SLO_{tag}_capped.pdb','w').write("\n".join(out)+"\n")
print(f"  {tag}: mutated residues (order#) {[f'{n}{i}->ALA' for n,i in mutated]} -> SLO_{tag}_capped.pdb")
