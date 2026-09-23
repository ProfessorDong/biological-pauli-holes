"""Cavity radius xi0 around the transferring hydrogen in SLO, measured over the FULL system.

Supersedes the wall-fragment-restricted measurement used earlier, which reported the
distance to the nearest atom of the single designated wall side chain (Leu732 or Asn672)
and therefore overestimated xi0 substantially.

DEFINITION (applied here and matched in the cross-system survey)
  xi0 = distance from the transferring hydrogen to the nearest EXTERNAL closed-shell
        heavy atom, where "external" excludes
          (i)   the donor heavy atom the hydrogen is covalently bound to,
          (ii)  atoms bonded to that donor, i.e. the substrate's own molecular framework
                (these are present in any C-H bond and are not a confining cage),
          (iii) the acceptor heavy atom, which is a reaction partner rather than a wall,
          (iv)  open-shell metals, which are outside the closed-shell model entirely,
          (v)   hydrogens.
  Elements are taken from atomic mass, because the MCPB-generated metal-site residues
  carry element fields that parmed cannot infer (it returns 'Y').

Usage: slo_xi0_full.py
"""
import json
import numpy as np
import parmed
from pathlib import Path

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
MD = ROOT / 'md' / 'mcpb'
EF = ROOT / 'results' / 'ensemble_fluctuation'
OUT = ROOT / 'results' / 'pauli_hole_survey'

SYSTEMS = {
    'WT':    dict(prm='SLO_sub_solv',       xferH=13031, donor=13002),
    'I553A': dict(prm='SLO_I553A_sub_solv', xferH=13022, donor=12993),
    'I552A': dict(prm='SLO_I552A_sub_solv', xferH=13022, donor=12993),
    'L754A': dict(prm='SLO_L754A_sub_solv', xferH=13023, donor=12993),
    'V750A': dict(prm='SLO_V750A_sub_solv', xferH=13025, donor=12996),
    'I538A': dict(prm='SLO_I538A_sub_solv', xferH=13022, donor=12993),
    'L546A': dict(prm='SLO_L546A_sub_solv', xferH=13022, donor=12993),
}
MASS = {1.008: 'H', 12.01: 'C', 14.01: 'N', 16.00: 'O', 32.06: 'S', 55.85: 'FE'}
CLOSED_SHELL = {'C', 'N', 'O', 'S'}


def elements(prm):
    out = []
    for a in prm.atoms:
        e = '?'
        for m, sym in MASS.items():
            if abs(a.mass - m) < 0.45:
                e = sym
                break
        out.append(e)
    return out


def acceptor_index(prm, els, pos, iFe):
    """Oxygen of the Fe-bound hydroxide, identified by RESIDUE not by proximity.

    Proximity to Fe is not a safe discriminator: the C-terminal Ile carboxylate OXT
    coordinates the iron more closely (1.66 A) than the hydroxide oxygen (1.77 A), so
    a nearest-O rule excludes the wrong atom and leaves the true acceptor in the wall set.
    """
    cand = [a.idx for a in prm.atoms
            if a.residue.name.upper().startswith('OH') and els[a.idx] == 'O']
    if not cand:
        raise RuntimeError('hydroxide acceptor residue not found')
    if len(cand) > 1:
        d = np.linalg.norm(pos[cand] - pos[iFe], axis=1)
        return int(cand[int(np.argmin(d))])
    return int(cand[0])


def run():
    rows = {}
    for tag, info in SYSTEMS.items():
        prm = parmed.load_file(str(MD / f"{info['prm']}.prmtop"))
        els = elements(prm)
        iH, iC = info['xferH'], info['donor']
        # exclusion set: donor, its bonded neighbours (substrate framework), metals
        excl = {iH, iC}
        for b in prm.atoms[iC].bonds:
            excl.add(b.atom1.idx); excl.add(b.atom2.idx)
        iFe = next(j for j in range(len(els)) if els[j] == 'FE')
        excl.add(iFe)
        keep = np.array([j for j in range(len(els))
                         if els[j] in CLOSED_SHELL and j not in excl])
        for clamp, cname in [('r255', 'reactive'), ('r340', 'reference')]:
            f = EF / f'{tag}_{clamp}_frames.npy'
            if not f.exists():
                continue
            fr = np.load(f)
            iO = acceptor_index(prm, els, fr[0], iFe)
            sel = keep[keep != iO] if iO is not None else keep
            ds, ids = [], []
            for k in range(fr.shape[0]):
                pos = fr[k]
                d = np.linalg.norm(pos[sel] - pos[iH], axis=1)
                j = int(np.argmin(d))
                ds.append(float(d[j])); ids.append(int(sel[j]))
            ds = np.array(ds)
            from collections import Counter
            top = Counter(ids).most_common(2)
            names = [f"{prm.atoms[i].residue.name}{prm.atoms[i].residue.number}:"
                     f"{prm.atoms[i].name} ({n}/{len(ids)})" for i, n in top]
            rows[f'{tag}_{clamp}'] = dict(system=tag, clamp=cname,
                                          xi0_A=float(ds.mean()), sd=float(ds.std(ddof=1)),
                                          n=len(ds), nearest=names,
                                          acceptor=f"{prm.atoms[iO].residue.name}:"
                                                   f"{prm.atoms[iO].name}" if iO else None)
            print(f"  {tag:6s} {cname:9s} xi0 = {ds.mean():.2f} +- {ds.std(ddof=1):.2f} A"
                  f"   nearest: {names[0]}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'slo_xi0_full.json').write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {OUT/'slo_xi0_full.json'}")
    return rows


if __name__ == '__main__':
    print('SLO cavity radius, measured over the full solvated system')
    print('(supersedes the wall-fragment-restricted values)\n')
    run()
