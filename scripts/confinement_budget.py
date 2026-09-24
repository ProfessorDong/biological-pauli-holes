#!/usr/bin/env python3
"""Where does the closed-shell confinement of the transferring hydrogen come from?

The universality test (confinement_universality.py) establishes that a hard wall
whose closest approach to a bound state is d perturbs that state by

      dE  =  A (kappa d)^p exp(-2 kappa d),

with the exponential rate 2 kappa fixed by the bound state's own decay constant
and only the polynomial prefactor set by the wall's shape. For a hydrogen 1s
state kappa = 1/a0, so each closed-shell atom near the transferring hydrogen
enters with weight

      w_i  =  exp(-2 d_i / a0),          a0 = 0.5292 A.

That weight falls by e^2 = 7.39 for every 0.53 A of extra distance, which makes
it a savage discriminator between atoms at bonding range and atoms across a
cavity. This script evaluates the budget on the real SLO geometries, namely the
clamped snapshots on which the SAPT k_exch values of the manuscript were
computed, and attributes it to chemically meaningful groups.

WHY THIS IS THE POINT
  The cavity-radius analysis excluded the acceptor and the substrate framework
  from the wall set, on the grounds that a reaction partner is not a cage. That
  is a defensible definition of a cavity radius but it is the wrong question for
  the physics: the transferring hydrogen's electron does not know which of its
  neighbors we have labeled reactants. It feels all of them, with the weight
  above. Reporting the full budget is what shows that the confinement is set at
  bonding range, where mutation of a distal side chain cannot reach.

TWO BUGS IN THE FIRST VERSION OF THIS SCRIPT, BOTH FIXED HERE
  (1) Residues were selected by PDB number. Amber renumbers residues
      sequentially from zero, so the JBC cavity positions matched unrelated
      residues 15 A away. Mutation sites are now found by DIFFING the mutant
      topology against the wild type, which needs no numbering convention.
  (2) parmed reports several MCPB metal-site atoms as yttrium (Z = 39). Elements
      are therefore taken from atomic mass, which is correct in the topology.
      Verified: HD*:NE2 mass 14.01, IE1:OXT and OH1:O mass 16.00.

Usage: confinement_budget.py     (needs the slomd env for parmed)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import parmed

ROOT = Path(_REPO)
MD = ROOT / 'md' / 'mcpb'
RG = ROOT / 'results' / 'reactive_geometry'
OUT = ROOT / 'results' / 'confinement_budget.json'

A0 = 0.529177210903

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
# peptide backbone heavy atoms, which a side-chain mutation does not remove
BACKBONE = {'N', 'CA', 'C', 'O', 'OXT'}


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


def coords(path):
    return np.asarray(parmed.load_file(str(path)).coordinates).reshape(-1, 3)


def mutation_site(prm_wt, prm_mut):
    """Residue index of the single ILE/LEU/VAL -> ALA substitution, found by
    comparing the two residue-name sequences. Returns the index in the MUTANT."""
    wt = [r.name for r in prm_wt.residues]
    mu = [r.name for r in prm_mut.residues]
    hits = [i for i in range(min(len(wt), len(mu)))
            if wt[i] != mu[i] and mu[i] == 'ALA']
    if len(hits) != 1:
        raise RuntimeError(f'expected exactly one X->ALA site, found {hits[:5]}')
    return hits[0]


def run():
    prm_wt = parmed.load_file(str(MD / 'SLO_sub_solv.prmtop'))

    # locate every JBC mutation site once, in the wild-type numbering
    sites = {}
    for tag, info in SYSTEMS.items():
        if tag == 'WT':
            continue
        prm_m = parmed.load_file(str(MD / f"{info['prm']}.prmtop"))
        i = mutation_site(prm_wt, prm_m)
        sites[tag] = i
        print(f'  {tag:6s} mutation at residue index {i:5d}  '
              f'({prm_wt.residues[i].name} -> ALA)')
    cavity_idx = set(sites.values())
    print()

    results = {}
    for tag, info in SYSTEMS.items():
        prm = parmed.load_file(str(MD / f"{info['prm']}.prmtop"))
        els = elements(prm)
        iH, iC = info['xferH'], info['donor']
        iFe = next(j for j in range(len(els)) if els[j] == 'FE')

        donor_bonded = set()
        for b in prm.atoms[iC].bonds:
            donor_bonded.add(b.atom1.idx)
            donor_bonded.add(b.atom2.idx)
        donor_bonded -= {iH, iC}

        for clamp, cname in [('r255', 'reactive'), ('r340', 'reference')]:
            f = RG / f'{tag}_{clamp}_final.rst7'
            if not f.exists():
                continue
            pos = coords(f)

            dFe = np.linalg.norm(pos - pos[iFe], axis=1)
            fe_shell = {j for j in range(len(els))
                        if els[j] in CLOSED_SHELL and 0 < dFe[j] < 2.6}
            cand = [a.idx for a in prm.atoms
                    if a.residue.name.upper().startswith('OH') and els[a.idx] == 'O']
            iO = int(cand[int(np.argmin(dFe[cand]))]) if cand else -1
            fe_shell.discard(iO)

            heavy = np.array([j for j in range(len(els))
                              if els[j] in CLOSED_SHELL and j != iC])
            d = np.linalg.norm(pos[heavy] - pos[iH], axis=1)
            w = np.exp(-2.0 * d / A0)

            def cat_of(j):
                if j == iO:
                    return 'acceptor hydroxide O'
                if j in fe_shell:
                    return 'cofactor first shell'
                a = prm.atoms[j]
                if a.residue.idx == prm.atoms[iC].residue.idx or j in donor_bonded:
                    return 'substrate framework'
                if a.residue.name.upper() in ('WAT', 'HOH', 'TIP3'):
                    return 'water'
                if a.residue.idx in cavity_idx:
                    # A cavity mutation to alanine removes the side chain beyond CB and leaves
                    # the backbone in place, so the atoms a mutation can actually remove are the
                    # side-chain heavy atoms. The original category assigned the WHOLE residue,
                    # backbone N, CA, C and O included, while calling itself a side chain. Both
                    # are reported now: the side-chain mask is the honest answer to "what can
                    # mutagenesis reach", and the whole-residue figure is kept as a sensitivity
                    # so the earlier number remains reproducible.
                    if a.name.strip().upper() in BACKBONE:
                        return 'JBC cavity backbone'
                    return 'JBC cavity side chain'
                return 'other protein'

            cat_w, cat_d = defaultdict(float), defaultdict(lambda: np.inf)
            for j, dj, wj in zip(heavy, d, w):
                c = cat_of(int(j))
                cat_w[c] += float(wj)
                cat_d[c] = min(cat_d[c], float(dj))
            total = sum(cat_w.values())

            order = np.argsort(-w)[:10]
            top = [dict(atom=f'{prm.atoms[heavy[k]].residue.name}'
                             f'{prm.atoms[heavy[k]].residue.idx}:'
                             f'{prm.atoms[heavy[k]].name}',
                        d_A=float(d[k]), fraction=float(w[k] / total))
                   for k in order]

            # cumulative weight inside a shell radius
            cum = {f'{R:.1f}': float(w[d <= R].sum() / total)
                   for R in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0)}

            # Tail beyond 8 A, computed from the FULL atom set before any
            # truncation.  The per-atom dump below is cut at 8 A for the figure,
            # so this fraction CANNOT be recovered from it afterwards; the table
            # column that reports cutoff independence has to be computed here or
            # it is zero by construction.
            tail_beyond_8A = float(w[d > 8.0].sum() / total)
            n_beyond_8A = int((d > 8.0).sum())

            # per-atom dump inside 8 A, for the figure only
            near = d <= 8.0
            per_atom = dict(
                d_A=[float(v) for v in d[near]],
                weight=[float(v) for v in w[near]],
                category=[cat_of(int(j)) for j in heavy[near]])

            results[f'{tag}_{clamp}'] = dict(
                system=tag, clamp=cname, total_weight=float(total),
                categories={c: dict(fraction=cat_w[c] / total,
                                    d_min_A=cat_d[c]) for c in cat_w},
                cumulative_fraction_within=cum, top_atoms=top,
                tail_beyond_8A=tail_beyond_8A, n_beyond_8A=n_beyond_8A,
                per_atom=per_atom)

            print(f'{tag}  ({cname} clamp)   total W = {total:.4e}')
            for c, wc in sorted(cat_w.items(), key=lambda kv: -kv[1]):
                print(f'    {c:24s} {100*wc/total:9.5f} %   nearest {cat_d[c]:5.2f} A')
            print(f'    cumulative within 2.5 A: {100*cum["2.5"]:.3f} %,  '
                  f'within 3.0 A: {100*cum["3.0"]:.3f} %')
            print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1, default=float))
    print(f'wrote {OUT}')
    return results


if __name__ == '__main__':
    import os
    import sys
    print('Exponential confinement budget of the transferring hydrogen')
    print('weight w = exp(-2 d / a0), a0 = 0.5292 A\n')
    run()
    # parmed holds many topologies open here and SIGSEGVs during interpreter
    # teardown on this machine, the same failure mode documented for OpenMM in
    # the project notes. The results are already on disk at this point, so skip
    # teardown rather than lose them.
    sys.stdout.flush()
    os._exit(0)
