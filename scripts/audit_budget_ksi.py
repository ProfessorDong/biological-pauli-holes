#!/usr/bin/env python3
"""Range-weight budget of the ketosteroid isomerase (KSI) oxyanion-hole hydrogen bond, with the
hydrogen on the residue that donates it.

scripts/confinement_budget_pdb.py builds the KSI entries of the cross-enzyme comparison on 2PZV
(chains A-D) with the hydrogen on the ligand phenol oxygen (IPH O1), pointing at Asp103 OD2. Kraut et
al. (PLoS Biol. 4, e99, 2006), the source of 2PZV, describe the bound ligands as phenolates that ACCEPT
two hydrogen bonds in the oxyanion hole, from Tyr16 and from protonated Asp103. The hydrogen of the
Asp103...O1 contact therefore belongs to Asp103.

Placement rule (the same construction as confinement_budget_pdb.py, with donor and acceptor swapped):
the hydrogen lies on the donor-acceptor axis, 0.97 A from the donor (BOND['O']), here from Asp103 OD2
toward IPH O1. Categories, the closed-shell set, the 12-A distance limit and the exclusion of the donor
atom itself are those of confinement_budget_pdb.budget(), which is imported unchanged.

As a check, the script first rebuilds the published placement (hydrogen on IPH O1) and compares every
category fraction, the within-3.0-A share and the fraction beyond 8 A with
results/confinement_budget_crossenzyme.json.

Writes results/audit_2026_11/budget_ksi.json.
Usage: audit_budget_ksi.py      (pauli env)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('PAULI_ROOT') or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / 'scripts'))
import confinement_budget_pdb as cbp  # noqa: E402

PDB = ROOT / 'pauli_data' / 'ksi' / '2PZV.pdb'
PUBLISHED = ROOT / 'results' / 'confinement_budget_crossenzyme.json'
OUT = ROOT / 'results' / 'audit_2026_11' / 'budget_ksi.json'
CHAINS = ('A', 'B', 'C', 'D')


def on_axis(donor, acceptor):
    u = acceptor['xyz'] - donor['xyz']
    u = u / np.linalg.norm(u)
    return donor['xyz'] + cbp.BOND[donor['el']] * u


def summary(b):
    f = {c: v['fraction'] for c, v in b['categories'].items()}
    return dict(
        bonding_shell_percent=100 * (f.get('donor framework', 0.0) + f.get('acceptor group', 0.0)),
        protein_percent=100 * f.get('protein', 0.0),
        water_percent=100 * f.get('water', 0.0),
        other_ligand_percent=100 * f.get('other ligand', 0.0),
        within_3A_percent=100 * b['cumulative']['3.0'],
        within_2p5A_percent=100 * b['cumulative']['2.5'],
        fraction_beyond_8A=b['tail_beyond_8A'],
        categories={c: dict(percent=100 * v['fraction'], d_min_A=v['d_min_A'])
                    for c, v in b['categories'].items()},
        top=b['top'])


def main():
    published = json.loads(PUBLISHED.read_text())
    atoms = cbp.parse_pdb(PDB)
    out = dict(source=str(PDB.relative_to(ROOT)), placement_rule=(
        'hydrogen on the donor-acceptor axis, 0.97 A from Asp103 OD2 toward IPH O1; budget() of '
        'scripts/confinement_budget_pdb.py unchanged'), chains={}, check={})
    worst = 0.0
    for c in CHAINS:
        phenol = cbp.find(atoms, 'IPH', None, 'O1', chain=c)
        asp = cbp.find(atoms, 'ASP', '103', 'OD2', chain=c)
        r_da = float(np.linalg.norm(phenol['xyz'] - asp['xyz']))
        # check: the published placement, hydrogen on the phenol oxygen
        old = cbp.budget(atoms, on_axis(phenol, asp), phenol, asp)
        pub = published[f'KSI-{c}']['placements']['on-axis']
        diffs = [abs(old['categories'][k]['fraction'] - pub['categories'][k]['fraction'])
                 for k in pub['categories']]
        diffs += [abs(old['cumulative']['3.0'] - pub['cumulative']['3.0']),
                  abs(old['tail_beyond_8A'] - pub['tail_beyond_8A'])]
        worst = max(worst, max(diffs))
        # corrected: hydrogen on Asp103 OD2, the residue that donates it
        new = cbp.budget(atoms, on_axis(asp, phenol), asp, phenol)
        out['chains'][c] = dict(r_DA_A=r_da, published_phenol_H=summary(old), asp103_H=summary(new))
    if worst > 1e-12:
        raise SystemExit(f'published placement not reproduced (max deviation {worst:.3e})')
    out['check'] = dict(published_placement_reproduced=True, max_abs_deviation=worst)

    def rng(key, sub='asp103_H'):
        v = [out['chains'][c][sub][key] for c in CHAINS]
        return [min(v), max(v)]
    rda = [out['chains'][c]['r_DA_A'] for c in CHAINS]
    out['ranges_asp103_H'] = dict(
        bonding_shell_percent=rng('bonding_shell_percent'), protein_percent=rng('protein_percent'),
        within_3A_percent=rng('within_3A_percent'), fraction_beyond_8A=rng('fraction_beyond_8A'))
    out['ranges_published_phenol_H'] = dict(
        bonding_shell_percent=rng('bonding_shell_percent', 'published_phenol_H'),
        protein_percent=rng('protein_percent', 'published_phenol_H'),
        within_3A_percent=rng('within_3A_percent', 'published_phenol_H'),
        fraction_beyond_8A=rng('fraction_beyond_8A', 'published_phenol_H'))
    out['r_DA_A'] = dict(per_chain=rda, mean=float(np.mean(rda)), min=min(rda), max=max(rda))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float) + '\n')
    for label, key in (('published, H on phenol O', 'ranges_published_phenol_H'),
                       ('corrected, H on Asp103 OD2', 'ranges_asp103_H')):
        r = out[key]
        print(f'  {label:28s} bonding shell {r["bonding_shell_percent"][0]:.2f}-{r["bonding_shell_percent"][1]:.2f} %'
              f'  protein {r["protein_percent"][0]:.3f}-{r["protein_percent"][1]:.3f} %'
              f'  within 3.0 A {r["within_3A_percent"][0]:.2f}-{r["within_3A_percent"][1]:.2f} %'
              f'  beyond 8 A {r["fraction_beyond_8A"][0]:.1e}-{r["fraction_beyond_8A"][1]:.1e}')
    print(f'  r_DA {min(rda):.3f}-{max(rda):.3f} A, mean {np.mean(rda):.3f} A')
    print(f'  check: published placement reproduced to {worst:.1e}; wrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
