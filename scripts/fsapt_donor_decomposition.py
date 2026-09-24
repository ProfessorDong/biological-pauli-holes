#!/usr/bin/env python3
"""Where does the real donor's extra exchange curvature come from? F-SAPT group partition.

WHY THIS EXISTS
  Swapping the methane surrogate for the real bis-allylic C5H8 unit raises the transverse
  exchange curvature at the WT wall from 7.885 to 12.598 N/m, a factor 1.60. Two very
  different readings give the same number:

    ELECTRONIC  the bis-allylic carbon is sp2 and its unpaired density is delocalized over
                the pentadienyl pi system, so the CH unit itself is a harder wall partner
                than a closed-shell sp3 methane carbon.
    CONTACT     C5H8 simply puts eleven atoms within 5 A of the wall where methane puts
                five. More atoms in contact, more exchange, no new physics.

  These are not distinguishable from the total. F-SAPT partitions the SAPT0 exchange term
  onto chemical groups exactly, with no fitting, so it separates them. Running it at each
  of the five transverse displacements and fitting the same quadratic gives a curvature
  per group, which is the quantity the manuscript actually reports.

  If the bis-allylic group alone carries roughly the methane value and the vinyl flanks
  supply the balance, the effect is contact and the surrogate is defensible as far as it
  goes. If the bis-allylic group alone exceeds methane, the surrogate understates the wall
  and the correction is electronic.

HELD IDENTICAL to sapt_pentadienyl_vs_methane.py: snapshot, wall fragment, scan axis, the
five displacements, basis, and the quadratic fit. F-SAPT replaces SAPT only in that it also
reports the partition; the total it produces is checked against the SAPT0 total.

Usage: fsapt_donor_decomposition.py [TAG [basis]]     (pauli env)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import psi4

# fsapt.py ships beside Psi4 as a script rather than an importable package member, but it
# guards its entry point, so importing it is safe and lets the partition be read in process
# at full precision. That matters: the fsapt.dat it writes rounds to three decimals, which
# is adequate for a system whose exchange is of order 1 kcal/mol and useless for L754A,
# whose open wall puts the total at roughly 0.003.
sys.path.insert(0, _os.environ.get('PSI4_FSAPT',
    _os.path.join(__import__('sys').prefix, 'share', 'psi4', 'fsapt')))
import fsapt  # noqa: E402

ROOT = Path(_REPO)
DIR = ROOT / 'results/sapt_bio/native_fragment'
OUT = ROOT / 'results/sapt_bio/donor_fragment'
SCRATCH = Path(os.environ.get('FSAPT_SCRATCH', Path.home() / 'scratch/fsapt_donor'))
Ha2kcal, CONV = 627.5094740631, 0.694770
DELTAS = [-0.20, -0.10, 0.00, +0.10, +0.20]

# F-SAPT builds and holds the localized-orbital three-index integrals, so it needs far more
# than the 500 MB Psi4 defaults to; the plain SAPT0 scan ran fine without this.
psi4.set_memory(os.environ.get('PSI4_MEM', '32 GB'))

# 1-based indices into the donor fragment, in the order extract_pentadienyl_donor.py wrote
# them. The bis-allylic group is the carbon that carries the transferring hydrogen plus
# both of its hydrogens; the flanks are the two vinyl CH=CH units with their link caps.
GROUPS_A = {
    'biallylic': [1, 2, 3],              # C14, H25 (transferring), H26
    'vinyl_C13': [4, 5, 6, 7, 12],       # C13 H24 = C10 H19, HcapC10
    'vinyl_C17': [8, 9, 10, 11, 13],     # C17 H28 = C15 H27, HcapC15
}
WALL = 'wall'   # the single monomer-B group; its atom list is built per system in run_point


def write_groups(path, groups):
    path.write_text(''.join(f'{k} ' + ' '.join(str(i) for i in v) + '\n'
                            for k, v in groups.items()))


def run_point(donor, native, disp, basis, tag, delta):
    """One F-SAPT point. Returns (total exchange kcal/mol, {groupA: kcal/mol})."""
    work = SCRATCH / f'{tag}_{delta:+.2f}'
    work.mkdir(parents=True, exist_ok=True)
    L = ['0 1'] + [f'{e} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}' for e, p in donor]
    L += ['--', '0 1']
    L += [f'{a["element"]} ' + ' '.join(f'{v:.6f}' for v in np.array(a['position']) + disp)
          for a in native]
    L.append('units angstrom\nsymmetry c1\nno_reorient\nno_com')
    psi4.core.set_output_file(str(OUT / f'fsapt_{tag}_{delta:+.2f}.out'), False)
    psi4.core.clean()
    psi4.set_options({'basis': basis, 'scf_type': 'df', 'freeze_core': True,
                      'FISAPT_FSAPT_FILEPATH': str(work) + '/'})
    psi4.energy('fisapt0', molecule=psi4.geometry('\n'.join(L)))
    total = float(psi4.variable('SAPT EXCH ENERGY') * Ha2kcal)

    # The wall is one group covering every atom of whatever side chain this system has:
    # six systems face a 14-atom Leu732 isobutane, L754A an 11-atom Asn672 acetamide.
    # fsapt.py indexes into the full dimer, so monomer-B atoms are shifted past the donor.
    write_groups(work / 'fA.dat', GROUPS_A)
    write_groups(work / 'fB.dat',
                 {WALL: [i + len(donor) for i in range(1, len(native) + 1)]})

    # compute_fsapt's second argument splits each link hydrogen 50-50 onto the two groups it
    # bridges, rather than leaving it as a free-standing Link-n group. That is what makes the
    # partition exhaustive over the three chemical groups, so the curvatures below sum to the
    # total. order2r holds those reduced numbers; order2 holds the unreduced ones.
    with contextlib.redirect_stdout(io.StringIO()):
        stuff = fsapt.compute_fsapt(str(work), True, print_output=False)
    exch = stuff['order2r']['Exch']
    per = {k: float(exch[k][WALL]) for k in GROUPS_A if k in exch}
    if set(per) != set(GROUPS_A):
        raise RuntimeError(f'F-SAPT returned groups {sorted(exch)} in {work}')
    return total, per


def quad_curvature(ds, ys):
    A = np.column_stack([np.ones_like(ds), ds, ds ** 2])
    p, *_ = np.linalg.lstsq(A, ys, rcond=None)
    return float(2 * p[2] * CONV), float(np.sqrt(np.mean((ys - A @ p) ** 2)))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'WT'
    basis = sys.argv[2] if len(sys.argv) > 2 else 'jun-cc-pVDZ'
    g = json.loads((DIR / f'{tag}_pentadienyl.json').read_text())
    dC, xH = np.array(g['donor_C']), np.array(g['xferH'])
    native = g['native_wall']
    heavy = [(a['name'], np.array(a['position'])) for a in native if a['element'] != 'H']
    wall = min(heavy, key=lambda t: np.linalg.norm(t[1] - xH))
    axis = (wall[1] - dC) / np.linalg.norm(wall[1] - dC)
    donor = [(a['element'], np.array(a['position'])) for a in g['donor_fragment']]
    assert len(donor) == 13, f'expected 13 donor atoms, got {len(donor)}'
    assert sorted(sum(GROUPS_A.values(), [])) == list(range(1, 14)), 'groups miss an atom'
    # GROUPS_A is positional, and nothing guarantees the extractor emits the same atom order
    # in every system: L754A's transferring hydrogen is H26 where the other six have H25.
    # So check the assignment chemically rather than trusting the order. Group 1 must be the
    # donor carbon plus exactly the two hydrogens bonded to it, one of which is the one being
    # transferred, and each flank must be two carbons with two hydrogens and one link cap.
    P = np.array([q for _, q in donor])
    assert np.allclose(P[0], dC), 'fragment atom 1 is not the donor carbon'
    bonded = [i for i in GROUPS_A['biallylic'][1:] if np.linalg.norm(P[i - 1] - dC) < 1.2]
    assert len(bonded) == 2, f'bis-allylic group holds {len(bonded)} H bonded to the donor C'
    assert any(np.allclose(P[i - 1], xH) for i in GROUPS_A['biallylic']), \
        'the transferring hydrogen is not in the bis-allylic group'
    for v in ('vinyl_C13', 'vinyl_C17'):
        el = [donor[i - 1][0] for i in GROUPS_A[v]]
        assert el.count('C') == 2 and el.count('H') == 3, f'{v} is not 2 C plus 2 H plus cap'
    print(f'{tag}  basis={basis}  wall={wall[0]}  groups={list(GROUPS_A)}\n')

    tot, grp = {}, {k: {} for k in GROUPS_A}
    for d in DELTAS:
        t, per = run_point(donor, native, axis * d, basis, tag, d)
        s = sum(per.values())
        # The reduced partition is exhaustive, so the groups must re-sum to the SAPT0 total.
        # fsapt.dat prints three decimals, so only rounding separates them.
        assert abs(s - t) < 0.01, f'partition {s:.4f} vs SAPT0 total {t:.4f}'
        tot[d] = t
        for k in GROUPS_A:
            grp[k][d] = per[k]
        print(f'  delta={d:+.2f}  E_exch={t:+8.4f}  ' +
              '  '.join(f'{k}={per[k]:+7.4f}' for k in GROUPS_A))

    ds = np.array(DELTAS)
    k_tot, rms = quad_curvature(ds, np.array([tot[d] for d in DELTAS]))
    ks = {k: quad_curvature(ds, np.array([grp[k][d] for d in DELTAS]))[0] for k in GROUPS_A}
    ref = json.loads((OUT / f'{tag}_donor_comparison.json').read_text())
    k_meth = ref['results']['methane']['k_Nm']

    print(f'\n{"group":>12}{"k_exch (N/m)":>15}{"share":>9}{"vs methane":>13}')
    for k, v in ks.items():
        print(f'{k:>12}{v:>15.3f}{v/k_tot:>8.1%}{v/k_meth:>12.3f}x')
    print(f'{"TOTAL":>12}{k_tot:>15.3f}{1.0:>8.1%}{k_tot/k_meth:>12.3f}x')
    print(f'{"methane":>12}{k_meth:>15.3f}')

    verdict = ('electronic: the bis-allylic group alone is a harder wall partner than '
               'methane, so the surrogate understates the wall'
               if ks['biallylic'] > 1.15 * k_meth else
               ('contact: the bis-allylic group reproduces methane and the flanks supply '
                'the balance'
                if ks['biallylic'] < 1.15 * k_meth and ks['biallylic'] > 0.85 * k_meth else
                'mixed: the bis-allylic group is below methane, see the table'))
    print(f'\nVERDICT  {verdict}')

    (OUT / f'{tag}_fsapt_decomposition.json').write_text(json.dumps(dict(
        tag=tag, basis=basis, wall_atom=wall[0], deltas=DELTAS,
        groups_A=GROUPS_A, exch_total_kcal=tot, exch_by_group_kcal=grp,
        k_total_Nm=k_tot, k_by_group_Nm=ks, k_methane_Nm=k_meth,
        rms_total_kcal=rms, verdict=verdict), indent=1))
    print(f'wrote {OUT}/{tag}_fsapt_decomposition.json')


if __name__ == '__main__':
    main()
