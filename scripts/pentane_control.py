#!/usr/bin/env python3
"""Do the substrate's pi bonds change the wall contact, at frozen heavy-atom geometry?

WHY THIS EXISTS
  Replacing the methane donor by the real C5H8 motif raises the transverse exchange
  curvature by a median factor of 1.31. That comparison cannot say WHY, because methane and
  C5H8 differ in atom count, in geometry and in electronic structure all at once. An earlier
  attempt to separate these compared the three-atom bis-allylic group against five-atom
  methane, which is not a controlled comparison and was withdrawn.

  This is the controlled version. The saturated analogue n-pentane is built on the IDENTICAL
  five-carbon skeleton, with the reactive CH2 hydrogens left exactly where the trajectory put
  them, and the four hydrogens needed to saturate the two C=C added along the directions that
  complete tetrahedral valence. Heavy atoms do not move. The only change is that two pi bonds
  become sigma bonds.

  The comparison is then made through F-SAPT on the group {C14, H25, H26}, the reactive CH2.
  That group has the SAME three atoms at the SAME coordinates in both molecules, so its
  exchange contribution differs only through the electronic structure of its neighbours. This
  sidesteps the atom-count confound entirely: pentane has four more hydrogens than pentadiene,
  but none of them are in the group being compared.

WHAT IS BEING TESTED, PRECISELY
  1,4-pentadiene is a SKIPPED diene. Its two double bonds are separated by the sp3 carbon
  that carries the transferring hydrogen, so they are isolated rather than conjugated;
  conjugation appears only in the pentadienyl radical formed after abstraction. This
  calculation therefore asks whether two ISOLATED pi bonds flanking the reactive carbon
  stiffen its wall contact, not whether conjugation does.

  The pentane built here is a hypothetical geometry, not a relaxed pentane: it inherits C=C
  bond lengths near 1.34 A where a saturated chain would want 1.54 A. That is deliberate,
  since relaxing it would reintroduce the geometric difference the control exists to remove,
  but it means the absolute pentane numbers are not those of a real pentane.

Usage: pentane_control.py [TAG [basis]]     (pauli env)
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

sys.path.insert(0, '/home/liang/anaconda3/envs/pauli/share/psi4/fsapt')
import fsapt  # noqa: E402

ROOT = Path(_REPO)
DIR = ROOT / 'results/sapt_bio/native_fragment'
OUT = ROOT / 'results/sapt_bio/donor_fragment'
SCRATCH = Path(os.environ.get('FSAPT_SCRATCH', Path.home() / 'scratch/fsapt_pentane'))
Ha2kcal, CONV = 627.5094740631, 0.694770
DELTAS = [-0.20, -0.10, 0.00, +0.10, +0.20]
CH = 1.09
psi4.set_memory(os.environ.get('PSI4_MEM', '32 GB'))

# The reactive CH2 is atoms 1..3 of the fragment in both molecules by construction below.
GROUPS_A = {'reactive_CH2': [1, 2, 3]}
WALL = 'wall'


def bonded(P, E, i, j):
    d = np.linalg.norm(P[i] - P[j])
    lim = 1.35 if 'H' in (E[i], E[j]) else 1.75
    return d < lim


def saturate(atoms, mode='anti'):
    """Add the hydrogens that turn each C=C into a C-C, leaving heavy atoms untouched.

    Every carbon needing a hydrogen here is a planar sp2 centre with three bonds, so the sum
    of its bond unit vectors is near zero and there is no tetrahedral vacancy to fill: the
    two vacancies are the pi lobes, perpendicular to the local plane. The added hydrogen
    therefore goes along the plane normal, which leaves a sign to choose. 'anti' alternates
    the sign between successive carbons, the staggered arrangement a real chain would adopt;
    'syn' puts them all on one face. The two are run against each other to confirm the
    reactive-CH2 comparison does not depend on the choice.
    """
    E = [a[0] for a in atoms]
    P = np.array([a[1] for a in atoms], float)
    N = [a[2] for a in atoms]
    nbr = {i: [j for j in range(len(E)) if j != i and bonded(P, E, i, j)]
           for i in range(len(E))}
    out, added = list(atoms), 0
    for i, e in enumerate(E):
        if e != 'C':
            continue
        need = 4 - len(nbr[i])
        assert need in (0, 1), f'{N[i]} has {len(nbr[i])} neighbours, expected 3 or 4'
        if not need:
            continue
        u = [(P[j] - P[i]) / np.linalg.norm(P[j] - P[i]) for j in nbr[i]]
        planar = np.linalg.norm(sum(u))
        assert planar < 0.35, f'{N[i]} is not planar (|sum of bond vectors|={planar:.3f})'
        n = np.cross(u[0], u[1])
        nn = np.linalg.norm(n)
        assert nn > 0.3, f'{N[i]}: bond vectors are collinear, no plane normal'
        s = 1.0 if (mode == 'syn' or added % 2 == 0) else -1.0
        out.append(('H', P[i] + CH * s * n / nn, f'Hsat{N[i]}'))
        added += 1
    return out, added


def write_groups(path, groups):
    path.write_text(''.join(f'{k} ' + ' '.join(str(i) for i in v) + '\n'
                            for k, v in groups.items()))


def run_point(donor, native, disp, basis, tag, label, delta):
    work = SCRATCH / f'{tag}_{label}_{delta:+.2f}'
    work.mkdir(parents=True, exist_ok=True)
    L = ['0 1'] + [f'{e} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}' for e, p, _ in donor]
    L += ['--', '0 1']
    L += [f'{a["element"]} ' + ' '.join(f'{v:.6f}' for v in np.array(a['position']) + disp)
          for a in native]
    L.append('units angstrom\nsymmetry c1\nno_reorient\nno_com')
    psi4.core.set_output_file(str(OUT / f'fsapt_{tag}_{label}_{delta:+.2f}.out'), False)
    psi4.core.clean()
    psi4.set_options({'basis': basis, 'scf_type': 'df', 'freeze_core': True,
                      'FISAPT_FSAPT_FILEPATH': str(work) + '/'})
    psi4.energy('fisapt0', molecule=psi4.geometry('\n'.join(L)))
    total = float(psi4.variable('SAPT EXCH ENERGY') * Ha2kcal)
    rest = [i for i in range(1, len(donor) + 1) if i not in GROUPS_A['reactive_CH2']]
    write_groups(work / 'fA.dat', dict(GROUPS_A, rest=rest))
    write_groups(work / 'fB.dat',
                 {WALL: [i + len(donor) for i in range(1, len(native) + 1)]})
    with contextlib.redirect_stdout(io.StringIO()):
        stuff = fsapt.compute_fsapt(str(work), True, print_output=False)
    e = stuff['order2r']['Exch']
    return total, float(e['reactive_CH2'][WALL]), float(e['rest'][WALL])


def curvature(ys):
    ds = np.array(DELTAS)
    A = np.column_stack([np.ones_like(ds), ds, ds ** 2])
    p, *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
    return float(2 * p[2] * CONV)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'WT'
    basis = sys.argv[2] if len(sys.argv) > 2 else 'jun-cc-pVDZ'
    g = json.loads((DIR / f'{tag}_pentadienyl.json').read_text())
    dC, xH = np.array(g['donor_C']), np.array(g['xferH'])
    native = g['native_wall']
    heavy = [(a['name'], np.array(a['position'])) for a in native if a['element'] != 'H']
    wall = min(heavy, key=lambda t: np.linalg.norm(t[1] - xH))
    axis = (wall[1] - dC) / np.linalg.norm(wall[1] - dC)

    diene = [(a['element'], np.array(a['position'], float), a['name'])
             for a in g['donor_fragment']]
    # reorder so the reactive CH2 is atoms 1..3 in BOTH molecules
    # select by index: these tuples hold numpy arrays, so identity tests on them are unsafe
    ridx = [i for i, a in enumerate(diene) if np.linalg.norm(a[1] - dC) < 1.2]
    assert len(ridx) == 3, f'reactive group has {len(ridx)} atoms, expected C plus two H'
    ridx.sort(key=lambda i: diene[i][0] != 'C')          # carbon first
    assert diene[ridx[0]][0] == 'C', 'reactive group does not start with the donor carbon'
    assert [diene[i][0] for i in ridx[1:]] == ['H', 'H'], 'reactive group is not a CH2'
    diene = [diene[i] for i in ridx] + [a for i, a in enumerate(diene) if i not in ridx]
    assert np.allclose(diene[0][1], dC), 'atom 1 is not the donor carbon'
    assert any(np.allclose(a[1], xH) for a in diene[:3]), 'xferH not in the reactive group'

    built = {}
    for mode in ('anti', 'syn'):
        pentane, added = saturate(diene, mode)
        nC = sum(1 for a in pentane if a[0] == 'C')
        nH = sum(1 for a in pentane if a[0] == 'H')
        assert (nC, nH) == (5, 12), f'{mode}: fragment is C{nC}H{nH}, expected C5H12'
        assert (6 * nC + nH) % 2 == 0, f'{mode}: odd electron count'
        P = np.array([a[1] for a in pentane])
        dmin = min(np.linalg.norm(P[i] - P[j])
                   for i in range(len(P)) for j in range(i + 1, len(P)))
        assert dmin > 0.85, f'{mode}: saturation created a clash at {dmin:.3f} A'
        for a, b in zip(diene[:3], pentane[:3]):
            assert np.allclose(a[1], b[1]), f'{mode}: {a[2]} moved during saturation'
        built[mode] = (pentane, dmin)
        print(f'{tag}  wall={wall[0]}  {mode}: added {added} H to give C{nC}H{nH}, '
              f'min interatomic {dmin:.3f} A')
    print(f'  reactive group held fixed in both: {[a[2] for a in diene[:3]]}\n')
    dmin = min(v[1] for v in built.values())

    res = {}
    for label, donor in ([('pentadiene', diene)] +
                         [(f'pentane_{m}', built[m][0]) for m in ('anti', 'syn')]):
        tot, grp, rest = [], [], []
        for d in DELTAS:
            t, r, o = run_point(donor, native, axis * d, basis, tag, label, d)
            tot.append(t), grp.append(r), rest.append(o)
            print(f'  {label:11s} delta={d:+.2f}  total={t:+8.4f}  '
                  f'CH2={r:+8.4f}  rest={o:+8.4f}', flush=True)
        res[label] = dict(n_atoms=len(donor), k_total=curvature(tot),
                          k_CH2=curvature(grp), k_rest=curvature(rest),
                          exch_total=tot, exch_CH2=grp, exch_rest=rest)
        print(f'   -> k_total={res[label]["k_total"]:.3f}  '
              f'k_CH2={res[label]["k_CH2"]:.3f} N/m\n')

    print(f'{"molecule":>14}{"atoms":>7}{"k_total":>10}{"k(CH2)":>10}{"CH2 share":>11}')
    for lbl in ('pentadiene', 'pentane_anti', 'pentane_syn'):
        r = res[lbl]
        print(f'{lbl:>14}{r["n_atoms"]:>7}{r["k_total"]:>10.3f}{r["k_CH2"]:>10.3f}'
              f'{r["k_CH2"]/r["k_total"]:>10.1%}')
    ratios = {m: res['pentadiene']['k_CH2'] / res[f'pentane_{m}']['k_CH2']
              for m in ('anti', 'syn')}
    print(f'\n  reactive CH2, pi flanks vs saturated flanks:')
    for m, v in ratios.items():
        print(f'    added H placed {m:5s}: {v:.3f}x')
    spread = abs(ratios['anti'] - ratios['syn'])
    print(f'  the two placement conventions differ by {spread:.3f}, so the comparison is '
          f'{"insensitive" if spread < 0.10 else "SENSITIVE"} to that choice')
    ratio = float(np.mean(list(ratios.values())))
    verdict = ('the pi bonds stiffen the reactive contact'
               if ratio > 1.10 else
               'the pi bonds soften the reactive contact' if ratio < 0.90 else
               'the pi bonds do not measurably change the reactive contact')
    print(f'  VERDICT  {verdict} (mean ratio {ratio:.3f})')
    print('  Note: the molecules differ in total atom count, so only the CH2 group '
          'comparison\n  is controlled; k_total is reported for completeness.')

    (OUT / f'{tag}_pentane_control.json').write_text(json.dumps(dict(
        tag=tag, basis=basis, wall_atom=wall[0], deltas=DELTAS, results=res,
        k_CH2_ratio_diene_over_pentane=ratio, k_CH2_ratio_by_mode=ratios,
        placement_spread=float(spread), min_distance=dmin, verdict=verdict,
        note=('n-pentane built on the frozen pentadiene heavy-atom skeleton; the reactive '
              'CH2 atoms are identical in both, so its F-SAPT exchange contribution is a '
              'controlled comparison, while total atom counts differ')), indent=1))
    print(f'\nwrote {OUT}/{tag}_pentane_control.json')


if __name__ == '__main__':
    main()
