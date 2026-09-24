#!/usr/bin/env python3
"""Panel summary of the total transverse restoring matrix and the exchange fraction.

Carries its own consistency check: the wall's contribution obtained by DIFFERENCING two
supermolecular total energies, K_tot(complex) - K_tot(donor), must agree in sign and order of
magnitude with the SAPT total-interaction Hessian computed independently on the same grid. The two
routes share no code and only one of them can see the fragment decomposition.

Usage: total_restoring_panel.py   (pauli env)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
from pathlib import Path

import numpy as np

D = Path(_REPO + '/results/native_donor_validation')
SYS = ['L754A', 'I552A', 'I538A', 'L546A', 'I553A', 'V750A', 'WT']


def main():
    have = [t for t in SYS if (D / f'{t}_r255_w015_total_restoring.json').exists()]
    if not have:
        print('no results yet'); return
    d = {t: json.loads((D / f'{t}_r255_w015_total_restoring.json').read_text()) for t in have}

    print('TOTAL CONSTRAINED TRANSVERSE RESTORING MATRIX, native donor, h = 0.15 A, jun-cc-pVDZ')
    print('all atoms but the transferring hydrogen frozen; not a normal-mode analysis\n')
    print(f'{"system":7s} | {"K_tot(complex) HF":>26s} | {"K_tot(complex) B3LYP":>26s} | pos.def.')
    print('-' * 88)
    for t in have:
        a, b = d[t]['HF']['K_tot_complex_eigs']
        c, e = d[t]['B3LYP']['K_tot_complex_eigs']
        print(f'{t:7s} | {a:+11.3f} , {b:+11.3f} | {c:+11.3f} , {e:+11.3f} | '
              f'{"yes" if a > 0 and c > 0 else "NO"}')

    print(f'\n{"system":7s} | {"exchange fraction":>18s} | {"interaction fraction":>20s} |'
          f' {"covalent share":>14s}')
    print('-' * 72)
    fr = []
    for t in have:
        fh = d[t]['HF']['exchange_fraction'] * 100
        fb = d[t]['B3LYP']['exchange_fraction'] * 100
        ih = d[t]['HF']['interaction_fraction'] * 100
        don = max(abs(np.array(d[t]['HF']['K_tot_donor_eigs'])))
        cpl = d[t]['HF']['spectral_norm_complex']
        fr.append((fh, fb))
        print(f'{t:7s} | HF {fh:6.3f}%  B3LYP {fb:5.3f}% | HF {ih:6.3f}%'
              f'{"":10s} | {100 * don / cpl:8.2f}%')
    a = np.array(fr)
    print(f'\nexchange fraction of the total transverse restoring matrix:')
    print(f'  HF    : {a[:,0].min():.3f}% to {a[:,0].max():.3f}%, median {np.median(a[:,0]):.3f}%')
    print(f'  B3LYP : {a[:,1].min():.3f}% to {a[:,1].max():.3f}%, median {np.median(a[:,1]):.3f}%')

    print('\nCONSISTENCY CHECK: wall contribution by supermolecular difference vs SAPT '
          'total interaction')
    print(f'{"system":7s} | {"difference HF":>22s} | {"difference B3LYP":>22s} | '
          f'{"SAPT K_perp^int":>22s}')
    print('-' * 84)
    for t in have:
        wh = d[t]['HF']['K_wall_from_difference_eigs']
        wb = d[t]['B3LYP']['K_wall_from_difference_eigs']
        si = d[t]['K_perp_int_eigs']
        print(f'{t:7s} | {wh[0]:+9.4f} , {wh[1]:+9.4f} | {wb[0]:+9.4f} , {wb[1]:+9.4f} | '
              f'{si[0]:+9.4f} , {si[1]:+9.4f}')
    agree = sum(1 for t in have
                if np.sign(d[t]['B3LYP']['K_wall_from_difference_eigs'][1])
                == np.sign(d[t]['K_perp_int_eigs'][1]))
    print(f'\n  sign of the larger wall eigenvalue agrees between the two independent routes '
          f'in {agree} of {len(have)}')

    out = dict(n_systems=len(have), systems=have,
               exchange_fraction_HF_percent=[float(x) for x in a[:, 0]],
               exchange_fraction_B3LYP_percent=[float(x) for x in a[:, 1]],
               per_system={t: dict(
                   K_tot_complex_HF=d[t]['HF']['K_tot_complex_eigs'],
                   K_tot_complex_B3LYP=d[t]['B3LYP']['K_tot_complex_eigs'],
                   K_tot_donor_HF=d[t]['HF']['K_tot_donor_eigs'],
                   wall_difference_HF=d[t]['HF']['K_wall_from_difference_eigs'],
                   wall_difference_B3LYP=d[t]['B3LYP']['K_wall_from_difference_eigs'],
                   sapt_K_perp_int=d[t]['K_perp_int_eigs'],
                   exchange_fraction_HF=d[t]['HF']['exchange_fraction'],
                   exchange_fraction_B3LYP=d[t]['B3LYP']['exchange_fraction']) for t in have},
               sign_agreement=f'{agree}/{len(have)}')
    (D / 'total_restoring_panel.json').write_text(json.dumps(out, indent=2))
    print(f'\nwrote {D}/total_restoring_panel.json')


if __name__ == '__main__':
    main()


def difference_series():
    """Fit the wall contribution as E(complex) - E(donor) POINTWISE, not as a difference of fits.

    Both surfaces are strongly anharmonic over +/-0.15 A of C-H displacement, with rms residuals
    of 0.2 to 0.9 kcal/mol against a quadratic. That anharmonicity is intramolecular and identical
    in the two series, so it cancels pointwise. Differencing the two fitted Hessians gives the same
    matrix but an uncertainty inflated by the cancelled anharmonicity; fitting the difference
    series gives the correct one.
    """
    print('\nWALL CONTRIBUTION FROM THE POINTWISE DIFFERENCE SERIES, HF and B3LYP')
    print(f'{"system":7s} | {"HF eigenvalues":>22s} {"rms":>9s} | '
          f'{"B3LYP eigenvalues":>22s} {"rms":>9s} | {"SAPT K_perp^int":>22s}')
    print('-' * 104)
    for t in SYS:
        p = D / f'{t}_r255_w015_total_restoring.json'
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        g = d['grid']
        A = np.array([[1, r['d1'], r['d2'], 0.5 * r['d1'] ** 2,
                       r['d1'] * r['d2'], 0.5 * r['d2'] ** 2] for r in g], float)
        row = [f'{t:7s} |']
        for lbl in ('HF', 'B3LYP'):
            z = np.array([r[f'{lbl}_complex'] - r[f'{lbl}_donor'] for r in g], float)
            c, *_ = np.linalg.lstsq(A, z, rcond=None)
            rms = float(np.sqrt(np.mean((A @ c - z) ** 2)))
            w = np.linalg.eigvalsh(np.array([[c[3], c[4]], [c[4], c[5]]]) * 0.694770)
            row.append(f' {w[0]:+10.4f} , {w[1]:+10.4f} {rms:9.2e} |')
        si = d['K_perp_int_eigs']
        row.append(f' {si[0]:+10.4f} , {si[1]:+10.4f}')
        print(''.join(row))
