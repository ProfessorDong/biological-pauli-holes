#!/usr/bin/env python3
"""Stage 4 summary: what moves K_perp and by how much.

Every row changes ONE setting from the published panel (WT, SAPT0/jun-cc-pVDZ, native C5H8 donor,
side-chain wall, 5x5 grid at half-width 0.15 A). Reports both eigenvalues and the fitted gradient,
as the pre-registration requires, and the percentage change of each eigenvalue from the baseline.
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
from pathlib import Path
import numpy as np

D = Path(_REPO + '/results/native_donor_validation')
S = D / 'stage4'


def load(p):
    g = json.loads(p.read_text())
    return (g['K_perp_exch_eigs'], g.get('K_perp_exch_grad_norm'),
            g['K_perp_exch_rms_kcal'], g.get('K_perp_int_eigs'))


base, gb, rb, ib = load(D / 'WT_r255_w015_native_transverse.json')
rows = [('BASELINE  SAPT0/jun-cc-pVDZ, C5H8 donor, side-chain wall', base, gb, rb, ib)]
for label, f in [
    ('basis   aug-cc-pVDZ',            S / 's4_basis_augdz.json'),
    ('basis   aug-cc-pVTZ',            S / 's4_basis_augtz.json'),
    ('order   SAPT2+',                 S / 's4_order_sapt2p.json'),
    ('wall    side chain + backbone',  S / 's4_wall_bb_WT.json'),
    ('donor   C7H12 (next shell)',     S / 's4_donor_shell2_WT.json'),
    ('grid    7x7 density',            D / 'WT_r255_w015_n7_native_transverse.json'),
]:
    if f.exists():
        rows.append((label,) + load(f))

print('WT, one variable changed per row\n')
h = (f"{'setting':46s} {'lambda_1':>10s} {'d%':>7s} {'lambda_2':>10s} {'d%':>7s} "
     f"{'|grad|':>7s} {'sign':>9s}")
print(h); print('-' * len(h))
for label, e, g, r, i in rows:
    d0 = 100 * (e[0] - base[0]) / abs(base[0])
    d1 = 100 * (e[1] - base[1]) / abs(base[1])
    gs = f'{g:7.4f}' if g is not None else '    n/a'
    print(f'{label:46s} {e[0]:+10.5f} {d0:+7.1f} {e[1]:+10.5f} {d1:+7.1f} {gs} '
          f'{"negative" if e[0] < 0 else "positive":>9s}')

print('\nL754A, the one positive-definite system')
lb, _, _, _ = load(D / 'L754A_r255_w015_native_transverse.json')
lx, gx, _, _ = load(S / 's4_wall_bb_L754A.json')
print(f'  side-chain wall        {lb[0]:+.5f} {lb[1]:+.5f}')
print(f'  side chain + backbone  {lx[0]:+.5f} {lx[1]:+.5f}   '
      f'({100*(lx[0]-lb[0])/abs(lb[0]):+.1f}% on lambda_1)')

print('\nRanked by |change in lambda_1|, the eigenvalue the paper\'s claim rests on')
rank = sorted(((abs(100 * (e[0] - base[0]) / abs(base[0])), lb2)
               for lb2, e, _, _, _ in rows[1:]), reverse=True)
for v, lb2 in rank:
    print(f'  {v:6.1f}%   {lb2}')
print(f'\nSIGN OF lambda_1: ' +
      ('NEGATIVE in every setting tested' if all(e[0] < 0 for _, e, _, _, _ in rows)
       else '*** CHANGES SIGN in some setting ***'))
