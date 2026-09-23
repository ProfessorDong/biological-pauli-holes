#!/usr/bin/env python3
"""Panel summary of the matched native-donor validation, against the published methane series.

Applies the decision rule pre-registered in prxlife/NATIVE_DONOR_VALIDATION.md section 8 before
looking at anything else, and reports every scalar summary of the Hessian rather than the one
that flatters the argument. Usage: native_donor_panel.py   (pauli env)
"""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
D = ROOT / 'results/native_donor_validation'
SYS = ['L754A', 'I552A', 'I538A', 'L546A', 'I553A', 'V750A', 'WT']   # ascending methane K_sep
KIE = {'WT': 66, 'V750A': 62, 'I552A': 66, 'I538A': 100, 'L754A': 106, 'L546A': 131, 'I553A': 148}


def exact_rho(x, y):
    xr, yr = rankdata(x) - (len(x) + 1) / 2, rankdata(y) - (len(y) + 1) / 2
    den = np.linalg.norm(xr) * np.linalg.norm(yr)
    obs = float(xr @ yr / den)
    perms = np.array(list(itertools.permutations(yr)))
    p = float(np.mean(np.abs(perms @ xr / den) >= abs(obs) - 1e-12))
    return obs, p


def main():
    meth = {r['tag']: r for r in json.loads(
        (ROOT / 'results/transverse_hessian/transverse_vs_separation.json').read_text())['records']
        if r['clamp'] == 'r255' and abs(r['half'] - 0.15) < 1e-9}
    nat = {t: json.loads((D / f'{t}_r255_w015_native_transverse.json').read_text()) for t in SYS}

    print('MATCHED NATIVE-DONOR PANEL, compressed clamp, half-width 0.15 A\n')
    print(f'{"system":7s} {"KIE":>4s} | {"K_sep^met":>9s} {"K_sep^nat":>9s} {"nat/met":>7s} |'
          f' {"Kperp^exch eigenvalues (native)":>31s} | {"posdef":>6s} {"Ksep/|K|2":>9s}')
    print('-' * 104)
    n_neg_nat = n_neg_met = 0
    for t in SYS:
        m, d = meth[t], nat[t]
        e = d['K_perp_exch_eigs']
        neg_nat, neg_met = e[0] < 0, min(m['w1'], m['w2']) < 0
        n_neg_nat += neg_nat
        n_neg_met += neg_met
        print(f'{t:7s} {KIE[t]:4d} | {m["Ksep"]:9.3f} {d["K_sep_Nm"]:9.3f} '
              f'{d["K_sep_Nm"]/m["Ksep"]:7.2f} | {e[0]:+14.4f} , {e[1]:+14.4f} | '
              f'{("no" if neg_nat else "yes"):>6s} {d["ratio_Ksep_over_specnorm_exch"]:9.1f}')

    print(f'\nPRE-REGISTERED PRIMARY ENDPOINT (section 8):')
    print(f'  negative K_perp^exch eigenvalue, native donor : {n_neg_nat} of 7')
    print(f'  negative K_perp^exch eigenvalue, methane      : {n_neg_met} of 7')
    verdict = ('SURVIVES (>=5 of 7)' if n_neg_nat >= 5 else
               'RESTRICTED TO THE SURROGATE (<=3 of 7)' if n_neg_nat <= 3 else
               'AMBIGUOUS (4 of 7), as declared in advance')
    print(f'  declared outcome                             : {verdict}')

    print('\nTOTAL INTERMOLECULAR INTERACTION, same grid and fragments')
    print(f'{"system":7s} {"Kperp^int eigenvalues":>30s}  {"class":>18s}')
    print('-' * 60)
    for t in SYS:
        a, b = nat[t]['K_perp_int_eigs']
        cls = ('negative definite' if b < 0 else
               'positive definite' if a > 0 else 'indefinite')
        print(f'{t:7s} {a:+14.4f} , {b:+14.4f}  {cls:>18s}')

    ks = np.array([nat[t]['K_sep_Nm'] for t in SYS])
    summaries = {
        'trace': np.array([sum(nat[t]['K_perp_exch_eigs']) for t in SYS]),
        'min eigenvalue': np.array([nat[t]['K_perp_exch_eigs'][0] for t in SYS]),
        'max eigenvalue': np.array([nat[t]['K_perp_exch_eigs'][1] for t in SYS]),
        'spectral norm': np.array([nat[t]['spectral_norm_exch'] for t in SYS]),
    }
    print('\nRANK CORRELATION OF K_perp^exch WITH K_sep, native donor (exploratory, n=7)')
    for k, v in summaries.items():
        r, p = exact_rho(ks, v)
        print(f'  {k:16s} rho = {r:+.3f}   exact two-sided p = {p:.3f}')

    rat = np.array([nat[t]['ratio_Ksep_over_specnorm_exch'] for t in SYS])
    rms = np.array([nat[t]['K_perp_exch_rms_kcal'] for t in SYS])
    print(f'\nK_sep / spectral norm: min {rat.min():.1f}, median {np.median(rat):.1f}, '
          f'max {rat.max():.1f}')
    print(f'fit residuals: {rms.min():.1e} to {rms.max():.1e} kcal/mol')
    print(f'K_sep native/methane ratio: {min(nat[t]["K_sep_Nm"]/meth[t]["Ksep"] for t in SYS):.2f}'
          f' to {max(nat[t]["K_sep_Nm"]/meth[t]["Ksep"] for t in SYS):.2f}')

    out = dict(clamp='r255', half=0.15, n_systems=len(SYS),
               n_negative_native=int(n_neg_nat), n_negative_methane=int(n_neg_met),
               primary_endpoint=verdict,
               per_system={t: dict(KIE=KIE[t], K_sep_methane=meth[t]['Ksep'],
                                   K_sep_native=nat[t]['K_sep_Nm'],
                                   K_perp_exch=nat[t]['K_perp_exch_eigs'],
                                   K_perp_int=nat[t]['K_perp_int_eigs'],
                                   ratio=nat[t]['ratio_Ksep_over_specnorm_exch']) for t in SYS},
               rank={k: dict(zip(('rho', 'p'), exact_rho(ks, v))) for k, v in summaries.items()})
    (D / 'panel_summary.json').write_text(json.dumps(out, indent=2))
    print(f'\nwrote {D}/panel_summary.json')


if __name__ == '__main__':
    main()
