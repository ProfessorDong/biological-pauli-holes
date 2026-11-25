#!/usr/bin/env python3
"""Side-by-side comparison of model S (published) and model P for the declared quantities.

Reads only result files: model S from results/ (published), model P from shadow_P/results/.
Applies the declared rules (results/protonation_sensitivity/DECLARATION.md): eigenvalue
classes at 2 sigma; the registered interpretation sets for the two tests; K_int classes
from the sign of both eigenvalues. Writes comparison.json.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PS = ROOT / 'results/protonation_sensitivity'
TAGS = ('I538A', 'I552A', 'I553A', 'L546A', 'L754A', 'V750A', 'WT')
SRC = {'S': ROOT / 'results', 'P': PS / 'shadow_P/results',
       'SpS': PS / 'shadow_SpS/results', 'PpS': PS / 'shadow_PpS/results'}
RUNS = {'S': (ROOT / 'results/reactive_geometry', ROOT / 'results/ensemble_fluctuation'),
        'P': (PS / 'reactive_geometry', PS / 'ensemble_fluctuation')}
RUNS['SpS'], RUNS['PpS'] = RUNS['S'], RUNS['P']   # the H25 sets reuse the same trajectories


def J(p):
    return json.loads(Path(p).read_text())


def cls2(l, se):
    if abs(l) < 2 * se:
        return 'unresolved'
    return 'negative' if l < 0 else 'positive'


def defin(e):
    e = sorted(e)
    if e[1] < 0:
        return 'negative definite'
    if e[0] > 0:
        return 'positive definite'
    return 'indefinite'


def model(m):
    R = SRC[m]
    out = {}
    rg, ef = RUNS[m]
    out['Q1_rDA'] = {f'{t}_{c}': dict(static=float(np.loadtxt(rg / f'{t}_{c}_colvar.dat').mean()),
                                      ensemble=float(np.loadtxt(ef / f'{t}_{c}_colvar.dat').mean()))
                     for t in TAGS for c in ('r255', 'r340')}
    xi = J(R / 'pauli_hole_survey/slo_xi0_full.json')
    out['Q2_contact'] = {k: round(v['xi0_A'], 3) for k, v in xi.items()}
    out['Q2_contact_range'] = [round(min(v['xi0_A'] for v in xi.values()), 3),
                               round(max(v['xi0_A'] for v in xi.values()), 3)]
    cb = J(R / 'confinement_budget.json')
    w25 = [v['cumulative_fraction_within']['2.5'] for v in cb.values()]
    side = [v['categories']['JBC cavity side chain']['fraction'] for v in cb.values()]
    shell = [v['categories']['substrate framework']['fraction'] + v['categories']['acceptor hydroxide O']['fraction']
             + v['categories']['cofactor first shell']['fraction'] for v in cb.values()]
    out['Q3_budget'] = dict(within_2p5A=[min(w25), max(w25)], mutated_side_chains=[min(side), max(side)],
                            bonding_shell=[min(shell), max(shell)],
                            tail_beyond_8A_max=max(v['tail_beyond_8A'] for v in cb.values()))
    out['Q4_static_test'] = J(R / 'reactive_geometry/analysis_result.json')
    out['Q5_fluctuation_test'] = J(R / 'ensemble_fluctuation/analysis_result.json')
    ep = J(R / 'sapt_bio/donor_fragment/exponential_law_panel.json')
    pf = J(R / 'sapt_bio/donor_fragment/exponential_law_prefactor.json')
    out['Q6_panel'] = dict(n=ep['n'], d=[ep['d_min'], ep['d_max']], k=[ep['k_min'], ep['k_max']],
                           common_rate_per_A=ep['common_rate_per_A'], ratio_to_solved=ep['ratio'],
                           r2=[min(v['r2'] for v in ep['per_system'].values()),
                               max(v['r2'] for v in ep['per_system'].values())],
                           vif=pf['vif'], identified=pf['identified'])
    th = {t: J(R / f'transverse_hessian/{t}_r255_w015_transverse_hessian.json') for t in TAGS}
    out['Q7_methane'] = {t: dict(K_sep=th[t]['K_sep_Nm'], eig=sorted(th[t]['eigenvalues_Nm'])) for t in TAGS}
    out['Q7_methane_n_negative'] = sum(1 for t in TAGS if min(th[t]['eigenvalues_Nm']) < 0)
    eu = J(R / 'native_donor_validation/eigen_uncertainty.json')
    nat = {t: J(R / f'native_donor_validation/{t}_r255_w015_native_transverse.json') for t in TAGS}
    q8 = {}
    for t in TAGS:
        l1, se = eu[t]['eigs'][0], eu[t]['se'][0]
        kint = sorted(nat[t]['K_perp_int_eigs'])
        q8[t] = dict(K_sep_native_axis=nat[t].get('K_sep_Nm'), lambda1=l1, lambda2=eu[t]['eigs'][1], se=se,
                     z=abs(l1) / se, cls=cls2(l1, se), K_int_eig=kint, K_int_cls=defin(kint) if kint else None)
    out['Q8_native'] = q8
    out['Q8_counts'] = dict(negative=sum(v['cls'] == 'negative' for v in q8.values()),
                            positive=sum(v['cls'] == 'positive' for v in q8.values()),
                            unresolved=sum(v['cls'] == 'unresolved' for v in q8.values()),
                            K_int_negative_definite=sum(v['K_int_cls'] == 'negative definite' for v in q8.values()))
    tr = J(R / 'native_donor_validation/total_restoring_panel.json')
    out['Q9_total'] = dict(per_system=tr['per_system'], sign_agreement=tr['sign_agreement'],
                           exchange_fraction_HF_percent=tr['exchange_fraction_HF_percent'],
                           exchange_fraction_B3LYP_percent=tr['exchange_fraction_B3LYP_percent'])
    s5 = J(R / 'native_donor_validation/stage5_ensemble_panel.json')
    q10 = {}
    for t in TAGS:
        files = sorted((R / 'stage5_configs' / t).glob(f'{t}_cfg*_w015_transverse.json'))
        lam, kint = [], []
        for f in files:
            d = J(f)
            lam.append(min(d['K_perp_exch_eigs']))
            kint.append(defin(d['K_perp_int_eigs']))
        q10[t] = dict(n=s5[t]['n'], frac_negative=s5[t]['frac_negative'], mean_smaller=s5[t]['mean_smaller'],
                      sem=s5[t]['sem'], K_int_classes={c: kint.count(c) for c in set(kint)} if kint else None)
    out['Q10_stage5'] = q10
    sp = PS / f'scalar_spearman_{m}.json'
    out['Q11_spearman'] = {k: v for k, v in J(sp).items() if not k.startswith('_')} if sp.exists() else None
    return out


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'proS':
        res = {m: model(m) for m in ('S', 'SpS', 'P', 'PpS')}
        (PS / 'proS/comparison_proS.json').write_text(json.dumps(res, indent=1, default=float))
        print('wrote proS/comparison_proS.json')
        return
    res = {'S': model('S'), 'P': model('P')}
    (PS / 'comparison.json').write_text(json.dumps(res, indent=1, default=float))
    S, P = res['S'], res['P']
    print('Q2 contact range  S', S['Q2_contact_range'], ' P', P['Q2_contact_range'])
    print('Q3 within 2.5 A   S', np.round(S['Q3_budget']['within_2p5A'], 3), ' P', np.round(P['Q3_budget']['within_2p5A'], 3))
    for c in ('reactive', 'reference'):
        print(f'Q4 static {c:9s} S partial {S["Q4_static_test"][c]["partial"]:+.3f} p {S["Q4_static_test"][c]["p_exact"]:.3f}'
              f'  P partial {P["Q4_static_test"][c]["partial"]:+.3f} p {P["Q4_static_test"][c]["p_exact"]:.3f}')
    print('Q6 rate', round(S['Q6_panel']['common_rate_per_A'], 2), round(P['Q6_panel']['common_rate_per_A'], 2),
          ' VIF', round(S['Q6_panel']['vif'], 1), round(P['Q6_panel']['vif'], 1))
    print('Q7 methane negatives S', S['Q7_methane_n_negative'], ' P', P['Q7_methane_n_negative'])
    print('Q8 counts S', S['Q8_counts'], '\n          P', P['Q8_counts'])
    for t in TAGS:
        print(f'   {t:6s} S {S["Q8_native"][t]["cls"]:10s} {S["Q8_native"][t]["lambda1"]:+.4f}  '
              f'P {P["Q8_native"][t]["cls"]:10s} {P["Q8_native"][t]["lambda1"]:+.4f} +/- {P["Q8_native"][t]["se"]:.4f}')
    print('Q9 sign agreement S', S['Q9_total']['sign_agreement'], ' P', P['Q9_total']['sign_agreement'])
    for t in TAGS:
        print(f'Q10 {t:6s} S frac_neg {S["Q10_stage5"][t]["frac_negative"]:.2f} n {S["Q10_stage5"][t]["n"]}  '
              f'P frac_neg {P["Q10_stage5"][t]["frac_negative"]:.2f} n {P["Q10_stage5"][t]["n"]}')


if __name__ == '__main__':
    main()
