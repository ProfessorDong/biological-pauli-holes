#!/usr/bin/env python3
"""Collect every L754A quantity, and every panel quantity it enters, for the pro-R (H26) and
pro-S (H25) hydrogens in both protonation models, from named fields of named result files.

Trees: S = published model S (H26), SpS = model S with H25, P = model P (H26), PpS = model P
with H25. Writes results/protonation_sensitivity/proS/summary_proS.json and prints it as a table.
Usage: psens_proS_summary.py
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PS = ROOT / 'results/protonation_sensitivity'
TREES = {'S': ROOT / 'results', 'SpS': PS / 'shadow_SpS/results',
         'P': PS / 'shadow_P/results', 'PpS': PS / 'shadow_PpS/results'}
TAGS = ('I538A', 'I552A', 'I553A', 'L546A', 'L754A', 'V750A', 'WT')
T = 'L754A'


def J(p):
    return json.loads(Path(p).read_text())


def cls2(l, se):
    if abs(l) < 2 * se:
        return 'unresolved'
    return 'negative' if l < 0 else 'positive'


def defin(e):
    e = sorted(e)
    return 'negative definite' if e[1] < 0 else ('positive definite' if e[0] > 0 else 'indefinite')


def ranks(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind='mergesort')
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[o[j + 1]] == x[o[i]]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def pearson_perm(x, y):
    """Pearson R and its exact two-sided permutation p over all orderings of x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r0 = float(np.corrcoef(x, y)[0, 1])
    hits = sum(abs(np.corrcoef(np.array(q), y)[0, 1]) >= abs(r0) - 1e-12 for q in itertools.permutations(x))
    return r0, hits / math.factorial(len(x))


def frames(R, c):
    """The collated per-frame list the analyses read (the published r255 per-frame files are not kept)."""
    v = np.array(J(R / f'ensemble_fluctuation/kexch_frames_{c}.json')[T], float)
    assert np.all(np.isfinite(v))
    return dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)), min=float(v.min()), max=float(v.max()),
                n_positive=int((v > 0).sum()))


def tree(m):
    R = TREES[m]
    o = {}
    o['static_Ksep'] = {c: J(R / f'reactive_geometry/{T}_{c}_native_result.json')['k_exch_Nm'] for c in ('r255', 'r340')}
    o['static_Ksep_all_positive'] = all(J(R / f'reactive_geometry/{t}_{c}_native_result.json')['k_exch_Nm'] > 0
                                        for t in TAGS for c in ('r255', 'r340'))
    o['ensemble_Ksep'] = {c: frames(R, c) for c in ('r255', 'r340')}
    if m in ('S', 'SpS'):
        nf = J(R / f'sapt_bio/native_fragment/{T}_native_result.json')
        un = J(R / 'sapt_bio/native_fragment_uncertainty.json')
        mw = J((PS / 'proS/methane_wall_validation' if m == 'S' else PS / 'proS/methane_wall_S') / f'{T}_methane_wall_result.json')
        dc = J(R / f'sapt_bio/donor_fragment/{T}_donor_comparison.json')
        o['near_attack'] = dict(k_native_wall=nf['k_exch_Nm'], rel_sigma=un['systems'][T]['rel_sigma'],
                                rel_sigma_range=un['rel_sigma_range'], r_HW=un['systems'][T]['r_HW_A'],
                                panel_k=[min(un['systems'][t]['k_exch_Nm'] for t in TAGS),
                                         max(un['systems'][t]['k_exch_Nm'] for t in TAGS)],
                                k_methane_wall=mw['k_Nm'], methane_wall_atom=mw['wall_atom'],
                                k_C5H8=dc['results']['pentadienyl']['k_Nm'], k_methane_donor=dc['results']['methane']['k_Nm'],
                                ratio_C5H8_over_methane=dc['ratio_pentadienyl_over_methane'], donor_wall_atom=dc['wall_atom'])
    th = {t: J(R / f'transverse_hessian/{t}_r255_w015_transverse_hessian.json') for t in TAGS}
    tr = {t: float(np.sum(th[t]['eigenvalues_Nm'])) for t in TAGS}
    o['methane_transverse'] = dict(eig=sorted(th[T]['eigenvalues_Nm']), trace=tr[T],
                                   panel_trace=[min(tr.values()), max(tr.values())],
                                   n_negative=sum(1 for t in TAGS if min(th[t]['eigenvalues_Nm']) < 0))
    tv = J(R / 'transverse_hessian/transverse_vs_separation.json')
    o['zpe_bound_percent_of_ladder'] = tv['max_percent_of_ladder']
    eu = J(R / 'native_donor_validation/eigen_uncertainty.json')
    nat = {t: J(R / f'native_donor_validation/{t}_r255_w015_native_transverse.json') for t in TAGS}
    l1, l2, se = eu[T]['eigs'][0], eu[T]['eigs'][1], eu[T]['se'][0]
    ratios = {t: nat[t]['K_sep_Nm'] / max(abs(x) for x in eu[t]['eigs']) for t in TAGS}
    classes = {t: cls2(eu[t]['eigs'][0], eu[t]['se'][0]) for t in TAGS}
    kint = {t: defin(nat[t]['K_perp_int_eigs']) for t in TAGS}
    o['native'] = dict(lambda1=l1, lambda2=l2, se=se, z=abs(l1) / se, cls=classes[T], K_sep_native=nat[T]['K_sep_Nm'],
                       ratio=ratios[T], panel_ratio=[min(ratios.values()), max(ratios.values())],
                       counts={c: sum(v == c for v in classes.values()) for c in ('negative', 'positive', 'unresolved')},
                       K_int_eig=sorted(nat[T]['K_perp_int_eigs']), K_int_cls=kint[T],
                       K_int_negative_definite=sum(v == 'negative definite' for v in kint.values()))
    if m in ('S', 'SpS'):
        hw = {}
        for h in ('005', '010', '015', '020'):
            d = J(R / f'native_donor_validation/{T}_r255_w{h}_native_transverse.json')
            hw[h] = sorted(d['K_perp_exch_eigs'])[0]
        o['half_width_lambda1'] = hw
        o['half_width_spread_percent'] = 100 * (max(hw.values()) - min(hw.values())) / abs(hw['015'])
        bb = J(R / f'native_donor_validation/stage4/s4_wall_bb_{T}.json')
        o['stage4_wall_bb'] = dict(lambda1=sorted(bb['K_perp_exch_eigs'])[0], lambda2=sorted(bb['K_perp_exch_eigs'])[1],
                                   change_percent=100 * (sorted(bb['K_perp_exch_eigs'])[0] - l1) / abs(l1))
    tp = J(R / 'native_donor_validation/total_restoring_panel.json')
    ps = tp['per_system']
    eigs = [x for t in TAGS for key in ('K_tot_complex_HF', 'K_tot_complex_B3LYP') for x in ps[t][key]]
    o['total_restoring'] = dict(L754A_HF=ps[T]['K_tot_complex_HF'], L754A_B3LYP=ps[T]['K_tot_complex_B3LYP'],
                                panel_eig=[min(eigs), max(eigs)], all_positive_definite=min(eigs) > 0,
                                L754A_share_HF_percent=100 * ps[T]['exchange_fraction_HF'],
                                median_share_HF_percent=float(np.median([100 * ps[t]['exchange_fraction_HF'] for t in TAGS])),
                                median_share_B3LYP_percent=float(np.median([100 * ps[t]['exchange_fraction_B3LYP'] for t in TAGS])),
                                share_HF_range=[min(100 * ps[t]['exchange_fraction_HF'] for t in TAGS),
                                                max(100 * ps[t]['exchange_fraction_HF'] for t in TAGS)],
                                sign_agreement=tp['sign_agreement'])
    du = J(R / 'native_donor_validation/duarte_projection_check.json')['summary']
    o['additive_projection'] = {k: dict(span=du[k]['additive_share_span_percent'],
                                        systems_negative=du[k]['systems_with_a_negative_share']) for k in ('HF', 'B3LYP')}
    s5 = J(R / 'native_donor_validation/stage5_ensemble_panel.json')
    kint5 = [defin(J(f)['K_perp_int_eigs']) for f in sorted((R / 'stage5_configs' / T).glob(f'{T}_cfg*_w015_transverse.json'))]
    o['stage5'] = dict(n=s5[T]['n'], mean=s5[T]['mean_smaller'], sem=s5[T]['sem'], frac_negative=s5[T]['frac_negative'],
                       min=s5[T]['min'], max=s5[T]['max'],
                       n_negative=int(round(s5[T]['frac_negative'] * s5[T]['n'])),
                       K_int={c: kint5.count(c) for c in sorted(set(kint5))},
                       panel_mean_negative=sum(1 for t in TAGS if s5[t]['mean_smaller'] < 0),
                       panel_median_negative=sum(1 for t in TAGS if float(np.median(s5[t]['all_smaller'])) < 0))
    xi = J(R / 'pauli_hole_survey/slo_xi0_full.json')
    o['contact'] = dict(L754A={k: (v['xi0_A'], v['sd']) for k, v in xi.items() if k.startswith(T)},
                        panel=[min(v['xi0_A'] for v in xi.values()), max(v['xi0_A'] for v in xi.values())])
    cb = J(R / 'confinement_budget.json')

    def shell(v):
        c = v['categories']
        return c['substrate framework']['fraction'] + c['acceptor hydroxide O']['fraction'] + c['cofactor first shell']['fraction']
    o['budget'] = dict(L754A={k: dict(within_2p5=v['cumulative_fraction_within']['2.5'], shell=shell(v),
                                      side=v['categories']['JBC cavity side chain']['fraction'])
                              for k, v in cb.items() if k.startswith(T)},
                       within_2p5=[min(v['cumulative_fraction_within']['2.5'] for v in cb.values()),
                                   max(v['cumulative_fraction_within']['2.5'] for v in cb.values())],
                       shell=[min(shell(v) for v in cb.values()), max(shell(v) for v in cb.values())],
                       side_max=max(v['categories']['JBC cavity side chain']['fraction'] for v in cb.values()))
    st, fl = J(R / 'reactive_geometry/analysis_result.json'), J(R / 'ensemble_fluctuation/analysis_result.json')
    o['static_test'] = {c: dict(partial=st[c]['partial'], p=st[c]['p_exact'], ci=st[c]['ci']) for c in ('reactive', 'reference')}
    o['fluctuation_test'] = {c: {e: dict(partial=fl[c][e]['partial'], p=fl[c][e]['p_exact'], ci=fl[c][e]['ci'])
                                 for e in ('D1_mean', 'D2_sd', 'D3_rel')} for c in ('reactive', 'reference')}
    ep = J(R / 'sapt_bio/donor_fragment/exponential_law_panel.json')
    pf = J(R / 'sapt_bio/donor_fragment/exponential_law_prefactor.json')
    o['decay_panel'] = dict(n=ep['n'], k=[ep['k_min'], ep['k_max']], rate=ep['common_rate_per_A'], ratio=ep['ratio'],
                            r2=[min(v['r2'] for v in ep['per_system'].values()), max(v['r2'] for v in ep['per_system'].values())],
                            vif=pf['vif'])
    sp = PS / f'scalar_spearman_{m}.json'
    o['spearman_methane'] = {k: v for k, v in J(sp).items() if not k.startswith('_')}
    inp = J(sp)['_inputs']
    loo = {}
    for drop in TAGS:
        keep = [t for t in TAGS if t != drop]
        loo[drop] = {s_: spearman([inp['K_sep'][t] for t in keep], [inp[s_][t] for t in keep]) for s_ in ('trace', 'spectral')}
    o['spearman_methane_leave_one_out'] = loo
    rhw = {}
    for t in TAGS:
        g = J(R / f'reactive_geometry/{t}_r255_geometry.json')
        h = np.array(g['xferH'])
        rhw[t] = float(min(np.linalg.norm(np.array(a['position']) - h) for a in g['native_wall'] if a['element'] != 'H'))
    o['r_HW_static_r255'] = rhw
    span = math.log(148 / 62)
    zb = J(R / 'transverse_hessian/transverse_vs_separation.json')
    o['zpe_percent_of_ln_span'] = {int(z['k_cov_Nm']): 100 * math.log(z['kie_factor']) / span for z in zb['zpe_bound']}
    o['methane_trace_by_variant'] = dict(zip(zb['rank']['tags'], zb['rank']['trace_K_perp']))
    if m in ('S', 'SpS'):
        kie = J(ROOT / 'results/sapt_bio/sapt_summary.json')
        y = np.log([kie[t]['KIE'] for t in TAGS])
        dc = {t: J(R / f'sapt_bio/donor_fragment/{t}_donor_comparison.json')['results'] for t in TAGS}
        o['near_attack_kie'] = {d: dict(zip(('R', 'p_exact'), pearson_perm([dc[t][d]['k_Nm'] for t in TAGS], y)))
                                for d in ('methane', 'pentadienyl')}
    return o


def main():
    out = {m: tree(m) for m in TREES}
    out['_orientation'] = J(PS / 'proS/orientation.json')
    (PS / 'proS/summary_proS.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({m: out[m] for m in TREES}, indent=1, default=float))


if __name__ == '__main__':
    main()
