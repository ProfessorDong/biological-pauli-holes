#!/usr/bin/env python3
r"""Audit every figure in the article and the supplement, against its data and its own artwork.

WHY THIS EXISTS
  verify_figure_claims.py covers three supplement figures and only the numbers introduced by the
  2026-08 audits. Nothing checked the other eight, nothing checked that a live figure is still the
  one its generator produces, and nothing checked the text baked INTO the artwork. Three real
  defects got through: two figures labelled the measured curvature "transverse" after the paper
  had retracted that description, a withdrawn correlation was drawn with an unhedged significance
  legend, and a panel title claimed six decades where one of the two curves spans 5.93.

WHAT IT CHECKS
  1. Every \includegraphics target resolves to a file that exists.
  2. Each live figure is byte-identical to the artwork its generator writes, so a regenerated
     figure that was never copied into the build is caught.
  3. The text embedded in each figure PDF: required strings must be present, retracted ones absent.
  4. Caption numbers against a NAMED FIELD of a NAMED FILE.
  5. Caption length, since a caption nobody reads documents nothing.

Usage: verify_figures.py     (pauli env; exit status is the failure count)
"""

import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(_REPO)
LIVE, FIGS, R = ROOT / 'prxlife', ROOT / 'figures', ROOT / 'results'
MAIN, APP = LIVE / 'main.tex', LIVE / 'appendices.tex'
CAPTION_MAX_WORDS = 245

# live figure -> the artwork its generator writes (None when the source is not tracked here)
GENERATED = {
    'fig1_selection_rule.pdf': FIGS / 'fig1_selection_rule.pdf',
    # the TikZ rebuild, which reads fig1_selection_rule.dat rather than carrying numbers
    'fig1_selection_rule_v2.pdf': FIGS / 'fig1_selection_rule_tikz.pdf',
    'fig1_pauli_pocket.pdf': FIGS / 'fig1_pauli_pocket_tikz.pdf',
    'fig2_index.pdf': FIGS / 'fig2_index_tikz.pdf',
    'fig3_fields_new.pdf': FIGS / 'fig3_fields_tikz.pdf',
    # rebuilt in TikZ 2026-09-23; reads fig_budget_{a,b}.dat rather than carrying numbers
    'fig_budget.pdf': FIGS / 'fig_budget_tikz.pdf',
    # rebuilt in TikZ 2026-09-23; reads fig_native_donor*.dat
    'fig_native_donor.pdf': FIGS / 'fig_native_donor_tikz.pdf',
    # rebuilt in TikZ 2026-09-23; reads fig_coordinate.dat
    'fig_coordinate.pdf': FIGS / 'fig_coordinate_tikz.pdf',
    # rebuilt in TikZ 2026-09-23; reads figED1_*.dat and generated label macros
    'figED1_sapt_scans.pdf': FIGS / 'figED1_sapt_scans_tikz.pdf',
    # rebuilt in TikZ 2026-09-23; reads fig6_*.dat and generated macros
    'fig6_pimd_density.pdf': FIGS / 'fig6_pimd_density_tikz.pdf',
    # rebuilt in TikZ 2026-09-23; reads figED2_*.dat
    'figED2_bead_convergence.pdf': FIGS / 'figED2_bead_convergence_tikz.pdf',
    'figED3_gating_KIE.pdf': FIGS / 'figED3_gating_KIE_tikz.pdf',
}

# text that must appear in the artwork, and text that must NOT
ARTWORK_REQUIRED = {
    # The panel once built a harmonic ZPE from the EXCHANGE curvature, which is not
    # positive definite in six of seven and so supplies no well. It now draws the total
    # and says what exchange contributes to it; both statements must stay in the artwork.
    'fig1_pauli_pocket.pdf': ['K⊥', 'tot', 'either sign'],
    'fig2_index.pdf': ['K⊥', 'Ksep'],
    'figED1_sapt_scans.pdf': ['wall displacement'],
    'fig_coordinate.pdf': ['H moves', 'wall moves'],
    'fig_native_donor.pdf': ['methane donor', 'native'],
}
ARTWORK_FORBIDDEN = {
    # the measured curvature is a fragment separation, so no figure may label it transverse
    'fig1_pauli_pocket.pdf': ['kexch'],
    # The revision-2 audit withdrew every biological bound on the confinement index.
    # The artwork went on printing "bounded Pi_H <= 8.2" while its own caption said the
    # opposite; only the caption had been corrected. Both forms are forbidden now.
    'fig2_index.pdf': ['kexch', 'bounded', '8.2'],
    # the sphere spans 5.93 decades, so "six decades" overstates it
    'fig1_selection_rule.pdf': ['six decades'],
    'fig1_selection_rule_v2.pdf': ['six decades'],
    # The 3.73 A line was labelled "equilibrium mean" IN THE ARTWORK while the caption
    # correctly called it a stratified snapshot mean. The caption was policed; the artwork
    # was not, so the contradiction survived. Both are checked now.
    'fig3_fields_new.pdf': ['160×', 'equilibrium'],
    # the 88-118 deg range is from the seven clamped geometries; the ensemble
    # frames store no acceptor oxygen, so 'sampled' invited a reading the data
    # cannot support
    'fig_coordinate.pdf': ['sampled C'],
    # the five snapshots are from BIASED windows; the artwork called them equilibrium
    # while the caption said the opposite. The generator's own variable was
    # r_DA_equilibrium, which is how the word reached the label.
    'figED3_gating_KIE.pdf': ['equilibrium'],
}


def txt(pdf):
    return re.sub(r'\s+', ' ', subprocess.run(
        ['pdftotext', str(pdf), '-'], capture_output=True, text=True).stdout)


def captions(texfile):
    s = texfile.read_text()
    out = []
    for m in re.finditer(r'\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}', s, re.S):
        blk = m.group(1)
        img = re.search(r'includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', blk)
        c = re.search(r'\\caption\{', blk)
        if not (img and c):
            continue
        i, d, j = c.end(), 1, c.end()
        while j < len(blk) and d:
            d += (blk[j] == '{') - (blk[j] == '}')
            j += 1
        out.append((img.group(1), blk[c.end():j - 1]))
    return out


def main():
    fails = []
    rows = []

    def ck(ok, label, detail=''):
        rows.append(('OK  ' if ok else 'FAIL', label, detail))
        if not ok:
            fails.append(label)

    # The separate Supplemental Material was retired on 2026-09-18 and folded into the
    # appendices, so every figure is now part of the single article document.
    allcaps = captions(LIVE / 'main.tex') + captions(LIVE / 'appendices.tex')
    capmap = dict(allcaps)

    def caption_of(name):
        return capmap.get(name, '')

    # fig4_dispersion was removed 2026-09-23: it plotted a withdrawn correlation, and a
    # figure is quoted more easily than its caption. The appendix section is retained.
    ck(len(allcaps) == 12, 'figure count is 12', f'found {len(allcaps)}')

    for name, cap in allcaps:
        f = LIVE / name
        ck(f.exists(), f'{name} exists')
        if not f.exists():
            continue
        src = GENERATED.get(name)
        if src and src.exists():
            same = hashlib.md5(f.read_bytes()).digest() == hashlib.md5(src.read_bytes()).digest()
            ck(same, f'{name} matches its generator', '' if same else f'differs from {src.name}')
        t = txt(f)
        for need in ARTWORK_REQUIRED.get(name, []):
            ck(need in t, f'{name} artwork carries "{need}"')
        for bad in ARTWORK_FORBIDDEN.get(name, []):
            ck(bad not in t, f'{name} artwork free of "{bad}"')
        w = len(cap.split())
        ck(w <= CAPTION_MAX_WORDS, f'{name} caption under {CAPTION_MAX_WORDS} words', f'{w} words')

    # caption numbers against named fields
    main_txt = txt(LIVE / 'main.pdf')
    supp_txt = main_txt
    dc = json.loads((R / 'derived_constants.json').read_text())['confinement_curve_decade_spans']
    ck(f'{dc["paraboloid"]:.2f}' in main_txt, 'fig1 paraboloid decade span',
       'derived_constants.json :: confinement_curve_decade_spans.paraboloid')
    ck(f'{dc["sphere"]:.2f}' in main_txt, 'fig1 sphere decade span',
       'derived_constants.json :: confinement_curve_decade_spans.sphere')

    # The two model ratios must still be traceable where they are quoted as MODEL results,
    # but they must NOT reappear as bounds on a biological index. An audit found the
    # inequality surviving in the appendix and in this figure's artwork after both had been
    # described as withdrawn, so the check is now inverted: presence of the numbers is
    # required in the model discussion, presence of the inequality is forbidden anywhere.
    kt = json.loads((R / 'kbt_ceiling_both_geometries.json').read_text())['by_distance']['2.4']
    ck(f'{kt["sphere"]["dE_over_kT"]:.1f}' in main_txt, 'model ratio 8.2 still traceable',
       'kbt_ceiling_both_geometries.json :: by_distance.2.4.sphere.dE_over_kT')
    ck(f'{kt["paraboloid"]["dE_over_kT"]:.2f}' in main_txt, 'model ratio 1.02 still traceable',
       'kbt_ceiling_both_geometries.json :: by_distance.2.4.paraboloid.dE_over_kT')
    for bad, what in [('\\Pi_H\\le8.2', 'Pi_H <= 8.2'), ('\\Pi_H\\le1.02', 'Pi_H <= 1.02')]:
        ck(bad not in APP.read_text() and bad not in MAIN.read_text(),
           f'no biological bound "{what}" in source')
    ck('accessible' not in txt(LIVE / 'fig5_decisive.pdf'),
       'decisive-figure artwork carries no accessible band')

    mb = json.loads((R / 'umbrella_reactplane/analysis/mbar_2d_regression.json').read_text())
    ck(f'{abs(mb["pearson_R"]):.2f}' in supp_txt, 'fig4 caption gives the MBAR result',
       'mbar_2d_regression.json :: pearson_R')

    # The stored field is named equilibrium_r_DA_mean, but the quantity is the unweighted
    # mean of five stratified umbrella snapshots and is not an equilibrium average. The
    # number must still trace to its file; the caption must not call it equilibrium.
    gt = json.loads((R / 'pcet_B34_dense/marcus_rate_A_v3.json').read_text())
    ck(f'{gt["equilibrium_r_DA_mean"]:.2f}' in main_txt, 'gating figure snapshot mean traceable',
       'marcus_rate_A_v3.json :: equilibrium_r_DA_mean (a stratified snapshot mean)')
    ck(f'{gt["r_reactive_KIE66"]:.2f}' in main_txt, 'gating figure inferred distance at WT',
       'marcus_rate_A_v3.json :: r_reactive_KIE66 (model-inferred, not measured)')
    cap = caption_of('figED3_gating_KIE.pdf')
    # The snapshots may be DESCRIBED as not being an equilibrium sample; what is forbidden
    # is labelling them as one, or claiming the inversion pins a population.
    low = cap.lower()
    for bad in ['equilibrium $r', 'equilibrium range', 'equilibrium mean',
                'equilibrium average', 'equilibrium distribution of']:
        ck(bad not in low, f'gating caption does not label snapshots "{bad}"')
    ck('not an equilibrium' in low, 'gating caption says the snapshots are not an equilibrium sample')
    ck('pins' not in low and 'rare tail' not in low,
       'gating caption claims no rare-tail population')

    # native-donor validation: every headline number traced to its result file
    nd = json.loads((R / 'native_donor_validation/panel_summary.json').read_text())
    eu = json.loads((R / 'native_donor_validation/eigen_uncertainty.json').read_text())
    ck(nd['n_negative_native'] == 5, 'native panel: 5 of 7 negative',
       'panel_summary.json :: n_negative_native')
    ck('SURVIVES' in nd['primary_endpoint'], 'native panel: declared endpoint survives',
       'panel_summary.json :: primary_endpoint')
    ck(f'{nd["n_negative_native"]} of seven' in main_txt or 'five of seven' in main_txt.lower(),
       'main text states five of seven')
    rat = [v['ratio'] for v in nd['per_system'].values()]
    ck(f'{min(rat):.1f}' in main_txt and f'{max(rat):.1f}' in main_txt,
       'native spectral-norm ratio range in text',
       'panel_summary.json :: per_system[*].ratio')
    ks = [v['K_sep_native'] / v['K_sep_methane'] for v in nd['per_system'].values()]
    ck(f'{min(ks):.2f}' in main_txt and f'{max(ks):.2f}' in main_txt,
       'donor sensitivity of K_sep in text',
       'panel_summary.json :: K_sep_native / K_sep_methane')
    nint = sum(1 for v in nd['per_system'].values() if v['K_perp_int'][1] < 0)
    ck(nint == 6, 'total interaction negative definite in six of seven',
       'panel_summary.json :: per_system[*].K_perp_int')
    v750 = eu['V750A']
    ck(abs(v750['eigs'][0]) / v750['se'][0] < 2, 'V750A unresolved at 2 sigma',
       'eigen_uncertainty.json :: V750A')
    ck('unresolved' in main_txt, 'main text reports V750A as unresolved')
    ck(f'{abs(v750["eigs"][0]):.4f}' in main_txt, 'V750A eigenvalue quoted',
       'eigen_uncertainty.json :: V750A.eigs[0]')

    # stage 3: the total constrained restoring matrix and the exchange fraction
    tr = json.loads((R / 'native_donor_validation/total_restoring_panel.json').read_text())
    ck(tr['n_systems'] == 7, 'total-restoring panel covers seven systems',
       'total_restoring_panel.json :: n_systems')
    posdef = all(v['K_tot_complex_HF'][0] > 0 and v['K_tot_complex_B3LYP'][0] > 0
                 for v in tr['per_system'].values())
    ck(posdef, 'total restoring matrix positive definite in every system',
       'total_restoring_panel.json :: per_system[*].K_tot_complex_*')
    lo = min(min(v['K_tot_complex_B3LYP']) for v in tr['per_system'].values())
    hi = max(max(v['K_tot_complex_HF']) for v in tr['per_system'].values())
    ck(f'{lo:.0f}' in main_txt and f'{hi:.0f}' in main_txt,
       'total-restoring eigenvalue range quoted', 'per_system[*].K_tot_complex_*')
    fh = sorted(tr['exchange_fraction_HF_percent'])
    med = fh[len(fh) // 2]
    ck(f'{med:.3f}' in main_txt, 'median exchange fraction quoted',
       'total_restoring_panel.json :: exchange_fraction_HF_percent median')
    ck(f'{min(fh):.3f}' in APP.read_text() and f'{max(fh):.3f}' in APP.read_text(),
       'exchange fraction range in the appendix',
       'total_restoring_panel.json :: exchange_fraction_HF_percent')
    ck(tr['sign_agreement'].startswith('6/'), 'wall check agrees in six of seven',
       'total_restoring_panel.json :: sign_agreement')

    # stage 2: transfer barrier against donor-acceptor distance, no kinetic data used
    tb = json.loads((R / 'transfer_barrier/barrier_result.json').read_text())
    import numpy as _np
    bx = _np.array([r['symm']['r_DA'] for r in tb])
    by = _np.array([r['barrier_kcal'] for r in tb])
    ck(len(tb) == 5, 'transfer-barrier scan covers five distances',
       'barrier_result.json :: length')
    ck(all(abs(r['symm']['r_DA'] - r['r_DA_target']) < 1e-3 and
           abs(r['react']['r_DA'] - r['r_DA_target']) < 1e-3 for r in tb),
       'r_DA constraint satisfied in both states', 'barrier_result.json :: r_DA')
    ck(all(by[i] < by[i+1] for i in range(len(by)-1)),
       'barrier increases monotonically with distance', 'barrier_result.json :: barrier_kcal')
    slope = _np.polyfit(bx, by, 1)[0]
    r2 = _np.corrcoef(bx, by)[0, 1] ** 2
    ck(f'{slope:.1f}' in main_txt, 'barrier slope quoted in main text',
       'barrier_result.json :: slope kcal/mol per A')
    ck(f'{r2:.3f}' in main_txt, 'barrier fit R^2 quoted', 'barrier_result.json :: R^2')
    ck(f'{by.max()-by.min():.1f}' in main_txt, 'barrier drop quoted',
       'barrier_result.json :: max-min')

    th = json.loads((R / 'transverse_hessian/transverse_vs_separation.json').read_text())
    prim = [r for r in th['records'] if r['clamp'] == 'r255' and abs(r['half'] - 0.15) < 1e-9]
    eigs = [w for r in prim for w in (r['w1'], r['w2'])]
    ck(f'{min(eigs):.3f}'.replace('-', '\u2212') in main_txt, 'fig_coordinate min eigenvalue',
       'transverse_vs_separation.json :: records[r255].w1')
    ck(f'+{max(eigs):.3f}' in main_txt, 'fig_coordinate max eigenvalue',
       'transverse_vs_separation.json :: records[r255].w2')

    pi = json.loads((R / 'pimd_prod/analysis/pimd_analysis_v2.json').read_text())
    for sysname in ('I553A', 'I552A'):
        v = pi['per_system'][sysname]['ln_N_eff_H_over_D_point']
        ck(f'{v:.3f}' in supp_txt, f'fig6 {sysname} isotope shift',
           f'pimd_analysis_v2.json :: per_system.{sysname}.ln_N_eff_H_over_D_point')

    w = max(len(r[1]) for r in rows)
    for status, label, detail in rows:
        print(f'  [{status}] {label:<{w}}  {detail}')
    print(f'\n  {len(rows) - len(fails)}/{len(rows)} checks passed')
    return len(fails)


if __name__ == '__main__':
    sys.exit(main())
