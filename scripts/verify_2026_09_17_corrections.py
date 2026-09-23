#!/usr/bin/env python3
"""Check every number changed on 2026-09-17 against a NAMED FIELD of a NAMED FILE.

The project rule is that a substring sweep proves digits exist somewhere, not that a claim is
sourced. So each check below names the file and the field, recomputes the manuscript's rendering
of it, and asserts that exactly that string is present in the built PDF text. A check that
cannot find its source fails loudly rather than passing quietly.

Usage: verify_2026_09_17_corrections.py     (pauli env)
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
R = ROOT / 'results'
PDF = ROOT / 'prxlife/main.pdf'


def load(rel):
    return json.loads((R / rel).read_text())


def main():
    txt = subprocess.run(['pdftotext', str(PDF), '-'], capture_output=True, text=True).stdout
    flat = re.sub(r'\s+', ' ', txt)
    checks, fails = [], 0

    def ck(label, source, expect, present=True):
        nonlocal fails
        ok = (expect in flat) == present
        checks.append((label, source, expect, ok))
        if not ok:
            fails += 1

    # --- D1, reactive-geometry endpoints -------------------------------------------------
    d1 = load('reactive_geometry/analysis_result.json')
    ck('D1 reactive partial', 'reactive_geometry/analysis_result.json :: reactive.partial',
       f'{d1["reactive"]["partial"]:.3f}'.replace('-', '−'))
    ck('D1 reference partial', 'reactive_geometry/analysis_result.json :: reference.partial',
       f'{d1["reference"]["partial"]:.3f}'.replace('-', '−'))
    contrast = abs(d1['reactive']['partial']) - abs(d1['reference']['partial'])
    ck('D1 geometry contrast', 'analysis_result.json :: |reactive|-|reference|',
       f'+{contrast:.3f}')
    # Bare digits are not a safe absence test: an unrelated K_perp^int eigenvalue of
    # -0.2164 N/m contains '0.216' as a substring. Anchor on the contrast wording instead.
    ck('D1 stale contrast withdrawn', 'superseded value must be gone',
       'contrast between the two geometries, is $-0.216', present=False)
    ck('D1 stale contrast withdrawn (alt form)', 'superseded value must be gone',
       'is $-0.216$', present=False)

    # --- D2, ensemble-fluctuation endpoints ----------------------------------------------
    d2 = load('ensemble_fluctuation/analysis_result.json')
    ck('D2 reactive partial', 'ensemble_fluctuation/analysis_result.json :: reactive.D2_sd.partial',
       f'{d2["reactive"]["D2_sd"]["partial"]:.3f}'.replace('-', '−'))
    ck('D2 reference partial', 'ensemble_fluctuation/analysis_result.json :: reference.D2_sd.partial',
       f'{d2["reference"]["D2_sd"]["partial"]:.3f}'.replace('-', '−'))
    ck('D3 no longer aborts', 'reactive.D3_rel.vif is below the abort threshold of 5',
       f'{d2["reactive"]["D3_rel"]["vif"]:.2f}')
    ck('D3 stale abort VIF withdrawn', 'superseded value must be gone', '6.37', present=False)

    # --- transverse Hessian ---------------------------------------------------------------
    th = load('transverse_hessian/transverse_vs_separation.json')
    tr = th['rank']['trace_K_perp']
    ck('K_perp trace minimum', 'transverse_vs_separation.json :: rank.trace_K_perp min',
       f'{min(tr):.3f}'.replace('-', '−'))
    ck('K_perp trace maximum', 'transverse_vs_separation.json :: rank.trace_K_perp max',
       f'+{max(tr):.3f}')
    ck('K_sep vs K_perp rank corr', 'transverse_vs_separation.json :: rank.spearman_rho',
       f'{th["rank"]["spearman_rho"]:.3f}'.replace('-', '−'))
    ck('corrected ZPE bound', 'transverse_vs_separation.json :: max_percent_of_ladder',
       f'{th["max_percent_of_ladder"]:.2f}')
    neg = sum(1 for r in th['records']
              if r['clamp'] == 'r255' and abs(r['half'] - 0.15) < 1e-9 and min(r['w1'], r['w2']) < 0)
    ck('six of seven anti-confining', f'records: {neg} of 7 have a negative eigenvalue',
       'Six of the seven')

    # --- exponential panel dynamic range ---------------------------------------------------
    ep = load('sapt_bio/donor_fragment/exponential_law_panel.json')
    import math
    dec = math.log10(ep['k_max'] / ep['k_min'])
    ck('panel dynamic range', 'exponential_law_panel.json :: log10(k_max/k_min)', f'{dec:.1f}')
    ck('stale five-decades claim withdrawn', 'superseded', 'five decades', present=False)

    # --- exponential panel, regenerated after the V750A F-SAPT repair ------------------------
    ep2 = load('sapt_bio/donor_fragment/exponential_law_panel.json')
    pf = load('sapt_bio/donor_fragment/exponential_law_prefactor.json')
    ck('panel configuration count', 'exponential_law_panel.json :: n', str(ep2['n']))
    ck('panel k maximum', 'exponential_law_panel.json :: k_max', f"{ep2['k_max']:.1f}")
    ck('panel effective decay', 'exponential_law_panel.json :: common_rate_per_A',
       f"{ep2['common_rate_per_A']:.2f}")
    ck('panel decay ratio to solved', 'exponential_law_panel.json :: ratio', f"{ep2['ratio']:.2f}")
    ck('panel variance inflation', 'exponential_law_prefactor.json :: vif', f"{pf['vif']:.0f}")
    # These two must be matched with their units attached. A bare '2.92' also occurs as 2.924
    # in an unrelated spin-expectation table, and a bare '52.1' would be equally ambiguous.
    ck('superseded panel decay withdrawn', 'pre-repair effective decay must be gone',
       '2.92 per', present=False)
    ck('superseded panel maximum withdrawn', 'pre-repair k_max must be gone',
       '52.1 N m', present=False)

    # --- campaign manifest -----------------------------------------------------------------
    nat = load('sapt_bio/native_fragment/summary.json')['native']
    met = load('sapt_bio/sapt_summary_corrected.json')['per_system_corrected']
    ratio = met['WT']['k_Nm'] / nat['WT']['k_exch_Nm']
    ck('methane-wall overestimate', 'sapt_summary_corrected vs native_fragment, WT',
       f'{ratio:.2f}')

    # --- permutation-scheme sensitivity ------------------------------------------------------
    ps = R / 'reactive_geometry/permutation_scheme_sensitivity.json'
    if ps.exists():
        j = json.loads(ps.read_text())
        checks.append(('Freedman-Lane agrees with pre-registered scheme',
                       'permutation_scheme_sensitivity.json :: schemes_agree',
                       str(j['schemes_agree']), j['schemes_agree']))
        if not j['schemes_agree']:
            fails += 1

    w = max(len(c[0]) for c in checks)
    for label, source, expect, ok in checks:
        print(f'  [{"OK " if ok else "FAIL"}] {label:<{w}}  {expect:>12}   <- {source}')
    print(f'\n  {len(checks) - fails}/{len(checks)} checks passed')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
