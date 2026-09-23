"""Analyse the corrected reactant proton-PES scans -> dZPE(H-D), k_donor, KIE_sc.

Consumes results/pcet_reactant_v2/*_scan.json (the corrected, capped-cluster
campaign) and emits per-snapshot numbers plus the LaTeX rows for the SI table.

REFUSES TO ANALYSE SPIN-CONTAMINATED POINTS. Every scan point carries its
<S^2>; any point deviating from the Fe(III) high-spin d5 value of 8.75 by more
than S2_TOL is excluded and reported. The superseded campaign silently averaged
over points sitting at <S^2> ~ 11-12.3 (a different electronic state entirely),
which is precisely what this guard exists to prevent. See AUDIT_2026-08-10.md.

Two independent routes to the reactant-well ZPE are computed and compared:
  (a) HARMONIC: local curvature k of the cubic spline at the donor-well minimum,
      ZPE = 0.5 * hbar * sqrt(k/m). Cheap, but assumes a parabolic well.
  (b) EXACT 1D: numerical solution of the nuclear Schrodinger equation on the
      spline (proton_schrodinger.solve_1d), which captures anharmonicity.
Route (b) is the reported value; their difference quantifies anharmonicity and
is worth stating in the SI rather than hiding.

Usage:  analyze_reactant_scans.py [--dir results/pcet_reactant_v2] [--tol 0.5]
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).parent))
from proton_schrodinger import solve_1d

Ha2kcal = 627.5094740631
KCAL_A2_TO_NM = 0.694770          # kcal/mol/A^2 -> N/m
HBAR = 1.054571817e-34
AMU_KG = 1.66053906660e-27
J_PER_KCAL = 4184.0 / 6.02214076e23
S2_TARGET = 8.75                  # Fe(III) high-spin d5, S = 5/2

M_H, M_D = 1.008, 2.014


def harmonic_zpe_kcal(k_Nm, m_amu):
    if k_Nm <= 0:
        return float('nan')
    omega = math.sqrt(k_Nm / (m_amu * AMU_KG))
    return 0.5 * HBAR * omega / J_PER_KCAL


def analyse_scan(path, s2_tol=0.5, verbose=True):
    d = json.loads(Path(path).read_text())
    pts = [p for p in d['points'] if p.get('E_hartree') is not None]
    n_raw = len(pts)

    # --- spin-state gate --------------------------------------------------
    clean, rejected = [], []
    for p in pts:
        s2 = p.get('S2')
        ref = p.get('S2_expected', S2_TARGET)
        if s2 is None or abs(s2 - ref) > s2_tol:
            rejected.append(p)
        else:
            clean.append(p)
    if rejected and verbose:
        print(f'    EXCLUDED {len(rejected)}/{n_raw} spin-contaminated points: '
              + ', '.join(f"a={r['alpha']:.2f}(S2={r.get('S2')})" for r in rejected[:6]))
    if len(clean) < 6:
        return None, dict(reason=f'only {len(clean)} clean points', rejected=len(rejected))

    q = np.array([p['q_along_A'] for p in clean])
    E = np.array([p['E_hartree'] for p in clean])
    order = np.argsort(q); q, E = q[order], E[order]
    V = (E - E.min()) * Ha2kcal

    cs = CubicSpline(q, V)
    qf = np.linspace(q[0], q[-1], 2001)
    Vf = cs(qf)

    # donor (reactant) well: minimum in the donor-side half of the scan
    half = len(qf) // 2
    i_min = int(np.argmin(Vf[:half]))
    q0 = qf[i_min]
    k_kcal = float(cs(q0, 2))                       # kcal/mol/A^2
    k_Nm = k_kcal * KCAL_A2_TO_NM

    # (a) harmonic route
    zpe_H_h = harmonic_zpe_kcal(k_Nm, M_H)
    zpe_D_h = harmonic_zpe_kcal(k_Nm, M_D)

    # (b) exact 1D route, solved on the reactant well only (up to the barrier
    #     top) so that eigenstates are bound states of the donor basin.
    #
    # The barrier is the FIRST LOCAL maximum after the reactant well, not the
    # global maximum: at large alpha the scan runs into the steeply repulsive
    # inner wall of the ACCEPTOR (H pushed to r_HO well below the equilibrium
    # O-H length), which is higher than the transfer barrier but is not a
    # barrier at all. Taking the global max both misreports the barrier and
    # opens the Schrodinger solve window across the whole double well, which
    # would let the ground state delocalise into the product basin and corrupt
    # the reactant-well ZPE.
    i_bar = None
    for j in range(i_min + 1, len(Vf) - 1):
        if Vf[j + 1] < Vf[j]:
            i_bar = j
            break
    if i_bar is None:                       # monotonic rise: no barrier in range
        i_bar = len(Vf) - 1
    q_lo = qf[0]
    q_hi = qf[min(i_bar, len(qf) - 1)]
    mask = (qf >= q_lo) & (qf <= q_hi)
    qw, Vw = qf[mask], Vf[mask]
    if len(qw) < 50:
        qw, Vw = qf, Vf
    epsH, _ = solve_1d(qw, Vw, M_H, K=3)
    epsD, _ = solve_1d(qw, Vw, M_D, K=3)
    zpe_H_e = float(epsH[0] - Vw.min())
    zpe_D_e = float(epsD[0] - Vw.min())

    res = dict(
        file=Path(path).name,
        r_DA=d['r_DA'],
        n_points_used=len(clean), n_points_raw=n_raw, n_rejected=len(rejected),
        S2_mean=float(np.mean([p['S2'] for p in clean])),
        S2_max_dev=float(max(abs(p['S2'] - p.get('S2_expected', S2_TARGET)) for p in clean)),
        charge=d.get('charge'), mult=d.get('mult'),
        q_donor=float(q0), k_donor_Nm=float(k_Nm),
        barrier_apparent_kcal=float(Vf[i_bar] - Vf[i_min]),
        zpe_H_harm=zpe_H_h, zpe_D_harm=zpe_D_h,
        dZPE_harm=zpe_H_h - zpe_D_h,
        zpe_H_exact=zpe_H_e, zpe_D_exact=zpe_D_e,
        dZPE_exact=zpe_H_e - zpe_D_e,
        anharmonicity=(zpe_H_e - zpe_D_e) - (zpe_H_h - zpe_D_h),
    )
    for T, tag in ((283.15, '10C'), (313.15, '40C')):
        kT = 1.987204259e-3 * T
        res[f'KIE_sc_{tag}'] = math.exp(res['dZPE_exact'] / kT)
    return res, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='/home/liang/Workspace/WritePaper/CatalysisQuamBio/results/pcet_reactant_v2')
    ap.add_argument('--tol', type=float, default=0.5)
    a = ap.parse_args()
    files = sorted(Path(a.dir).glob('*_scan.json'))
    if not files:
        print(f'no scans in {a.dir}'); return

    print(f'Corrected reactant scans in {a.dir}\n')
    rows, skipped = [], []
    for f in files:
        print(f'  {f.name}')
        r, err = analyse_scan(f, s2_tol=a.tol)
        if r is None:
            skipped.append((f.name, err)); print(f'    SKIPPED: {err}'); continue
        rows.append(r)
        print(f'    r_DA={r["r_DA"]:.3f} A  k_donor={r["k_donor_Nm"]:.0f} N/m  '
              f'<S^2>={r["S2_mean"]:.3f} (max dev {r["S2_max_dev"]:.3f})')
        print(f'    ZPE_H={r["zpe_H_exact"]:.3f}  ZPE_D={r["zpe_D_exact"]:.3f}  '
              f'dZPE={r["dZPE_exact"]:.3f} kcal/mol   KIE_sc(10C)={r["KIE_sc_10C"]:.1f}')
        print(f'    apparent barrier={r["barrier_apparent_kcal"]:.1f} kcal/mol  '
              f'anharmonic correction to dZPE={r["anharmonicity"]:+.3f}')

    if not rows:
        print('\nno analysable scans'); return
    dz = np.array([r['dZPE_exact'] for r in rows])
    kd = np.array([r['k_donor_Nm'] for r in rows])
    ba = np.array([r['barrier_apparent_kcal'] for r in rows])
    n = len(rows)
    print(f'\n=== ensemble over {n} snapshots ===')
    print(f'  dZPE(H-D)   = {dz.mean():.3f} +/- {dz.std(ddof=1):.3f} kcal/mol '
          f'(SEM {dz.std(ddof=1)/math.sqrt(n):.3f})')
    print(f'  k_donor     = {kd.mean():.0f} +/- {kd.std(ddof=1):.0f} N/m')
    print(f'  barrier     = {ba.mean():.1f} +/- {ba.std(ddof=1):.1f} kcal/mol')
    for T, tag, exp in ((283.15, '10C', 66), (313.15, '40C', 52)):
        kT = 1.987204259e-3 * T
        k = np.exp(dz / kT)
        print(f'  KIE_sc({tag}) = {k.mean():.1f} +/- {k.std(ddof=1):.1f}   '
              f'(experimental {exp}; tunnelling amplification {exp/k.mean():.1f}x)')

    print('\n=== LaTeX rows for the SI table ===')
    for r in rows:
        w = r['file'].replace('wt_win', '').replace('_scan.json', '')
        w = f'{w[0]}.{w[1:]}'
        print(f'$r_0\\!=\\!{w}$ & {r["r_DA"]:.3f} & ${r["k_donor_Nm"]:.0f}$ & '
              f'{r["barrier_apparent_kcal"]:.1f} & {r["zpe_H_exact"]:.3f} & '
              f'{r["zpe_D_exact"]:.3f} & {r["S2_mean"]:.2f} & {r["KIE_sc_10C"]:.1f} \\\\')

    out = Path(a.dir) / 'analysis_v2.json'
    out.write_text(json.dumps(dict(
        per_snapshot=rows, skipped=skipped, s2_tol=a.tol,
        ensemble=dict(n=n,
                      dZPE_mean=float(dz.mean()), dZPE_sd=float(dz.std(ddof=1)),
                      dZPE_sem=float(dz.std(ddof=1)/math.sqrt(n)),
                      k_donor_mean=float(kd.mean()), k_donor_sd=float(kd.std(ddof=1)),
                      barrier_mean=float(ba.mean()), barrier_sd=float(ba.std(ddof=1)))
    ), indent=2))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
