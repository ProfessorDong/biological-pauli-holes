"""Assemble the CDFT diabatic surfaces and report ONLY what is defensible.

Collects every point from the original scans and the continuation extensions,
applies one uniform validity filter to all of them, splines each diabat, locates
the crossing, and states explicitly which Marcus quantities are measurable and
which are not.

VALIDITY FILTER (both required):
  1. 'CDFT final multipliers' present  -> the constraint optimiser converged
  2. Mulliken spin on the constrained fragment within SPIN_TOL of target
     -> the constraint was physically ACHIEVED
Points failing either are excluded and listed, never averaged in.

WHY THERE IS NO ENERGY-BASED COLLAPSE TEST:
An earlier version of this filter also rejected any point with
|E_diabat - E_unconstrained| < 1e-5 Ha as "collapsed onto the adiabatic
surface". That test is WRONG for the reactant diabat in the reactant region:
there the reactant genuinely IS the electronic ground state, so the constraint
is legitimately a near-no-op and the two energies coincide by correct physics.
It discarded five valid points (alpha 0.24-0.41), moved the reactant minimum to
alpha=0.460 (r_CH = 1.57 A, chemically impossible) and flipped the frozen dG
sign. The spin measurement alone is the correct discriminator: a genuinely
collapsed point past the crossing relaxes onto a product-like ground state and
shows fragment spin ~1 instead of ~0, which criterion 2 catches. Energy
coincidence is diagnostic only where the diabat is expected to be the UPPER
surface, and is redundant with the spin test there.
"""
import json, re, sys
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
Ha = 627.5094740631
SPIN_TOL = 0.15
FRAG_FIRST, FRAG_LAST = 2, 13


def fragment_spin(out_path):
    try:
        txt = Path(out_path).read_text(errors='ignore')
    except Exception:
        return None
    i = txt.rfind('Spin Density - Mulliken Population Analysis')
    if i < 0:
        return None
    tot, seen = 0.0, 0
    for line in txt[i:].splitlines():
        m = re.match(r'\s*(\d+)\s+([A-Za-z]{1,2})\s+\d+\s+(-?\d+\.\d+)', line)
        if not m:
            continue
        idx = int(m.group(1))
        if FRAG_FIRST <= idx <= FRAG_LAST:
            tot += float(m.group(3)); seen += 1
        elif idx > FRAG_LAST and seen:
            break
    return tot if seen else None


def harvest(out_path, spin_target):
    txt = Path(out_path).read_text(errors='ignore')
    Es = re.findall(r'Total DFT energy\s*=\s*(-?\d+\.\d+)', txt)
    E_un = float(Es[0]) if Es else None
    E_di = float(Es[1]) if len(Es) > 1 else None
    conv = 'CDFT final multipliers' in txt
    spin = fragment_spin(out_path)
    if E_di is None or not conv or spin is None:
        return None, 'not converged / no spin output'
    if abs(spin - spin_target) > SPIN_TOL:
        return None, f'spin {spin:+.3f} vs target {spin_target:.1f}'
    return dict(E=E_di, E_un=E_un, spin=spin), None


def collect(dirs, spin_target, alpha_from_name):
    pts, rejected = {}, []
    for d in dirs:
        p = ROOT / 'results' / d
        if not p.exists():
            continue
        for f in sorted(p.glob('*.out')):
            a = alpha_from_name(f.stem)
            if a is None:
                continue
            rec, why = harvest(f, spin_target)
            if rec is None:
                rejected.append((d, f.stem, round(a, 4), why))
            else:
                pts[round(a, 4)] = rec          # extension overwrites duplicates
    return pts, rejected


def alpha_orig(stem):
    m = re.fullmatch(r'pt(\d+)', stem)
    return int(m.group(1)) / 100 if m else None


def alpha_ext(stem):
    m = re.fullmatch(r'x(\d+)', stem)
    return int(m.group(1)) / 10000 if m else None


def alpha_any(stem):
    return alpha_orig(stem) if stem.startswith('pt') else alpha_ext(stem)


def main():
    r_DA = 3.4074
    reac, rej_r = collect(['cdft_reactant_diabat', 'cdft_reactant_extend'], 0.0, alpha_any)
    prod, rej_p = collect(['cdft_product_diabat', 'cdft_product_extend'], 1.0, alpha_any)

    print('=== VALID POINTS (uniform filter applied to all runs) ===')
    for tag, d in (('reactant (spin 0)', reac), ('product (spin 1)', prod)):
        a = sorted(d)
        print(f'  {tag}: {len(a)} points, alpha {min(a):.4f}..{max(a):.4f}')
        print(f'      spins {min(d[k]["spin"] for k in a):+.3f}..{max(d[k]["spin"] for k in a):+.3f}')
    print(f'\n  rejected: {len(rej_r)} reactant, {len(rej_p)} product')
    for r in (rej_r + rej_p)[:6]:
        print(f'      {r[0]}/{r[1]} alpha={r[2]}: {r[3]}')
    if len(rej_r) + len(rej_p) > 6:
        print(f'      ... and {len(rej_r)+len(rej_p)-6} more')

    ar, Er = np.array(sorted(reac)), np.array([reac[k]['E'] for k in sorted(reac)])
    ap, Ep = np.array(sorted(prod)), np.array([prod[k]['E'] for k in sorted(prod)])
    csr, csp = CubicSpline(ar, Er), CubicSpline(ap, Ep)

    lo, hi = max(ar.min(), ap.min()), min(ar.max(), ap.max())
    print(f'\n=== OVERLAP (both diabats valid) ===')
    print(f'  alpha in [{lo:.4f}, {hi:.4f}]  (width {hi-lo:.4f}), '
          f'r_H {lo*r_DA:.3f}..{hi*r_DA:.3f} A')

    g = np.linspace(lo, hi, 4001)
    diff = csr(g) - csp(g)
    sign = np.sign(diff)
    idx = np.where(np.diff(sign) != 0)[0]
    print('\n=== CROSSING ===')
    if len(idx):
        a_x = float(np.interp(0.0, [diff[idx[0]], diff[idx[0]+1]], [g[idx[0]], g[idx[0]+1]]))
        E_x = float(csr(a_x))
        print(f'  alpha_cross = {a_x:.4f}   r_H = {a_x*r_DA:.3f} A   '
              f'r_HO = {r_DA - a_x*r_DA:.3f} A')
        print(f'  E at crossing = {E_x:.6f} Ha')
        print(f'  bracketed by converged, spin-verified points on BOTH surfaces')
    else:
        print('  no sign change within the overlap')
        a_x = None

    ir, ip = int(np.argmin(Er)), int(np.argmin(Ep))
    print('\n=== MINIMA ===')
    print(f'  reactant: alpha={ar[ir]:.4f}  r_H={ar[ir]*r_DA:.3f} A (C-H)   E={Er[ir]:.6f}')
    print(f'  product : alpha={ap[ip]:.4f}  r_H={ap[ip]*r_DA:.3f} A  '
          f'r_HO={r_DA-ap[ip]*r_DA:.3f} A (O-H)   E={Ep[ip]:.6f}')
    dG = (Ep[ip] - Er[ir]) * Ha
    print(f'  frozen-geometry dG = {dG:+.1f} kcal/mol  '
          f'(NOT the experimental dG; heavy atoms fixed, no environment relaxation)')

    print('\n=== LAMBDA: NOT DIRECTLY MEASURABLE ===')
    print(f'  lambda needs each diabat at the OTHER minimum:')
    print(f'    reactant at alpha={ap[ip]:.4f}: outside its valid domain '
          f'(max {ar.max():.4f})')
    print(f'    product  at alpha={ar[ir]:.4f}: outside its valid domain '
          f'(min {ap.min():.4f})')
    print(f'  Adaptive continuation was attempted to reach both and terminated at')
    print(f'  the true constraint boundaries. Any lambda from these data would be')
    print(f'  an extrapolation well beyond the sampled region and is NOT reported.')

    out = ROOT / 'results' / 'diabat_analysis.json'
    out.write_text(json.dumps(dict(
        r_DA=r_DA, spin_tol=SPIN_TOL,
        reactant=dict(alphas=ar.tolist(), E=Er.tolist(),
                      spins=[reac[k]['spin'] for k in sorted(reac)],
                      domain=[float(ar.min()), float(ar.max())]),
        product=dict(alphas=ap.tolist(), E=Ep.tolist(),
                     spins=[prod[k]['spin'] for k in sorted(prod)],
                     domain=[float(ap.min()), float(ap.max())]),
        overlap=[float(lo), float(hi)],
        crossing_alpha=a_x,
        dG_frozen_kcal=float(dG),
        lambda_measurable=False,
        n_rejected=len(rej_r) + len(rej_p)), indent=2))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
