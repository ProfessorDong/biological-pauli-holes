"""Extend a CDFT diabat's domain by adaptive continuation in alpha.

MOTIVATION
Each CDFT diabat converges only where its constraint is enforceable, and the
first scans found hard boundaries: the reactant diabat (fragment spin 0) died
above alpha=0.629 and the product diabat (fragment spin 1) died below
alpha=0.586. Beyond those points the optimiser cannot reach the required
Lagrange multiplier from the warm start and silently relaxes onto the
unconstrained (adiabatic) solution. That leaves lambda inaccessible, because
lambda needs each diabat evaluated at the OTHER diabat's minimum
(reactant at alpha=0.714, product at alpha=0.329).

METHOD: continuation (homotopy) in geometry
The multiplier changes smoothly with alpha, so the failures are a step-size
problem, not a fundamental one. This script walks outward from the last VALID
point in small steps, halving the step whenever a point fails and retrying,
down to MIN_STEP. Each successful point becomes the warm start for the next.

BUG THIS FIXES
The original scanner warm-started from the previous point unconditionally. Once
one point collapsed onto the adiabatic solution, every subsequent point started
from a collapsed guess, guaranteeing a cascade of failures that looked like a
hard physical boundary but was partly self-inflicted. Here the warm start is
taken ONLY from points that passed validation.

VALIDATION (two independent signals, both required)
  1. NWChem prints 'CDFT final multipliers' on convergence.
  2. The Mulliken spin-density population summed over the constrained fragment
     must equal the target within SPIN_TOL. This is the DIRECT measurement that
     the constraint was achieved; on converged points it reads 1.00-1.06 for the
     product diabat and 0.00 to -0.06 for the reactant, and it is what
     distinguishes a genuine diabat from an adiabatic collapse wearing a diabat
     label. A third, weaker check is also recorded: a collapsed point reproduces
     the unconstrained energy to ~1e-5 Ha.

Usage:
  pcet_cdft_extend.py --spin 0.0 --from-dir cdft_reactant_diabat \
      --out cdft_reactant_extend --start 0.629 --target 0.714
  pcet_cdft_extend.py --spin 1.0 --from-dir cdft_product_diabat \
      --out cdft_product_extend  --start 0.586 --target 0.329
"""
import argparse, subprocess, os, sys, json, time, re, shutil
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from proton_pes_scanner import load_snapshot
from qm_cluster import build_cluster
from pcet_cdft_diabats import reorder, write_input, is_substrate
from proton_pes_scanner_cdft_nwchem import (
    NWCHEM_ENV, NWCHEM_BIN, MPIRUN, NWCHEM_BASIS_LIBRARY)

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
PRM = ROOT / 'md/mcpb/SLO_sub_solv.prmtop'
SEED = ROOT / 'results/umbrella/_RTX4060_rep1_archive_2026-08-10/WT/win_2.70_final.rst7'
DONOR, ACCEPTOR, XFERH = 13002, 12985, 13031

START_STEP = 0.0214      # half the original 0.0429 spacing
MIN_STEP   = 0.0027      # keep halving to ~1/16 of the original spacing
SPIN_TOL   = 0.15        # |achieved fragment spin - target|
TIMEOUT_S  = 21600


def fragment_spin(out_path, first=2, last=13):
    """Mulliken spin-density summed over the constrained fragment."""
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
        if first <= idx <= last:
            tot += float(m.group(3)); seen += 1
        elif idx > last and seen:
            break
    return tot if seen else None


def parse_point(out_path, spin_target):
    txt = Path(out_path).read_text(errors='ignore')
    Es = re.findall(r'Total DFT energy\s*=\s*(-?\d+\.\d+)', txt)
    mults = re.findall(r'CDFT multipliers:\s*\n\s*\d+\s+(-?\d+\.\d+)', txt)
    E_un = float(Es[0]) if Es else None
    E_di = float(Es[1]) if len(Es) > 1 else None
    conv = 'CDFT final multipliers' in txt
    spin = fragment_spin(out_path)
    collapsed = (E_un is not None and E_di is not None
                 and abs(E_di - E_un) < 1e-5)
    valid = bool(conv and E_di is not None and spin is not None
                 and abs(spin - spin_target) < SPIN_TOL and not collapsed)
    return dict(E_unconstrained=E_un, E_diabat=E_di, cdft_converged=conv,
                fragment_spin=spin, collapsed_to_adiabatic=collapsed,
                multiplier=float(mults[-1]) if mults else None, valid=valid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spin', type=float, required=True)
    ap.add_argument('--from-dir', required=True, help='dir holding the last valid point')
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=float, required=True, help='alpha of last valid point')
    ap.add_argument('--target', type=float, required=True, help='alpha to reach')
    ap.add_argument('--nproc', type=int, default=8)
    a = ap.parse_args()

    outdir = ROOT / 'results' / a.out
    outdir.mkdir(parents=True, exist_ok=True)
    direction = 1.0 if a.target > a.start else -1.0

    prm, elems, pos = load_snapshot(PRM, SEED)
    dpos, apos = pos[DONOR], pos[ACCEPTOR]
    r_DA = float(np.linalg.norm(dpos - apos))
    axis = (apos - dpos) / r_DA
    atoms, charge, mult, rep = build_cluster(prm, elems, pos, DONOR, XFERH, verbose=False)
    ordered, n_sub = reorder(atoms)

    # seed the continuation from the last VALID point of the original scan
    src = ROOT / 'results' / a.from_dir / f'pt{int(round(a.start*100)):03d}.cdft.movecs'
    prev = None
    if src.exists():
        shutil.copy(src, outdir / src.name)
        prev = src.name
        print(f'warm start from {a.from_dir}/{src.name}')
    else:
        print(f'WARNING: {src} not found; first point will be cold-started')

    print(f'r_DA={r_DA:.3f} A  fragment = atoms 2..{1+n_sub}  spin target {a.spin:.2f}')
    print(f'continuation: alpha {a.start:.4f} -> {a.target:.4f}  '
          f'step {START_STEP:.4f} (min {MIN_STEP:.4f})\n', flush=True)

    env = os.environ.copy()
    env['NWCHEM_BASIS_LIBRARY'] = str(NWCHEM_BASIS_LIBRARY) + '/'
    env['PATH'] = f'{NWCHEM_ENV/"bin"}:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = f'{NWCHEM_ENV/"lib"}:' + env.get('LD_LIBRARY_PATH', '')
    env['OMP_NUM_THREADS'] = '1'

    js = outdir / 'extend_scan.json'
    results = []
    alpha = a.start
    step = START_STEP
    while (a.target - alpha) * direction > 1e-6:
        nxt = alpha + direction * step
        if (nxt - a.target) * direction > 0:
            nxt = a.target
        stem = f'x{int(round(nxt*10000)):05d}'
        inp, out = outdir / f'{stem}.nw', outdir / f'{stem}.out'
        xh = dpos + nxt * r_DA * axis
        write_input(ordered, n_sub, xh, inp, charge, mult, a.spin, prev)
        print(f'alpha={nxt:.4f} (step {step:.4f}) ...', end=' ', flush=True)
        t0 = time.time()
        with open(out, 'wb') as fo:
            try:
                rc = subprocess.run([str(MPIRUN), '-n', str(a.nproc), str(NWCHEM_BIN),
                                     inp.name], env=env, cwd=str(outdir), stdout=fo,
                                    stderr=subprocess.STDOUT, timeout=TIMEOUT_S).returncode
            except subprocess.TimeoutExpired:
                fo.write(b'\nTIMEOUT\n'); rc = 124
        w = time.time() - t0
        r = parse_point(out, a.spin)
        r.update(alpha=float(nxt), r_H=float(nxt*r_DA), step=float(step),
                 wall_s=w, rc=rc)
        results.append(r)
        js.write_text(json.dumps(dict(r_DA=r_DA, spin_target=a.spin,
                                      n_substrate=n_sub, charge=charge, mult=mult,
                                      start=a.start, target=a.target,
                                      points=results), indent=2))
        if r['valid']:
            print(f'OK  E={r["E_diabat"]:.6f}  spin={r["fragment_spin"]:.3f}  '
                  f'mult={r["multiplier"]}  {w/60:.0f} min', flush=True)
            alpha = nxt
            prev = f'{stem}.cdft.movecs'          # warm start ONLY from valid points
            step = min(START_STEP, step * 2.0)    # relax the step again after success
        else:
            why = ('collapsed to adiabatic' if r['collapsed_to_adiabatic']
                   else f'spin={r["fragment_spin"]} vs target {a.spin}'
                   if r['fragment_spin'] is not None else 'no spin output / aborted')
            print(f'FAIL ({why})  {w/60:.0f} min', flush=True)
            step /= 2.0
            if step < MIN_STEP:
                print(f'\nSTOP: step below MIN_STEP={MIN_STEP} at alpha={alpha:.4f}. '
                      f'This is the true domain boundary for this constraint.',
                      flush=True)
                break
            # prev deliberately unchanged: retry from the last VALID point

    nvalid = sum(1 for r in results if r['valid'])
    print(f'\n{nvalid}/{len(results)} attempts valid; reached alpha={alpha:.4f} '
          f'(target {a.target:.4f})')
    print(f'wrote {js}')


if __name__ == '__main__':
    main()
