"""Symmetric CDFT diabatic surfaces for the SLO PCET reaction (reactant + product).

WHY NOT A PLAIN UKS SCAN FOR THE REACTANT:
An unconstrained UKS scan follows the ADIABATIC ground state. Past the crossing
seam it ceases to be reactant-like and turns into the product state, so it is not
a reactant diabat and would corrupt lambda and dG precisely where they matter.
Marcus theory needs two DIABATIC surfaces, each of which retains its electronic
character across the whole proton coordinate.

CONSTRUCTION:
Both diabats are computed with the SAME constraint operator on the SAME fragment
and differ only in the target value, so systematic errors largely cancel in the
difference that defines lambda and dG:

  reactant diabat : spin density on the pentadienyl fragment = 0
                    (closed-shell 1,4-diene + Fe(III) high-spin d5, S=5/2)
  product  diabat : spin density on the pentadienyl fragment = 1
                    (delocalised pentadienyl radical + Fe(II) high-spin d6, S=2)

THE FRAGMENT, NOT A SINGLE ATOM:
The SLO product radical is a delocalised pentadienyl spread over
C10=C13-C14-C17=C15. Constraining a single donor carbon understates that
delocalisation. We therefore constrain the whole pentadienyl fragment (5 C, their
5 hydrogens, and the 2 truncation caps = 12 atoms), EXCLUDING the transferring
hydrogen, which leaves with the acceptor in the product state.

NWChem's `cdft` directive takes contiguous atom ranges, so the geometry is
reordered as [transferring H] + [12 substrate-fragment atoms] + [rest], making
the constrained region exactly atoms 2-13. Reordering is a pure input-ordering
change and does not affect the energy.

Both diabats run on the corrected 80-atom capped cluster (charge +1, 299 e-,
mult 6; see qm_cluster.py and AUDIT_2026-08-10.md) and on the SAME archived
snapshot the reactant ORCA scans used (r_DA = 3.407 A), so all surfaces share
byte-identical nuclei.

SCAN DIRECTION MATTERS:
A CDFT diabat must be initialised where it IS the electronic ground state and then
continued adiabatically into the region where it is not; started from the wrong
end, the constraint optimiser cannot reach the required Lagrange multiplier and
silently relaxes back onto the unconstrained (adiabatic) solution. Observed here:
scanning the product diabat forward from alpha=0.20, where forcing a pentadienyl
radical onto a hydrogen still bonded at 0.68 A costs ~75 kcal/mol, gave
CDFTconv=False and an energy equal to the unconstrained value at every point.

  reactant diabat (spin 0): ground state at SMALL alpha -> scan FORWARD
  product  diabat (spin 1): ground state at LARGE alpha -> scan REVERSE

Usage:
  pcet_cdft_diabats.py --spin 0.0 --out reactant_diabat
  pcet_cdft_diabats.py --spin 1.0 --out product_diabat --reverse
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import argparse, subprocess, os, sys, json, time, re
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from proton_pes_scanner import load_snapshot
from qm_cluster import build_cluster
from proton_pes_scanner_cdft_nwchem import (
    NWCHEM_ENV, NWCHEM_BIN, MPIRUN, NWCHEM_BASIS_LIBRARY)

ROOT = Path(_REPO)
PRM = ROOT / 'md/mcpb/SLO_sub_solv.prmtop'
SEED = ROOT / 'results/umbrella/_RTX4060_rep1_archive_2026-08-10/WT/win_2.70_final.rst7'
DONOR, ACCEPTOR, XFERH = 13002, 12985, 13031
ALPHAS = np.linspace(0.20, 0.80, 15)
TIMEOUT_S = 21600


def is_substrate(label):
    """Pentadienyl fragment: substrate heavy atoms, their hydrogens, and the two
    truncation caps. The transferring H is inserted separately and excluded."""
    return (label.startswith('LIG820') or label == 'donor-C'
            or label.startswith('cap:LIG820'))


def reorder(atoms):
    """Return (ordered_atoms, n_substrate) with substrate-fragment atoms first so
    they occupy the contiguous NWChem range 2..(1+n_substrate)."""
    sub = [a for a in atoms if is_substrate(a[2])]
    rest = [a for a in atoms if not is_substrate(a[2])]
    return sub + rest, len(sub)


def write_input(ordered, n_sub, xferH_pos, out_file, charge, mult, spin,
                prev_movecs=None):
    stem = Path(out_file).stem
    L = []
    L.append(f'title "SLO diabat: pentadienyl-fragment spin = {spin:.2f}"')
    L.append('memory total 6 gb')
    L.append(f'start {stem}')
    L.append('geometry noautosym units angstroms')
    L.append(f'  H  {xferH_pos[0]:.6f}  {xferH_pos[1]:.6f}  {xferH_pos[2]:.6f}')
    for e, p, _ in ordered:
        L.append(f'  {e}  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}')
    L.append('end')
    L.append('basis')
    L.append('  * library "def2-SVP"')
    L.append('end')
    L.append(f'charge {charge}')
    # Step 1: relax orbitals at this geometry, staying on the current diabat
    L.append('dft')
    L.append('  xc b3lyp'); L.append('  disp vdw 4')
    L.append(f'  mult {mult}'); L.append('  odft')
    if prev_movecs:
        L.append(f'  vectors input {prev_movecs} output {stem}.pre.movecs')
    else:
        L.append(f'  vectors output {stem}.pre.movecs')
    L.append('  convergence energy 1e-5'); L.append('  iterations 200')
    L.append('end'); L.append('task dft')
    # Step 2: impose the fragment spin constraint -> the diabat
    L.append('dft')
    L.append('  xc b3lyp'); L.append('  disp vdw 4')
    L.append(f'  mult {mult}'); L.append('  odft')
    L.append(f'  vectors input {stem}.pre.movecs output {stem}.cdft.movecs')
    L.append(f'  cdft 2 {1 + n_sub} spin {spin:.3f}')
    L.append('  convergence energy 1e-6'); L.append('  iterations 300')
    L.append('end'); L.append('task dft')
    Path(out_file).write_text('\n'.join(L) + '\n')


def parse(out_path):
    txt = Path(out_path).read_text(errors='ignore')
    Es = re.findall(r'Total DFT energy\s*=\s*(-?\d+\.\d+)', txt)
    iters = re.findall(
        r'd=\s*\d,ls=[\d\.]+,diis\s+(\d+)\s+(-?\d+\.\d+)\s+\S+\s+\S+\s+(\S+)\s+(\S+)', txt)
    mults = re.findall(r'CDFT multipliers:\s*\n\s*\d+\s+(-?\d+\.\d+)', txt)
    s2 = re.findall(r'<S2>\s*=\s*(-?\d+\.\d+)', txt)
    return dict(E_unconstrained=float(Es[0]) if Es else None,
                E_diabat=float(Es[1]) if len(Es) > 1 else (
                    float(iters[-1][1]) if iters else None),
                E_diabat_formally_converged=len(Es) > 1,
                cdft_converged='CDFT final multipliers' in txt,
                multiplier=float(mults[-1]) if mults else None,
                S2=float(s2[-1]) if s2 else None,
                n_iter=len(iters))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spin', type=float, required=True,
                    help='0.0 = reactant diabat, 1.0 = product diabat')
    ap.add_argument('--out', required=True, help='output subdirectory name')
    ap.add_argument('--nproc', type=int, default=6)
    ap.add_argument('--reverse', action='store_true',
                    help='scan from large alpha to small. REQUIRED for the product '
                         'diabat: each diabat must be initialised where it is the '
                         'ground state and continued into the region where it is not.')
    a = ap.parse_args()

    outdir = ROOT / 'results' / a.out
    outdir.mkdir(parents=True, exist_ok=True)

    prm, elems, pos = load_snapshot(PRM, SEED)
    dpos, apos = pos[DONOR], pos[ACCEPTOR]
    r_DA = float(np.linalg.norm(dpos - apos))
    axis = (apos - dpos) / r_DA
    qpos = np.array([dpos + al * r_DA * axis for al in ALPHAS])

    atoms, charge, mult, rep = build_cluster(prm, elems, pos, DONOR, XFERH, verbose=False)
    ordered, n_sub = reorder(atoms)
    print(f'r_DA={r_DA:.3f} A   cluster {rep["n_atoms"]} atoms, charge {charge:+d}, '
          f'mult {mult}, <S^2>_target {rep["S2_expected"]:.2f}')
    print(f'constrained fragment = NWChem atoms 2..{1+n_sub} ({n_sub} atoms, '
          f'pentadienyl excl. transferring H), spin target {a.spin:.2f}')
    print(f'diabat: {"REACTANT (closed-shell diene)" if a.spin < 0.5 else "PRODUCT (pentadienyl radical)"}\n',
          flush=True)

    env = os.environ.copy()
    env['NWCHEM_BASIS_LIBRARY'] = str(NWCHEM_BASIS_LIBRARY) + '/'
    env['PATH'] = f'{NWCHEM_ENV/"bin"}:' + env.get('PATH', '')
    env['LD_LIBRARY_PATH'] = f'{NWCHEM_ENV/"lib"}:' + env.get('LD_LIBRARY_PATH', '')
    env['OMP_NUM_THREADS'] = '1'

    js = outdir / 'diabat_scan.json'
    results, done = [], set()
    if js.exists():
        try:
            prior = json.loads(js.read_text())
            for p in prior.get('points', []):
                if p.get('E_diabat') is not None:
                    results.append(p); done.add(round(p['alpha'], 3))
            if results: print(f'RESUME: {len(results)} points already done', flush=True)
        except Exception:
            pass
    prev = None
    if done:
        edge = min(done) if a.reverse else max(done)
        c = outdir / f'pt{int(round(edge*100)):03d}.cdft.movecs'
        if c.exists(): prev = c.name

    order = list(range(len(ALPHAS)))
    if a.reverse:
        order = order[::-1]
    for i in order:
        al, xh = ALPHAS[i], qpos[i]
        if round(float(al), 3) in done:
            continue
        stem = f'pt{int(round(al*100)):03d}'
        inp, out = outdir / f'{stem}.nw', outdir / f'{stem}.out'
        write_input(ordered, n_sub, xh, inp, charge, mult, a.spin, prev)
        print(f'=== point {i:02d}/15  alpha={al:.3f}  r_H={al*r_DA:.3f} A '
              f'({"warm" if prev else "COLD"}) ===', flush=True)
        t0 = time.time()
        with open(out, 'wb') as fo:
            try:
                rc = subprocess.run([str(MPIRUN), '-n', str(a.nproc), str(NWCHEM_BIN),
                                     inp.name], env=env, cwd=str(outdir), stdout=fo,
                                    stderr=subprocess.STDOUT, timeout=TIMEOUT_S).returncode
            except subprocess.TimeoutExpired:
                fo.write(b'\nTIMEOUT\n'); rc = 124
        w = time.time() - t0
        r = parse(out)
        r.update(idx=i, alpha=float(al), r_H=float(al*r_DA), wall_s=w, rc=rc,
                 spin_target=a.spin)
        results.append(r)
        js.write_text(json.dumps(dict(r_DA=r_DA, spin_target=a.spin,
                                      n_substrate=n_sub, charge=charge, mult=mult,
                                      cluster=rep['composition'],
                                      points=results), indent=2))
        print(f'  E_diabat={r["E_diabat"]}  CDFTconv={r["cdft_converged"]}  '
              f'mult={r["multiplier"]}  wall={w/60:.1f} min  rc={rc}', flush=True)
        c = outdir / f'{stem}.cdft.movecs'
        p_ = outdir / f'{stem}.pre.movecs'
        prev = c.name if c.exists() else (p_.name if p_.exists() else prev)

    print(f'\nwrote {js}')


if __name__ == '__main__':
    main()
