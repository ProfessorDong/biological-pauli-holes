"""B: NWChem-CDFT product-state proton-PES scanner.

Companion to proton_pes_scanner.py (reactant, ORCA UKS) — this script computes
the SECOND diabatic Born-Oppenheimer surface, corresponding to the PCET product
state Fe(II)-OH_2 + [substrate C radical], by constraining the spin density on
the donor carbon to +1.0 via NWChem's %cdft block.

Why NWChem: the project's ORCA 6.1.0 install does not expose CDFT (no %cdft
block, no orca_cdft binary). NWChem 7.3.1 (conda-forge) has native CDFT with
per-atom charge OR spin-density constraints, converges cleanly on LiH and on
small transition-metal clusters, and runs via MPI on multiple cores.

Cluster construction is IDENTICAL to proton_pes_scanner.py (reactant): the
same load_snapshot() and cluster_atoms() from that file are reused, so the two
diabatic surfaces sit on byte-identical nuclear geometries and the Marcus
crossing seam is well-defined by construction.

Product-state constraints:
  - spin density on donor C = +1.0  (alpha-C radical, one unpaired e-)
  - total multiplicity kept at the reactant mult=6 (ferromagnetic coupling of
    Fe(II) HS d6 S=2 with the C-alpha radical S=1/2, matching the reactant
    Fe(III) HS S=5/2).
  - net charge kept at the reactant charge (+1), because PCET is (H atom +
    electron) transfer with no net charge change on the cluster.

CHARGE (corrected 2026-08-10): the SLO cluster's formal charge is +1, not +2.
Accounting: Fe(III) +3, OH- -1, Ile817 C-terminal carboxylate -1, three neutral
His, neutral Asn672, neutral substrate fragment. sum(Z)=376, so charge +1 gives
375 electrons (odd), which supports mult=6 with <S^2>=8.75. The earlier +2
assignment omitted the Ile carboxylate, gave 374 electrons (even), made mult=6
parity-impossible, and produced a spurious ligand radical (<S^2>~11-12, S=3) in
both ORCA and NWChem. All scans run before 2026-08-10 used the wrong charge.

Convergence: NWChem CDFT typically needs 20-50 outer iterations for the
Lagrange multiplier to converge, each of which is one full SCF. On the
60-80 atom cluster, one point takes 15-45 min on 4 MPI ranks.

Usage:
  proton_pes_scanner_cdft_nwchem.py --seed R.rst7 --prmtop P.prmtop \
      --xferH i --donor j --acceptor k \
      --npts 1 --alpha_center 0.5 --out OUT     # single-point convergence test
  proton_pes_scanner_cdft_nwchem.py --seed R.rst7 --prmtop P.prmtop \
      --xferH i --donor j --acceptor k \
      --npts 15 --out OUT                       # full 15-point pilot
"""
import argparse, subprocess, numpy as np, os, sys, json, shlex, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from proton_pes_scanner import (
    load_snapshot, cluster_atoms, _atom_z, choose_charge_mult,
)

NWCHEM_ENV = Path(os.environ.get('NWCHEM_ENV', '/home/liang/anaconda3/envs/nwchem'))
NWCHEM_BIN = NWCHEM_ENV / 'bin' / 'nwchem'
MPIRUN     = NWCHEM_ENV / 'bin' / 'mpirun'
NWCHEM_BASIS_LIBRARY = NWCHEM_ENV / 'share' / 'nwchem' / 'libraries.bse'


def find_donor_C_nwchem_idx(cluster):
    """The transferring H is atom 1 in the NWChem geometry ordering; cluster
    atoms follow at indices 2, 3, ....  NWChem CDFT uses 1-indexed atom
    numbers. Returns the 1-indexed NWChem atom number for the donor C.
    """
    for i, (elem, p, name) in enumerate(cluster):
        if elem == 'C' and name.startswith('donor-C'):
            return i + 2
    for i, (elem, p, name) in enumerate(cluster):
        if elem == 'C':
            return i + 2
    raise RuntimeError('no donor C found in cluster')


def write_nwchem_input_cdft(cluster, xferH_pos, donor_C_nwchem_idx, out_file,
                            charge=1, mult=6, cdft_spin=1.0,
                            basis='def2-SVP', xc='b3lyp', dispersion='vdw 4'):
    """Write an NWChem input file for a UKS CDFT single-point energy
    calculation. The transferring H is atom #1; the cluster atoms follow.
    CDFT constrains the donor C spin density to +cdft_spin.
    """
    stem = Path(out_file).stem
    lines = []
    lines.append(f'title "SLO PCET product diabat, spin({donor_C_nwchem_idx})={cdft_spin:.2f}"')
    lines.append('memory total 6 gb')
    lines.append(f'start {stem}')
    lines.append('geometry noautosym units angstroms')
    lines.append(f'  H  {xferH_pos[0]:.6f}  {xferH_pos[1]:.6f}  {xferH_pos[2]:.6f}')
    for elem, p, name in cluster:
        lines.append(f'  {elem}  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}')
    lines.append('end')
    lines.append('basis')
    lines.append(f'  * library "{basis}"')
    lines.append('end')
    lines.append(f'charge {charge}')
    # STEP 1: converge the unconstrained UKS SCF for a good starting guess
    lines.append('dft')
    lines.append(f'  xc {xc}')
    if dispersion:
        lines.append(f'  disp {dispersion}')
    lines.append(f'  mult {mult}')
    lines.append('  odft')
    lines.append(f'  vectors output {stem}.pre.movecs')
    lines.append('  convergence energy 1e-5')
    lines.append('  iterations 200')
    lines.append('end')
    lines.append('task dft')
    # STEP 2: switch on the CDFT constraint, restarting from the converged orbitals
    lines.append('dft')
    lines.append(f'  xc {xc}')
    if dispersion:
        lines.append(f'  disp {dispersion}')
    lines.append(f'  mult {mult}')
    lines.append('  odft')
    lines.append(f'  vectors input {stem}.pre.movecs output {stem}.cdft.movecs')
    lines.append(f'  cdft {donor_C_nwchem_idx} {donor_C_nwchem_idx} spin {cdft_spin:.3f}')
    lines.append('  convergence energy 1e-6')
    lines.append('  iterations 300')
    lines.append('end')
    lines.append('task dft')
    with open(out_file, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def parse_nwchem_cdft(out_file):
    """Extract 'Total DFT energy' AND presence of 'CDFT final multipliers'
    (indicates converged CDFT Lagrange multiplier). Returns (E_hartree, converged)."""
    E, cdft_converged = None, False
    try:
        for line in Path(out_file).read_text().splitlines():
            if 'Total DFT energy' in line:
                E = float(line.split('=')[-1].strip())
            if 'CDFT final multipliers' in line:
                cdft_converged = True
    except Exception:
        pass
    return E, cdft_converged


def run_nwchem(inp, out, workdir, nproc=4, timeout_s=7200):
    env = os.environ.copy()
    env['NWCHEM_BASIS_LIBRARY'] = str(NWCHEM_BASIS_LIBRARY) + '/'
    env['PATH'] = f'{NWCHEM_ENV/"bin"}:' + env.get('PATH','')
    env['LD_LIBRARY_PATH'] = f'{NWCHEM_ENV/"lib"}:' + env.get('LD_LIBRARY_PATH','')
    env['OMP_NUM_THREADS'] = '1'
    cmd = [str(MPIRUN), '-n', str(nproc), str(NWCHEM_BIN), str(Path(inp).name)]
    with open(out, 'wb') as fo:
        try:
            proc = subprocess.run(cmd, env=env, cwd=str(workdir),
                                  stdout=fo, stderr=subprocess.STDOUT,
                                  timeout=timeout_s)
        except subprocess.TimeoutExpired:
            fo.write(b'\nTIMEOUT\n')
            return 999
    return proc.returncode


def run_scan_cdft(prmtop, rst7, xferH_idx, donor_idx, acceptor_idx,
                  npts=15, alpha_center=None, mode='small', out_prefix='pes_cdft',
                  charge=1, mult=6, cdft_spin=1.0, nproc=4):
    prm, elems, pos = load_snapshot(prmtop, rst7)
    xferH_pos = pos[xferH_idx]
    donor_pos = pos[donor_idx]
    acc_pos = pos[acceptor_idx]
    r_DA = np.linalg.norm(donor_pos - acc_pos)
    axis = (acc_pos - donor_pos) / r_DA
    print(f'r_DA = {r_DA:.3f} A')

    if npts == 1 and alpha_center is not None:
        alpha = np.array([alpha_center])
    else:
        alpha = np.linspace(0.20, 0.80, npts)
    q_H_positions = np.array([donor_pos + a * r_DA * axis for a in alpha])
    q_grid_along = alpha * r_DA

    from qm_cluster import build_cluster
    cluster, charge_use, mult_use, cluster_report = build_cluster(
        prm, elems, pos, donor_idx=donor_idx, xferH_idx=xferH_idx)
    donor_C_nw_idx = find_donor_C_nwchem_idx(cluster)
    print(f'  donor-C is NWChem atom index {donor_C_nw_idx}')
    if (charge_use, mult_use) != (charge, mult):
        print(f'  NOTE: builder-derived (charge, mult) = ({charge_use}, {mult_use}); '
              f'CLI requested ({charge}, {mult}). Using the builder value.')

    outdir = Path(out_prefix).parent.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    scan_results = []
    for i, (a_val, xh_pos) in enumerate(zip(alpha, q_H_positions)):
        inp = outdir / f'{Path(out_prefix).name}_pt{i:02d}.nw'
        out = outdir / f'{Path(out_prefix).name}_pt{i:02d}.out'
        write_nwchem_input_cdft(cluster, xh_pos, donor_C_nw_idx, inp,
                                charge=charge_use, mult=mult_use,
                                cdft_spin=cdft_spin)
        print(f'  point {i:02d}: alpha={a_val:.3f} q={q_grid_along[i]:.3f} nwchem CDFT ...', flush=True)
        rc = run_nwchem(inp, out, workdir=outdir, nproc=nproc)
        E, converged = parse_nwchem_cdft(out)
        scan_results.append({'point':i, 'alpha':float(a_val),
                             'q_along_A':float(q_grid_along[i]),
                             'xferH_position':xh_pos.tolist(),
                             'E_hartree':E, 'cdft_converged':converged,
                             'return_code':rc})
        status = f'E={E:.6f} Ha (CDFT converged={converged}, rc={rc})' if E else f'ORCA/CDFT FAILED (rc={rc})'
        print(f'    {status}', flush=True)

    result = {
        'r_DA': float(r_DA), 'alpha_grid': alpha.tolist(),
        'q_along_A_grid': q_grid_along.tolist(),
        'n_cluster_atoms': len(cluster), 'mode': mode,
        'engine': 'nwchem-7.3.1-cdft',
        'cdft_spin_constraint': cdft_spin,
        'cdft_donor_C_nwchem_idx': donor_C_nw_idx,
        'charge': charge_use, 'mult': mult_use,
        'points': scan_results,
    }
    with open(outdir / f'{Path(out_prefix).name}_cdft_scan.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'wrote {outdir / (Path(out_prefix).name + "_cdft_scan.json")}')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prmtop', required=True)
    ap.add_argument('--rst7', required=True)
    ap.add_argument('--xferH', type=int, required=True)
    ap.add_argument('--donor', type=int, required=True)
    ap.add_argument('--acceptor', type=int, required=True)
    ap.add_argument('--npts', type=int, default=15)
    ap.add_argument('--alpha_center', type=float, default=None,
                    help='if npts=1, run a single point at this alpha')
    ap.add_argument('--mode', choices=['small','full'], default='small')
    ap.add_argument('--charge', type=int, default=1)
    ap.add_argument('--mult', type=int, default=6)
    ap.add_argument('--cdft_spin', type=float, default=1.0)
    ap.add_argument('--nproc', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    run_scan_cdft(a.prmtop, a.rst7, a.xferH, a.donor, a.acceptor,
                  npts=a.npts, alpha_center=a.alpha_center, mode=a.mode,
                  out_prefix=a.out, charge=a.charge, mult=a.mult,
                  cdft_spin=a.cdft_spin, nproc=a.nproc)
