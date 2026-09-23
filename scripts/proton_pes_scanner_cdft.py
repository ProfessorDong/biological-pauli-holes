"""B: CDFT product-state proton-PES scanner. Companion to proton_pes_scanner.py.

STATUS (2026-08-08): This script is a fully-written skeleton but is NOT
runnable on the project's ORCA 6.1.0 install because that build does not
expose CDFT. The %cdft block is not recognised; the %scf sub-block does not
accept a CDFT child block; header keywords CDFT / Constrained / ConstraintDFT
are silently ignored; and there is no orca_cdft executable in the install.
Two viable ports exist:
  - NWChem: has native CDFT (set dft:cdft_charge_type mulliken; constraint
    on atomic charge or spin density). conda install -c conda-forge nwchem;
    ~1-4 h per point on 4 cores for a 60-80 atom cluster.
  - CP2K: has native &CDFT block since CP2K 8. conda install cp2k; ~30-60 min
    per point on 4 cores.
Both require re-implementing write_orca_input_cdft() to write the target
package's input syntax while keeping the cluster-extraction, spin-density
constraint on the donor C, and json output schema unchanged. The rest of
this file (cluster construction from prmtop+rst7, product-state parity
handling, output aggregation) is written to be package-agnostic.

For each snapshot + proton-scan grid point, compute the UKS energy of the
constrained-DFT PRODUCT electronic state:
    Fe(II)-OH_2 + [substrate carbon radical]
by adding an ORCA %cdft block that pins one unit of spin density onto the
donor carbon (the alpha-C radical of the pentadienyl product). This forces the
electronic configuration in which one electron has transferred from the
substrate pi-system to the Fe(III) centre, giving the PCET product diabat.

Multiplicity: reactant is Fe(III) HS d5 S=5/2 (mult=6, parity-adjusted to
mult=5 fallback where cluster e-count requires). Product adds one radical
electron on the organic donor while Fe drops to d6 S=2, so overall unpaired
count is unchanged in the ferromagnetic-coupling limit; but because CDFT can
find either the ferromagnetic (mult=6) or the antiferromagnetic (mult=4)
product-state solution depending on the initial SCF guess, we scan at mult
choices matching the reactant parity so the two diabats live at the same
spin-space sector.

The product energy at each grid point defines the second diabatic surface;
combined with the B3.4-dense reactant-diabat surfaces, this gives the two
parabolas needed to extract Marcus reorganisation energy lambda, driving
force DeltaG, and (via the seam crossing) an approximate electronic coupling
V_el.

Convergence expectations: CDFT on a 60-80-atom cluster with UKS
B3LYP-D3BJ/def2-SVP + TightSCF + SlowConv typically requires 200-500
iterations; wall time 40-90 min per point on 4 cores. A 15-point scan is
therefore 10-25 h per snapshot. This pilot runs a single-point convergence
test at the transition state (alpha=0.5) of WT window 2.70 first, then
optionally extends to a full 15-point scan of that snapshot.

Usage:
  # single-point convergence test at alpha = 0.5
  proton_pes_scanner_cdft.py --seed R.rst7 --prmtop P.prmtop --xferH i --donor j --acceptor k \
       --npts 1 --alpha_center 0.5 --out OUT
  # full 15-point pilot
  proton_pes_scanner_cdft.py --seed R.rst7 --prmtop P.prmtop --xferH i --donor j --acceptor k \
       --npts 15 --out OUT
"""
import argparse, subprocess, numpy as np, os, sys, json
from pathlib import Path

# Reuse the reactant scanner's infrastructure to keep the cluster definition
# byte-identical between reactant and product surfaces.
sys.path.insert(0, str(Path(__file__).parent))
from proton_pes_scanner import (
    ORCA_DIR, ORCA_BIN, load_snapshot, cluster_atoms, _atom_z,
    parse_orca_energy, choose_charge_mult,
)


def write_orca_input_cdft(cluster, xferH_pos, donor_C_cluster_idx, out_file,
                          charge=2, mult=6, cdft_spin=1.0, alpha_mix=0.02,
                          functional='B3LYP', basis='def2-SVP', dispersion='D3BJ'):
    """Write an ORCA UKS input file with a CDFT constraint pinning ~cdft_spin
    unit of spin density on the donor carbon atom (0-indexed in the ORCA xyz
    ordering, where the xferH is atom 0 and the cluster atoms follow).
    """
    lines = []
    lines.append(f'! UKS {functional} {basis} {dispersion} TightSCF SlowConv')
    lines.append('%scf')
    lines.append('  maxiter 500')
    lines.append('  DIISMaxEq 15')
    lines.append('end')
    lines.append('%cdft')
    lines.append(f'  Atoms = {{{donor_C_cluster_idx}}}')
    lines.append('  ConstrainType = SpinDensity')
    lines.append(f'  Constraint = {cdft_spin:.3f}')
    lines.append(f'  Alpha = {alpha_mix:.3f}')
    lines.append('  MaxIter = 100')
    lines.append('  ConvThresh = 1.0e-4')
    lines.append('end')
    lines.append('%pal nprocs 4 end')
    lines.append(f'* xyz {charge} {mult}')
    lines.append(f'H  {xferH_pos[0]:.6f}  {xferH_pos[1]:.6f}  {xferH_pos[2]:.6f}')
    for elem, p, name in cluster:
        lines.append(f'{elem}  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}')
    lines.append('*')
    with open(out_file, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def find_donor_C_cluster_idx(cluster):
    """The donor C atom is the first C in the cluster whose source tag starts
    with 'donor-C'. In our ORCA xyz ordering the transferring H is atom 0
    (index 0), so cluster indices are shifted by +1 relative to the cluster list.
    ORCA CDFT uses 0-indexed atom numbers.
    """
    for i, (elem, p, name) in enumerate(cluster):
        if elem == 'C' and name.startswith('donor-C'):
            return i + 1
    for i, (elem, p, name) in enumerate(cluster):
        if elem == 'C':
            return i + 1
    raise RuntimeError('no donor C found in cluster')


def parse_cdft_energy(out_file):
    """Extract the CDFT-converged FINAL SINGLE POINT ENERGY, or None if
    CDFT did not converge. In ORCA 6.1 the CDFT block prints
    'CDFT CONVERGED' before the final SCF; a converged run therefore has
    both markers."""
    E = None
    cdft_converged = False
    txt = ''
    try:
        txt = Path(out_file).read_text()
    except Exception:
        return None, False
    for line in txt.splitlines():
        if 'FINAL SINGLE POINT ENERGY' in line:
            E = float(line.split()[-1])
        if 'CDFT CONVERGED' in line or 'CDFT converged' in line:
            cdft_converged = True
    return E, cdft_converged


def run_scan_cdft(prmtop, rst7, xferH_idx, donor_idx, acceptor_idx,
                  npts=15, alpha_center=None, mode='small', out_prefix='pes_cdft',
                  charge=2, mult=6, cdft_spin=1.0, alpha_mix=0.02):
    """Product-state PES scanner. Same grid geometry as proton_pes_scanner.py
    (alpha in [0.20, 0.80] * r_DA), but each ORCA input adds the %cdft block.

    If alpha_center is provided AND npts==1, a single-point convergence test is
    run at the specified alpha (default 0.5, i.e. transition-state proton
    midpoint) instead of a full 15-point grid.
    """
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

    cluster_full = cluster_atoms(prm, elems, pos, xferH_idx, donor_idx, acceptor_idx, mode=mode)
    cluster = [c for c in cluster_full if not np.allclose(c[1], xferH_pos, atol=1e-3)]
    donor_C_orca_idx = find_donor_C_cluster_idx(cluster)
    print(f'QM cluster: {len(cluster)} atoms (excluding scanned xferH); '
          f'donor-C is ORCA atom index {donor_C_orca_idx}')

    charge_use, mult_use = choose_charge_mult(cluster, xferH_present=True,
                                              prefer_charge=charge, prefer_mult=mult)
    if (charge_use, mult_use) != (charge, mult):
        print(f'  cluster electron parity requires (charge, mult) = ({charge_use}, {mult_use}) '
              f'(originally requested {charge}, {mult})')

    outdir = Path(out_prefix).parent
    outdir.mkdir(parents=True, exist_ok=True)
    OPENMPI_DIR = Path(os.environ.get('OPENMPI_DIR', '/home/liang/software/openmpi-4.1.8-install'))
    MULTIWFN_DIR = Path(os.environ.get('MULTIWFN_DIR', '/home/liang/software/Multiwfn_3.8_bin_Linux_noGUI/Multiwfn_3.8_bin_Linux_noGUI'))
    env = os.environ.copy()
    env['PATH'] = f'{ORCA_DIR}:{OPENMPI_DIR}/bin:{MULTIWFN_DIR}:{env["PATH"]}'
    env['LD_LIBRARY_PATH'] = f'{ORCA_DIR}:{OPENMPI_DIR}/lib:{env.get("LD_LIBRARY_PATH","")}'

    scan_results = []
    for i, (a_val, xh_pos) in enumerate(zip(alpha, q_H_positions)):
        inp = f'{out_prefix}_pt{i:02d}.inp'
        out = f'{out_prefix}_pt{i:02d}.out'
        write_orca_input_cdft(cluster, xh_pos, donor_C_orca_idx, inp,
                              charge=charge_use, mult=mult_use,
                              cdft_spin=cdft_spin, alpha_mix=alpha_mix)
        print(f'  point {i:02d}: alpha={a_val:.3f} q={q_grid_along[i]:.3f} running CDFT ORCA...', flush=True)
        try:
            proc = subprocess.run([str(ORCA_BIN), inp], capture_output=True,
                                  timeout=7200, env=env, cwd=str(Path(inp).parent))
            with open(out, 'wb') as f: f.write(proc.stdout)
            E, converged = parse_cdft_energy(out)
        except subprocess.TimeoutExpired:
            E, converged = None, False
            with open(out, 'w') as f: f.write('TIMEOUT\n')
        scan_results.append({'point':i, 'alpha':float(a_val),
                             'q_along_A':float(q_grid_along[i]),
                             'xferH_position':xh_pos.tolist(),
                             'E_hartree':E, 'cdft_converged':converged})
        status = f'E={E:.6f} Ha (CDFT converged={converged})' if E else 'ORCA/CDFT FAILED'
        print(f'    {status}')

    result = {
        'r_DA': float(r_DA), 'alpha_grid': alpha.tolist(),
        'q_along_A_grid': q_grid_along.tolist(),
        'n_cluster_atoms': len(cluster), 'mode': mode,
        'cdft_spin_constraint': cdft_spin, 'cdft_donor_C_orca_idx': donor_C_orca_idx,
        'charge': charge_use, 'mult': mult_use,
        'points': scan_results,
    }
    with open(f'{out_prefix}_cdft_scan.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'wrote {out_prefix}_cdft_scan.json')
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
    ap.add_argument('--charge', type=int, default=2)
    ap.add_argument('--mult', type=int, default=6)
    ap.add_argument('--cdft_spin', type=float, default=1.0,
                    help='CDFT spin-density target on donor C (1.0 = alpha-radical)')
    ap.add_argument('--alpha_mix', type=float, default=0.02,
                    help='CDFT Lagrange multiplier mixing coefficient')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    run_scan_cdft(a.prmtop, a.rst7, a.xferH, a.donor, a.acceptor,
                  npts=a.npts, alpha_center=a.alpha_center, mode=a.mode,
                  out_prefix=a.out, charge=a.charge, mult=a.mult,
                  cdft_spin=a.cdft_spin, alpha_mix=a.alpha_mix)
