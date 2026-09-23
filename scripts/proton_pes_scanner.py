"""B3.2: proton-PES scanner. For a given SLO snapshot, extract a small QM cluster
around the reactive C-H-O triad, scan the transferring H along the C-O axis at N
grid points, run ORCA UKS B3LYP-D3BJ/def2-SVP at each point, and output V(q_H)
suitable for the B3.1 Schrödinger solver.

Small-cluster mode: QM region = substrate C-H (donor), Fe-OH (acceptor), and only
the imidazole ring of one His ligand + one water. ~30 atoms. ~15-30 min per point,
~4-8 hours per full scan.

Full-cluster mode: adds all three His imidazoles + Asn694 amide + Ile-COO + waters.
~60-80 atoms. ~30-60 min per point, ~10-15 hours per scan.

Usage: proton_pes_scanner.py --seed R.rst7 --prmtop P.prmtop --xferH i --donor j --acceptor k
       --npts 15 --range 0.6 --mode small --out OUT.json
"""
import argparse, subprocess, numpy as np, os, sys, json
from pathlib import Path

ORCA_DIR = Path(os.environ.get('ORCA_DIR', '/home/liang/software/orca_6_1_0_linux_x86-64_shared_openmpi418'))
ORCA_BIN = ORCA_DIR / 'orca'

# --- coordinate extraction ---
def load_snapshot(prmtop, rst7):
    """Return per-atom elements + coordinates from an Amber prmtop+rst7."""
    import parmed
    from openmm import app, unit
    prm = parmed.load_file(str(prmtop))
    seed = app.AmberInpcrdFile(str(rst7))
    pos = np.array([[v.x, v.y, v.z] for v in seed.positions.value_in_unit(unit.angstrom)])
    # Element by atomic number
    Z_to_symbol = {1:'H', 6:'C', 7:'N', 8:'O', 12:'Mg', 15:'P', 16:'S', 26:'Fe'}
    elements = []
    for a in prm.atoms:
        # parmed atom.element is not always reliable; use atom.name
        z = a.atomic_number if hasattr(a, 'atomic_number') else None
        if z in Z_to_symbol:
            elements.append(Z_to_symbol[z])
        else:
            # Fallback: use first char of atom name
            name = a.name
            if name.startswith('H'): elements.append('H')
            elif name.startswith('C'): elements.append('C')
            elif name.startswith('N'): elements.append('N')
            elif name.startswith('O'): elements.append('O')
            elif name.startswith('S'): elements.append('S')
            elif name.startswith('Fe'): elements.append('Fe')
            elif name.startswith('P'): elements.append('P')
            else: elements.append('X')
    return prm, elements, pos


def cluster_atoms(prm, elements, pos, xferH_idx, donor_idx, acceptor_idx, mode='small'):
    """Return list of (element, position, source_name) for the QM cluster."""
    cluster = []
    # Donor carbon + its H atoms (linoleate C11 with all H)
    donor_bonded_H = []
    for b in prm.atoms[donor_idx].bonds:
        other = b.atom2 if b.atom1.idx == donor_idx else b.atom1
        if other.element == 1 or other.name.startswith('H'):
            donor_bonded_H.append(other.idx)
    cluster.append((elements[donor_idx], pos[donor_idx].copy(), f'donor-C:{prm.atoms[donor_idx].name}'))
    for h in donor_bonded_H:
        cluster.append((elements[h], pos[h].copy(), f'donor-{prm.atoms[h].name}'))
    # Fe-OH acceptor: Fe + O + H
    # Find Fe residue
    fe_res = [r for r in prm.residues if 'FE' in r.name.upper()]
    if fe_res:
        for a in fe_res[0].atoms:
            cluster.append((elements[a.idx], pos[a.idx].copy(), f'FE1:{a.name}'))
    oh_res = [r for r in prm.residues if 'OH' in r.name.upper() and r.name in ('OH1','OH','OHF')]
    if oh_res:
        for a in oh_res[0].atoms:
            cluster.append((elements[a.idx], pos[a.idx].copy(), f'OH:{a.name}'))
    # First-shell Fe ligands (needed to properly coordinate the Fe(III) electronic state):
    # 3 histidines + Asn amide side-chain O + Ile C-terminal carboxylate.
    # Add these for BOTH small and full modes — otherwise the Fe electronic state is unphysical.
    Fe_idx = None
    for r in prm.residues:
        if r.name.startswith('FE'):
            for a in r.atoms:
                if a.name.startswith('FE') or elements[a.idx] == 'Fe':
                    Fe_idx = a.idx; break
        if Fe_idx is not None: break
    added_atoms = {c[1].tobytes() for c in cluster}
    if Fe_idx is not None:
        Fe_pos = pos[Fe_idx]
        # For all residues, if any atom is within 3.5 A of Fe, include the SUFFICIENT part of the residue
        for r in prm.residues:
            if r.name.startswith(('FE','OH','WAT','HOH','Na','Cl')): continue
            if r.name == prm.residues[donor_idx // 999999 if False else 0].name and False: continue
            any_near = False
            for a in r.atoms:
                if np.linalg.norm(pos[a.idx] - Fe_pos) < 3.5:
                    any_near = True; break
            if not any_near: continue
            # This is a first-shell ligand. Add the side-chain heavy atoms + their H, up to ~10 A from Fe.
            if r.name.startswith('HD') or r.name == 'HIS':
                keep_names = ['CB','HB2','HB3','CG','ND1','HD1','CE1','HE1','NE2','HE2','CD2','HD2']
            elif r.name.startswith('ASN'):
                keep_names = ['CB','HB2','HB3','CG','OD1','ND2','HD21','HD22']
            elif r.name.startswith('IE') or r.name == 'ILE':
                # Ile-COO C-terminus: include the whole residue's side chain + COO
                keep_names = ['CB','HB','CG1','HG12','HG13','CG2','HG21','HG22','HG23',
                              'CD1','HD11','HD12','HD13','C','O','OXT']
            else:
                keep_names = [a.name for a in r.atoms if not a.name.startswith('H')]
            for a in r.atoms:
                if a.name not in keep_names: continue
                key = pos[a.idx].tobytes()
                if key in added_atoms: continue
                cluster.append((elements[a.idx], pos[a.idx].copy(), f'{r.name}{r.number}:{a.name}'))
                added_atoms.add(key)
    if mode == 'full':
        # Full mode: also add nearest waters within 4 A of any cluster atom
        cluster_pos = np.array([c[1] for c in cluster])
        for r in prm.residues:
            if r.name not in ('WAT','HOH'): continue
            for a in r.atoms:
                d = np.min(np.linalg.norm(cluster_pos - pos[a.idx], axis=1))
                if d <= 4.0:
                    # add whole water
                    for aa in r.atoms:
                        key = pos[aa.idx].tobytes()
                        if key in added_atoms: continue
                        cluster.append((elements[aa.idx], pos[aa.idx].copy(),
                                        f'WAT{r.number}:{aa.name}'))
                        added_atoms.add(key)
                    break
    return cluster


def _atom_z(elem):
    """Atomic number for common elements in these clusters."""
    return {'H':1, 'C':6, 'N':7, 'O':8, 'S':16, 'Fe':26, 'P':15}.get(elem, 0)


def choose_charge_mult(cluster_atoms_list, xferH_present=True, prefer_charge=1, prefer_mult=6):
    """Validate that (charge, mult) is consistent with the cluster's electron parity.

    For Fe(III) high-spin d5 the reactant state has 5 unpaired electrons (S=5/2,
    mult=6), which requires an ODD total electron count.

    HISTORY / WHY THIS RAISES INSTEAD OF FALLING BACK: an earlier version of this
    function silently downgraded mult 6 -> 5 whenever parity forbade the requested
    multiplicity. That masked a wrong cluster charge (+2 instead of +1, i.e. the
    Ile C-terminal carboxylate's -1 was omitted from the formal-charge accounting).
    With charge=+2 the cluster carries 374 electrons (even), mult=6 is impossible,
    and the SCF relaxed instead to <S^2> ~ 11-12.3 (S=3) in BOTH ORCA and NWChem:
    Fe(III) S=5/2 ferromagnetically coupled to a SPURIOUS ligand radical created by
    removing one electron too many. The parity mismatch is a hard error about the
    charge, never something to paper over by changing the spin state.

    Correct SLO cluster accounting: Fe(III) +3, OH- -1, Ile817 C-terminal
    carboxylate -1, three neutral His, neutral Asn672, neutral substrate
    fragment  =>  net +1, 375 electrons (odd), mult 6, <S^2> = 8.75.

    Returns (charge, mult) unchanged when consistent; raises ValueError otherwise.
    """
    Z_sum = sum(_atom_z(e) for e, _, _ in cluster_atoms_list)
    if xferH_present:
        Z_sum += 1
    n_e = Z_sum - prefer_charge
    unpaired = prefer_mult - 1
    if (unpaired % 2) == (n_e % 2):
        return prefer_charge, prefer_mult
    ok = [q for q in range(prefer_charge - 2, prefer_charge + 3)
          if ((Z_sum - q) % 2) == (unpaired % 2)]
    raise ValueError(
        f'Electron-parity mismatch: sum(Z)={Z_sum}, charge={prefer_charge:+d} gives '
        f'{n_e} electrons ({"even" if n_e % 2 == 0 else "odd"}), which cannot support '
        f'mult={prefer_mult} ({unpaired} unpaired). Fix the CHARGE, not the multiplicity. '
        f'Charges compatible with mult={prefer_mult}: {ok}. '
        f'For the SLO Fe(III)-OH cluster the correct value is +1.')


def write_orca_input(cluster, xferH_pos, out_file, charge=1, mult=6, functional='B3LYP',
                     basis='def2-SVP', dispersion='D3BJ'):
    """Write an ORCA input file for a UKS single-point energy calculation.
    For an Fe(III)-OH^- cluster + neutral organic: net charge = +2, S=5/2 -> mult=6.
    """
    lines = []
    lines.append(f'! UKS {functional} {basis} {dispersion} TightSCF SlowConv')
    lines.append('%scf')
    lines.append('  maxiter 300')
    lines.append('end')
    lines.append('%pal nprocs 4 end')
    lines.append(f'* xyz {charge} {mult}')
    # Write the xferH first (we'll edit this atom's position per scan point)
    lines.append(f'H  {xferH_pos[0]:.6f}  {xferH_pos[1]:.6f}  {xferH_pos[2]:.6f}')
    for elem, p, name in cluster:
        lines.append(f'{elem}  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}')
    lines.append('*')
    with open(out_file, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def parse_orca_energy(out_file):
    """Extract 'FINAL SINGLE POINT ENERGY' from an ORCA output. Returns Hartrees."""
    E = None
    try:
        for line in Path(out_file).read_text().splitlines():
            if 'FINAL SINGLE POINT ENERGY' in line:
                E = float(line.split()[-1])
    except Exception:
        pass
    return E


def parse_orca_s2_conv(out_file):
    """Extract <S**2> and the SCF-converged flag from an ORCA output.

    <S^2> is the single most informative diagnostic for this system: the target
    Fe(III) high-spin d5 reactant (S=5/2, mult 6) must give <S^2> = 8.75. A value
    far above that means the SCF found a different electronic state, which is how
    the 2026-08-10 cluster defect announced itself (<S^2> ~ 11-12.3 against a
    requested 6.0). Never report an energy from this pipeline without it.
    """
    s2, converged = None, False
    try:
        for line in Path(out_file).read_text(errors='ignore').splitlines():
            if 'Expectation value of <S**2>' in line:
                try:
                    s2 = float(line.split()[-1])
                except ValueError:
                    pass
            if 'SCF CONVERGED AFTER' in line:
                converged = True
    except Exception:
        pass
    return s2, converged


def run_scan(prmtop, rst7, xferH_idx, donor_idx, acceptor_idx,
             npts=15, q_range=0.6, mode='small', out_prefix='pes',
             charge=1, mult=6, alpha_center=None):
    """Scan the transferring H along the donor→acceptor axis, at (npts) evenly-spaced
    q_H positions spanning [q_H_min - q_range/2, q_H_min + q_range/2]...
    Actually just span the C-O axis from 0.9 A (near donor C) to r_DA-0.9 A (near acceptor O),
    parameterised by fraction alpha in [0.15, 0.85]."""
    prm, elems, pos = load_snapshot(prmtop, rst7)
    xferH_pos = pos[xferH_idx]
    donor_pos = pos[donor_idx]
    acc_pos = pos[acceptor_idx]
    r_DA = np.linalg.norm(donor_pos - acc_pos)
    axis = (acc_pos - donor_pos) / r_DA
    print(f'r_DA = {r_DA:.3f} A  axis={axis}')
    # xfer H's projection onto the C-O axis, relative to donor C
    proj0 = float(np.dot(xferH_pos - donor_pos, axis))
    print(f'current xferH projection = {proj0:.3f} A along C->O')
    # Grid alpha in [0.20, 0.80] * r_DA -> q_H position along axis.
    # alpha_center with npts=1 runs a single point (used for electronic-state
    # validation at the reactant well, alpha ~ r_CH / r_DA).
    if npts == 1 and alpha_center is not None:
        alpha = np.array([alpha_center])
    else:
        alpha = np.linspace(0.20, 0.80, npts)
    q_H_positions = np.array([donor_pos + a * r_DA * axis for a in alpha])
    q_grid_along = alpha * r_DA   # 1D coordinate along C-O axis
    # Now for the QM cluster: build once (excluding the transferring H), then per-point
    # scan the position of the xferH.
    # Remove the transferring H from the cluster (we'll insert it per scan point)
    # Build the QM cluster with the VALIDATED builder (qm_cluster.build_cluster):
    # every severed bond is closed with an H cap, the formal charge comes from an
    # explicit per-fragment table, and the builder raises rather than returning a
    # cluster with under-coordinated heavy atoms. The legacy cluster_atoms() path
    # emitted 35 severed bonds with zero caps; see qm_cluster.py for the history.
    from qm_cluster import build_cluster
    cluster, charge_use, mult_use, cluster_report = build_cluster(
        prm, elems, pos, donor_idx=donor_idx, xferH_idx=xferH_idx)
    outdir = Path(out_prefix).parent
    outdir.mkdir(parents=True, exist_ok=True)
    if (charge_use, mult_use) != (charge, mult):
        print(f'  NOTE: builder-derived (charge, mult) = ({charge_use}, {mult_use}); '
              f'CLI requested ({charge}, {mult}). Using the builder value.')
    scan_results = []
    for i, (a_val, xh_pos) in enumerate(zip(alpha, q_H_positions)):
        inp = f'{out_prefix}_pt{i:02d}.inp'
        out = f'{out_prefix}_pt{i:02d}.out'
        write_orca_input(cluster, xh_pos, inp, charge=charge_use, mult=mult_use)
        print(f'  point {i:02d}: alpha={a_val:.3f}  q={q_grid_along[i]:.3f}  running ORCA...', flush=True)
        # Run ORCA
        env = os.environ.copy()
        OPENMPI_DIR = Path(os.environ.get('OPENMPI_DIR', '/home/liang/software/openmpi-4.1.8-install'))
        MULTIWFN_DIR = Path(os.environ.get('MULTIWFN_DIR', '/home/liang/software/Multiwfn_3.8_bin_Linux_noGUI/Multiwfn_3.8_bin_Linux_noGUI'))
        env['PATH'] = f'{ORCA_DIR}:{OPENMPI_DIR}/bin:{MULTIWFN_DIR}:{env["PATH"]}'
        env['LD_LIBRARY_PATH'] = f'{ORCA_DIR}:{OPENMPI_DIR}/lib:{env.get("LD_LIBRARY_PATH","")}'
        proc = subprocess.run([str(ORCA_BIN), inp], capture_output=True, timeout=3600, env=env,
                              cwd=str(Path(inp).parent))
        # Write ORCA stdout to .out
        with open(out, 'wb') as f: f.write(proc.stdout)
        E = parse_orca_energy(out)
        s2, converged = parse_orca_s2_conv(out)
        s2_ref = cluster_report['S2_expected']
        scan_results.append({'point':i, 'alpha':float(a_val), 'q_along_A':float(q_grid_along[i]),
                             'xferH_position':xh_pos.tolist(), 'E_hartree':E,
                             'S2': s2, 'S2_expected': s2_ref, 'scf_converged': converged,
                             'spin_contamination': (None if s2 is None else round(s2 - s2_ref, 4))})
        if E is None:
            print('    ORCA failed')
        else:
            flag = ''
            if s2 is not None and abs(s2 - s2_ref) > 0.5:
                flag = f'   *** SPIN CONTAMINATION {s2 - s2_ref:+.2f} ***'
            print(f'    E = {E:.6f} Ha   <S^2> = {s2}  (expect {s2_ref:.2f})'
                  f'  SCF_conv={converged}{flag}')
    # Save
    result = {
        'r_DA': float(r_DA), 'alpha_grid': alpha.tolist(),
        'q_along_A_grid': q_grid_along.tolist(),
        'n_cluster_atoms': len(cluster) + 1, 'mode': mode,
        'cluster_report': {k: v for k, v in cluster_report.items()
                           if k != 'undercoordinated'},
        'charge': charge_use, 'mult': mult_use,
        'points': scan_results,
    }
    with open(f'{out_prefix}_scan.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'wrote {out_prefix}_scan.json')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prmtop', required=True)
    ap.add_argument('--rst7', required=True)
    ap.add_argument('--xferH', type=int, required=True)
    ap.add_argument('--donor', type=int, required=True)
    ap.add_argument('--acceptor', type=int, required=True)
    ap.add_argument('--npts', type=int, default=11)
    ap.add_argument('--alpha_center', type=float, default=None,
                    help='if npts=1, run a single point at this alpha')
    ap.add_argument('--mode', choices=['small','full'], default='small')
    ap.add_argument('--charge', type=int, default=1)
    ap.add_argument('--mult', type=int, default=6)
    ap.add_argument('--out', required=True, help='output prefix (files will be OUT_pt00.inp/out, OUT_scan.json)')
    a = ap.parse_args()
    run_scan(a.prmtop, a.rst7, a.xferH, a.donor, a.acceptor,
             npts=a.npts, mode=a.mode, out_prefix=a.out,
             charge=a.charge, mult=a.mult, alpha_center=a.alpha_center)
