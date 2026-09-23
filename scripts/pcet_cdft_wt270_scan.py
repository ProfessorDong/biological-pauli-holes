"""15-point NWChem-CDFT product-diabat scan of WT window 2.70 (v2, CORRECTED cluster).

Uses qm_cluster.build_cluster: 80 atoms, 7 H caps, charge +1, 299 electrons, mult 6.
The superseded v1 run used the uncapped 82-atom cluster at charge +2 and is archived
under results/_INVALID_charge_plus2_2026-08-10/. Runs on CPU/MPI so it can execute
concurrently with GPU umbrella sampling.

Runs 15 alpha values in [0.20, 0.80] sequentially, chaining the CDFT-converged
movecs from point i-1 into point i's initial guess so the SCF at each new
geometry starts from a warm orbital set. First point (alpha=0.20) is
cold-started from the reactant B3.4-dense atomic guess.

Wall time estimate: 5 h/point cold-started, 1.5 h/point warm-started;
total ~25-30 h for the full 15 points.
"""
import subprocess, os, sys, json, time, shutil
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from proton_pes_scanner import load_snapshot, cluster_atoms, choose_charge_mult
from proton_pes_scanner_cdft_nwchem import (
    find_donor_C_nwchem_idx, write_nwchem_input_cdft, parse_nwchem_cdft,
    NWCHEM_ENV, NWCHEM_BIN, MPIRUN, NWCHEM_BASIS_LIBRARY,
)

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
OUTDIR = ROOT / 'results' / 'pcet_cdft_v2'
PRM = ROOT / 'md/mcpb/SLO_sub_solv.prmtop'
# GEOMETRY CONSISTENCY: this MUST be the identical snapshot used by the
# reactant campaign (results/pcet_reactant_v2, r_DA = 3.407 A). The live
# results/umbrella/WT/ is being regenerated on Blackwell, so we read the
# archived snapshot the reactant scans actually used. A snapshot is an INPUT
# geometry, not a GPU-dependent result; what matters is that both diabats sit
# on byte-identical nuclei so that lambda, dG and the crossing seam are defined.
SEED = ROOT / 'results/umbrella/_RTX4060_rep1_archive_2026-08-10/WT/win_2.70_final.rst7'
DONOR, ACCEPTOR, XFERH = 13002, 12985, 13031

ALPHAS = np.linspace(0.20, 0.80, 15)
NPROC = 8
TIMEOUT_S = 21600     # 6 h per point


def write_input_with_prev(cluster, xferH_pos, donor_C_nw_idx, out_file,
                          charge, mult, cdft_spin, prev_cdft_movecs=None):
    """Same as write_nwchem_input_cdft but uses prev_cdft_movecs (if given) as
    the input orbitals for both steps, which lets the plain UKS pre-converge
    from a warm guess and the CDFT step continue from a converged constraint
    field."""
    stem = Path(out_file).stem
    lines = []
    lines.append(f'title "SLO CDFT WT-2.70 alpha step, spin({donor_C_nw_idx})={cdft_spin:.2f}"')
    lines.append('memory total 6 gb')
    lines.append(f'start {stem}')
    lines.append('geometry noautosym units angstroms')
    lines.append(f'  H  {xferH_pos[0]:.6f}  {xferH_pos[1]:.6f}  {xferH_pos[2]:.6f}')
    for elem, p, name in cluster:
        lines.append(f'  {elem}  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}')
    lines.append('end')
    lines.append('basis')
    lines.append('  * library "def2-SVP"')
    lines.append('end')
    lines.append(f'charge {charge}')
    # Step 1: unconstrained UKS to relax orbitals to the new geometry
    lines.append('dft')
    lines.append('  xc b3lyp')
    lines.append('  disp vdw 4')
    lines.append(f'  mult {mult}')
    lines.append('  odft')
    if prev_cdft_movecs is not None:
        lines.append(f'  vectors input {prev_cdft_movecs} output {stem}.pre.movecs')
    else:
        lines.append(f'  vectors output {stem}.pre.movecs')
    lines.append('  convergence energy 1e-5')
    lines.append('  iterations 200')
    lines.append('end')
    lines.append('task dft')
    # Step 2: switch on CDFT
    lines.append('dft')
    lines.append('  xc b3lyp')
    lines.append('  disp vdw 4')
    lines.append(f'  mult {mult}')
    lines.append('  odft')
    lines.append(f'  vectors input {stem}.pre.movecs output {stem}.cdft.movecs')
    lines.append(f'  cdft {donor_C_nw_idx} {donor_C_nw_idx} spin {cdft_spin:.3f}')
    lines.append('  convergence energy 1e-6')
    lines.append('  iterations 300')
    lines.append('end')
    lines.append('task dft')
    Path(out_file).write_text('\n'.join(lines) + '\n')


def run_nwchem_point(inp_path, out_path, cwd):
    env = os.environ.copy()
    env['NWCHEM_BASIS_LIBRARY'] = str(NWCHEM_BASIS_LIBRARY) + '/'
    env['PATH'] = f'{NWCHEM_ENV/"bin"}:' + env.get('PATH','')
    env['LD_LIBRARY_PATH'] = f'{NWCHEM_ENV/"lib"}:' + env.get('LD_LIBRARY_PATH','')
    env['OMP_NUM_THREADS'] = '1'
    cmd = [str(MPIRUN), '-n', str(NPROC), str(NWCHEM_BIN), Path(inp_path).name]
    t0 = time.time()
    with open(out_path, 'wb') as fo:
        try:
            proc = subprocess.run(cmd, env=env, cwd=str(cwd), stdout=fo,
                                  stderr=subprocess.STDOUT, timeout=TIMEOUT_S)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            fo.write(b'\nTIMEOUT\n')
            rc = 124
    return rc, time.time() - t0


def parse_energies(out_path):
    """Return dict with plain-UKS step1 energy, CDFT step2 last-iter energy,
    last multiplier, and CDFT-converged flag. Robust to timeout mid-CDFT."""
    import re
    txt = Path(out_path).read_text(errors='ignore')
    Es_step1_lines = re.findall(r'Total DFT energy\s*=\s*(-?\d+\.\d+)', txt)
    # First 'Total DFT energy' = step 1 plain UKS; second (if any) = step 2 CDFT converged
    E_step1 = float(Es_step1_lines[0]) if len(Es_step1_lines) >= 1 else None
    E_step2_converged = float(Es_step1_lines[1]) if len(Es_step1_lines) >= 2 else None
    # Last DIIS energy of the CDFT step (whether formally converged or not)
    cdft_iters = re.findall(
        r'd=\s*\d,ls=[\d\.]+,diis\s+(\d+)\s+(-?\d+\.\d+)\s+\S+\s+\S+\s+(\S+)\s+(\S+)', txt)
    # step 2 is any iter beyond step 1's last one; take the block after the first Total DFT energy
    if E_step1 is not None and 'Total DFT energy' in txt:
        idx = txt.index('Total DFT energy')
        after = txt[idx:]
        step2_iters = re.findall(
            r'd=\s*\d,ls=[\d\.]+,diis\s+(\d+)\s+(-?\d+\.\d+)\s+\S+\s+\S+\s+(\S+)\s+(\S+)', after)
    else:
        step2_iters = cdft_iters
    E_step2_lastiter = float(step2_iters[-1][1]) if step2_iters else None
    mults = re.findall(r'CDFT multipliers:\s*\n\s*\d+\s+(-?\d+\.\d+)', txt)
    mult_last = float(mults[-1]) if mults else None
    cdft_converged_marker = 'CDFT final multipliers' in txt
    return dict(E_step1=E_step1,
                E_step2_converged=E_step2_converged,
                E_step2_lastiter=E_step2_lastiter,
                mult_last=mult_last,
                cdft_converged=cdft_converged_marker,
                cdft_iter_count=len(step2_iters))


def main():
    prm, elems, pos = load_snapshot(PRM, SEED)
    xferH_pos = pos[XFERH]
    donor_pos = pos[DONOR]
    acc_pos = pos[ACCEPTOR]
    r_DA = float(np.linalg.norm(donor_pos - acc_pos))
    axis = (acc_pos - donor_pos) / r_DA
    q_H_positions = np.array([donor_pos + a * r_DA * axis for a in ALPHAS])
    from qm_cluster import build_cluster
    cluster, charge_use, mult_use, cluster_report = build_cluster(
        prm, elems, pos, donor_idx=DONOR, xferH_idx=XFERH)
    donor_C_nw_idx = find_donor_C_nwchem_idx(cluster)
    print(f'r_DA={r_DA:.3f} A; cluster={len(cluster)} atoms; donor_C_nw={donor_C_nw_idx}; '
          f'charge={charge_use}, mult={mult_use}')
    OUTDIR.mkdir(parents=True, exist_ok=True)

    campaign_json = OUTDIR / 'wt270_15pt_cdft_scan.json'
    # ---- resume: reload any points already completed in a previous run ----
    all_results = []
    done_alphas = set()
    if campaign_json.exists():
        try:
            prior = json.loads(campaign_json.read_text())
            for p in prior.get('points', []):
                E = p.get('E_step2_converged') or p.get('E_step2_lastiter')
                if E is not None:
                    all_results.append(p)
                    done_alphas.add(round(float(p['alpha']), 3))
            if all_results:
                print(f'RESUME: {len(all_results)} points already complete '
                      f'(alphas {sorted(done_alphas)})', flush=True)
        except Exception as e:
            print(f'  WARN: could not parse prior campaign json ({e}); starting fresh')
    # warm-start from the highest-alpha completed point's cdft movecs
    prev_movecs = None
    if done_alphas:
        last_done = max(done_alphas)
        cand = OUTDIR / f'wt270_alpha{int(round(last_done*100)):03d}.cdft.movecs'
        if cand.exists():
            prev_movecs = cand
            print(f'RESUME: warm-starting from {cand.name}', flush=True)

    for i, (alpha, xh_pos) in enumerate(zip(ALPHAS, q_H_positions)):
        if round(float(alpha), 3) in done_alphas:
            continue                          # already computed in a prior run
        stem = f'wt270_alpha{int(round(alpha*100)):03d}'
        inp = OUTDIR / f'{stem}.nw'
        out = OUTDIR / f'{stem}.out'
        if inp.exists():
            pass                              # allow re-run: fresh input each time
        write_input_with_prev(cluster, xh_pos, donor_C_nw_idx, inp,
                              charge=charge_use, mult=mult_use, cdft_spin=1.0,
                              prev_cdft_movecs=(prev_movecs.name if prev_movecs else None))
        # If we have a previous cdft.movecs, stage it next to this input under the expected name
        if prev_movecs is not None:
            (OUTDIR / prev_movecs.name).exists() or print(f'  WARN: prev movecs missing: {prev_movecs}')
        print(f'\n=== point {i:02d}/15  alpha={alpha:.3f}  r_H={alpha*r_DA:.3f} A '
              f'({"warm" if prev_movecs else "COLD"}) ===', flush=True)
        rc, wall = run_nwchem_point(inp, out, cwd=OUTDIR)
        parsed = parse_energies(out)
        E_report = parsed['E_step2_converged'] or parsed['E_step2_lastiter']
        pt = dict(idx=i, alpha=float(alpha), r_H=float(alpha*r_DA),
                  xferH_position=xh_pos.tolist(),
                  wall_s=float(wall), rc=int(rc), **parsed)
        all_results.append(pt)
        Path(campaign_json).write_text(json.dumps(
            dict(r_DA=r_DA, cluster_atoms=len(cluster),
                 donor_C_nw_idx=donor_C_nw_idx, charge=charge_use, mult=mult_use,
                 alphas=ALPHAS.tolist(), points=all_results), indent=2))
        cdft_movecs = OUTDIR / f'{stem}.cdft.movecs'
        pre_movecs = OUTDIR / f'{stem}.pre.movecs'
        if cdft_movecs.exists():
            prev_movecs = cdft_movecs
        elif pre_movecs.exists():
            prev_movecs = pre_movecs
        m = parsed["mult_last"]
        m_str = f'{m:.5f}' if m is not None else 'None'
        print(f'  E_step1={parsed["E_step1"]}, E_step2={E_report}, mult={m_str}, '
              f'CDFTconv={parsed["cdft_converged"]}, wall={wall/60:.1f} min, rc={rc}', flush=True)


if __name__ == '__main__':
    main()
