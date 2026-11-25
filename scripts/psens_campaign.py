#!/usr/bin/env python3
"""Run the model-P molecular dynamics for a list of variants (protonation sensitivity).

Per variant, in order (results/protonation_sensitivity/DECLARATION.md, Protocol):
  validate the topology against model S          psens_validate_topology.py
  minimize and equilibrate                        psens_equilibrate.py
  2.70 A umbrella window, 0.5 + 5.0 ns, k 12      umbrella_window.py
  static clamped runs r255 and r340, 0.3 + 3.0 ns, k 150     umbrella_window.py
  ensemble clamped runs r255 and r340, frames every 75 ps    umbrella_window_frames.py
  stage-5 analog, 3.0 ns at 2.55 A from the static r255 frame, frames every 50 ps
Every seed is drawn from seeds.json, written before the first run. A failed run is
repeated (at most three attempts, each with its own seed); every attempt is logged.
Each run's seed structure is checked for a periodic-image split between the substrate
and the acceptor, which the non-periodic bias would otherwise see as a huge distance.
Usage: psens_campaign.py TAG [TAG ...]      (slomd env)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import parmed

ROOT = Path(os.environ.get('PAULI_ROOT') or Path(__file__).resolve().parents[1])
PS = ROOT / 'results/protonation_sensitivity'
PY = sys.executable
LOG = PS / 'campaign.log'
SEEDS = json.loads((PS / 'seeds.json').read_text())


def log(msg):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def indices(prmtop):
    s = parmed.load_file(str(prmtop))
    d = [a.idx for a in s.atoms if a.residue.name == 'LIG' and a.name == 'C14']
    a = [a.idx for a in s.atoms if a.residue.name == 'OH1' and a.name == 'O']
    assert len(d) == 1 and len(a) == 1
    return d[0], a[0]


def check_seed(prmtop, rst7, don, acc):
    """Return a seed file whose substrate and acceptor share a periodic image."""
    s = parmed.load_file(str(prmtop), str(rst7))
    X = np.array(s.coordinates)
    r = float(np.linalg.norm(X[don] - X[acc]))
    if r < 8.0:
        return rst7, r
    box = np.array(s.box[:3], float)
    lig = [a.idx for a in s.residues[[at.residue.idx for at in s.atoms if at.idx == don][0]].atoms]
    best = None
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                sh = np.array([i, j, k]) * box
                rr = float(np.linalg.norm(X[don] + sh - X[acc]))
                if best is None or rr < best[0]:
                    best = (rr, sh)
    X[lig] += best[1]
    s.coordinates = X
    fixed = Path(str(rst7).replace('_final.rst7', '_final_unwrapped.rst7'))
    s.save(str(fixed), format='rst7', overwrite=True)
    log(f'  image split in {rst7.name}: r_DA {r:.2f} -> {best[0]:.3f} A after shifting LIG; using {fixed.name}')
    return fixed, best[0]


def run(step, cmd, outputs, tag):
    if all(Path(o).exists() for o in outputs):
        log(f'{tag} {step}: outputs present, skipped')
        return True
    for att in range(3):
        seed = SEEDS[tag][step][att]
        c = [x.replace('{SEED}', str(seed)) for x in cmd]
        log(f'{tag} {step}: attempt {att + 1}, seed {seed}')
        t0 = time.time()
        p = subprocess.run(c, capture_output=True, text=True, timeout=3 * 3600)
        with open(PS / 'logs' / f'{tag}_{step}_att{att + 1}.log', 'w') as f:
            f.write(' '.join(c) + '\n' + p.stdout + p.stderr)
        ok = all(Path(o).exists() for o in outputs)
        log(f'{tag} {step}: rc {p.returncode}, {"ok" if ok else "FAILED"}, {(time.time() - t0) / 60:.1f} min')
        if ok:
            return True
    return False


def campaign(tag):
    b, md = PS / 'build' / tag, PS / 'md' / tag
    for d in (md, PS / 'reactive_geometry', PS / 'ensemble_fluctuation', PS / 'stage5_md', PS / 'logs'):
        d.mkdir(parents=True, exist_ok=True)
    val = b / 'topology_validation.json'
    if not (val.exists() and json.loads(val.read_text())['passed']):
        p = subprocess.run([PY, str(ROOT / 'scripts/psens_validate_topology.py'), tag],
                           capture_output=True, text=True)
        if p.returncode != 0:
            log(f'{tag}: topology validation FAILED; variant stopped')
            return
    log(f'{tag}: topology validation passed')
    prm = b / 'P_solv.prmtop'
    don, acc = indices(prm)
    win = str(ROOT / 'scripts/umbrella_window.py')
    winf = str(ROOT / 'scripts/umbrella_window_frames.py')
    if not run('equilibrate', [PY, str(ROOT / 'scripts/psens_equilibrate.py'), tag, '{SEED}'],
               [md / 'P_eq.rst7'], tag):
        return
    if not run('window270', [PY, win, '--prmtop', str(prm), '--seed', str(md / 'P_eq.rst7'),
                             '--donor', str(don), '--acceptor', str(acc), '--r0', '2.70', '--k', '12.0',
                             '--ns-eq', '0.5', '--ns-prod', '5.0', '--out', str(md / 'win_2.70'),
                             '--rngseed', '{SEED}'],
               [md / 'win_2.70_final.rst7', md / 'win_2.70_colvar.dat'], tag):
        return
    seed270, r = check_seed(prm, md / 'win_2.70_final.rst7', don, acc)
    log(f'{tag}: 2.70 A window final r_DA {r:.3f} A')
    for clamp, r0 in (('r255', '2.55'), ('r340', '3.40')):
        out = PS / 'reactive_geometry' / f'{tag}_{clamp}'
        run(f'static_{clamp}', [PY, win, '--prmtop', str(prm), '--seed', str(seed270), '--donor', str(don),
                                '--acceptor', str(acc), '--r0', r0, '--k', '150.0', '--ns-eq', '0.3',
                                '--ns-prod', '3.0', '--out', str(out), '--rngseed', '{SEED}'],
            [f'{out}_final.rst7', f'{out}_colvar.dat'], tag)
        out = PS / 'ensemble_fluctuation' / f'{tag}_{clamp}'
        run(f'ensemble_{clamp}', [PY, winf, '--prmtop', str(prm), '--seed', str(seed270), '--donor', str(don),
                                  '--acceptor', str(acc), '--r0', r0, '--k', '150.0', '--ns-eq', '0.3',
                                  '--ns-prod', '3.0', '--frame-stride', '75', '--out', str(out),
                                  '--rngseed', '{SEED}'],
            [f'{out}_frames.npy', f'{out}_colvar.dat'], tag)
    st = PS / 'reactive_geometry' / f'{tag}_r255_final.rst7'
    if st.exists():
        seed5, r = check_seed(prm, st, don, acc)
        out = PS / 'stage5_md' / f'{tag}_r255'
        run('stage5', [PY, winf, '--prmtop', str(prm), '--seed', str(seed5), '--donor', str(don),
                       '--acceptor', str(acc), '--r0', '2.55', '--k', '150.0', '--ns-eq', '0.3',
                       '--ns-prod', '3.0', '--frame-stride', '50', '--out', str(out), '--rngseed', '{SEED}'],
            [f'{out}_frames.npy', f'{out}_colvar.dat'], tag)
    log(f'{tag}: campaign finished')


if __name__ == '__main__':
    for t in sys.argv[1:]:
        campaign(t)
    os._exit(0)
