#!/usr/bin/env python3
"""Minimize and equilibrate a freshly built model-P system (protonation sensitivity).

Mirrors the model-S protocol (md/mcpb/minimize_sub.py, md/mcpb/equilibrate_sub.py):
  minimization: solute heavy atoms restrained at 10 kcal/mol/A^2 (3000 iterations),
                then free (5000 iterations)
  heating:      10 -> 300 K in 10 K steps of 3 ps (90 ps) at 1 bar, solute heavy atoms
                restrained (E = k dx^2, k = 5 kcal/mol/A^2), 2 fs, no mass repartitioning
  equilibration 1 ns at 1 bar while the restraint is released in 20 steps
The one deliberate difference: the system starts from the 2.70 A umbrella window of
model S, so the model-S r_DA restraints (flat bottom 3.0-3.5 A while minimizing, upper
wall at 5.5 A while equilibrating) are replaced by that window's own bias,
0.5 k (r - 2.70 A)^2 with k = 12 kcal/mol/A^2.
The new cap atoms are left unrestrained in the first minimization stage.
The final state is written UNWRAPPED, so the substrate and the acceptor stay in one
periodic image for the non-periodic bias of the next run.
Usage: psens_equilibrate.py TAG RNGSEED      (slomd env)
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from openmm import (CustomBondForce, CustomExternalForce, LangevinMiddleIntegrator,
                    MonteCarloBarostat, Platform, app, unit)

ROOT = Path(os.environ.get('PAULI_ROOT') or Path(__file__).resolve().parents[1])
KCAL_A2 = 4.184 * 100.0          # kcal/mol/A^2 -> kJ/mol/nm^2


def write_rst7(path, positions_A, box_A, title):
    n = len(positions_A)
    lines = [title[:80], f'{n:5d}']
    flat = np.asarray(positions_A, float).reshape(-1)
    for i in range(0, len(flat), 6):
        lines.append(''.join(f'{v:12.7f}' for v in flat[i:i + 6]))
    lines.append(''.join(f'{v:12.7f}' for v in list(box_A) + [90.0, 90.0, 90.0]))
    fd, tmp = tempfile.mkstemp(dir=str(Path(path).parent), suffix='.rst7.part')
    with os.fdopen(fd, 'w') as f:
        f.write('\n'.join(lines) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    tag, seed = sys.argv[1], int(sys.argv[2])
    b = ROOT / 'results/protonation_sensitivity/build' / tag
    out = ROOT / 'results/protonation_sensitivity/md' / tag
    out.mkdir(parents=True, exist_ok=True)
    prm = app.AmberPrmtopFile(str(b / 'P_solv.prmtop'))
    crd = app.AmberInpcrdFile(str(b / 'P_solv.inpcrd'))
    top = prm.topology
    atoms = list(top.atoms())
    don = [a.index for a in atoms if a.residue.name == 'LIG' and a.name == 'C14']
    acc = [a.index for a in atoms if a.residue.name == 'OH1' and a.name == 'O']
    assert len(don) == 1 and len(acc) == 1
    don, acc = don[0], acc[0]
    # solute = every residue up to and including the substrate; OpenMM renames Amber WAT to HOH,
    # so a solvent filter by residue name would silently restrain the whole solvent
    lig_res = [r.index for r in top.residues() if r.name == 'LIG']
    assert len(lig_res) == 1
    solute_heavy = [a.index for a in atoms if a.residue.index <= lig_res[0]
                    and a.element is not None and a.element.symbol != 'H']
    assert all(a.residue.name not in ('HOH', 'WAT', 'NA', 'Na+') for a in atoms if a.index in set(solute_heavy))
    caps = {a.index for a in atoms if a.residue.name in ('ACE', 'NME')}

    sysm = prm.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=10 * unit.angstrom,
                            constraints=app.HBonds, rigidWater=True)
    bias = CustomBondForce('0.5*kb*(r-r0nm)^2')
    bias.addPerBondParameter('kb')
    bias.addPerBondParameter('r0nm')
    bias.addBond(don, acc, [12.0 * KCAL_A2, 0.270])
    sysm.addForce(bias)
    pos0 = np.array(crd.positions.value_in_unit(unit.nanometer))
    # periodicdistance: the tleap box has its origin at the corner, so OpenMM may translate the
    # protein by a box vector during the run; an absolute-coordinate restraint would then explode
    posr = CustomExternalForce('kref*periodicdistance(x,y,z,x0,y0,z0)^2')
    posr.addGlobalParameter('kref', 0.0)
    for nm in ('x0', 'y0', 'z0'):
        posr.addPerParticleParameter(nm)
    for i in solute_heavy:
        if i not in caps:
            posr.addParticle(i, list(pos0[i]))
    sysm.addForce(posr)
    sysm.addForce(MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin, 25))

    integ = LangevinMiddleIntegrator(10 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    integ.setRandomNumberSeed(seed)
    sim = app.Simulation(top, sysm, integ, Platform.getPlatformByName('CUDA'), {'Precision': 'mixed'})
    sim.context.setPositions(crd.positions)
    sim.context.setPeriodicBoxVectors(*crd.boxVectors)

    def rda():
        p = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        return float(np.linalg.norm(p[don] - p[acc]))

    log = dict(tag=tag, rngseed=seed, donor=don, acceptor=acc, n_restrained=posr.getNumParticles(),
               start=time.strftime('%Y-%m-%dT%H:%M:%S'), rDA={})
    log['rDA']['start'] = rda()
    sim.context.setParameter('kref', 10.0 * KCAL_A2)
    sim.minimizeEnergy(maxIterations=3000)
    log['rDA']['min1'] = rda()
    sim.context.setParameter('kref', 0.0)
    sim.minimizeEnergy(maxIterations=5000)
    log['rDA']['min2'] = rda()
    log['E_min_kJ'] = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    sim.context.setParameter('kref', 5.0 * KCAL_A2)
    for T in range(10, 301, 10):
        integ.setTemperature(T * unit.kelvin)
        sim.step(1500)
    log['rDA']['heated'] = rda()
    nsteps, nblk = 500000, 20
    trace = []
    for k in range(nblk):
        sim.context.setParameter('kref', 5.0 * (1 - (k + 1) / nblk) * KCAL_A2)
        sim.step(nsteps // nblk)
        trace.append(rda())
    log['rDA']['npt_blocks'] = trace
    st = sim.context.getState(getPositions=True, enforcePeriodicBox=False)
    p = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    bv = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
    box = [float(bv[0][0]), float(bv[1][1]), float(bv[2][2])]
    write_rst7(out / 'P_eq.rst7', p, box, f'model-P {tag} equilibrated, seed {seed}')
    log.update(end=time.strftime('%Y-%m-%dT%H:%M:%S'), box_A=box,
               rDA_final=float(np.linalg.norm(p[don] - p[acc])))
    (out / 'equilibration_log.json').write_text(json.dumps(log, indent=1))
    print(json.dumps({k: log[k] for k in ('tag', 'rngseed', 'rDA_final', 'box_A')}))
    os._exit(0)


if __name__ == '__main__':
    main()
