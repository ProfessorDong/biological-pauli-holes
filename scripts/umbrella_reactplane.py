#!/usr/bin/env python3
"""Short umbrella-window re-run that dumps per-frame (r_DA, r_HO, theta_CHO) for MBAR
2D reactive-plane reweighting (Category A polish task A3-proper).

Restarts from the existing NPT-equilibrated win_r0_final.rst7 files produced by the
primary umbrella campaign, applies the same harmonic bias on r_DA, and records the
three observables every 1 ps during 200 ps of production. No new equilibration is
needed because the rst7 is the endpoint of a 5 ns biased run at exactly the same k, r0.

Usage: umbrella_reactplane.py --prmtop P --seed S.rst7 --donor D --acceptor A
       --xferH H --r0 R --k K --ns-prod 0.2 --out OUT
"""
import os as _os
import argparse, numpy as np
from openmm import app, unit, MonteCarloBarostat, LangevinMiddleIntegrator, CustomBondForce, Platform

ap = argparse.ArgumentParser()
ap.add_argument("--prmtop", required=True)
ap.add_argument("--seed",   required=True)
ap.add_argument("--donor",   type=int, required=True)
ap.add_argument("--acceptor",type=int, required=True)
ap.add_argument("--xferH",   type=int, required=True)
ap.add_argument("--r0",      type=float, required=True)
ap.add_argument("--k",       type=float, default=12.0)
ap.add_argument("--ns-prod", type=float, default=0.2)
ap.add_argument("--temp",    type=float, default=300.0)
ap.add_argument("--out",     required=True)
ap.add_argument("--rngseed", type=int, default=0)
a = ap.parse_args()

prm = app.AmberPrmtopFile(a.prmtop)
seed = app.AmberInpcrdFile(a.seed)
sys = prm.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer,
                       constraints=app.HBonds, hydrogenMass=4.0*unit.amu)
sys.addForce(MonteCarloBarostat(1.0*unit.bar, a.temp*unit.kelvin, 25))
kbias = a.k * 4.184 * 100.0
bias = CustomBondForce("0.5*kb*(r-r0nm)^2")
bias.addPerBondParameter("kb"); bias.addPerBondParameter("r0nm")
bias.addBond(a.donor, a.acceptor, [kbias, a.r0/10.0]); sys.addForce(bias)

integ = LangevinMiddleIntegrator(a.temp*unit.kelvin, 1.0/unit.picosecond, 0.004*unit.picoseconds)
if a.rngseed > 0: integ.setRandomNumberSeed(a.rngseed)
sim = app.Simulation(prm.topology, sys, integ,
                     Platform.getPlatformByName('CUDA'), {'Precision':'mixed'})
sim.context.setPositions(seed.positions)
if seed.boxVectors is not None:
    sim.context.setPeriodicBoxVectors(*seed.boxVectors)
sim.context.setVelocitiesToTemperature(a.temp*unit.kelvin, a.rngseed if a.rngseed>0 else None)

# No equilibration - the seed rst7 is already NPT-equilibrated under the same bias.
n_steps  = int(a.ns_prod * 250000)         # 4 fs timestep
n_rec    = 250                              # every 1 ps
n_frames = n_steps // n_rec

rDA_ts = np.zeros(n_frames, dtype=np.float32)
rHO_ts = np.zeros(n_frames, dtype=np.float32)
theta_ts = np.zeros(n_frames, dtype=np.float32)

D, A, H = a.donor, a.acceptor, a.xferH
for i in range(n_frames):
    sim.step(n_rec)
    p = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    dvec = p[D] - p[H]; avec = p[A] - p[H]
    rDA_ts[i] = np.linalg.norm(p[D] - p[A])
    rHO_ts[i] = np.linalg.norm(p[A] - p[H])
    cost = np.dot(dvec, avec) / (np.linalg.norm(dvec)*np.linalg.norm(avec) + 1e-12)
    theta_ts[i] = np.degrees(np.arccos(np.clip(cost, -1, 1)))

np.savez_compressed(a.out + "_reactplane.npz",
                    r_DA=rDA_ts, r_HO=rHO_ts, theta_CHO=theta_ts,
                    r0=a.r0, k=a.k, ns_prod=a.ns_prod)

print(f'  r0={a.r0:.2f} n={n_frames}  <r_DA>={rDA_ts.mean():.3f}  '
      f'<r_HO>={rHO_ts.mean():.3f}  <theta>={theta_ts.mean():.2f}', flush=True)

import sys as _sys, os as _os
_sys.stdout.flush(); _sys.stderr.flush()
_os._exit(0)
