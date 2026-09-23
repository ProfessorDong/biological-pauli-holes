#!/usr/bin/env python3
"""
md_run.py -- OpenMM driver for the SLO gating-axis MD (Part B / MD_protocol.md).

Reads an AMBER prmtop+inpcrd (built by tleap with the MCPB Fe(III)-OH params +
GAFF2/RESP linoleate), then: minimize -> NVT heat -> NPT equilibrate (with
restraints ramped down) -> production. Keeps a one-sided flat-bottom wall on the
donor-acceptor distance so the substrate cannot unbind during plain MD (remove it
/ replace with PLUMED metadynamics for the free-energy version -- see MD_protocol).

Run (in the slomd env, once the system is built):
  python scripts/md_run.py --prmtop sys.prmtop --inpcrd sys.inpcrd \
      --temp 300 --prod-ns 200 --replica 1 \
      --posres-mask protein_and_metal_indices.txt \
      --rda 1234 5678 --rda-wall 6.0 --rda-k 30 \
      --out results/md/WT_T300_r1

Notes:
 * --rda i j  are 0-based atom indices for C11(substrate) and O(Fe-OH).
 * HMR + 4 fs enabled with --hmr (otherwise 2 fs).
 * Picks CUDA -> OpenCL -> CPU automatically.
"""
import os, sys, argparse, time
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prmtop", required=True); ap.add_argument("--inpcrd", required=True)
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--prod-ns", type=float, default=200.0)
    ap.add_argument("--equil-ns", type=float, default=5.0)
    ap.add_argument("--replica", type=int, default=1)
    ap.add_argument("--posres-mask", default=None, help="file of 0-based indices to position-restrain during equil")
    ap.add_argument("--rda", type=int, nargs=2, default=None, help="donor acceptor atom indices (0-based)")
    ap.add_argument("--rda-wall", type=float, default=6.0, help="upper flat-bottom wall (A)")
    ap.add_argument("--rda-k", type=float, default=30.0, help="wall force const kcal/mol/A^2")
    ap.add_argument("--hmr", action="store_true")
    ap.add_argument("--out", default="results/md/run")
    a = ap.parse_args()

    from openmm import app, unit, openmm as mm
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    prm = app.AmberPrmtopFile(a.prmtop); crd = app.AmberInpcrdFile(a.inpcrd)
    dt = 0.004 if a.hmr else 0.002
    hmass = 1.5*unit.amu if not a.hmr else 4.0*unit.amu
    system = prm.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer,
                              constraints=app.HBonds, hydrogenMass=hmass)

    # one-sided flat-bottom wall on r_DA (prevents unbinding in plain MD)
    if a.rda:
        k = a.rda_k * unit.kilocalories_per_mole/unit.angstrom**2
        wall = a.rda_wall*unit.angstrom
        f = mm.CustomBondForce("step(r-r0)*0.5*k*(r-r0)^2")
        f.addPerBondParameter("k"); f.addPerBondParameter("r0")
        f.addBond(a.rda[0], a.rda[1], [k.value_in_unit(unit.kilojoule_per_mole/unit.nanometer**2),
                                       wall.value_in_unit(unit.nanometer)])
        system.addForce(f)

    # positional restraints (equilibration only), ramped down later
    posres = None
    if a.posres_mask and os.path.exists(a.posres_mask):
        idx = [int(x) for x in open(a.posres_mask).read().split()]
        posres = mm.CustomExternalForce("0.5*kpr*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        posres.addGlobalParameter("kpr", 10.0*unit.kilocalories_per_mole/unit.angstrom**2)
        for p in ("x0","y0","z0"): posres.addPerParticleParameter(p)
        ref = crd.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        for i in idx: posres.addParticle(i, ref[i])
        system.addForce(posres)

    system.addForce(mm.MonteCarloBarostat(1.0*unit.bar, a.temp*unit.kelvin, 25))
    integ = mm.LangevinMiddleIntegrator(a.temp*unit.kelvin, 1.0/unit.picosecond, dt*unit.picoseconds)

    plat = None
    for name in ("CUDA","OpenCL","CPU"):
        try: plat = mm.Platform.getPlatformByName(name); break
        except Exception: continue
    sim = app.Simulation(prm.topology, system, integ, plat)
    sim.context.setPositions(crd.positions)
    if crd.boxVectors is not None: sim.context.setPeriodicBoxVectors(*crd.boxVectors)

    print(f"[md_run] platform={plat.getName()} dt={dt}ps T={a.temp}K prod={a.prod_ns}ns")
    print("minimizing..."); sim.minimizeEnergy(maxIterations=10000)
    sim.context.setVelocitiesToTemperature(a.temp*unit.kelvin)

    # NVT/NPT equilibration with restraints, then ramp kpr -> 0
    eq_steps = int(a.equil_ns*unit.nanoseconds/(dt*unit.picoseconds))
    if posres is not None:
        for kpr in (10.0, 5.0, 2.0, 0.5, 0.0):
            sim.context.setParameter("kpr", kpr*4.184*100)  # kcal/mol/A^2 -> kJ/mol/nm^2
            sim.step(max(1, eq_steps//5))
    else:
        sim.step(eq_steps)

    # production
    sim.reporters.append(app.DCDReporter(a.out+".dcd", 5000))
    sim.reporters.append(app.StateDataReporter(a.out+".log", 5000, step=True, time=True,
        potentialEnergy=True, temperature=True, density=True, speed=True))
    prod_steps = int(a.prod_ns*unit.nanoseconds/(dt*unit.picoseconds))
    t0=time.time(); sim.step(prod_steps)
    sim.saveState(a.out+".state.xml")
    print(f"[md_run] done {a.prod_ns} ns in {(time.time()-t0)/3600:.2f} h -> {a.out}.dcd")

if __name__ == "__main__":
    main()
