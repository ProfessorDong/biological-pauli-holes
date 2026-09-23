#!/usr/bin/env python3
"""One umbrella-sampling window along the donor-acceptor distance r_DA = |C11...O(Fe-OH)|,
for the SLO gating PMF. Holds r_DA near r0 with a harmonic bias so the substrate stays in
the bound reactive complex (no unbinding), and records the r_DA time series for WHAM/MBAR.

The DAD PMF W(r_DA) and its curvature at the minimum (the gating force constant) are the
quantities the vibronically-nonadiabatic SLO rate theory connects to the KIE.

Seeding: reads an Amber rst7 (coords+box). Writes its final rst7 so the driver can seed the
next (adjacent) window adiabatically -> smooth pulling, no hysteresis / clashes.

Usage: umbrella_window.py --prmtop P --seed S.rst7 --donor i --acceptor j
       --r0 <Angstrom> --k <kcal/mol/A^2> --ns-eq 0.5 --ns-prod 5 --out <prefix>"""
import argparse, numpy as np
from openmm import app, unit, MonteCarloBarostat, LangevinMiddleIntegrator, CustomBondForce, Platform
import openmm as mm

ap=argparse.ArgumentParser()
ap.add_argument("--prmtop",required=True); ap.add_argument("--seed",required=True)
ap.add_argument("--donor",type=int,required=True); ap.add_argument("--acceptor",type=int,required=True)
ap.add_argument("--r0",type=float,required=True)          # Angstrom
ap.add_argument("--k",type=float,default=15.0)            # kcal/mol/A^2
ap.add_argument("--ns-eq",type=float,default=0.5); ap.add_argument("--ns-prod",type=float,default=5.0)
ap.add_argument("--temp",type=float,default=300.0); ap.add_argument("--out",required=True)
ap.add_argument("--rngseed",type=int,default=0)           # replica velocity/integrator seed (0 = nondeterministic)
a=ap.parse_args()

prm=app.AmberPrmtopFile(a.prmtop); seed=app.AmberInpcrdFile(a.seed)
system=prm.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer,
                        constraints=app.HBonds, hydrogenMass=4.0*unit.amu)
system.addForce(MonteCarloBarostat(1.0*unit.bar, a.temp*unit.kelvin, 25))
# harmonic umbrella on r_DA: 0.5*k*(r-r0)^2 ; k in kJ/mol/nm^2, r,r0 in nm
kbias=a.k*4.184*100.0
bias=CustomBondForce("0.5*kb*(r-r0nm)^2"); bias.addPerBondParameter("kb"); bias.addPerBondParameter("r0nm")
bias.addBond(a.donor,a.acceptor,[kbias, a.r0/10.0]); system.addForce(bias)

integ=LangevinMiddleIntegrator(a.temp*unit.kelvin,1.0/unit.picosecond,0.004*unit.picoseconds)
if a.rngseed>0: integ.setRandomNumberSeed(a.rngseed)            # reproducible replica
plat=Platform.getPlatformByName('CUDA'); props={'Precision':'mixed'}
sim=app.Simulation(prm.topology, system, integ, plat, props)
sim.context.setPositions(seed.positions)
if seed.boxVectors is not None: sim.context.setPeriodicBoxVectors(*seed.boxVectors)
sim.minimizeEnergy(maxIterations=1000)                    # relieve the pull to r0
sim.context.setVelocitiesToTemperature(a.temp*unit.kelvin, a.rngseed if a.rngseed>0 else None)
sim.step(int(a.ns_eq*250000))                             # equilibrate (4 fs -> 250k steps/ns)

d,acc=a.donor,a.acceptor
colvar=[]; nprod=int(a.ns_prod*250000); rec=250          # record every 1 ps
for done in range(0,nprod,rec):
    sim.step(min(rec,nprod-done))
    p=sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    colvar.append(np.linalg.norm(p[d]-p[acc]))
colvar=np.array(colvar)
np.savetxt(a.out+"_colvar.dat", colvar, header=f"r0={a.r0} k={a.k} mean={colvar.mean():.3f} std={colvar.std():.3f}")
# save final state as rst7 seed for the next window
# Amber restart: title(1) + "%5d %15.7e"(1) + coord lines (natoms/2 rounded up, 6 fields %12.7f) + box(1)
# Write via a temp file + atomic rename so a partial write / SIGSEGV never leaves a truncated .rst7.
import os as _os, tempfile as _tempfile
st=sim.context.getState(getPositions=True, enforcePeriodicBox=True)
# Extract positions robustly: iterate Vec3 (angstrom units) directly, avoiding an intermittent
# numpy "0-d array" bug from .value_in_unit(...).ravel() on some Blackwell+openmm 8.5.2 combos.
_positions_ang = st.getPositions()   # Quantity[list of Vec3 in Å per rst7 convention]
pos = np.array([[v.x, v.y, v.z] for v in _positions_ang.value_in_unit(unit.angstrom)]).flatten()
bv=st.getPeriodicBoxVectors().value_in_unit(unit.angstrom)
natom=len(pos)//3
tmpfd, tmpname = _tempfile.mkstemp(dir=_os.path.dirname(a.out) or ".", suffix=".rst7.part")
try:
    if pos.ndim != 1 or pos.size < 3:
        raise ValueError(f"bad positions shape {pos.shape} (expected 1-D length {3*natom})")
    with _os.fdopen(tmpfd, "w") as fh:
        fh.write(f"umbrella window r0={a.r0}\n{natom:5d} {0.0:15.7e}\n")
        for i in range(0,len(pos),6):
            fh.write("".join(f"{v:12.7f}" for v in pos[i:i+6])+"\n")
        fh.write(f"{bv[0][0]:12.7f}{bv[1][1]:12.7f}{bv[2][2]:12.7f}{90.0:12.7f}{90.0:12.7f}{90.0:12.7f}\n")
        fh.flush(); _os.fsync(fh.fileno())
    _os.replace(tmpname, a.out+"_final.rst7")
except Exception as _e:
    print(f"  [warn] final rst7 save failed: {_e}", flush=True)
    try: _os.unlink(tmpname)
    except Exception: pass
print(f"  window r0={a.r0:.2f} A: <r_DA>={colvar.mean():.3f} +/- {colvar.std():.3f} A  (n={len(colvar)})", flush=True)
import sys as _sys
_sys.stdout.flush(); _sys.stderr.flush()
# Bypass Python teardown to avoid the intermittent OpenMM+CUDA cleanup SIGSEGV — the
# scientific outputs (colvar.dat + final rst7) are already flushed to disk above.
_os._exit(0)
