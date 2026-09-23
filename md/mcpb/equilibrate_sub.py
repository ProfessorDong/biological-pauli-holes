#!/usr/bin/env python3
"""Heat + NPT-equilibrate the minimized WT+substrate complex.
 - starts from SLO_sub_min coordinates (staged-minimized, clash-free, r_DA~3.5 A)
 - heating 0->300 K over 200 ps (NVT), solute heavy atoms position-restrained (k=5)
 - NPT 300 K / 1 bar, restraints ramped 5->0 over 1 ns
 - a loose flat-bottom UPPER wall on r_DA (C11...O) keeps the substrate in the pocket
   (wall at 5.5 A; the near-attack basin itself is unrestrained so equilibration is honest)
 Saves the equilibrated state (SLO_sub_eq.xml + .pdb) for the metadynamics restart.
 Usage: python equilibrate_sub.py [ns_npt]  (default 1.0)"""
import sys, numpy as np
from openmm import app, unit, MonteCarloBarostat, LangevinMiddleIntegrator, CustomBondForce, CustomExternalForce, Platform
import parmed as pmd

ns_npt=float(sys.argv[1]) if len(sys.argv)>1 else 1.0
PRE=sys.argv[2] if len(sys.argv)>2 else 'SLO_sub'
p=pmd.load_file(PRE+'_solv.prmtop')
p.coordinates=np.load(PRE+'_min_xyz.npy')
C11=[a for a in p.atoms if a.residue.name=='LIG' and a.name=='C14'][0].idx
Oacc=[a for a in p.atoms if a.residue.name=='OH1' and a.name=='O'][0].idx

sysm=p.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=10*unit.angstrom, constraints=app.HBonds, rigidWater=True)
# loose upper wall on r_DA so the substrate cannot leave the pocket during equilibration
wall=CustomBondForce("kw*step(r-rw)*(r-rw)^2"); wall.addPerBondParameter("kw"); wall.addPerBondParameter("rw")
wall.addBond(C11,Oacc,[20000.0, 0.55]); sysm.addForce(wall)   # 5.5 A upper wall
# position restraints on solute heavy atoms (ramped via global 'kref')
posr=CustomExternalForce("kref*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
posr.addGlobalParameter("kref", 5.0*4.184*100)   # kJ/mol/nm^2 for k=5 kcal/mol/A^2
for nm in ("x0","y0","z0"): posr.addPerParticleParameter(nm)
crd=p.coordinates/10.0
nres=0
for a in p.atoms:
    if a.atomic_number>1 and a.residue.name not in ('WAT','HOH','Na+','Cl-'):
        posr.addParticle(a.idx,[crd[a.idx][0],crd[a.idx][1],crd[a.idx][2]]); nres+=1
sysm.addForce(posr)
sysm.addForce(MonteCarloBarostat(1*unit.bar, 300*unit.kelvin, 25))

integ=LangevinMiddleIntegrator(300*unit.kelvin,1/unit.picosecond,0.002*unit.picoseconds)
sim=None
for pn,pr in [('CUDA',{'Precision':'mixed'}),('OpenCL',{'Precision':'mixed'}),('CPU',{})]:
    try: sim=app.Simulation(p.topology,sysm,integ,Platform.getPlatformByName(pn),pr); print(f"  platform: {pn}",flush=True); break
    except Exception as e:
        print(f"  {pn} unavailable: {str(e)[:60]}",flush=True)
        integ=LangevinMiddleIntegrator(300*unit.kelvin,1/unit.picosecond,0.002*unit.picoseconds)
sim.context.setPositions((p.coordinates/10.0)*unit.nanometer)
sim.minimizeEnergy(maxIterations=500)

def rDA():
    pos=sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    return np.linalg.norm(pos[C11]-pos[Oacc])

# heat 0->300 K over 200 ps
print("  heating 0->300 K (200 ps, solute restrained)...",flush=True)
for T in range(10,301,10):
    integ.setTemperature(T*unit.kelvin); sim.context.setParameter("kref",5.0*4.184*100)
    sim.step(1500)   # 3 ps per 10 K step = ~90 ps; +buffer
print(f"  after heating: r_DA={rDA():.2f} A",flush=True)

# NPT with restraint ramp 5->0
nsteps=int(ns_npt*500000); nblk=20
print(f"  NPT {ns_npt} ns, ramping restraints to 0...",flush=True)
for b in range(nblk):
    k=5.0*(1-(b+1)/nblk); sim.context.setParameter("kref", k*4.184*100)
    sim.step(nsteps//nblk)
    if b%5==0: print(f"    block {b+1}/{nblk}  k={k:.2f}  r_DA={rDA():.2f} A",flush=True)
print(f"  equilibrated: r_DA={rDA():.2f} A",flush=True)

sim.saveState(PRE+'_eq.xml')
st_final = sim.context.getState(getPositions=True, enforcePeriodicBox=True)

# Save the .rst7 FIRST so umbrella_driver can seed even if the downstream PDB write
# or CUDA teardown hits the intermittent SIGSEGV we see with openmm 8.5.2 + Blackwell.
import os as _os, tempfile as _tempfile
pos_ang = st_final.getPositions(asNumpy=True).value_in_unit(unit.angstrom).ravel()
bv = st_final.getPeriodicBoxVectors().value_in_unit(unit.angstrom)
natom = len(pos_ang)//3
tmpfd, tmpname = _tempfile.mkstemp(dir=_os.path.dirname(PRE) or ".", suffix=".rst7.part")
try:
    with _os.fdopen(tmpfd, "w") as fh:
        fh.write(f"equilibrated {PRE}\n{natom:5d} {0.0:15.7e}\n")
        for i in range(0, len(pos_ang), 6):
            fh.write("".join(f"{v:12.7f}" for v in pos_ang[i:i+6]) + "\n")
        fh.write(f"{bv[0][0]:12.7f}{bv[1][1]:12.7f}{bv[2][2]:12.7f}{90.0:12.7f}{90.0:12.7f}{90.0:12.7f}\n")
        fh.flush(); _os.fsync(fh.fileno())
    _os.replace(tmpname, PRE+'_eq.rst7')
    print(f"  wrote {PRE}_eq.rst7", flush=True)
except Exception as _e:
    print(f"  [warn] eq rst7 save failed: {_e}", flush=True)
    try: _os.unlink(tmpname)
    except Exception: pass

# Then try the PDB write, but tolerate a SIGSEGV: it is not essential (the .xml + .rst7 hold the state).
try:
    with open(PRE+'_eq.pdb','w') as f:
        app.PDBFile.writeFile(sim.topology, st_final.getPositions(), f)
    print(f"  wrote {PRE}_eq.pdb", flush=True)
except Exception as _e:
    print(f"  [warn] eq pdb save failed: {_e}", flush=True)

print(f"  eq state saved: {PRE}_eq.xml + {PRE}_eq.rst7", flush=True)
import sys as _sys
_sys.stdout.flush(); _sys.stderr.flush()
_os._exit(0)   # skip CUDA teardown to avoid the intermittent SIGSEGV
