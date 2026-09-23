#!/usr/bin/env python3
"""Staged minimization of the WT+substrate complex to relax the template-docking clashes.
 Stage 1: protein heavy atoms restrained (k=10) -> waters/H/substrate periphery relax.
 Stage 2: release protein, keep a flat-bottom r_DA restraint on C11(LIG.C14)...O(OH1.O)
          so the near-attack geometry is preserved while clashes anneal out.
Reports energy, r_DA, and the worst substrate-protein contact before/after."""
import sys, numpy as np
from openmm import app, unit, LangevinMiddleIntegrator, CustomBondForce, CustomExternalForce, Platform
import parmed as pmd

PRE=sys.argv[1] if len(sys.argv)>1 else 'SLO_sub'
p=pmd.load_file(PRE+'_solv.prmtop',PRE+'_solv.inpcrd')
C11=[a for a in p.atoms if a.residue.name=='LIG' and a.name=='C14'][0].idx
Oacc=[a for a in p.atoms if a.residue.name=='OH1' and a.name=='O'][0].idx
lig_idx=[a.idx for a in p.atoms if a.residue.name=='LIG']
print(f"  C11(LIG.C14) idx={C11}  Oacc(OH1.O) idx={Oacc}  LIG atoms={len(lig_idx)}")

sysm=p.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=10*unit.angstrom, constraints=app.HBonds)

# flat-bottom near-attack restraint on r_DA (harmonic outside 3.0-3.6 A)
rest=CustomBondForce("k*step(r-hi)*(r-hi)^2 + k*step(lo-r)*(lo-r)^2")
rest.addPerBondParameter("k"); rest.addPerBondParameter("lo"); rest.addPerBondParameter("hi")
rest.addBond(C11,Oacc,[50000.0, 0.30, 0.35])   # kJ/mol/nm^2 (~120 kcal/mol/A^2); keep 3.0-3.5 A
sysm.addForce(rest)

# positional restraint on protein heavy atoms (stage 1)
posr=CustomExternalForce("kp*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
posr.addPerParticleParameter("kp"); posr.addPerParticleParameter("x0"); posr.addPerParticleParameter("y0"); posr.addPerParticleParameter("z0")
crd=p.get_coordinates(0)/10.0  # nm
prot_res=set(range(len(p.residues)))
for a in p.atoms:
    if a.atomic_number>1 and a.residue.name not in ('WAT','HOH','LIG','Na+','Cl-'):
        posr.addParticle(a.idx,[10.0*4.184*100, crd[a.idx][0],crd[a.idx][1],crd[a.idx][2]])  # kJ/nm^2
posr_idx=sysm.addForce(posr)

integ=LangevinMiddleIntegrator(300*unit.kelvin,1/unit.picosecond,0.002*unit.picoseconds)
sim=None
for pn,pr in [('CUDA',{'Precision':'mixed'}),('OpenCL',{'Precision':'mixed'}),('CPU',{})]:
    try:
        sim=app.Simulation(p.topology, sysm, integ, Platform.getPlatformByName(pn), pr)
        print(f"  platform: {pn}"); break
    except Exception as e:
        print(f"  {pn} unavailable ({str(e)[:50]})"); integ=LangevinMiddleIntegrator(300*unit.kelvin,1/unit.picosecond,0.002*unit.picoseconds)
assert sim is not None
sim.context.setPositions(p.positions)

def rDA():
    pos=sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    return np.linalg.norm(pos[C11]-pos[Oacc]), pos
def worst_contact(pos):
    from scipy.spatial import cKDTree
    prot=np.array([pos[a.idx] for a in p.atoms if a.atomic_number>1 and a.residue.name not in ('WAT','HOH','LIG','Na+','Cl-')])
    lig=np.array([pos[i] for i in lig_idx])
    return cKDTree(prot).query(lig)[0].min()

d0,pos0=rDA(); print(f"  BEFORE: r_DA={d0:.2f} A, worst LIG-protein contact={worst_contact(pos0):.2f} A")
e0=sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
print(f"  initial PE={e0:.3e} kcal/mol  -> stage 1 minimize (protein restrained)...")
sim.minimizeEnergy(maxIterations=3000)
d1,pos1=rDA(); print(f"  after stage1: r_DA={d1:.2f} A, worst contact={worst_contact(pos1):.2f} A")

# stage 2: release protein positional restraint
sim.context.getState(getPositions=True)
for i in range(posr.getNumParticles()):
    idx,params=posr.getParticleParameters(i); params=list(params); params[0]=0.0; posr.setParticleParameters(i,idx,params)
posr.updateParametersInContext(sim.context)
sim.minimizeEnergy(maxIterations=5000)
d2,pos2=rDA(); ef=sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
print(f"  after stage2: r_DA={d2:.2f} A, worst contact={worst_contact(pos2):.2f} A, PE={ef:.3e} kcal/mol")

state=sim.context.getState(getPositions=True)
# Save the numpy coords FIRST (equilibrate needs this); the PDB is decorative.
np.save(PRE+'_min_xyz.npy', state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
try:
    with open(PRE+'_min.pdb','w') as f: app.PDBFile.writeFile(sim.topology, state.getPositions(), f)
except Exception as _e:
    print(f"  [warn] min pdb save failed: {_e}", flush=True)
print(f"  wrote {PRE}_min_xyz.npy + {PRE}_min.pdb", flush=True)
import os as _os, sys as _sys
_sys.stdout.flush(); _sys.stderr.flush()
_os._exit(0)   # skip CUDA teardown to avoid the intermittent SIGSEGV
