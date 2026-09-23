#!/usr/bin/env python3
"""One ring-polymer molecular-dynamics (RPMD) near-attack window for the SLO transferring
hydrogen (or deuterium) at fixed donor-acceptor distance r_DA.

Motivation. The exact algebraic decomposition of the vibronic PCET rate is
  ln(k_H/k_D) = DeltaS2^(H-D)/kB + ln(g_H/g_D),
where DeltaS2 is the isotope-specific Renyi-2 reactive-flux entropy (path multiplicity)
and g the per-path potency. Classical MD is mass-independent by construction (equilibrium
P(q) prop exp(-beta U)) and cannot distinguish these. RPMD samples the quantum imaginary-time
path of the transferring particle, so its reactive-basin marginal density in (r_HO, theta_CHO)
is genuinely isotope-specific through the ring-polymer bead spring constant m*(N/(beta hbar))^2.

Strategy. Sample the transferring particle's quantum distribution while holding r_DA harmonically
near a near-attack target r0 in {3.10, 3.35, 3.60 Ang}. Use N beads (default 8), 0.5 fs timestep,
NVT at 300 K with the OpenMM RPMDIntegrator on CUDA. Record the transferring-particle bead
positions (all N per frame) so post-hoc analysis can compute the quantum marginal density and
its differential entropy.

RPMD constraints: no HydrogenMassRepartitioning (defeats the point); no barostat (NVT only, but
seeded from the equilibrated NPT rst7 so the box is realistic); constrain only the non-transferring
X-H bonds (HBonds) to keep the timestep at 0.5 fs; leave the transferring particle unconstrained.

Usage: pimd_window.py --prmtop P --seed S.rst7 --donor i --acceptor j --xferH k
       --mass 1.008 --nbeads 8 --r0 3.35 --k 15 --ps-eq 5 --ps-prod 50 --out <prefix>
"""
import argparse, numpy as np, time
from openmm import app, unit, RPMDIntegrator, CustomBondForce, Platform
import openmm as mm

ap = argparse.ArgumentParser()
ap.add_argument("--prmtop", required=True)
ap.add_argument("--seed", required=True)
ap.add_argument("--donor", type=int, required=True, help="donor heavy atom index (C11 substrate)")
ap.add_argument("--acceptor", type=int, required=True, help="acceptor heavy atom index (O of Fe-OH)")
ap.add_argument("--xferH", type=int, required=True, help="transferring H atom index on donor")
ap.add_argument("--mass", type=float, default=1.008, help="mass of transferring particle in amu (1.008=H, 2.014=D, 3.016=T)")
ap.add_argument("--nbeads", type=int, default=8, help="ring-polymer bead count")
ap.add_argument("--r0", type=float, required=True, help="target r_DA in Angstrom")
ap.add_argument("--k", type=float, default=15.0, help="harmonic bias on r_DA in kcal/mol/A^2")
ap.add_argument("--ps-eq", type=float, default=5.0, help="ps of RPMD equilibration")
ap.add_argument("--ps-prod", type=float, default=50.0, help="ps of RPMD production sampling")
ap.add_argument("--dt-fs", type=float, default=0.5, help="RPMD timestep in fs")
ap.add_argument("--temp", type=float, default=300.0)
ap.add_argument("--record-every-fs", type=float, default=50.0, help="record cadence in fs")
ap.add_argument("--out", required=True)
ap.add_argument("--rngseed", type=int, default=0)
a = ap.parse_args()

t0 = time.time()

prm = app.AmberPrmtopFile(a.prmtop)
seed = app.AmberInpcrdFile(a.seed)

# System: PME, NO constraints (OpenMM's RPMDIntegrator does not support any constraints),
# no HMR, NVT (no barostat). The lack of HBonds constraints forces a short timestep to
# resolve the fastest X-H stretch (~10 fs period), typically dt <= 0.25 fs.
system = prm.createSystem(nonbondedMethod=app.PME,
                          nonbondedCutoff=1.0 * unit.nanometer,
                          constraints=None,
                          rigidWater=False)

# Set the transferring particle mass (H, D, or T).
system.setParticleMass(a.xferH, a.mass * unit.amu)

# Harmonic umbrella on r_DA in nm-units (0.5*k*(r-r0)^2)
kbias = a.k * 4.184 * 100.0  # kcal/mol/A^2 -> kJ/mol/nm^2
bias = CustomBondForce("0.5*kb*(r-r0nm)^2")
bias.addPerBondParameter("kb"); bias.addPerBondParameter("r0nm")
bias.addBond(a.donor, a.acceptor, [kbias, a.r0 / 10.0])
system.addForce(bias)

# RPMD integrator
dt = a.dt_fs * unit.femtoseconds
integ = RPMDIntegrator(a.nbeads, a.temp * unit.kelvin, 1.0 / unit.picosecond, dt)
if a.rngseed > 0:
    integ.setRandomNumberSeed(a.rngseed)

plat = Platform.getPlatformByName('CUDA')
props = {'Precision': 'mixed'}
sim = app.Simulation(prm.topology, system, integ, plat, props)

# All copies start from the classical seed positions; the ring polymer expands via thermal
# fluctuations during equilibration.  Seed the box first, positions second (order matters
# for periodic systems).
if seed.boxVectors is not None:
    sim.context.setPeriodicBoxVectors(*seed.boxVectors)
for copy in range(a.nbeads):
    integ.setPositions(copy, seed.positions)

# Do NOT call sim.minimizeEnergy() with an RPMDIntegrator: minimize uses the integrator's
# position accessors which are defined per-copy for RPMD but the Simulation wrapper's minimizer
# is not RPMD-aware and can leave inconsistent states. Instead, do a short low-friction warm-up.
# Waters and X-H bonds are unconstrained (RPMD requires this) so we rely on the seed already
# being NPT-equilibrated and use a gentle warm-up.
steps_eq = int(a.ps_eq * 1000.0 / a.dt_fs)
sim.step(steps_eq)
t_eq = time.time() - t0
print(f"  RPMD equilibrated {a.ps_eq:.1f} ps ({steps_eq} steps, N={a.nbeads} beads, m={a.mass:.3f}) in {t_eq:.1f} s", flush=True)

# Production: record all-bead transferring-H positions, plus r_DA (centroid distance).
steps_prod = int(a.ps_prod * 1000.0 / a.dt_fs)
steps_record = int(a.record_every_fs / a.dt_fs)
n_records = steps_prod // steps_record

xferH_positions = np.zeros((n_records, a.nbeads, 3), dtype=np.float32)  # A
donor_pos = np.zeros((n_records, 3), dtype=np.float32)
acceptor_pos = np.zeros((n_records, 3), dtype=np.float32)
r_DA_series = np.zeros(n_records, dtype=np.float32)

for i in range(n_records):
    sim.step(steps_record)
    # Read per-copy positions via asNumpy=True (fast, robust). We only need the transferring
    # H (all beads), the donor centroid and the acceptor centroid, so avoid materialising a
    # (natoms, 3) array 8 times per frame.
    donor_sum = np.zeros(3); acceptor_sum = np.zeros(3)
    for copy in range(a.nbeads):
        st = integ.getState(copy, getPositions=True, enforcePeriodicBox=True)
        p = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        xferH_positions[i, copy] = p[a.xferH]
        donor_sum += p[a.donor]
        acceptor_sum += p[a.acceptor]
    donor_pos[i] = donor_sum / a.nbeads
    acceptor_pos[i] = acceptor_sum / a.nbeads
    r_DA_series[i] = np.linalg.norm(donor_pos[i] - acceptor_pos[i])

t_prod = time.time() - t0 - t_eq
print(f"  RPMD production {a.ps_prod:.1f} ps ({steps_prod} steps, {n_records} records) in {t_prod:.1f} s", flush=True)

# Save the per-frame all-bead transferring-particle positions along with the donor/acceptor
# centroids, so downstream analysis can compute the quantum reactive-basin marginal density
# in the (r_HO, theta_CHO) plane rigorously.
np.savez_compressed(a.out + "_pimd.npz",
                    xferH_positions=xferH_positions,
                    donor_pos=donor_pos,
                    acceptor_pos=acceptor_pos,
                    r_DA=r_DA_series,
                    nbeads=a.nbeads,
                    mass_amu=a.mass,
                    r0=a.r0,
                    kbias=a.k,
                    dt_fs=a.dt_fs,
                    ps_prod=a.ps_prod,
                    temp_K=a.temp)

# Compact console summary
r_HO = np.linalg.norm(xferH_positions.mean(axis=1) - acceptor_pos, axis=1)   # centroid HO
sigma_xferH_beads_A = np.linalg.norm(xferH_positions - xferH_positions.mean(axis=1, keepdims=True), axis=2).std()
print(f"  m={a.mass:.3f} N={a.nbeads} r0={a.r0:.2f} r_DA=<{r_DA_series.mean():.3f}> "
      f"<r_HO_centroid>={r_HO.mean():.3f}  sigma_bead(HD)={sigma_xferH_beads_A:.4f} A", flush=True)

import sys, os as _os
sys.stdout.flush(); sys.stderr.flush()
_os._exit(0)
