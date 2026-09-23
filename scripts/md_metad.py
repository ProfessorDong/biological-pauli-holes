#!/usr/bin/env python3
"""
md_metad.py -- OpenMM well-tempered metadynamics on the donor-acceptor distance
r_DA = |C11(substrate) - O(Fe-OH)| for the SLO gating-axis campaign (Part B).

Publication-grade protocol:
  * INDEPENDENT replicas (--replica/--seed) per system  -> mean +/- SE error bars
    on P(r_DA) and the "tunneling-ready" population (aggregate in md_analysis.py).
  * per-block free-energy CONVERGENCE logging + a stationarity metric, so we can
    show G(r_DA) has stopped drifting rather than trusting a single fixed-time curve.
  * well-tempered metadynamics via OpenMM's NATIVE Metadynamics (no PLUMED),
    CUDA on the RTX 4060.

Converges G(r_DA) -> P(r_DA) ~ exp(-G/kT) far faster than brute-force MD, so the
catalytically relevant short-r_DA population is sampled in ~150-300 ns/replica.

Run ONE replica (slomd env, after the system is built + equilibrated):
  python scripts/md_metad.py --prmtop sys.prmtop --inpcrd eq.rst7 \
      --donor 1234 --acceptor 5678 --ns 200 --replica 1 --out results/metad/WT/rep1
Run >= 3 replicas per system (--replica 1,2,3 -> different seeds), then aggregate
P(r_DA) mean +/- SE across replicas with md_analysis.py.
"""
import os, argparse, numpy as np
try:
    from numpy import trapezoid as _trapz   # numpy >= 2.0
except ImportError:
    from numpy import trapz as _trapz        # numpy < 2.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prmtop", required=True); ap.add_argument("--inpcrd", required=True)
    ap.add_argument("--donor", type=int, required=True)     # 0-based C11 index
    ap.add_argument("--acceptor", type=int, required=True)  # 0-based O(Fe-OH) index
    ap.add_argument("--rmin", type=float, default=0.25)     # nm
    ap.add_argument("--rmax", type=float, default=0.70)     # nm
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--ns", type=float, default=200.0)
    ap.add_argument("--biasfactor", type=float, default=8.0)
    ap.add_argument("--height", type=float, default=1.0)    # kJ/mol
    ap.add_argument("--sigma", type=float, default=0.02)    # nm
    ap.add_argument("--pace", type=int, default=1000)       # steps between gaussians
    ap.add_argument("--trs-cut", type=float, default=3.1)   # tunneling-ready cutoff (Angstrom)
    ap.add_argument("--replica", type=int, default=1)       # replica id (-> seed if --seed 0)
    ap.add_argument("--seed", type=int, default=0)          # 0 -> use replica id as seed
    ap.add_argument("--hmr", action="store_true")
    ap.add_argument("--out", default="results/metad/run")
    a = ap.parse_args()

    from openmm import app, unit, openmm as mm
    seed = a.seed if a.seed > 0 else a.replica
    outdir = os.path.dirname(a.out) or "."
    biasdir = a.out + "_bias"           # UNIQUE per replica -> genuinely independent runs
    os.makedirs(biasdir, exist_ok=True); os.makedirs(outdir, exist_ok=True)

    prm = app.AmberPrmtopFile(a.prmtop)
    rst = app.AmberInpcrdFile(a.inpcrd)
    dt = 0.004 if a.hmr else 0.002
    hmass = 4.0*unit.amu if a.hmr else 1.5*unit.amu
    system = prm.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer,
                              constraints=app.HBonds, hydrogenMass=hmass)
    system.addForce(mm.MonteCarloBarostat(1.0*unit.bar, a.temp*unit.kelvin, 25))

    # CV: distance C11...O  (CustomBondForce energy == r in nm)
    cv = mm.CustomBondForce("r"); cv.addBond(a.donor, a.acceptor, [])
    bias = app.metadynamics.BiasVariable(cv, a.rmin, a.rmax,
                                         biasWidth=a.sigma, periodic=False, gridWidth=200)
    meta = app.metadynamics.Metadynamics(system, [bias], a.temp*unit.kelvin,
                                         biasFactor=a.biasfactor,
                                         height=a.height*unit.kilojoules_per_mole,
                                         frequency=a.pace, saveFrequency=a.pace, biasDir=biasdir)
    integ = mm.LangevinMiddleIntegrator(a.temp*unit.kelvin, 1.0/unit.picosecond, dt*unit.picoseconds)
    integ.setRandomNumberSeed(seed)     # reproducible per replica
    plat, props = _platform(mm)
    sim = app.Simulation(prm.topology, system, integ, plat, props)
    sim.context.setPositions(rst.positions)
    if rst.boxVectors is not None: sim.context.setPeriodicBoxVectors(*rst.boxVectors)
    sim.minimizeEnergy(maxIterations=2000)
    sim.context.setVelocitiesToTemperature(a.temp*unit.kelvin, seed)
    sim.reporters.append(app.StateDataReporter(a.out+".log", 5000, step=True, time=True,
        temperature=True, density=True, speed=True))

    steps = int(a.ns*unit.nanoseconds/(dt*unit.picoseconds))
    block = 50000
    kT = (unit.MOLAR_GAS_CONSTANT_R*a.temp*unit.kelvin).value_in_unit(unit.kilojoule_per_mole)
    rgrid = np.linspace(a.rmin, a.rmax, 200)*10.0   # Angstrom
    conv = []; prevP = None
    for done in range(0, steps, block):
        meta.step(sim, min(block, steps-done))
        fe = meta.getFreeEnergy().value_in_unit(unit.kilojoule_per_mole)  # G(r), kJ/mol
        fe = fe - fe.min()
        P = np.exp(-fe/kT); P /= _trapz(P, rgrid)
        trs = float(_trapz(P[rgrid <= a.trs_cut], rgrid[rgrid <= a.trs_cut]))  # tunneling-ready pop
        drmsd = float(np.sqrt(np.mean((P-prevP)**2))) if prevP is not None else np.nan
        prevP = P.copy()
        ns_done = (done+block)*dt/1000.0
        conv.append((ns_done, trs, drmsd))
        np.savetxt(a.out+"_fes.csv", np.c_[rgrid, fe, P], delimiter=",",
                   header="r_DA_A,G_kJ/mol,P(r)", comments="")
        np.savetxt(a.out+"_conv.csv", np.array(conv), delimiter=",",
                   header="ns,tunneling_ready,P_rmsd_vs_prev_block", comments="")
        print(f"  rep{a.replica} {ns_done:.1f}/{a.ns} ns  TRS(<={a.trs_cut}A)={trs:.3f}  dP_rmsd={drmsd:.2e}")
    print(f"[md_metad] rep{a.replica} done -> {a.out}_fes.csv (+_conv.csv). "
          f"Converged when P_rmsd_vs_prev_block is small and flat. "
          f"Run >=3 replicas, then aggregate mean+/-SE in md_analysis.py.")

def _platform(mm):
    for n in ("CUDA", "OpenCL", "CPU"):
        try:
            p = mm.Platform.getPlatformByName(n)
            return p, ({"Precision": "mixed"} if n in ("CUDA", "OpenCL") else {})
        except Exception:
            continue
    raise RuntimeError("no OpenMM platform available")

if __name__ == "__main__":
    main()
