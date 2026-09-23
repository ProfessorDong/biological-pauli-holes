#!/usr/bin/env python3
"""
md_analysis.py -- gating-axis analysis for the SLO Part-B campaign.

TWO modes:

(1) AGGREGATE metadynamics replicas (the PRIMARY, publication result):
    P(r_DA) for a well-tempered-metadynamics run is the reweighted exp(-G/kT)
    that md_metad.py writes to each replica's *_fes.csv -- NOT a raw histogram
    (the biased trajectory histogram is meaningless). This mode reads >= 3
    replica *_fes.csv files for one system and reports:
      * tunneling-ready population per replica, and its MEAN +/- SE across replicas
      * the mean P(r_DA) curve with a per-bin SE band
    The mean +/- SE tunneling-ready population is the gating descriptor (with an
    honest error bar) to regress against the measured ln(KIE) ladder.

    python scripts/md_analysis.py --aggregate results/metad/WT/rep*_fes.csv \
        --rtrs 3.1 --out results/metad/WT_summary.csv

(2) UNBIASED trajectory (for a plain MD control or ensemble crowd_Ht only):
    histograms r_DA directly -- valid ONLY for unbiased MD, never for metadynamics.

    python scripts/md_analysis.py TOPOL TRAJ --donor "resname LIG and name C11" \
        --acceptor "name FE and around 2.6 (name O* and resname HOH WAT)" [--rtrs 3.1]
"""
import os, glob, argparse
import numpy as np
try:
    from numpy import trapezoid as _trapz   # numpy >= 2.0
except ImportError:
    from numpy import trapz as _trapz        # numpy < 2.0

def _trs(r, P, rtrs):
    """tunneling-ready population = integral of P(r) up to rtrs (Angstrom)."""
    P = P / _trapz(P, r)
    m = r <= rtrs
    return float(_trapz(P[m], r[m]))

def aggregate(fes_files, rtrs, out):
    files = []
    for pat in fes_files:
        files += sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
    files = [f for f in files if os.path.exists(f)]
    if len(files) < 1:
        print("[md_analysis] no *_fes.csv files matched."); return
    Ps, grids, trs = [], [], []
    for f in files:
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        r, P = d[:, 0], d[:, 2]
        trs.append(_trs(r, P, rtrs)); Ps.append(P / _trapz(P, r)); grids.append(r)
    r = grids[0]
    if not all(np.allclose(g, r) for g in grids):
        print("[!] replicas have different r grids -- rerun md_metad.py with matching rmin/rmax/grid.")
        return
    Ps = np.array(Ps); trs = np.array(trs); n = len(trs)
    se = lambda x: (x.std(ddof=1)/np.sqrt(n)) if n > 1 else np.nan
    trs_mean, trs_se = trs.mean(), se(trs)
    P_mean = Ps.mean(0)
    P_se = (Ps.std(0, ddof=1)/np.sqrt(n)) if n > 1 else np.zeros_like(Ps[0])
    print(f"[aggregate] {n} replicas: {', '.join(os.path.basename(f) for f in files)}")
    for f, t in zip(files, trs):
        print(f"    {os.path.basename(f):24s} tunneling-ready(<= {rtrs} A) = {t:.3f}")
    print(f"  MEAN tunneling-ready = {trs_mean:.3f} +/- {trs_se:.3f} (SE, n={n})"
          + ("   [<3 replicas: SE weak, run more]" if n < 3 else ""))
    out = out or "results/metad/summary.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savetxt(out, np.c_[r, P_mean, P_se], delimiter=",",
               header="r_DA_A,P_mean,P_SE", comments="")
    with open(os.path.splitext(out)[0] + "_trs.csv", "w") as fh:
        fh.write("n_replicas,tunneling_ready_mean,tunneling_ready_SE,rtrs_A\n")
        fh.write(f"{n},{trs_mean:.5f},{trs_se:.5f},{rtrs}\n")
    print(f"  wrote {out} (P mean+/-SE) and *_trs.csv (descriptor for the ln(KIE) regression).")

def trajectory(topol, traj, donor, acceptor, rtrs, out):
    import MDAnalysis as mda
    u = mda.Universe(topol, traj)
    D = u.select_atoms(donor); A = u.select_atoms(acceptor)
    if not len(D) or not len(A):
        print(f"[!] donor ({len(D)}) / acceptor ({len(A)}) selection empty -- adjust selections.")
        return
    r = np.array([np.linalg.norm(D.center_of_geometry() - A.center_of_geometry())
                  for _ in u.trajectory])
    hist, edges = np.histogram(r, bins=60, range=(2.0, 8.0), density=True)
    centers = 0.5*(edges[:-1]+edges[1:])
    print(f"[UNBIASED traj] frames={len(r)}  <r_DA>={r.mean():.2f} A  sd={r.std():.2f}  "
          f"min={r.min():.2f}  tunneling-ready(<= {rtrs} A)={float(np.mean(r <= rtrs)):.3f}")
    out = out or f"results/Pr_{os.path.splitext(os.path.basename(traj))[0]}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savetxt(out, np.c_[centers, hist], delimiter=",", header="r_DA_A,P(r)", comments="")
    print(f"  wrote {out}  (valid only for UNBIASED MD -- for metadynamics use --aggregate).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topol", nargs="?"); ap.add_argument("traj", nargs="?")
    ap.add_argument("--aggregate", nargs="+", default=None,
                    help="replica *_fes.csv files/globs for one system (metadynamics)")
    ap.add_argument("--donor", default="resname LIG and name C11")
    ap.add_argument("--acceptor", default="name O5 OW OH2 and around 2.6 name FE")
    ap.add_argument("--rtrs", type=float, default=3.1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.aggregate:
        aggregate(a.aggregate, a.rtrs, a.out)
    elif a.topol and a.traj and os.path.exists(a.topol) and os.path.exists(a.traj):
        trajectory(a.topol, a.traj, a.donor, a.acceptor, a.rtrs, a.out)
    else:
        print(__doc__); print("[md_analysis] nothing to do -- pass --aggregate <fes csvs> or TOPOL TRAJ.")

if __name__ == "__main__":
    main()
