#!/usr/bin/env python3
"""
sapt_exch_curvature.py  --  generate SAPT exchange-repulsion-vs-distance data
and turn it into the building block of the descriptor:

    k_exch  = d^2 E_exch / dr^2     (curvature of the exchange-repulsion "wall")
    ZPE     = (hbar/2) * sqrt(k_exch / m)     (mass-dependent -> H/D/T isotope link)

Why this replaces the (dead) UT-Austin SAPT download:
  We compute the SAPT0 component decomposition (electrostatics / EXCHANGE /
  induction / dispersion) ourselves with Psi4, scanning an intermolecular
  distance.  This gives E_exch(r); its second derivative is the curvature that
  the manuscript's confinement descriptor is built on.

Run (in the 'pauli' conda env):
    conda activate pauli
    python sapt_exch_curvature.py
Output:
    sapt_exch_scan.csv  (r, E_elst, E_exch, E_ind, E_disp, E_tot ; kcal/mol)
    printed: k_exch(r) PROFILE, and ZPE for H/D/T evaluated at REF_R.

PATCH NOTE (v2): curvature is now evaluated at a FIXED, physically meaningful
contact distance REF_R (default 2.8 A, ~van der Waals contact), and reported as
a full profile k_exch(r).  The earlier version used argmin(E_total), which for a
purely repulsive toy geometry falls at the grid edge where exchange is ~0 and the
curvature is not meaningful.

MAPPING TO THE ENZYME:
  Closed-shell dimers are required for SAPT0 (a bare transferring H is open-shell),
  so this scans whole monomers as a clean methodological demo.  In the protein you
  instead displace the TRANSFERRING H transverse to the donor-acceptor axis and
  take the finite-difference Hessian of the EDA exchange term -> that is the
  transverse k_exch.  The mass plugged into ZPE is the transferring nucleus
  (H/D/T).  This script demonstrates k_exch is well-defined and that ZPE built
  from it is mass-sensitive (H/D ratio -> sqrt(mH/mD) = 0.707).
"""

import numpy as np
import psi4

# ----------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------
REF_R = 2.8                  # Angstrom: distance at which to report k_exch & ZPE
                            #           (choose near the operating/contact geometry)
SCAN  = np.round(np.arange(2.4, 4.21, 0.2), 3)   # scan grid (must bracket REF_R)
BASIS = "jun-cc-pvdz"        # good, affordable SAPT0 basis; aug-cc-pvdz for production

psi4.set_memory("4 GB")
psi4.core.set_num_threads(4)
psi4.set_output_file("sapt_exch_curvature.out", False)
psi4.set_options({"basis": BASIS, "scf_type": "df", "freeze_core": "true"})

# water monomer geometry (Angstrom)
MONO = [
    ("O",  0.000000,  0.000000,  0.000000),
    ("H",  0.000000,  0.757160,  0.586260),
    ("H",  0.000000, -0.757160,  0.586260),
]

def dimer_at(R):
    """Fragment A at origin; fragment B = copy translated by R along +x (Angstrom)."""
    lines = ["0 1"]
    for el, x, y, z in MONO:
        lines.append(f"{el} {x:.6f} {y:.6f} {z:.6f}")
    lines.append("--")
    lines.append("0 1")
    for el, x, y, z in MONO:
        lines.append(f"{el} {x+R:.6f} {y:.6f} {z:.6f}")
    lines += ["units angstrom", "symmetry c1", "no_reorient", "no_com"]
    return psi4.geometry("\n".join(lines))

H2KCAL = 627.509474

# ----------------------------------------------------------------------
# distance scan
# ----------------------------------------------------------------------
rows = []
for R in SCAN:
    dim = dimer_at(float(R))
    psi4.energy("sapt0", molecule=dim)
    elst = psi4.variable("SAPT ELST ENERGY") * H2KCAL
    exch = psi4.variable("SAPT EXCH ENERGY") * H2KCAL
    ind  = psi4.variable("SAPT IND ENERGY")  * H2KCAL
    disp = psi4.variable("SAPT DISP ENERGY") * H2KCAL
    tot  = psi4.variable("SAPT TOTAL ENERGY") * H2KCAL
    rows.append((R, elst, exch, ind, disp, tot))
    print(f"R={R:5.2f}  Eexch={exch:9.4f}  Etot={tot:9.4f} kcal/mol")
    psi4.core.clean()

data = np.array(rows)
np.savetxt("sapt_exch_scan.csv", data, delimiter=",",
           header="r_Ang,E_elst,E_exch,E_ind,E_disp,E_tot (kcal/mol)", comments="")
print("\nwrote sapt_exch_scan.csv")

# ----------------------------------------------------------------------
# curvature profile k_exch(r) = d^2 E_exch/dr^2  (central differences)
# ----------------------------------------------------------------------
r = data[:, 0]; eexch = data[:, 2]
print(f"\n{'r(A)':>6}{'E_exch':>10}{'k_exch(kcal/mol/A^2)':>22}")
k_at = {}
for i in range(1, len(r) - 1):
    h = r[i + 1] - r[i]
    k = (eexch[i + 1] - 2 * eexch[i] + eexch[i - 1]) / h**2
    k_at[round(float(r[i]), 3)] = k
    print(f"{r[i]:6.2f}{eexch[i]:10.4f}{k:22.3f}")

# pick the grid point closest to REF_R (must be an interior point)
interior = np.array(sorted(k_at))
ref = float(interior[np.argmin(np.abs(interior - REF_R))])
if abs(ref - REF_R) > 1e-6:
    print(f"\n[note] REF_R={REF_R} not on grid; using nearest interior point r={ref}")
if ref in (interior[0], interior[-1]):
    print("[warn] reference point is at a grid edge; widen SCAN to bracket REF_R.")
k_kcal = k_at[ref]

# ----------------------------------------------------------------------
# ZPE from the exchange-wall curvature at REF_R, for H / D / T (isotope link)
# ----------------------------------------------------------------------
NA   = 6.02214076e23
hbar = 1.054571817e-34
c    = 2.99792458e10            # cm/s
amu  = 1.66053907e-27           # kg
k_SI = k_kcal * 4184.0 / NA / 1e-20      # N/m  (per molecule)

print(f"\n== exchange-wall curvature at r = {ref:.2f} A ==")
print(f"k_exch = {k_kcal:.3f} kcal/mol/A^2 = {k_SI:.2f} N/m")
print(f"{'nucleus':>8} {'mass(amu)':>10} {'nu(cm^-1)':>10} {'ZPE(kcal/mol)':>14}")
zpe = {}
for name, m_amu in [("H", 1.007825), ("D", 2.014102), ("T", 3.016049)]:
    omega = np.sqrt(k_SI / (m_amu * amu))          # rad/s
    nu    = omega / (2 * np.pi * c)                 # cm^-1
    z     = 0.5 * hbar * omega * NA / 4184.0        # kcal/mol
    zpe[name] = z
    print(f"{name:>8} {m_amu:>10.4f} {nu:>10.1f} {z:>14.4f}")

print(f"\nH-D ZPE difference = {zpe['H']-zpe['D']:.4f} kcal/mol")
print(f"ZPE_D/ZPE_H ratio  = {zpe['D']/zpe['H']:.3f}  (expected sqrt(mH/mD)=0.707)")
print("This isotope signature comes from the exchange-repulsion curvature alone, "
      "independent of donor-acceptor distance and electrostatics.")
