#!/usr/bin/env python3
"""
dock_substrate.py -- build a SUBSTRATE-BOUND near-attack model of SLO by
constrained placement of the reactive substrate moiety into 1F8N.

METHOD (defensible for a buried, geometrically-defined reaction; this is what
QM/MM studies of SLO do, NOT blind flexible docking):
  * Reactive model = (Z,Z)-1,4-pentadiene fragment (C1..C5); the central
    bis-allylic carbon C3 is the analog of linoleic-acid C11 (the pro-S H donor).
    (A 5-carbon truncation is adequate for a LOCAL descriptor around the
    transferring H; the rest of the C18 chain is >6 A away. Note the truncation.)
  * Acceptor = Fe(III)-OH oxygen (HOH841 O, 2.11 A from Fe).
  * Constraints (from Tresadern et al.; Meyer & Klinman 2005):
        C3...O_acc = 3.0 A  (paper: C-O ~ 2.8-3.3 A)
        transferring H placed ON the C3->O_acc line  => near-linear C-H...O (180 deg),
        H...O = 3.0 - 1.09 = 1.91 A.
  * Chain orientation: the three non-transferring sp3 bonds of C3 lie on the
    109.47-deg tetrahedral cone away from O_acc (i.e. the chain splays into the
    hydrophobic cavity, perpendicular-ish to the C-H...O axis -- realistic).
    The azimuth about the C-H...O axis is chosen to MINIMIZE protein clashes.
  * Ordered waters are treated as displaceable (excluded from clash scoring).

Outputs:
  pauli_data/slo/substrate_model.pdb   (fragment only, HETATM resname LIG)
  pauli_data/slo/1F8N_sub.pdb          (protein + fragment, for visualization)
  results/_transfer_site.npz           (C3, transferring-H, O_acc positions)
and prints the near-attack geometry + a transfer-site descriptor table for
WT and the I553 truncation models.

Run:  python scripts/dock_substrate.py
"""
import os, numpy as np, MDAnalysis as mda

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLO  = os.path.join(ROOT, "pauli_data", "slo")
RES  = os.path.join(ROOT, "results"); os.makedirs(RES, exist_ok=True)

CO   = 3.0      # C3...O_acc target (A)
CH   = 1.09     # C-H (A)
CC1  = 1.50     # C3-C2 single (A)
CC2  = 1.34     # C2=C1 double (A)
TET  = np.deg2rad(109.47)

def altloc_heavy(u, exclude_water=True):
    al = getattr(u.atoms, "altLocs", np.array([""]*len(u.atoms)))
    ag = u.atoms[np.isin(al, ["", " ", "A"])]
    sel = "not name H* and not name FE FE2 FE3"
    if exclude_water: sel += " and not resname HOH WAT"
    return ag.select_atoms(sel)

def perp_frame(n):
    n = n/np.linalg.norm(n)
    a = np.array([1.,0.,0.]) if abs(n[0])<0.9 else np.array([0.,1.,0.])
    u = np.cross(n, a); u/=np.linalg.norm(u); v = np.cross(n, u)
    return u, v

def build_fragment(C3, hhat, u, v, az):
    """hhat = C3->O_acc unit (transfer H direction). Three sp3 bonds on the
    109.47-deg cone about (-hhat), at azimuth az (+120,+240)."""
    base = -hhat
    cone = []
    for k in range(3):
        ang = az + k*2*np.pi/3
        d = np.cos(np.pi-TET)*(-base)  # component along hhat
        # direction at TET from hhat:
        d = np.cos(TET)*hhat + np.sin(TET)*(np.cos(ang)*u + np.sin(ang)*v)
        cone.append(d/np.linalg.norm(d))
    # choose the two cone dirs whose mean best points 'into cavity' (away from O):
    # here simply: arms = cone[0],cone[1]; second H = cone[2]
    armA, armB, h2 = cone[0], cone[1], cone[2]
    Ht = C3 + CH*hhat
    H2 = C3 + CH*h2
    C2 = C3 + CC1*armA; C4 = C3 + CC1*armB
    C1 = C2 + CC2*(C2-C3)/np.linalg.norm(C2-C3)
    C5 = C4 + CC2*(C4-C3)/np.linalg.norm(C4-C3)
    atoms = [("C1",C1),("C2",C2),("C3",C3),("C4",C4),("C5",C5),("H1",Ht),("H2",H2)]
    return atoms, Ht

def clash_score(atoms, prot_pos):
    cpos = np.array([p for n,p in atoms if n.startswith("C")])
    d = np.linalg.norm(cpos[:,None,:]-prot_pos[None,:,:], axis=2)
    return int(np.sum(d<2.4)), float(d.min())

def write_pdb(path, atoms, header_lines=None):
    with open(path,"w") as f:
        if header_lines:
            for h in header_lines: f.write(h)
        for i,(nm,p) in enumerate(atoms,1):
            el = "H" if nm.startswith("H") else "C"
            f.write(f"HETATM{i:5d} {nm:<4s}LIG X 900    "
                    f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00          {el:>2s}\n")
        f.write("END\n")

def main():
    d = np.load(os.path.join(RES,"_active_site.npz"))
    Oacc = d["Oacc"]
    uWT = mda.Universe(os.path.join(SLO,"1F8N.pdb"))
    prot = altloc_heavy(uWT).positions
    # transferring-H direction = C3 -> O_acc ; C3 sits CO away from O on that line,
    # on the cavity side (use channel dir to choose the side).
    chan = d["chan"]
    C3 = Oacc + CO*chan                     # 3.0 A from O, toward cavity
    hhat = (Oacc - C3); hhat/=np.linalg.norm(hhat)   # H points back at O (near-linear)
    uf, vf = perp_frame(hhat)
    # scan azimuth to minimize clashes
    best=None
    for az in np.deg2rad(np.arange(0,360,10)):
        atoms,Ht = build_fragment(C3,hhat,uf,vf,az)
        ncl,dmin = clash_score(atoms,prot)
        key=(ncl,-dmin)
        if best is None or key<best[0]:
            best=(key,az,atoms,Ht,ncl,dmin)
    _,az,atoms,Ht,ncl,dmin = best
    # geometry report
    C3p=dict(atoms)["C3"]; HtO=np.linalg.norm(Ht-Oacc); C3O=np.linalg.norm(C3p-Oacc)
    ang=np.degrees(np.arccos(np.dot((C3p-Ht)/np.linalg.norm(C3p-Ht),
                                    (Oacc-Ht)/np.linalg.norm(Oacc-Ht))))
    print("== near-attack geometry ==")
    print(f"  C3...O_acc = {C3O:.2f} A   H...O = {HtO:.2f} A   C-H...O angle = {ang:.1f} deg")
    print(f"  best azimuth = {np.degrees(az):.0f} deg | clashes(<2.4A)={ncl}  min C-prot dist={dmin:.2f} A")

    write_pdb(os.path.join(SLO,"substrate_model.pdb"), atoms)
    # combined protein+fragment for visualization
    with open(os.path.join(SLO,"1F8N.pdb")) as f: pl=[l for l in f if l.startswith(("ATOM","HETATM","TER"))]
    with open(os.path.join(SLO,"1F8N_sub.pdb"),"w") as f:
        f.writelines(pl)
        for i,(nm,p) in enumerate(atoms,1):
            el="H" if nm.startswith("H") else "C"
            f.write(f"HETATM{9000+i:5d} {nm:<4s}LIG X 900    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00          {el:>2s}\n")
        f.write("END\n")
    np.savez(os.path.join(RES,"_transfer_site.npz"), C3=C3p, Ht=Ht, Oacc=Oacc)
    print(f"  wrote substrate_model.pdb, 1F8N_sub.pdb, results/_transfer_site.npz")

    # context: distance from transfer H to cavity residues
    print("\n== distance from transferring H to cavity Calpha ==")
    for rid in (546,553,754):
        ca=uWT.select_atoms(f"resid {rid} and name CA")
        ca=ca[np.isin(getattr(ca,'altLocs',np.array(['']*len(ca))),['',' ','A'])]
        print(f"  {ca.resnames[0]}{rid}: {np.linalg.norm(ca.positions[0]-Ht):.1f} A")

    # transfer-site descriptor across WT + I553 models (protein-only exchange wall)
    print("\n== transfer-site crowding (protein heavy atoms in 2-6 A shell of transferring H) ==")
    LAM=1.5
    for stem in ("1F8N","I553V_model","I553A_model","I553G_model"):
        p=os.path.join(SLO,stem+".pdb")
        if not os.path.exists(p): continue
        ag=altloc_heavy(mda.Universe(p)).positions
        r=np.linalg.norm(ag-Ht,axis=1); m=(r>2.0)&(r<6.0)
        crowd=float(np.sum(np.exp(-r[m]/LAM))); n6=int(np.sum(r<6.0))
        print(f"  {stem:14s} crowd_Ht={crowd:.3f}  nHeavy6A={n6}")
    print("\n[interpretation] if crowd_Ht is ~flat across the I553 series, that "
          "confirms I553 is too distal to change the static exchange wall at the "
          "transferring H -- i.e. it acts through GATING/dynamics, which needs an "
          "MD ensemble + the substrate, not a single near-attack snapshot.")

if __name__ == "__main__":
    main()
