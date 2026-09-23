#!/usr/bin/env python3
"""
seminario_fe.py -- Fe(III)-OH bonded force constants by the MODIFIED SEMINARIO METHOD
(Allen, Payne & Cole, JCTC 2018), driven with the ORCA analytic Hessian.

Uses the published reference functions (aa840/ModSeminario_Py) verbatim for the
projection math; this script only (i) parses the ORCA .hess, (ii) converts units
exactly as the reference does (Hartree/Bohr^2 -> kcal/mol/Ang^2 via *627.509391/0.529^2;
Bohr -> Ang via *0.529), (iii) builds the 3x3 sub-Hessian eigen-blocks, and
(iv) calls force_constant_bond / force_angle_constant for every Fe-centred bond and angle.
Output: Fe-ligand bond k (kcal/mol/Ang^2) + r0, and X-Fe-Y / Fe-L-X angle k (kcal/mol/rad^2) + theta0.
"""
import sys, numpy as np
REF="/home/liang/Workspace/WritePaper/CatalysisQuamBio/md/mcpb/ModSeminario_Py/Python_Modified_Seminario_Method"
sys.path.insert(0, REF)
from force_constant_bond import force_constant_bond
from force_angle_constant import force_angle_constant

HESS="/home/liang/Workspace/WritePaper/CatalysisQuamBio/md/mcpb/orca_fe/fe_freq.hess"
BOHR=0.529                       # exact constant used by the reference (for consistency)
H2KCAL=627.509391

def parse_hess(path):
    L=open(path).read().splitlines()
    # ---- $atoms ----
    ia=L.index('$atoms'); nat=int(L[ia+1].split()[0])
    els=[]; xyz=[]
    for k in range(ia+2, ia+2+nat):
        p=L[k].split(); els.append(p[0]); xyz.append([float(p[2]),float(p[3]),float(p[4])])
    xyz=np.array(xyz)            # Bohr
    # ---- $hessian ----
    ih=L.index('$hessian'); dim=int(L[ih+1].split()[0])
    H=np.zeros((dim,dim)); i=ih+2
    while i < len(L):
        parts=L[i].split()
        if not parts or parts[0].startswith('$'): break
        if all(('.' not in p and 'e' not in p.lower()) for p in parts):   # column-index header row
            cols=[int(p) for p in parts]; i+=1
            for _ in range(dim):
                row=L[i].split(); ri=int(row[0])
                for kk,c in enumerate(cols): H[ri][c]=float(row[1+kk])
                i+=1
        else:
            i+=1
    return els, xyz, H, dim

els, xyz_bohr, H_au, dim = parse_hess(HESS)
N=len(els)
coords = xyz_bohr*BOHR                       # Angstrom
hess  = H_au * H2KCAL/(BOHR**2)              # kcal/mol/Ang^2
hess  = 0.5*(hess+hess.T)                     # symmetrize (as reference does)

# eigen-blocks for every atom pair (exactly as the reference builds them)
eigenvalues=np.empty((N,N,3),dtype=complex); eigenvectors=np.empty((3,3,N,N),dtype=complex)
for i in range(N):
    for j in range(N):
        a,b=np.linalg.eig(hess[3*i:3*i+3,3*j:3*j+3]); eigenvalues[i,j,:]=a; eigenvectors[:,:,i,j]=b

d=lambda i,j: float(np.linalg.norm(coords[i]-coords[j]))
bl=np.zeros((N,N))
for i in range(N):
    for j in range(N): bl[i,j]=d(i,j)

fe=[i for i,e in enumerate(els) if e.lower()=='fe'][0]
ligs=[j for j in range(N) if j!=fe and els[j]!='H' and d(fe,j)<2.6]  # heavy-atom ligands only
def neigh(a):   # covalent neighbours of atom a (exclude Fe)
    return [j for j in range(N) if j!=a and j!=fe and d(a,j)< (1.35 if els[j]=='H' else 1.85)]

print(f"# Fe atom index {fe} ({els[fe]}); {len(ligs)} ligating atoms within 2.6 A\n")
print("## Fe-ligand BONDS  (modified-Seminario)")
print(f"{'pair':<10}{'k (kcal/mol/A^2)':>18}{'r0 (A)':>10}")
for j in ligs:
    AB=force_constant_bond(fe,j,eigenvalues,eigenvectors,coords)
    BA=force_constant_bond(j,fe,eigenvalues,eigenvectors,coords)
    k=float(np.real((AB+BA)/2))
    print(f"Fe-{els[j]:<7}{k:18.2f}{d(fe,j):10.3f}")

print("\n## L-Fe-L' ANGLES (Fe central)")
print(f"{'angle':<14}{'k (kcal/mol/rad^2)':>20}{'theta0 (deg)':>14}")
for a in range(len(ligs)):
    for c in range(a+1,len(ligs)):
        A,C=ligs[a],ligs[c]
        kth,th0=force_angle_constant(A,fe,C,bl,eigenvalues,eigenvectors,coords,1.0,1.0)
        print(f"{els[A]}-Fe-{els[C]:<6}{float(np.real(kth)):20.2f}{np.degrees(th0) if th0<7 else th0:14.1f}")

print("\n## Fe-L-X ANGLES (Fe terminal)")
for L in ligs:
    for X in neigh(L):
        kth,th0=force_angle_constant(fe,L,X,bl,eigenvalues,eigenvectors,coords,1.0,1.0)
        print(f"Fe-{els[L]}-{els[X]:<5}{float(np.real(kth)):20.2f}{np.degrees(th0) if th0<7 else th0:14.1f}")
