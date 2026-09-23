#!/usr/bin/env python3
"""
orca_to_gms_bridge.py -- write ORCA's optimized geometry + analytic Hessian into the
exact GAMESS-US log blocks that MCPB.py's parser (pymsmt/mol/gmsio.py) reads, so MCPB
does the metal-site assembly (per-ligand atom typing, per-term Seminario force constants,
charge embedding, tleap) using ORCA's tightly-converged, spin-pure QM.

The ORCA cluster IS MCPB's small model (geometry was extracted from SLO_small_opt.inp),
so atom order matches MCPB's expectation. Writes SLO_small_opt.log (geometry) and
SLO_small_fc.log (geometry + Hessian). Units: geometry in BOHR, Hessian in Eh/Bohr^2
(GAMESS native), values in fixed 9-char fields starting at column 20 (0-based).
"""
import numpy as np
BOHR=0.52917721
Z={'H':1,'C':6,'N':7,'O':8,'Fe':26,'FE':26}

L=open('orca_fe/fe_opt.xyz').read().splitlines(); nat=int(L[0])
els=[l.split()[0] for l in L[2:2+nat]]
xb=np.array([[float(x) for x in l.split()[1:4]] for l in L[2:2+nat]])/BOHR

def parse_hess(p):
    Lh=open(p).read().splitlines(); ih=Lh.index('$hessian'); dim=int(Lh[ih+1].split()[0])
    H=np.zeros((dim,dim)); i=ih+2
    while i<len(Lh):
        pp=Lh[i].split()
        if not pp or pp[0].startswith('$'): break
        if all(('.' not in x and 'e' not in x.lower()) for x in pp):
            cols=[int(x) for x in pp]; i+=1
            for _ in range(dim):
                r=Lh[i].split(); ri=int(r[0])
                for kk,c in enumerate(cols): H[ri][c]=float(r[1+kk])
                i+=1
        else: i+=1
    return H,dim
H,dim=parse_hess('orca_fe/fe_freq.hess')
H=0.5*(H+H.T)

def geom_block():
    s=" ATOM      ATOMIC                      COORDINATES (BOHR)\n"
    s+="           CHARGE         X                   Y                   Z\n"
    for i in range(nat):
        s+=f" {els[i]:<5s}{float(Z[els[i]]):5.1f}{xb[i,0]:20.10f}{xb[i,1]:20.10f}{xb[i,2]:20.10f}\n"
    s+="\n          INTERNUCLEAR DISTANCES (ANGS.)\n"
    return s

def ihdr(i,nc):   # atom-index header + coord header, 4-space? values start col 20
    idx="".join(f"{(i*6+k)//3+1:9d}" for k in range(nc))
    crd="".join(f"{'XYZ'[(i*6+k)%3]:>9s}" for k in range(nc))
    return " "*20+idx+"\n"+" "*20+crd+"\n"

def datarow(j,i,nc):
    lbl=(f"{j//3+1:5d}  {els[j//3]:<3s}{'XYZ'[j%3]}"+" "*20)[:20]
    return lbl+"".join(f"{H[j][i*6+k]:9.6f}" for k in range(nc))

def hess_block():
    msize=dim; cyc=msize//6
    s="          CARTESIAN FORCE CONSTANT MATRIX\n"
    s+="          "+"-"*31+"\n\n"          # marker+1, marker+2(blank)
    s+=ihdr(0,6)+"\n"                        # marker+3,+4, +5(blank) ; data at marker+6
    for j in range(0,msize): s+=datarow(j,0,6)+"\n"
    for i in range(1,cyc):
        s+="\n"+ihdr(i,6)+"\n"               # 4 separator lines: blank,idx,crd,blank
        for j in range(i*6,msize): s+=datarow(j,i,6)+"\n"
    if msize%6==3:
        i=cyc; s+="\n"+ihdr(i,3)+"\n"
        for j in range(i*6,msize): s+=datarow(j,i,3)+"\n"
    return s

with open('SLO_small_opt.log','w') as f:
    f.write("          ORCA->GAMESS geometry bridge\n\n"+geom_block()+
            "\n EXECUTION OF GAMESS TERMINATED NORMALLY\n")
with open('SLO_small_fc.log','w') as f:
    f.write("          ORCA->GAMESS Hessian bridge\n\n"+geom_block()+"\n"+hess_block()+
            "\n EXECUTION OF GAMESS TERMINATED NORMALLY\n")
print(f"  wrote SLO_small_opt.log + SLO_small_fc.log ({nat} atoms, {dim}x{dim} Hessian)")
