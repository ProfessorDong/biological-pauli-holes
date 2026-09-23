#!/usr/bin/env python3
"""Dock the full linoleate ANION into the WT SLO active site in a near-attack pose:
   C11 (bis-allylic reactive carbon) ~3.3 A beyond the Fe-OH oxygen along the reaction axis,
   one C11-H (pro-S) pointing at the acceptor O (near-linear C-H...O), and the chain rolled
   about that axis to minimize protein clashes (methyl end into the hydrophobic channel).
   Writes the docked substrate (LIG residue) as a PDB. MD equilibration relaxes the pose."""
import numpy as np, parmed as pmd

d=np.load('_dock_target.npz'); O_acc=d['oacc']; T=d['target_C11']; Fe=d['fe']

# --- linoleate ANION from LIG.mol2 (drop the acid 'ho' H) ---
L=open('LIG.mol2').read().splitlines()
ai=L.index('@<TRIPOS>ATOM'); bi=L.index('@<TRIPOS>BOND')
names=[]; types=[]; xyz=[]
for ln in L[ai+1:bi]:
    q=ln.split()
    if q[5]=='ho': continue
    names.append(q[1]); types.append(q[5]); xyz.append([float(q[2]),float(q[3]),float(q[4])])
xyz=np.array(xyz); N=len(names)
iC11=names.index('C14')                                   # bis-allylic reactive carbon
hbond=[i for i in range(N) if names[i][0]=='H' and np.linalg.norm(xyz[i]-xyz[iC11])<1.2]

# --- protein heavy atoms (exclude H, water, and the OH/Fe acceptor itself) for clash scoring ---
p=pmd.load_file('../mcpb/SLO_dry.prmtop','../mcpb/SLO_dry.inpcrd')
prot=np.array([[a.xx,a.xy,a.xz] for a in p.atoms
               if a.atomic_number>1 and a.residue.name not in ('WAT','HOH','OH1','FE1')])

def Rvec(u,v):
    u=u/np.linalg.norm(u); v=v/np.linalg.norm(v); ax=np.cross(u,v); s=np.linalg.norm(ax)
    if s<1e-8: return np.eye(3)
    ax/=s; a=np.arccos(np.clip(u@v,-1,1)); K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K
def Rax(ax,a):
    ax=ax/np.linalg.norm(ax); K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K
def clashes(c):
    from scipy.spatial import cKDTree
    t=cKDTree(prot); return sum(len(t.query_ball_point(pt,2.6)) for pt in c)

best=None
for iH in hbond:                                          # try each C11-H as the pro-S donor
    c=xyz - xyz[iC11] + T                                  # C11 -> target
    c=(c-T)@Rvec(c[iH]-T, O_acc-T).T + T                  # align C11->H with C11->O_acc
    axis=O_acc-T
    for deg in range(0,360,15):                           # roll about the C-H...O axis
        cc=(c-T)@Rax(axis, np.radians(deg)).T + T
        nc=clashes(cc)
        rDA=np.linalg.norm(cc[iC11]-O_acc); rHO=np.linalg.norm(cc[iH]-O_acc)
        if best is None or nc<best[0]:
            best=(nc, cc.copy(), iH, deg, rDA, rHO)
nc,cc,iH,deg,rDA,rHO=best
print(f"  best pose: clashes(<2.6A)={nc}  proS-H={names[iH]}  roll={deg}deg")
print(f"  near-attack: r(C11...O_acc)={rDA:.2f} A   r(H...O_acc)={rHO:.2f} A")

# write docked substrate PDB (resname LIG)
with open('linoleate_docked.pdb','w') as f:
    for i in range(N):
        el=names[i][0]
        f.write(f"HETATM{i+1:5d} {names[i]:<4s} LIG A 900    {cc[i,0]:8.3f}{cc[i,1]:8.3f}{cc[i,2]:8.3f}  1.00  0.00          {el:>2s}\n")
    f.write("END\n")
print("  wrote linoleate_docked.pdb")
