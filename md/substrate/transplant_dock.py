#!/usr/bin/env python3
"""Template-based substrate docking (chosen approach).
 1. Kabsch-superpose the 8R-LOX (4QWT chain C) Fe first-coordination sphere onto SLO's
    (Fe + 3 His-NE2 + Ile-carboxylate O + water/OH), trying all His permutations.
 2. Apply that transform to the bound arachidonate (ACD) -> AA now sits in SLO's channel.
 3. Order both fatty-acid backbones (carboxyl C1 -> methyl terminus) by graph walk, then
    Kabsch-fit linoleate onto AA with the REACTIVE carbons aligned (linoleate C11 <-> AA C10,
    each enzyme's H-abstraction carbon), so linoleate follows the experimental channel path.
 4. Rigid-slide linoleate along the local chain axis so C11 sits 3.30 A from the Fe-OH O.
 5. Report near-attack geometry + protein clashes."""
import numpy as np, parmed as pmd

def xyzl(l): return np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])])
def kabsch(P,Q):
    Pc=P-P.mean(0); Qc=Q-Q.mean(0); H=Pc.T@Qc; U,S,Vt=np.linalg.svd(H)
    d=np.sign(np.linalg.det(Vt.T@U.T)); R=Vt.T@np.diag([1,1,d])@U.T
    t=Q.mean(0)-R@P.mean(0); rmsd=np.sqrt((((R@P.T).T+t-Q)**2).sum(1).mean())
    return R,t,rmsd

# ---------- 1. template metal site + arachidonate (4QWT chain C) ----------
Tm={}; acd={}
for l in open('../template/4QWT.pdb'):
    if l[:6].strip() not in ('ATOM','HETATM') or l[21]!='C': continue
    try: pos=xyzl(l)
    except: continue
    rn=l[17:20].strip(); ri=l[22:26].strip(); an=l[12:16].strip()
    if rn=='FE2': Tm['FE']=pos
    elif rn=='HIS' and an=='NE2' and ri in ('384','389','570'): Tm['H'+ri]=pos
    elif rn=='ILE' and ri=='693' and an=='O': Tm['CBX']=pos
    elif rn=='HOH' and ri=='815' and an=='O': Tm['OH']=pos
    elif rn=='ACD' and an[0]=='C': acd[int(an[1:])]=pos
aa=np.array([acd[i] for i in range(1,21)])          # AA C1..C20

# ---------- SLO metal site (Fe + ligands within 2.6 A) ----------
p=pmd.load_file('../mcpb/SLO_dry.prmtop','../mcpb/SLO_dry.inpcrd')
crd={a.idx:np.array([a.xx,a.xy,a.xz]) for a in p.atoms}
fe=[a for a in p.atoms if a.type=='M1'][0]; feS=crd[fe.idx]
Sm={'FE':feS}
his=[];
for a in p.atoms:
    d=np.linalg.norm(crd[a.idx]-feS)
    if 1.5<d<2.6 and a.idx!=fe.idx:
        if a.name=='NE2': his.append(crd[a.idx])
        elif a.name=='OXT': Sm['CBX']=crd[a.idx]
        elif a.residue.name=='OH1' and a.name=='O': Sm['OH']=crd[a.idx]

# ---------- best His permutation for the superposition ----------
from itertools import permutations
Hkeys=['H384','H389','H570']
best=None
for perm in permutations(range(3)):
    P=np.array([Tm['FE'],Tm['H384'],Tm['H389'],Tm['H570'],Tm['CBX'],Tm['OH']])
    Q=np.array([Sm['FE'],his[perm[0]],his[perm[1]],his[perm[2]],Sm['CBX'],Sm['OH']])
    R,t,rmsd=kabsch(P,Q)
    if best is None or rmsd<best[0]: best=(rmsd,R,t,perm)
rmsd,R,t,perm=best
print(f"  metal-site superposition RMSD = {rmsd:.2f} A (His perm {perm})")
aa_S=(R@aa.T).T+t                                   # arachidonate in SLO frame
oacc=Sm['OH']
print(f"  transplanted AA: C10(reactive) is {np.linalg.norm(aa_S[9]-oacc):.2f} A from SLO Fe-OH O")

# ---------- linoleate backbone order (graph walk carboxyl C -> methyl) ----------
L=open('LIG.mol2').read().splitlines()
ai=L.index('@<TRIPOS>ATOM'); bi=L.index('@<TRIPOS>BOND')
be=next(i for i in range(bi+1,len(L)) if L[i].startswith('@'))
nm=[]; ty=[]; xy=[]
for ln in L[ai+1:bi]:
    q=ln.split()
    if q[5]=='ho': continue
    nm.append(q[1]); ty.append(q[5]); xy.append([float(q[2]),float(q[3]),float(q[4])])
xy=np.array(xy); idx={n:i for i,n in enumerate(nm)}
nbr={i:[] for i in range(len(nm))}
for ln in L[bi+1:be]:
    q=ln.split()
    if len(q)>=4 and q[1].isdigit():
        a,b=int(q[1])-1,int(q[2])-1
        # map original mol2 index -> anion index via name
        pass
# rebuild neighbor by name (mol2 indices are 1-based over the NEUTRAL set incl 'ho')
namemap={}
i0=0
for ln in L[ai+1:bi]:
    q=ln.split(); namemap[int(q[0])]=q[1]
for ln in L[bi+1:be]:
    q=ln.split()
    if len(q)>=4 and q[1].isdigit():
        na,nb=namemap[int(q[1])],namemap[int(q[2])]
        if na in idx and nb in idx: nbr[idx[na]].append(idx[nb]); nbr[idx[nb]].append(idx[na])
carbons=[i for i in range(len(nm)) if nm[i][0]=='C']
carbC=[i for i in carbons if sum(1 for j in nbr[i] if nm[j][0]=='O')>=2][0]   # C1 carboxyl
# walk the carbon chain
order=[carbC]; prev=-1; cur=carbC
while True:
    nxt=[j for j in nbr[cur] if nm[j][0]=='C' and j!=prev and j not in order]
    if not nxt: break
    prev=cur; cur=nxt[0]; order.append(cur)
print(f"  linoleate backbone carbons ordered: {len(order)} (expect 18)")
lin_bb=np.array([xy[i] for i in order])             # linoleate C1..C18

# ---------- fit linoleate onto AA with reactive carbons aligned (linC11<->aaC10) ----------
# match linoleate Ck (k=1..18) to AA C(k-1) so lin C11 -> AA C10
pairs=[(k, k) for k in range(1,18)]                 # lin index k (1-based) -> AA index k+?
# lin C11 (order idx 10) <-> AA C10 (aa_S idx 9): offset so lin[j] ~ aa[j-1]
Lp=np.array([lin_bb[j] for j in range(1,18)])       # lin C2..C18
Ap=np.array([aa_S[j-1] for j in range(1,18)])       # AA C1..C17
R2,t2,r2=kabsch(Lp,Ap)
lin_all=(R2@xy.T).T+t2                              # whole linoleate anion into SLO channel
iC11=order[10]                                      # linoleate C11 (11th backbone carbon)
print(f"  linoleate onto AA-path RMSD={r2:.2f} A; C11 now {np.linalg.norm(lin_all[iC11]-oacc):.2f} A from O")

# ---------- slide along local chain axis so C11 = 3.30 A from O ----------
axis=(lin_all[order[11]]-lin_all[order[9]]); axis/=np.linalg.norm(axis)  # C10->C12 local tangent
# move C11 toward target distance 3.30 along the O->C11 direction
for _ in range(200):
    v=lin_all[iC11]-oacc; d=np.linalg.norm(v)
    if abs(d-3.30)<0.02: break
    lin_all += (3.30-d)*(v/d)*0.5
print(f"  after slide: r(C11...O)={np.linalg.norm(lin_all[iC11]-oacc):.2f} A")

# clashes
from scipy.spatial import cKDTree
prot=np.array([crd[a.idx] for a in p.atoms if a.atomic_number>1 and a.residue.name not in ('WAT','HOH','OH1','FE1')])
tk=cKDTree(prot); dmin=np.array([tk.query(pt)[0] for pt in lin_all])
print(f"  min substrate-protein dist={dmin.min():.2f} A; atoms<2.0A={int((dmin<2.0).sum())}; <2.5A={int((dmin<2.5).sum())}")

with open('linoleate_docked.pdb','w') as f:
    for i in range(len(nm)):
        f.write(f"HETATM{i+1:5d} {nm[i]:<4s} LIG A 900    {lin_all[i,0]:8.3f}{lin_all[i,1]:8.3f}{lin_all[i,2]:8.3f}  1.00  0.00          {nm[i][0]:>2s}\n")
    f.write("END\n")
print("  wrote linoleate_docked.pdb (template-guided pose)")
