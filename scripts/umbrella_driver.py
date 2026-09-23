#!/usr/bin/env python3
"""Drive a full umbrella-sampling PMF along r_DA for ONE system. Windows are seeded
adiabatically outward from the equilibrated structure (start at the window nearest the
equilibrated r_DA, then step inward toward the near-attack and outward toward the resting
distance, each window seeded from its completed neighbor) -> smooth pulling, no hysteresis.
Usage: umbrella_driver.py <tag> <prmtop> <eq_rst7> <donor> <acceptor> [eqr_DA]"""
import sys, os, subprocess, numpy as np

tag,prmtop,eqrst,donor,acceptor = sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]
eqr = float(sys.argv[6]) if len(sys.argv)>6 else 5.0
rep = int(sys.argv[7]) if len(sys.argv)>7 else 1          # replica id (independent seeds)
WIN = os.path.dirname(os.path.abspath(__file__))+"/umbrella_window.py"
PY  = "/home/liang/anaconda3/envs/slomd/bin/python"
suffix = "" if rep==1 else f"_rep{rep}"
outdir = f"/home/liang/Workspace/WritePaper/CatalysisQuamBio/results/umbrella/{tag}{suffix}"
os.makedirs(outdir, exist_ok=True)

centers = np.round(np.arange(2.7, 6.21, 0.25), 2)          # 2.70 .. 6.20 A, 15 windows
K = 12.0                                                    # kcal/mol/A^2
start = int(np.argmin(np.abs(centers-eqr)))                # nearest window to equilibrated r_DA
order = [start]
lo,hi = start-1, start+1
while lo>=0 or hi<len(centers):                            # interleave inward/outward from start
    if lo>=0: order.append(lo); lo-=1
    if hi<len(centers): order.append(hi); hi+=1

seedmap = {}                                               # window index -> seed rst7
seedmap[start] = eqrst
def run(i):
    r0 = centers[i]; out = f"{outdir}/win_{r0:.2f}"
    if os.path.exists(out+"_colvar.dat"):
        print(f"  [skip] window {r0:.2f} already done", flush=True); return
    seedf = seedmap[i]                                     # coordinate seed (rst7)
    rngseed = rep*100000 + i*7 + 1                          # velocity RNG seed, unique per replica+window
    cmd = [PY, WIN, "--prmtop",prmtop, "--seed",seedf, "--donor",donor, "--acceptor",acceptor,
           "--r0",f"{r0}", "--k",f"{K}", "--ns-eq","0.5", "--ns-prod","5.0", "--out",out, "--rngseed",str(rngseed)]
    print(f"  window {r0:.2f} A (seed from r_DA index {seedmap.get(i,'?')})", flush=True)
    # OpenMM+CUDA can intermittently SIGSEGV in interpreter teardown OR (rarely) in CUDA
    # context init after a previous crash. umbrella_window.py now uses os._exit(0) so a
    # clean run cannot SIGSEGV on teardown; a real init crash gets sleep+retry, and if
    # a window still fails after 3 attempts we record it and continue so the batch
    # doesn't abort at 5% completion.
    import time
    # A healthy window can take up to ~20 min on the Blackwell (5 ns HMR + 0.5 ns eq).
    # Cap at 45 min so an occasional hang after a SIGSEGV cannot block the whole batch,
    # while still letting slow-but-progressing windows finish.
    WINDOW_TIMEOUT_S = 2700
    for attempt in (1, 2, 3):
        try:
            rc = subprocess.run(cmd, timeout=WINDOW_TIMEOUT_S).returncode
        except subprocess.TimeoutExpired:
            rc = -998
            print(f"  [warn] window {r0:.2f} attempt {attempt} TIMEOUT after {WINDOW_TIMEOUT_S}s", flush=True)
        if os.path.exists(out+"_colvar.dat") and os.path.exists(out+"_final.rst7"):
            if rc != 0:
                print(f"  [note] window {r0:.2f} exit={rc} but outputs OK -> accepting", flush=True)
            return
        print(f"  [warn] window {r0:.2f} attempt {attempt} rc={rc}, outputs missing", flush=True)
        time.sleep(5 * attempt)
    with open(outdir+"/FAILED_WINDOWS.txt", "a") as fh:
        fh.write(f"{r0:.2f} rc={rc}\n")
    print(f"  [FAIL] window {r0:.2f} failed 3x — recorded, continuing batch", flush=True)

for i in order:
    # seed each window from its nearest already-run neighbor's final frame
    if i not in seedmap:
        for j in (i-1, i+1):
            f=f"{outdir}/win_{centers[j]:.2f}_final.rst7" if 0<=j<len(centers) else None
            if f and os.path.exists(f): seedmap[i]=f; break
        else: seedmap[i]=eqrst
    run(i)
print(f"UMBRELLA_DONE {tag}", flush=True)
