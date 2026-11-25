#!/usr/bin/env python3
"""Schedule every per-variant extraction and quantum-chemistry step on model-P data.

Runs the unchanged model-S scripts inside shadow_P (PAULI_ROOT), starting each job as soon
as its inputs exist, so the CPU work overlaps the GPU campaign. Every job runs in its own
process; a job is retried at most twice; everything is logged in downstream.log.
Resources: at most THREADS CPU threads and MEM_GB of declared Psi4 memory at once.
Usage: psens_downstream.py [TAG ...]
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PS = ROOT / 'results/protonation_sensitivity'
SH = PS / 'shadow_P'
SLOMD = '/home/liang/anaconda3/envs/slomd/bin/python'
PAULI = '/home/liang/anaconda3/envs/pauli/bin/python'
THREADS, MEM_GB = 16, 100
LOG = PS / 'downstream.log'
TAGS = ('WT', 'I538A', 'I552A', 'I553A', 'L546A', 'L754A', 'V750A')
R = SH / 'results'
RG, EF, NF, DF = R / 'reactive_geometry', R / 'ensemble_fluctuation', R / 'sapt_bio/native_fragment', R / 'sapt_bio/donor_fragment'
NDV, TH, CFG = R / 'native_donor_validation', R / 'transverse_hessian', R / 'stage5_configs'


def log(m):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {m}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def job(jid, py, script, args, deps, outs, threads=1, mem=6, env=None):
    return dict(id=jid, cmd=[py, str(SH / 'scripts' / script), *map(str, args)], deps=deps, outs=outs,
                threads=threads, mem=mem, env=env or {}, tries=0, state='wait')


def static_jobs(t):
    J = []
    for c in ('r255', 'r340'):
        J.append(job(f'{t}:extract_clamped:{c}', SLOMD, 'extract_clamped_fragment.py', [t, c],
                     [RG / f'{t}_{c}_final.rst7'], [RG / f'{t}_{c}_geometry.json']))
        J.append(job(f'{t}:ck_static:{c}', PAULI, 'sapt_clamped_fragment.py', [t, c],
                     [RG / f'{t}_{c}_geometry.json'], [RG / f'{t}_{c}_native_result.json'], 4, 6))
        J.append(job(f'{t}:extract_frames:{c}', SLOMD, 'extract_frames_geometry.py', [t, c],
                     [EF / f'{t}_{c}_frames.npy', EF / f'{t}_{c}_colvar.dat'],
                     [EF / f'{t}_{c}_frames_geometry.json']))
        for f in range(40):
            J.append(job(f'{t}:ck_frame:{c}:{f:02d}', PAULI, 'sapt_one_frame.py', [t, c, f],
                         [EF / f'{t}_{c}_frames_geometry.json'], [EF / f'{t}_{c}_f{f:03d}.json'], 4, 6))
        for w in range(8):
            J.append(job(f'{t}:fsapt:{c}:{w}', PAULI, 'fsapt_ensemble.py', [t, c, 40, f'{w}/8'],
                         [EF / f'{t}_{c}_frames_geometry.json', NF / f'{t}_fragment_indices.json'],
                         [DF / f'{t}_{c}_fsapt_ensemble_part{w}of8.json'], 1, 6, {'PSI4_MEM': '6 GB'}))
        J.append(job(f'{t}:fsapt_merge:{c}', PAULI, 'merge_fsapt_shards.py', [t, c, 40],
                     [DF / f'{t}_{c}_fsapt_ensemble_part{w}of8.json' for w in range(8)],
                     [DF / f'{t}_{c}_fsapt_ensemble.json']))
    J.append(job(f'{t}:transverse_methane', PAULI, 'transverse_proton_hessian.py', [t, 'r255'],
                 [RG / f'{t}_r255_geometry.json', RG / f'{t}_r255_native_result.json'],
                 [TH / f'{t}_r255_w015_transverse_hessian.json'], 1, 8))
    J.append(job(f'{t}:extract_native_donor', SLOMD, 'extract_native_donor_clamped.py', [t, 'r255'],
                 [RG / f'{t}_r255_geometry.json'], [NDV / f'{t}_r255_native_donor.json']))
    J.append(job(f'{t}:transverse_native', PAULI, 'transverse_native_donor.py', [t],
                 [NDV / f'{t}_r255_native_donor.json'], [NDV / f'{t}_r255_w015_native_transverse.json'], 1, 8))
    J.append(job(f'{t}:total_restoring', PAULI, 'total_restoring_hessian.py', [t],
                 [NDV / f'{t}_r255_native_donor.json'], [NDV / f'{t}_r255_w015_total_restoring.json'], 1, 8))
    J.append(job(f'{t}:stage5_extract', SLOMD, 'stage5_extract.py', [t, 16],
                 [CFG / f'{t}_frames.npy', PS / 'stage5_md' / f'{t}_r255_colvar.dat', RG / f'{t}_r255_geometry.json'],
                 [CFG / t / 'selection.json']))
    return J


def stage5_jobs(t):
    sel = json.loads((CFG / t / 'selection.json').read_text())
    J = []
    for f in sorted((CFG / t).glob(f'{t}_cfg*_native_donor.json')):
        out = f.with_name(f.name.replace('_native_donor.json', '_w015_transverse.json'))
        J.append(job(f'{t}:stage5:{f.name[len(t) + 1:len(t) + 6]}', PAULI, 'stage5_transverse.py', [f],
                     [f], [out], 1, 4))
    log(f'{t}: stage-5 selection has {len(J)} configurations ({sel.get("stride", "?")} stride)')
    return J


def main():
    tags = tuple(sys.argv[1:]) or TAGS
    jobs = [j for t in tags for j in static_jobs(t)]
    s5_added = set()
    running = {}
    while True:
        for t in tags:
            if t not in s5_added and (CFG / t / 'selection.json').exists():
                jobs += stage5_jobs(t)
                s5_added.add(t)
        for jid, (p, j, t0) in list(running.items()):
            if p.poll() is None:
                continue
            del running[jid]
            ok = all(Path(o).exists() for o in j['outs'])
            log(f'{jid}: rc {p.returncode}, {"ok" if ok else "FAILED"}, {(time.time() - t0) / 60:.1f} min')
            j['state'] = 'done' if ok else ('wait' if j['tries'] < 3 else 'failed')
        used_t = sum(j['threads'] for _, j, _ in running.values())
        used_m = sum(j['mem'] for _, j, _ in running.values())
        for j in jobs:
            if j['state'] != 'wait':
                continue
            if all(Path(o).exists() for o in j['outs']):
                j['state'] = 'done'
                continue
            if not all(Path(d).exists() for d in j['deps']):
                continue
            if used_t + j['threads'] > THREADS or used_m + j['mem'] > MEM_GB:
                continue
            j['tries'] += 1
            env = dict(os.environ, PAULI_ROOT=str(SH), OMP_NUM_THREADS=str(j['threads']), **j['env'])
            lf = open(PS / 'logs' / f'ds_{j["id"].replace(":", "_")}_try{j["tries"]}.log', 'w')
            p = subprocess.Popen(j['cmd'], env=env, cwd=str(SH), stdout=lf, stderr=subprocess.STDOUT)
            running[j['id']] = (p, j, time.time())
            j['state'] = 'running'
            used_t += j['threads']
            used_m += j['mem']
        states = [j['state'] for j in jobs]
        if not running and all(s in ('done', 'failed') for s in states) and len(s5_added) == len(tags):
            break
        time.sleep(20)
    failed = [j['id'] for j in jobs if j['state'] == 'failed']
    log(f'downstream finished: {states.count("done")} done, {len(failed)} failed {failed[:10]}')


if __name__ == '__main__':
    main()
