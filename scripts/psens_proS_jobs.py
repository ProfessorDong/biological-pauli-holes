#!/usr/bin/env python3
"""Schedule every L754A recomputation with the pro-S hydrogen (H25), in both models.

Same job machinery as psens_downstream.py: unchanged model-S scripts run inside the shadow roots
built by psens_proS_setup.py, each job starting when its inputs exist, at most THREADS threads and
MEM_GB of declared Psi4 memory at once, two retries, everything logged in proS/jobs.log.
Model S also redoes the near-attack set and the stage-4 checks run for L754A; model P has neither.
Usage: psens_proS_jobs.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PS = ROOT / 'results/protonation_sensitivity'
SLOMD = '/home/liang/anaconda3/envs/slomd/bin/python'
PAULI = '/home/liang/anaconda3/envs/pauli/bin/python'
THREADS, MEM_GB = 16, 100
LOG = PS / 'proS/jobs.log'
T = 'L754A'


def log(m):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {m}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def jobs_for(model):
    sh = PS / ('shadow_SpS' if model == 'S' else 'shadow_PpS')
    R = sh / 'results'
    RG, EF, NF, DF = R / 'reactive_geometry', R / 'ensemble_fluctuation', R / 'sapt_bio/native_fragment', R / 'sapt_bio/donor_fragment'
    NDV, TH, CFG = R / 'native_donor_validation', R / 'transverse_hessian', R / 'stage5_configs'
    J = []

    def job(name, py, script, args, deps, outs, threads=1, mem=6, env=None, scripts_dir=None):
        sd = scripts_dir or (sh / 'scripts')
        J.append(dict(id=f'{model}:{name}', cmd=[py, str(sd / script), *map(str, args)], deps=deps, outs=outs,
                      threads=threads, mem=mem, env=dict(env or {}, PAULI_ROOT=str(sh)), cwd=str(sh),
                      tries=0, state='wait'))

    if model == 'S':
        job('near:extract', SLOMD, 'extract_native_fragment.py', [T], [], [NF / f'{T}_geometry.json'])
        job('near:ck', PAULI, 'sapt_native_fragment.py', [T], [NF / f'{T}_geometry.json'], [NF / f'{T}_native_result.json'], 4, 6)
        job('near:methane_wall', PAULI, 'sapt_methane_wall.py', [T, PS / 'proS/methane_wall_S'],
            [NF / f'{T}_geometry.json'], [PS / 'proS/methane_wall_S' / f'{T}_methane_wall_result.json'], 4, 6,
            scripts_dir=ROOT / 'scripts')
        job('near:pentadienyl', SLOMD, 'extract_pentadienyl_donor.py', [T], [], [NF / f'{T}_pentadienyl.json'])
        job('near:fragment_indices', SLOMD, 'donor_fragment_indices.py', [T], [NF / f'{T}_pentadienyl.json'],
            [NF / f'{T}_fragment_indices.json'])
        job('near:donor_comparison', PAULI, 'sapt_pentadienyl_vs_methane.py', [T], [NF / f'{T}_pentadienyl.json'],
            [DF / f'{T}_donor_comparison.json'], 4, 8)
        job('near:fsapt_decomposition', PAULI, 'fsapt_donor_decomposition.py', [T],
            [NF / f'{T}_pentadienyl.json', DF / f'{T}_donor_comparison.json'], [DF / f'{T}_fsapt_decomposition.json'],
            1, 16, {'PSI4_MEM': '16 GB'})
    else:
        SNF = PS / 'shadow_SpS/results/sapt_bio/native_fragment'
        SDF = PS / 'shadow_SpS/results/sapt_bio/donor_fragment'
        job('map_fragment_indices', SLOMD, 'psens_map_fragment_indices.py', [T], [SNF / f'{T}_fragment_indices.json'],
            [NF / f'{T}_fragment_indices.json'], scripts_dir=ROOT / 'scripts')
        job('copy_decomposition', SLOMD, 'psens_map_fragment_indices.py', [T, 'copy-decomposition'],
            [SDF / f'{T}_fsapt_decomposition.json'], [DF / f'{T}_fsapt_decomposition.json'], scripts_dir=ROOT / 'scripts')
    for c in ('r255', 'r340'):
        job(f'extract_clamped:{c}', SLOMD, 'extract_clamped_fragment.py', [T, c], [RG / f'{T}_{c}_final.rst7'],
            [RG / f'{T}_{c}_geometry.json'])
        job(f'ck_static:{c}', PAULI, 'sapt_clamped_fragment.py', [T, c], [RG / f'{T}_{c}_geometry.json'],
            [RG / f'{T}_{c}_native_result.json'], 4, 6)
        job(f'extract_frames:{c}', SLOMD, 'extract_frames_geometry.py', [T, c],
            [EF / f'{T}_{c}_frames.npy', EF / f'{T}_{c}_colvar.dat'], [EF / f'{T}_{c}_frames_geometry.json'])
        for f in range(40):
            job(f'ck_frame:{c}:{f:02d}', PAULI, 'sapt_one_frame.py', [T, c, f], [EF / f'{T}_{c}_frames_geometry.json'],
                [EF / f'{T}_{c}_f{f:03d}.json'], 4, 6)
        for w in range(8):
            job(f'fsapt:{c}:{w}', PAULI, 'fsapt_ensemble.py', [T, c, 40, f'{w}/8'],
                [EF / f'{T}_{c}_frames_geometry.json', NF / f'{T}_fragment_indices.json', DF / f'{T}_fsapt_decomposition.json'],
                [DF / f'{T}_{c}_fsapt_ensemble_part{w}of8.json'], 1, 6,
                {'PSI4_MEM': '6 GB', 'FSAPT_SCRATCH': str(PS / f'proS/fsapt_scratch_{model}')})
        job(f'fsapt_merge:{c}', PAULI, 'merge_fsapt_shards.py', [T, c, 40],
            [DF / f'{T}_{c}_fsapt_ensemble_part{w}of8.json' for w in range(8)], [DF / f'{T}_{c}_fsapt_ensemble.json'])
    job('transverse_methane', PAULI, 'transverse_proton_hessian.py', [T, 'r255'],
        [RG / f'{T}_r255_geometry.json', RG / f'{T}_r255_native_result.json'], [TH / f'{T}_r255_w015_transverse_hessian.json'], 1, 8)
    job('extract_native_donor', SLOMD, 'extract_native_donor_clamped.py', [T, 'r255'], [RG / f'{T}_r255_geometry.json'],
        [NDV / f'{T}_r255_native_donor.json'])
    halves = (0.05, 0.10, 0.15, 0.20) if model == 'S' else (0.15,)
    for h in halves:
        job(f'transverse_native:w{int(round(h * 100)):03d}', PAULI, 'transverse_native_donor.py', [T, 'r255', h, 5],
            [NDV / f'{T}_r255_native_donor.json'], [NDV / f'{T}_r255_w{int(round(h * 100)):03d}_native_transverse.json'], 1, 8)
    job('total_restoring', PAULI, 'total_restoring_hessian.py', [T], [NDV / f'{T}_r255_native_donor.json'],
        [NDV / f'{T}_r255_w015_total_restoring.json'], 1, 8)
    if model == 'S':
        job('stage4:fragments', SLOMD, 'stage4_fragment_variants.py', [T, 'r255'], [NDV / f'{T}_r255_native_donor.json'],
            [NDV / f'{T}_r255_stage4_fragments.json'])
        pts = []
        for i in range(5):
            for j in range(5):
                out = NDV / 'stage4/points' / f's4_wall_bb_L754A_{i:02d}_{j:02d}.json'
                pts.append(out)
                job(f'stage4:point:{i}{j}', PAULI, 'stage4_point.py',
                    ['s4_wall_bb_L754A', T, 'sapt0', 'jun-cc-pVDZ', 'native', 'backbone', '0.15', '5', i, j, '10 GB'],
                    [NDV / f'{T}_r255_stage4_fragments.json'], [out], 1, 10)
        job('stage4:fit', PAULI, 'stage4_fit.py', ['s4_wall_bb_L754A'], pts, [NDV / 'stage4/s4_wall_bb_L754A.json'])
    job('stage5_extract', SLOMD, 'stage5_extract.py', [T, 16], [CFG / f'{T}_frames.npy', RG / f'{T}_r255_geometry.json'],
        [CFG / T / 'selection.json'])
    return J, CFG


def stage5_jobs(model, CFG):
    sh = PS / ('shadow_SpS' if model == 'S' else 'shadow_PpS')
    J = []
    for f in sorted((CFG / T).glob(f'{T}_cfg*_native_donor.json')):
        out = f.with_name(f.name.replace('_native_donor.json', '_w015_transverse.json'))
        J.append(dict(id=f'{model}:stage5:{f.name[len(T) + 1:len(T) + 6]}', cmd=[PAULI, str(sh / 'scripts/stage5_transverse.py'), str(f)],
                      deps=[f], outs=[out], threads=1, mem=4, env=dict(PAULI_ROOT=str(sh)), cwd=str(sh), tries=0, state='wait'))
    log(f'{model}: stage-5 selection has {len(J)} configurations')
    return J


def main():
    (PS / 'proS/logs').mkdir(parents=True, exist_ok=True)
    jobs, cfg = [], {}
    for m in ('S', 'P'):
        J, CFG = jobs_for(m)
        jobs += J
        cfg[m] = CFG
    added, running = set(), {}
    while True:
        for m in ('S', 'P'):
            if m not in added and (cfg[m] / T / 'selection.json').exists():
                jobs += stage5_jobs(m, cfg[m])
                added.add(m)
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
            env = dict(os.environ, OMP_NUM_THREADS=str(j['threads']), **j['env'])
            lf = open(PS / 'proS/logs' / f'{j["id"].replace(":", "_")}_try{j["tries"]}.log', 'w')
            p = subprocess.Popen(j['cmd'], env=env, cwd=j['cwd'], stdout=lf, stderr=subprocess.STDOUT)
            running[j['id']] = (p, j, time.time())
            j['state'] = 'running'
            used_t += j['threads']
            used_m += j['mem']
        states = [j['state'] for j in jobs]
        if not running and all(s in ('done', 'failed') for s in states) and len(added) == 2:
            break
        time.sleep(15)
    failed = [j['id'] for j in jobs if j['state'] == 'failed']
    log(f'pro-S jobs finished: {states.count("done")} done, {len(failed)} failed {failed[:10]}')


if __name__ == '__main__':
    main()
