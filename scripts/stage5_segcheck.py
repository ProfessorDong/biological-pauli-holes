#!/usr/bin/env python3
"""Did segmenting the MD perturb the sampled ensemble?

Chunking re-minimizes and reassigns velocities at each boundary; the original continuous
3 ns runs did not. The test statistic is the spread of the three sub-run means of r_DA.
Comparing that to its own SEM is not enough, because a continuous trajectory drifts too,
so the control is the five CONTINUOUS runs cut into equal thirds and put through the
identical statistic. If the chunked spread sits inside the continuous spread, segmenting
added nothing.
"""
import sys, numpy as np, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from stage5_extract import acf_tau

OUT = pathlib.Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio/results/stage5_configs')
CHUNKED, CONTINUOUS = ['I538A', 'L546A'], ['WT', 'I553A', 'I552A', 'L754A', 'V750A']

def stats(parts, name, kind):
    means = np.array([p.mean() for p in parts])
    allc = np.concatenate(parts)
    sem = np.mean([p.std()/np.sqrt(max(1.0, len(p)/acf_tau(p))) for p in parts])
    print(f'  {name:7s} [{kind:10s}] thirds = ' + ' '.join(f'{m:.3f}' for m in means) +
          f'  spread={means.std():.4f}  SEM={sem:.4f}  tau_cat={acf_tau(allc):5.2f}  '
          f'<r>={allc.mean():.3f}')
    return means.std(), sem

print('Chunked systems, real segment boundaries:')
chunk = [stats([np.loadtxt(s) for s in sorted(OUT.glob(f'{t}_seg*_colvar.dat'))], t, 'segmented')
         for t in CHUNKED]
print('\nControl: continuous runs cut into thirds (no boundary exists here):')
cont = [stats(np.array_split(np.loadtxt(OUT / f'{t}_colvar.dat'), 3), t, 'continuous')
        for t in CONTINUOUS]

cs = np.array([c[0] for c in chunk]); ks = np.array([c[0] for c in cont])
print(f'\n  between-third spread: chunked {cs.min():.4f}-{cs.max():.4f} A, '
      f'continuous {ks.min():.4f}-{ks.max():.4f} A')
if cs.max() <= ks.max():
    print('  -> chunked spread lies INSIDE the continuous range: segmenting added no detectable '
          'perturbation.')
else:
    print(f'  -> chunked spread EXCEEDS the continuous maximum. Boundary effect not excluded; '
          f'do not merge these systems into the panel without further checks.')
