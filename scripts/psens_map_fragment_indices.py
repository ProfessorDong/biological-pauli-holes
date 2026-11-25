#!/usr/bin/env python3
"""Model-P copies of model-S near-attack donor files, for the L754A pro-S correction.

  psens_map_fragment_indices.py TAG                       map shadow_SpS fragment indices to model P
  psens_map_fragment_indices.py TAG copy-decomposition    copy the single-snapshot F-SAPT decomposition
Indices are mapped by residue and atom name (psens_shadow.atom_map); the transferring flag is kept.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PS = ROOT / 'results/protonation_sensitivity'
sys.path.insert(0, str(ROOT / 'scripts'))
from psens_shadow import atom_map   # noqa: E402

tag = sys.argv[1]
if len(sys.argv) > 2 and sys.argv[2] == 'copy-decomposition':
    src = PS / 'shadow_SpS/results/sapt_bio/donor_fragment' / f'{tag}_fsapt_decomposition.json'
    shutil.copy(src, PS / 'shadow_PpS/results/sapt_bio/donor_fragment' / src.name)
    sys.exit(0)
fi = json.loads((PS / 'shadow_SpS/results/sapt_bio/native_fragment' / f'{tag}_fragment_indices.json').read_text())
m, S, P = atom_map(tag)
fi['donor_C'], fi['xferH'] = m[fi['donor_C']], m[fi['xferH']]
fi['carbons'] = [m[i] for i in fi['carbons']]
fi['hydrogens'] = [dict(h, idx=m[h['idx']]) for h in fi['hydrogens']]
fi['caps'] = [dict(host=m[c['host']], beyond=m[c['beyond']]) for c in fi['caps']]
fi['_model'] = 'P (indices mapped from model S by residue and atom name)'
assert P.atoms[fi['xferH']].name == 'H25', P.atoms[fi['xferH']].name
(PS / 'shadow_PpS/results/sapt_bio/native_fragment' / f'{tag}_fragment_indices.json').write_text(json.dumps(fi, indent=1))
print('mapped', tag, 'xferH', fi['xferH'], P.atoms[fi['xferH']].name)
