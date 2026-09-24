#!/usr/bin/env python3
"""Stage 5, phase B: choose configurations beyond the autocorrelation time and extract fragments.

WHY THIS EXISTS
  The coordinate result of the paper is computed on ONE configuration per system, and its quoted
  uncertainties are fit uncertainties from the least-squares covariance of a single quadratic.
  The paper's own ensemble section shows that a single configuration is a poor estimator of the
  wall stiffness: the published single-configuration k_exch values sit -1.09 to +0.58 standard
  deviations from their ensemble means. So the sign classification of K_perp is currently quoted
  against the wrong uncertainty. This supplies the right one.

WHAT IT DOES
  For each system, reads the freshly sampled trajectory frames, measures the autocorrelation time
  of the hydrogen-to-nearest-wall distance, selects configurations spaced beyond it, and writes
  one fragment file per configuration in the format transverse_native_donor.py already consumes:
  native C5H8 donor, native wall fragment, acceptor oxygen. Every chemical assertion of the
  single-configuration extractor is repeated per frame.

Usage: stage5_extract.py TAG [n_configs]     (slomd env: parmed + openmm)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
import sys
from pathlib import Path

import numpy as np
import parmed

ROOT = Path(_REPO)
MD, CFG = ROOT / 'md/mcpb', ROOT / 'results/stage5_configs'
RG = ROOT / 'results/reactive_geometry'
LEU = ['CB', 'HB2', 'HB3', 'CG', 'HG', 'CD1', 'HD11', 'HD12', 'HD13',
       'CD2', 'HD21', 'HD22', 'HD23']
ASN = ['CB', 'HB2', 'HB3', 'CG', 'OD1', 'ND2', 'HD21', 'HD22']


def neighbours(a):
    return [(b.atom2 if b.atom1 is a else b.atom1) for b in a.bonds]


def acf_tau(x):
    """Integrated autocorrelation time, in frames, by the initial-positive-sequence estimator."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    c = np.correlate(x, x, 'full')[n - 1:] / (np.arange(n, 0, -1) * np.var(x))
    tau, k = 1.0, 1
    while k < n - 1 and c[k] > 0:
        tau += 2 * c[k] * (1 - k / n)
        k += 1
    return max(1.0, tau)


def donor_fragment(prm, pos, dC, xH):
    flank = [o for o in neighbours(dC) if o.element_name == 'C']
    assert len(flank) == 2
    keep, caps = [dC], []
    for f in flank:
        assert len([o for o in neighbours(f) if o.element_name == 'H']) == 1
        far = [o for o in neighbours(f) if o.element_name == 'C' and o is not dC][0]
        assert len([o for o in neighbours(far) if o.element_name == 'H']) == 1
        keep += [f, far]
        caps.append((far, [o for o in neighbours(far)
                           if o.element_name == 'C' and o is not f][0]))
    at = []
    for c in keep:
        at.append((c.element_name, pos[c.idx], c.name))
        at += [('H', pos[o.idx], o.name) for o in neighbours(c) if o.element_name == 'H']
    for far, beyond in caps:
        v = pos[beyond.idx] - pos[far.idx]
        at.append(('H', pos[far.idx] + v / np.linalg.norm(v) * 1.09, f'Hcap{far.name}'))
    nC = sum(e == 'C' for e, _, _ in at)
    nH = sum(e == 'H' for e, _, _ in at)
    assert (nC, nH) == (5, 8), f'C{nC}H{nH}'
    P = np.array([p for _, p, _ in at])
    dmin = min(np.linalg.norm(P[i] - P[j]) for i in range(len(P)) for j in range(i + 1, len(P)))
    assert dmin > 0.85, f'dmin {dmin:.3f}'
    assert any(n == xH.name for _, _, n in at), 'transferring H dropped'
    return at, dmin


def main():
    tag = sys.argv[1]
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    src = (ROOT / 'scripts/extract_native_fragment.py').read_text()
    ns = {}
    exec(src[src.index('ARCH = '):src.index('}\n', src.index('SYSTEMS = {')) + 1], {}, ns)
    info = ns['SYSTEMS'][tag]
    base = json.loads((RG / f'{tag}_r255_geometry.json').read_text())
    acc_idx = base['acceptor_idx']

    prm = parmed.load_file(str(MD / f'{info["prm"]}.prmtop'))
    frames = np.load(CFG / f'{tag}_frames.npy')
    dC, xH = prm.atoms[info['donor_C_idx']], prm.atoms[info['xferH_idx']]
    wres = next(r for r in prm.residues
                if r.name == info['wall_res_name'] and r.number == info['wall_res_id'])
    keep = LEU if info['wall_res_name'] == 'LEU' else ASN
    byname = {a.name: a for a in wres.atoms}
    CA = byname.get('CA')

    # autocorrelation of the hydrogen-to-nearest-wall distance
    rhw = []
    for p in frames:
        w = np.array([p[byname[n].idx] for n in keep if n in byname])
        rhw.append(np.min(np.linalg.norm(w - p[xH.idx], axis=1)))
    tau = acf_tau(rhw)
    stride = max(1, int(np.ceil(tau)))
    sel = list(range(0, len(frames), stride))[:want]
    print(f'{tag}: {len(frames)} frames, r_HW tau = {tau:.2f} frames '
          f'({tau*50:.0f} ps), stride {stride}, taking {len(sel)}')

    out = CFG / tag
    out.mkdir(exist_ok=True)
    for k, fi in enumerate(sel):
        pos = frames[fi].astype(float)
        d = float(np.linalg.norm(pos[dC.idx] - pos[xH.idx]))
        assert 1.0 < d < 1.2, f'frame {fi}: C-H {d:.3f}'
        frag, dmin = donor_fragment(prm, pos, dC, xH)
        wall = []
        for n in keep:
            if n in byname:
                a = byname[n]
                wall.append(dict(element='H' if n.startswith('H') else n[0],
                                 position=pos[a.idx].tolist(), name=n))
        if CA is not None:
            v = pos[byname['CB'].idx] - pos[CA.idx]
            wall.append(dict(element='H',
                             position=(pos[byname['CB'].idx] - v / np.linalg.norm(v) * 1.09).tolist(),
                             name='HcapCB'))
        rec = dict(TAG=tag, clamp='r255', config=k, source_frame=int(fi),
                   wall_residue=f'{info["wall_res_name"]}{info["wall_res_id"]}',
                   donor_C=pos[dC.idx].tolist(), xferH=pos[xH.idx].tolist(),
                   xferH_name=xH.name, acceptor_O=pos[acc_idx].tolist(),
                   acceptor_idx=acc_idx, native_wall=wall,
                   donor_fragment=[dict(element=e, position=p.tolist(), name=n)
                                   for e, p, n in frag],
                   donor_fragment_formula='C5H8', r_CH=d, dmin=dmin,
                   r_DA=float(np.linalg.norm(pos[acc_idx] - pos[dC.idx])),
                   tau_frames=float(tau), stride=stride)
        (out / f'{tag}_cfg{k:02d}_native_donor.json').write_text(json.dumps(rec, indent=2))
    (out / 'selection.json').write_text(json.dumps(
        dict(tag=tag, n_frames=len(frames), tau_frames=float(tau), frame_spacing_ps=50,
             tau_ps=float(tau * 50), stride=stride, selected=sel,
             r_HW_mean=float(np.mean(rhw)), r_HW_sd=float(np.std(rhw, ddof=1))), indent=2))
    print(f'  wrote {len(sel)} configurations to {out}')


if __name__ == '__main__':
    main()
