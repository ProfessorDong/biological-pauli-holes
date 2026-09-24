#!/usr/bin/env python3
"""Emit the two endpoint table bodies from the data, and check what the manuscript prints.

WHY THIS EXISTS
  Both tables were typed by hand. When the V750A restraint was repaired the adjacent statistical
  paragraphs were corrected and the tables were not, so the appendix printed a curvature of
  0.135 N/m where the corrected file holds 8.469. A cell-by-cell comparison then found three
  further defects that no sweep had caught, all single-digit rounding: L754A 0.507 against 0.506,
  WT 15.070 against 15.069, I553A 10.000 against 10.002. A table is a claim per cell, and the only
  reliable way to keep it true is to generate it.

  --check compares every printed cell against its source and exits non-zero on any disagreement.
  --emit prints the LaTeX row bodies to paste or \\input.

Usage: generate_endpoint_tables.py [--check | --emit]     (pauli env)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(_REPO)
RG, EF = ROOT / 'results/reactive_geometry', ROOT / 'results/ensemble_fluctuation'
TEX = ROOT / 'prxlife/appendices.tex'
# printed order, which is by ascending isotope effect
SYS = ['V750A', 'WT', 'I552A', 'I538A', 'L754A', 'L546A', 'I553A']
KIE = {'WT': 66, 'V750A': 62, 'I552A': 66, 'I538A': 100, 'L754A': 106, 'L546A': 131, 'I553A': 148}


def pad(x, width=6):
    """Match the table's \\phantom{1} alignment for values narrower than the column."""
    s = f'{x:.3f}'
    return '\\phantom{1}' + s if len(s) < width else s


def static_rows():
    k255 = json.loads((RG / 'kexch_r255.json').read_text())
    k340 = json.loads((RG / 'kexch_r340.json').read_text())
    out = []
    for s in SYS:
        r255 = float(np.loadtxt(RG / f'{s}_r255_colvar.dat').mean())
        r340 = float(np.loadtxt(RG / f'{s}_r340_colvar.dat').mean())
        out.append((s, [f'{k255[s]:.3f}', f'{r255:.3f}', f'{k340[s]:.3f}', f'{r340:.3f}']))
    return out


def ensemble_rows():
    out = []
    for s in SYS:
        v, n = [], []
        for c in ('r255', 'r340'):
            d = json.loads((EF / f'{s}_{c}_kexch_frames.json').read_text())
            a = np.array([x for x in d['k_exch_Nm'] if x is not None and np.isfinite(x)])
            v += [a.mean(), a.std(ddof=1)]
            n.append(len(a))
        out.append((s, [f'{v[0]:.3f}', f'{v[1]:.3f}', f'{v[2]:.3f}', f'{v[3]:.3f}',
                        f'{n[0]}/{n[1]}']))
    return out


def printed(ncols, label):
    """Find the table by its \\label and read the variant rows that follow it."""
    lines = TEX.read_text().splitlines()
    start = next((i for i, ln in enumerate(lines) if f'\\label{{{label}}}' in ln), None)
    assert start is not None, f'table label {label} not found'
    got = {}
    for ln in lines[start:start + 60]:
        if '\\end{tabular}' in ln:
            break
        s = ln.split('&')[0].strip()
        if s in SYS and ln.count('&') == ncols:
            got[s] = [c.strip().replace('\\phantom{1}', '').replace('\\\\', '').strip()
                      for c in ln.split('&')][2:2 + ncols - 1]
    assert len(got) == len(SYS), f'{label}: found {len(got)} of {len(SYS)} rows'
    return got


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    if mode == '--emit':
        print('% Table: static separation curvature at the two clamps')
        for s, c in static_rows():
            print(f'{s:<5} & {KIE[s]:>3} & {pad(float(c[0]))} & {c[1]} & '
                  f'{pad(float(c[2]))} & {c[3]} \\\\')
        print('\n% Table: ensemble mean and spread at the two clamps')
        for s, c in ensemble_rows():
            print(f'{s:<5} & {KIE[s]:>3} & {c[0]} & {c[1]} & {pad(float(c[2]))} & '
                  f'{c[3]} & {c[4]} \\\\')
        total = sum(int(x) for _, c in ensemble_rows() for x in c[4].split('/'))
        print(f'\n% total finite ensemble calculations: {total}')
        return 0

    bad = 0
    for label, rows, ncols, tabel in [
            ('static', static_rows(), 5, 'tab:reactgeom'),
            ('ensemble', ensemble_rows(), 6, 'tab:ensfluct')]:
        got = printed(ncols, tabel)
        print(f'\n{label} table')
        for s, want in rows:
            g = got.get(s)
            ok = g == want
            bad += not ok
            print(f'  [{"OK " if ok else "BAD"}] {s:<6} printed {g}  data {want}')
    total = sum(int(x) for _, c in ensemble_rows() for x in c[4].split('/'))
    body = TEX.read_text() + (ROOT / 'prxlife/main.tex').read_text()
    ok_total = str(total) in body
    bad += not ok_total
    print(f'\n  [{"OK " if ok_total else "BAD"}] total finite calculations = {total} present in text')
    print(f'\n  {"all cells agree" if not bad else f"{bad} disagreements"}')
    return bad


if __name__ == '__main__':
    sys.exit(main())
