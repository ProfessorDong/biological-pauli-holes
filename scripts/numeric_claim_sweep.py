#!/usr/bin/env python3
"""Trace every distinctive numeric claim in the manuscript back to the file that produced it.

WHY THIS EXISTS
  Five separate defects in this manuscript have now been traced to numbers that
  travelled between the main text and the supplement without anyone re-checking
  the source: a factor-of-two convention error in xi0, a figure plotting the
  methane series while the text quoted the native series, four stale section
  cross-references, a benchmark described as transverse when the scan was
  longitudinal, and a ratio formed between two different second derivatives.
  Each was found by hand. This does the search exhaustively instead.

WHAT IT DOES
  Pulls every number from the LaTeX sources that is distinctive enough to have
  come from a computation rather than from prose (a decimal with two or more
  places, or an integer of three or more digits), then looks for that number in
  the committed result files. A claim that cannot be located is not necessarily
  wrong: it may be rounded, derived, quoted from a paper, or defined rather than
  measured. The output is a WORKLIST, not a verdict.

KNOWN LIMITATION, AND IT IS NOT MINOR
  This is a SUBSTRING matcher, so FOUND is necessary but not sufficient: "6.43"
  matches inside "16.437" and inside "0.64312". A pass therefore means only that
  the digits occur somewhere in the corpus, not that the claim is sourced. Two
  caption numbers passed this way and had to be given a real home in
  derived_constants.json once the coincidence was noticed. Treat FOUND as
  evidence of absence-of-fabrication, not as proof of provenance; a number that
  matters should still be traced by hand to the file that computes it.

WHAT COUNTS AS A HIT
  The number must appear in a result file at the precision quoted, or at one
  more decimal place (so 7.885 matches a stored 7.8851683...). Rounded quotations
  are searched for separately and flagged as ROUNDED rather than FOUND, because
  those are exactly the cases where a stale value hides.

Usage: numeric_claim_sweep.py [--full]
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
TEX = [ROOT / 'prxlife/main.tex', ROOT / 'prxlife/appendices.tex',
       ROOT / 'prxlife/cover-letter.tex']  # supplement.tex retired 2026-09-18, folded into appendices
# md/ and qm/ hold quantum-chemistry outputs that results/ does not mirror, and
# ~/scratch/orca_rerun holds the ORCA runs the project convention keeps out of the
# repo. All three are legitimate provenance and all three must be searched.
DATA_DIRS = [ROOT / 'results', ROOT / 'md', ROOT / 'qm',
             Path.home() / 'scratch']
DATA_EXT = {'.json', '.csv', '.dat', '.txt', '.log', '.out',
            '.engrad', '.xyz', '.input', '.hess'}
MAX_BYTES = 6_000_000

# numbers that are structural rather than measured, and would only add noise
IGNORE = {
    '100', '1000', '2026', '2024', '2022', '2019', '2014', '2005', '2002',
    '1990', '1980', '1979', '1938', '1937', '0.05', '0.01', '1.00', '0.00',
    '283', '313', '298', '273', '360', '106', '108',
}


def load_corpus():
    """Read every committed result file once, as text."""
    corpus = {}
    for d in DATA_DIRS:
        if not d.exists():
            continue
        for f in d.rglob('*'):
            if f.is_file() and f.suffix.lower() in DATA_EXT:
                try:
                    if f.stat().st_size > MAX_BYTES:
                        continue
                    corpus[str(f.relative_to(ROOT))] = f.read_text(errors='ignore')
                except Exception:
                    pass
    return corpus


def strip_latex(t):
    """Remove comments and the bibliography, which are not claims."""
    t = re.sub(r'(?<!\\)%.*', '', t)
    t = re.sub(r'\\cite\{[^}]*\}', ' ', t)
    t = re.sub(r'\\label\{[^}]*\}', ' ', t)
    t = re.sub(r'\\ref\{[^}]*\}', ' ', t)
    t = re.sub(r'\\includegraphics(\[[^]]*\])?\{[^}]*\}', ' ', t)
    return t


NUM = re.compile(r'(?<![\w.])(\d+\.\d{2,}|\d{3,})(?![\w])')


def claims(path):
    t = strip_latex(Path(path).read_text())
    out = []
    for m in NUM.finditer(t):
        v = m.group(1)
        if v in IGNORE:
            continue
        a, b = max(0, m.start() - 85), min(len(t), m.end() + 55)
        ctx = ' '.join(t[a:b].split())
        out.append((v, ctx))
    return out


def find(v, corpus):
    """Exact hit, then a one-more-digit hit, then a rounded hit."""
    exact = [f for f, txt in corpus.items() if v in txt]
    if exact:
        return 'FOUND', exact
    if '.' in v:
        deeper = re.compile(re.escape(v) + r'\d')
        hit = [f for f, txt in corpus.items() if deeper.search(txt)]
        if hit:
            return 'FOUND', hit
        # rounded: drop the last decimal and look for any continuation
        stem = v[:-1]
        if len(stem.split('.')[-1]) >= 1:
            r = re.compile(re.escape(stem) + r'\d')
            hit = [f for f, txt in corpus.items() if r.search(txt)]
            if hit:
                return 'ROUNDED', hit
    return 'UNLOCATED', []


def main():
    full = '--full' in sys.argv
    corpus = load_corpus()
    print(f'corpus: {len(corpus)} result files\n')

    tally = defaultdict(int)
    unlocated = []
    for tex in TEX:
        cl = claims(tex)
        seen = set()
        for v, ctx in cl:
            if v in seen:
                continue
            seen.add(v)
            status, files = find(v, corpus)
            tally[status] += 1
            if status == 'UNLOCATED':
                unlocated.append((tex.name, v, ctx))
            elif status == 'ROUNDED' or full:
                print(f'  [{status:8s}] {v:>12s}  {files[0]}')
        print(f'{tex.name}: {len(seen)} distinct distinctive numbers')

    print(f'\nFOUND {tally["FOUND"]}   ROUNDED {tally["ROUNDED"]}   '
          f'UNLOCATED {tally["UNLOCATED"]}')
    print(f'\n{"="*78}\nUNLOCATED WORKLIST (each needs a source, or a reason it has none)\n{"="*78}')
    for f, v, ctx in unlocated:
        print(f'\n  {v}   [{f}]')
        print(f'    ...{ctx}...')

    out = ROOT / 'results' / 'numeric_claim_sweep.json'
    out.write_text(json.dumps(
        dict(n_files=len(corpus), tally=dict(tally),
             unlocated=[dict(file=f, value=v, context=c) for f, v, c in unlocated]),
        indent=1))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
