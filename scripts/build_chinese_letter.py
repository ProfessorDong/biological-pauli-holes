#!/usr/bin/env python3
"""Build a Chinese letter PDF from its markdown source, reproducibly.

WHY THIS EXISTS
  The letters to Prof. You are drafted in markdown and typeset with ctexart + XeLaTeX. That
  conversion had been done ad hoc each time, which invites two failure modes that have both
  already happened once: editing the markdown and rebuilding the STALE .tex, so the PDF
  silently lacks the new material; and hand-escaping CJK punctuation differently on each
  pass. This makes the .tex a build artefact of the .md, so the markdown is the single source.

  Requires Noto Serif CJK SC and xelatex. Two passes, because a one-pass ctexart build can
  leave CJK line breaking unsettled.

Usage: build_chinese_letter.py <name-without-extension>   (pauli env; run from prxlife/)
       e.g.  build_chinese_letter.py 致尤教授-关于泡利口袋尺寸
"""
import re
import subprocess
import sys
from pathlib import Path

PREAMBLE = r"""\documentclass[12pt,a4paper]{ctexart}
\usepackage{amsmath,amssymb}\usepackage{booktabs}
\usepackage[margin=2.5cm]{geometry}
\setCJKmainfont{Noto Serif CJK SC}
\linespread{1.28}\setlength{\parskip}{5pt}\setlength{\parindent}{0pt}
\punctstyle{quanjiao}
\begin{document}"""

SUBS = [('→', r'$\rightarrow$'), ('≈', r'$\approx$'), ('±', r'$\pm$'), ('×', r'$\times$'),
        ('~', r'$\sim$'), ('ξ₀', r'$\xi_0$'), ('a₀', r'$a_0$'), ('Å', r'\AA{}'),
        ('Nε2', r'N$\epsilon$2'), ('λ', r'$\lambda$'), ('≥', r'$\ge$'), ('≤', r'$\le$'),
        ('°', r'$^\circ$'), ('·', r'$\cdot$'), ('ν₁', r'$\nu_1$'), ('α', r'$\alpha$'),
        ('∇', r'$\nabla$'), ('−', r'$-$'), ('δ', r'$\delta$'), ('ε', r'$\epsilon$'),
        ('μ', r'$\mu$'), ('σ', r'$\sigma$'), ('Δ', r'$\Delta$'), ('π', r'$\pi$'),
        ('κ', r'$\kappa$'), ('ρ', r'$\rho$'), ('θ', r'$\theta$'), ('ω', r'$\omega$'),
        ('√', r'$\surd$'), ('∝', r'$\propto$'), ('⟨', r'$\langle$'), ('⟩', r'$\rangle$')]


def esc(t):
    t = t.replace('\\', r'\textbackslash{}').replace('&', r'\&').replace('%', r'\%')
    t = t.replace('_', r'\_').replace('#', r'\#').replace('$', r'\$')
    t = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', t)
    t = re.sub(r'`(.+?)`', r'\\texttt{\1}', t)
    # scientific notation written plainly in the markdown (10^5, 10^-8, 2.7 x 10^5) is a
    # LaTeX error outside math mode, so lift it into math before anything else sees it
    t = re.sub(r'(\d)\^\{?(-?\d+)\}?', r'\1$^{\2}$', t)
    t = t.replace('^', r'\textasciicircum{}')   # any bare caret left is literal
    for a, b in SUBS:
        t = t.replace(a, b)
    return t


def convert(md):
    lines, out, i = md.split('\n'), [], 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if ln.startswith('# '):
            out.append(r'\begin{center}\Large\bfseries ' + esc(ln[2:])
                       + r'\end{center}\vspace{4pt}')
        elif ln.startswith('## '):
            out.append(r'\section*{' + esc(ln[3:]) + '}')
        elif ln.strip() == '---':
            out.append(r'\vspace{6pt}\hrule\vspace{6pt}')
        elif ln.startswith('|'):
            tbl = []
            while i < len(lines) and lines[i].startswith('|'):
                tbl.append(lines[i])
                i += 1
            rows = [[c.strip() for c in r.strip('|').split('|')]
                    for r in tbl if not set(r) <= set('|-: ')]
            n = len(rows[0])
            out.append(r'\begin{center}\begin{tabular}{' + 'l' * n + r'}\toprule')
            out.append(' & '.join(esc(c) for c in rows[0]) + r' \\ \midrule')
            for r in rows[1:]:
                out.append(' & '.join(esc(c) for c in r) + r' \\')
            out.append(r'\bottomrule\end{tabular}\end{center}')
            continue
        elif ln.startswith('> '):
            out.append(r'\begin{quote}' + esc(ln[2:]) + r'\end{quote}')
        else:
            out.append(esc(ln))
        i += 1
    return PREAMBLE + '\n' + '\n'.join(out) + '\n\\end{document}\n'


def main():
    name = sys.argv[1]
    md = Path(f'{name}.md')
    assert md.exists(), f'{md} not found; run from the directory holding the markdown'
    Path(f'{name}.tex').write_text(convert(md.read_text()))
    for _ in range(2):
        r = subprocess.run(['xelatex', '-interaction=nonstopmode', f'{name}.tex'],
                           capture_output=True, text=True)
    pdf = Path(f'{name}.pdf')
    assert pdf.exists(), f'xelatex produced no PDF:\n{r.stdout[-2000:]}'
    n = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True).stdout
    pages = re.search(r'Pages:\s+(\d+)', n).group(1)
    errs = [l for l in r.stdout.split('\n') if l.startswith('!')]
    print(f'{pdf}: {pages} pages, {len(errs)} LaTeX errors')
    for e in errs[:5]:
        print('  ', e)
    sys.exit(len(errs))


if __name__ == '__main__':
    main()
