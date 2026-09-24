# Biological Pauli Holes — analysis code and numerical results

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22927663.svg)](https://doi.org/10.5281/zenodo.22927663)

Code and numerical results for a study of whether Pauli confinement of a
transferring hydrogen can contribute to enzyme catalysis, tested in soybean
lipoxygenase (SLO).

**Authors:** Liang Dong (corresponding, Liang_Dong@baylor.edu), Baylor
University and UT Southwestern; Jun-Han You, Shanghai Jiao Tong University.

> **Status: pre-publication.** The associated manuscript is in preparation for
> submission; the paper reference will be added on acceptance. The code and
> numerical results here are released under the licences below, so you are free
> to use and redistribute them under those terms. If you use them before the
> paper appears, please cite this repository via `CITATION.cff`.

## What is here

| Path | Contents |
|---|---|
| `scripts/` | All analysis code: SAPT campaigns, the confined-hydrogen solver, the range-weight budget, the transverse Hessian, the stage-4/5 sensitivity work, and the verification suite. |
| `results/**/*.json` | The numerical results the paper's claims are checked against, 1,000+ files. |
| `md/` | Molecular-dynamics build inputs (leap scripts, frcmod, mol2). |
| `setup/` | Environment setup for the QM stack. |

Two scripts are worth knowing about:

- `scripts/verify_document_consistency.py` — 156 checks tying statements in the
  manuscript to named fields of named result files.
- `scripts/verify_figures.py` — 95 checks tying each figure to its generator and
  its underlying data.

**Both need the manuscript sources and figures, which this repository does not
contain.** The manuscript is not published here: the paper itself is the
published artefact, and this repository is its code and numbers. The document
checks therefore report `SKIP` on a bare checkout and say plainly that nothing
was verified, rather than printing a passing-looking score. To run them, point
`PAULI_ROOT` at a tree that has the sources in `prxlife/` (see Paths below).

Everything that reads only `results/` runs from a bare checkout as-is, and
reproduces the committed numbers exactly.

## Paths

Scripts locate the repository from their own location, so a checkout works
anywhere. To point them at a different tree, set `PAULI_ROOT`:

    PAULI_ROOT=/path/to/tree python scripts/verify_document_consistency.py

That is how to run the manuscript checks if you have the manuscript sources
separately: put them in `prxlife/` under the tree `PAULI_ROOT` names.

The simulation and quantum-chemistry drivers additionally need interpreters and
external programs. Those are read from the environment, with this machine's
paths as fallbacks, so override whichever you need:

| Variable | What it points at |
|---|---|
| `PAULI_ROOT` | the repository tree |
| `PAULI_PYTHON` | Python for analysis (psi4, ASE, MDAnalysis, statsmodels) |
| `SLOMD_PYTHON` | Python for MD (AmberTools, OpenMM, parmed, pymbar) |
| `ORCA_DIR`, `OPENMPI_DIR`, `MULTIWFN_DIR` | external program installations |
| `PSI4_FSAPT` | Psi4's `share/psi4/fsapt`; defaults to the running interpreter's |

## What is deliberately not here

This repository is code and numbers only. Excluded, by category:

- **Manuscript sources and figures.** Held separately.
- **Large primary data** — MD trajectories, Amber topologies and restarts,
  ORCA `.gbw` wavefunctions and outputs, bead-frame arrays (~7 GB). These
  exceed GitHub's limits and will be deposited on Zenodo, which mints a DOI.
- **Published literature.** Third-party copyright; consult the citations.
- **Third-party software and benchmark sets** (ORCA, i-pi, GMTKN55,
  ModSeminario_Py). Obtain these from their own distributors under their own
  licences.
- **Superseded runs.** Several campaigns were invalidated during the work and
  their numbers are withheld so they cannot be mistaken for current results.
  The scripts that produced them are kept, because the paper names these runs
  as superseded and the code is part of that record. Superseded: the 5-point /
  multiplicity-5 / charge +2 proton scans (`pcet_B34*`, `pcet_pilot`), and
  V750A results predating the restraint repair. The production proton scan is
  `results/pcet_reactant_v2` (15 points, charge +1, multiplicity 6).

## Reproducing

Two conda environments are used:

- `pauli` (Python 3.12): psi4, ASE, MDAnalysis, statsmodels, rdkit, prody,
  scikit-learn, pdb2pqr, xtb — analysis and energy decomposition.
- `slomd` (Python 3.13): AmberTools 24, OpenMM 8.5.2 (CUDA), parmed, mdtraj,
  pymbar, propka — molecular dynamics and PMF analysis.

Invoke the environment's interpreter directly, for example
`~/anaconda3/envs/pauli/bin/python scripts/verify_document_consistency.py`.

Analyses that read only the JSON files in `results/` run from this repository
alone. Anything that regenerates a trajectory or a wavefunction needs the
Zenodo deposit and the QM stack in `setup/`.

## Licence

Code is MIT (`LICENSE`); the numerical results under `results/` are CC BY 4.0
(`LICENSE-DATA`). Neither covers the excluded third-party material above.

## Citing

See `CITATION.cff`. The paper reference will be added on acceptance.
