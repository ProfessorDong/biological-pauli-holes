# Biological Pauli Holes — analysis code and numerical results

Code and numerical results for a study of whether Pauli confinement of a
transferring hydrogen can contribute to enzyme catalysis, tested in soybean
lipoxygenase (SLO).

**Authors:** Liang Dong (corresponding, Liang_Dong@baylor.edu), Baylor
University and UT Southwestern; Jun-Han You, Shanghai Jiao Tong University.

> **Status: private repository, pre-publication.** The associated manuscript is
> under preparation for submission. Please do not redistribute.

## What is here

| Path | Contents |
|---|---|
| `scripts/` | All analysis code: SAPT campaigns, the confined-hydrogen solver, the range-weight budget, the transverse Hessian, the stage-4/5 sensitivity work, and the verification suite. |
| `results/**/*.json` | The numerical results the paper's claims are checked against, 1,000+ files. |
| `md/` | Molecular-dynamics build inputs (leap scripts, frcmod, mol2). |
| `setup/` | Environment setup for the QM stack. |

Two scripts are worth knowing about:

- `scripts/verify_document_consistency.py` — 138 checks tying statements in the
  manuscript to named fields of named result files.
- `scripts/verify_figures.py` — 95 checks tying each figure to its generator and
  its underlying data.

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
