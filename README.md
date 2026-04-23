# Interpretable-fingerprint
This repository provides the code, source datasets, and run instructions for the manuscript:

**Interpretable Fingerprint-Based Prediction of Power Conversion Efficiency for Donor-Acceptor Pairs in Organic Solar Cells with External Fingerprint-Only Data Integration**

## Repository scope

This repository is intended for:
- reviewers who need a direct reproducibility path;
- readers who want to inspect the workflow step by step;
- users who want to rerun the baseline, radius-control, Gao-only, and joint in-house + Gao branches.

## Source datasets

The public workflow is organized around the following source files:

- `D-A.csv`
- `Donor.csv`
- `Acceptor.csv`
- `gao_fd_fp.npy`
- `gao_fa_fp.npy`
- `gao_fp_Y.npy`

The refactored `Baseline.py` works directly around `D-A.csv`, `Donor.csv`, and `Acceptor.csv` and no longer depends on the older internal filenames.

## Code modules

- `Baseline.py` — modules **A**, **B**, and **E**
- `Gao.py` — module **C**
- `Combine.py` — module **D**
- `Explanation.py` — interpretation and feature-level analysis

## Workflow map

| Module | Script | Purpose |
|---|---|---|
| A | `Baseline.py` | Main interpretable baseline (`main_r3`) |
| B | `Baseline.py` | Radius-control branch (`core_r2`) |
| C | `Gao.py` | Gao-only transfer branch |
| D | `Combine.py` | Joint in-house + Gao TextCNN(FPtand) branch |
| E | `Baseline.py` | Split × seed × preset stability search |

## Recommended execution order

1. Prepare the in-house fingerprint arrays from the source datasets
2. Run the split/seed search
3. Re-run the best `main_r3` configuration
4. Re-run the best `core_r2` configuration
5. Run the Gao-only branch
6. Run the joint in-house + Gao branch

## Commands

### Step 0 — prepare in-house fingerprint arrays

```bash
python Baseline.py --dataset "D-A.csv" --donor-col donor_smiles --acceptor-col acceptor_smiles --pce-col PCE --auto-prepare-only
```

This step prepares the downstream fingerprint files, including:

- `fd_fp1.npy`
- `fa_fp1.npy`
- `fd_fp_augmented.npy`
- `fa_fp_augmented.npy`
- `fp_Y_augmented.npy`

### Step E — split × seed × preset search

```bash
python Baseline.py --main-search 0 100 --output-tag E_seed_search
```

### Step A — main interpretable baseline

```bash
python Baseline.py --run-best-main --output-tag A_main_r3
```

### Step B — radius-control branch

```bash
python Baseline.py --run-best-core-r2 --output-tag B_core_r2
```

### Step C — Gao-only transfer branch

Run the Gao-only script in an isolated subfolder so that its fixed input filenames do not interfere with other steps.

```bash
mkdir C_gao_only
cp Gao.py C_gao_only/
cp gao_fd_fp.npy C_gao_only/fd_fp.npy
cp gao_fa_fp.npy C_gao_only/fa_fp.npy
cp gao_fp_Y.npy C_gao_only/fp_Y.npy
cd C_gao_only
python Gao.py
```

### Step D — joint in-house + Gao branch

```bash
python Baseline.py --run-best-core-r2-plus-gao --gao-fd-path gao_fd_fp.npy --gao-fa-path gao_fa_fp.npy --gao-y-path gao_fp_Y.npy --output-tag D_core_r2_plus_gao
```

If `Combine.py` is used as the dedicated joint TextCNN(FPtand) runner, feed it the merged arrays produced in the previous step.

## Suggested repository structure

```text
.
├── README.md
├── Baseline.py
├── Gao.py
├── Combine.py
├── Explanation.py
├── D-A.csv
├── Donor.csv
├── Acceptor.csv
├── gao_fd_fp.npy
├── gao_fa_fp.npy
├── gao_fp_Y.npy
├── docs/
│   ├── index.md
│   ├── run_commands.md
│   └── _config.yml
└── results/
```

## Suggested GitHub Pages URL

If your GitHub username is `YOUR_USERNAME`, a clean public link would be:

`https://YOUR_USERNAME.github.io/osc-pce-fingerprint-workflow/`

## Notes for reviewers and readers

This public-facing workflow is intentionally described using the original source datasets rather than old intermediate filenames. Intermediate arrays may still be generated automatically during execution, but they are implementation details rather than the main user entry points.
