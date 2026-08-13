# Interpretable, Leakage-Controlled Fingerprint Prediction of Organic Solar-Cell Donor-Acceptor Pair Efficiency

This repository contains the complete revised analysis workflow, processed inputs, exact split definitions, leakage/collision audits, model outputs, interpretation archives, figure source data, and graphical abstract supporting the manuscript:

**Interpretable, Leakage-Controlled Fingerprint Prediction of Organic Solar-Cell Donor-Acceptor Pair Efficiency with External Fingerprint-Only Data Integration**

The revision separates two scientifically distinct tasks:

1. **Structure-resolved interpretation and chemical extrapolation diagnostics** using the in-house donor-acceptor dataset.
2. **Prediction-oriented fingerprint-only integration** using the in-house radius-2 representation together with 535 Gao fingerprint-only records.

The joint Gao + in-house branch is an **internal prediction branch**, not an independent external-validation set.

## Revised study design at a glance

```text
1,076 structure-available literature records
                |
                v
836 unique in-house D-A pairs
                |
        strict scaffold control
       /                       \
20 scaffold-disjoint       101 scaffold-overlap
hold-out pairs             quarantine pairs
       \                       /
        -------- excluded -----
                |
                v
715 in-house development pairs
       |                          \
       | radius-3                 \ radius-2 alignment
       v                           v
Random Forest + SHAP       715 in-house + 535 Gao
interpretation branch            = 1,250 samples
       |                           |
       v                           v
strict scaffold test       role-aware TextCNN
                           fixed + nested evaluation
```

## Main revised findings

- The structure-resolved source contains **1,076 records** and **836 unique donor-acceptor pairs** after pair-level aggregation.
- A fixed **20-pair donor- and acceptor-scaffold-disjoint hold-out** was constructed before model development. An additional **101 pairs** sharing either donor or acceptor scaffold with the hold-out were quarantined, leaving **715 development pairs**.
- The validation-selected radius-3 Random Forest achieved **Pearson r = 0.6257, R² = 0.3655, RMSE = 2.1760, MAE = 1.5760** on its 143-pair target-blind fixed test set.
- On the strict 20-pair scaffold-disjoint hold-out, performance dropped to **r = 0.2037, R² = -0.0878, RMSE = 2.6887, MAE = 2.2966**, with wide bootstrap intervals. The manuscript therefore does **not** claim broad scaffold-level generalization.
- Radius-2 alignment combines **715 in-house + 535 Gao = 1,250 samples** for prediction-oriented integration.
- The role-aware 101-model joint TextCNN ensemble achieved **r = 0.8326, Spearman rho = 0.7993, R² = 0.6872, RMSE = 2.2578, MAE = 1.6266** on the fixed target-blind joint test set.
- Across ten target-blind split-specific five-model ensembles, the joint branch gave **r = 0.8364 ± 0.0193, R² = 0.6963 ± 0.0337, RMSE = 2.2628 ± 0.1527, MAE = 1.7282 ± 0.1002**.
- Partition choice contributes more variability than model initialization for the principal predictive metrics.
- Exact overlap/collision auditing found **no cross-source exact duplicate complete paired fingerprint or complete sequence representation**. Two cross-source exact-overlap groups occur only at the acceptor-fingerprint level.
- **49 complete paired-representation groups (111 samples)** are duplicated within sources; 48 contain heterogeneous PCE labels. These groups are kept intact across train/validation/test boundaries.
- No FPtand sequence is truncated at `max_len = 200`. The role-aware representation uses disjoint donor/acceptor token namespaces and a separator.
- SHAP interpretation is restricted to the structure-resolved radius-3 Random Forest and uses one predefined top-10-by-mean-|SHAP| rule consistently for feature ranking, correlation analysis, and mapped-environment interpretation.

## Repository structure

```text
.
├── README.md
├── CITATION.cff
├── DATA_DESCRIPTION.md
├── REPRODUCIBILITY_STATUS.md
├── RELEASE_NOTES_v2.0-review-revision.md
├── RUN_ENVIRONMENT.json
├── CHECKSUMS.sha256
├── FILE_INDEX.csv
├── requirements.txt
├── requirements-lock.txt
├── environment.yml
│
├── Baseline.py
├── Combine.py
├── common_utils.py
├── 00_prepare_datasets.py
├── 01_run_101_seeds.py
├── 02_summarize_results.py
├── 03_make_submission_figures.py
├── 04_shap_interpretation.py
├── 05_export_split_indices.py
├── 06_aggregation_sensitivity.py
├── make_graphical_abstract.py
│
├── config_fixed_structure_only_roleaware.json
├── config_fixed_structure_only_legacy.json
├── config_nested_roleaware.json
├── config_legacy_hspxy_benchmark.json
├── run_revision_workflow.sh
├── run_revision_workflow.ps1
│
├── data/
├── baseline/
├── prepared_data/
├── split_indices/
├── aggregation_sensitivity/
├── results_roleaware/
├── results_legacy/
├── interpretation/
├── figures_roleaware/
└── graphical_abstract/
```

See `DATA_DESCRIPTION.md` for file-level interpretation.

## Input data

The revised repository package is self-contained for the reported analysis:

```text
data/D-A.csv
data/gao_fd_fp.npy
data/gao_fa_fp.npy
data/gao_fp_Y.npy
```

- `D-A.csv` contains the **1,076 structure-available in-house records** used before pair-level aggregation.
- The Gao files contain the **535 retained fingerprint-only records** used in the radius-2 prediction branch.

The external Gao records are used for **prediction-oriented integration only** because complete donor/acceptor structures are not uniformly recoverable in the present implementation.

## Exact train/validation/test archive

Reviewer-level reproducibility files are provided under `split_indices/`.

### Baseline fixed split

The completed-run `baseline/split_indices.npz` is archived directly and expanded to a human-readable 836-row assignment table:

```text
split_indices/baseline_fixed_split_indices.npz
split_indices/baseline_parent_split_assignments.csv
```

The 836 parent pairs are exhaustively assigned as:

| Partition | n |
|---|---:|
| Development train | 500 |
| Development validation | 72 |
| Development fixed test | 143 |
| Strict scaffold-disjoint hold-out | 20 |
| Scaffold-overlap quarantine | 101 |

### Fixed role-aware TextCNN splits

The fixed TextCNN partitions are reconstructed deterministically from the archived fingerprint arrays, conservative safe representation-group IDs, the exact structure-only split implementation, and split seed 12.

| Branch | Train | Validation | Test | Group overlap |
|---|---:|---:|---:|---:|
| Gao only | 375 | 53 | 107 | 0 |
| In-house radius-2 | 501 | 71 | 143 | 0 |
| Joint in-house + Gao | 875 | 125 | 250 | 0 |

For all three branches, the regenerated fixed test indices were checked against the completed-run `ensemble_predictions.csv` tables and **matched exactly**.

Files:

```text
split_indices/gao_only_fixed_split_assignments.csv
split_indices/inhouse_r2_fixed_split_assignments.csv
split_indices/joint_inhouse_gao_fixed_split_assignments.csv
split_indices/*_fixed_split_indices.npz
```

### Nested split archive

For each TextCNN branch, exact row-level assignments are archived for split seeds **0-9**, with five model seeds **0-4** trained within each split:

```text
split_indices/gao_only_nested_split_assignments.csv
split_indices/inhouse_r2_nested_split_assignments.csv
split_indices/joint_inhouse_gao_nested_split_assignments.csv
split_indices/*_nested_split_indices.npz
```

Because exact representation groups can contain multiple rows, nested train/validation/test counts vary slightly by seed. All train-validation, train-test, and validation-test group overlaps are zero.

`split_indices/exact_split_manifest.json` records methods, seeds, counts, overlap audits, and fixed-test verification. `split_indices/CHECKSUMS.sha256` provides integrity hashes.

## Exact software environment of the completed manuscript run

The completed run recorded:

```text
Python       3.11.15
OS           Windows 10 (10.0.22631)
NumPy        2.4.6
pandas       3.0.5
SciPy        1.17.1
scikit-learn 1.9.0
RDKit        2026.03.4
PyTorch      2.11.0+cu126
SHAP         0.51.0
joblib       1.5.3
Matplotlib   3.11.1
```

See `RUN_ENVIRONMENT.json`, `environment.yml`, and `requirements-lock.txt`.

## Full reproduction workflow

Run all commands from the repository root.

### 1. Structure-resolved baseline and strict scaffold-disjoint evaluation

```bash
python Baseline.py \
  --da-csv data/D-A.csv \
  --output-dir baseline \
  --aggregation-rule maximum \
  --scaffold-test-size 20 \
  --scaffold-seed 42 \
  --fixed-split-seed 12 \
  --bootstrap-repeats 2000 \
  --run-nested-baseline
```

This creates the 836-pair parent set, the 20-pair strict scaffold hold-out, 101-pair quarantine, 715-pair development pool, fixed radius-3 split, conventional-model comparison, and scaffold-distance diagnostics.

### 2. Leakage-controlled SHAP interpretation

```bash
python 04_shap_interpretation.py \
  --pipeline baseline/interpretation_pipeline.joblib \
  --output-dir interpretation \
  --top-k 10 \
  --background-size 200 \
  --background-seed 2026 \
  --explain-scope test
```

The interpretation model is the pre-specified no-augmentation Random Forest. Mutual-information feature selection and model fitting use training data only. Interventional TreeExplainer explains all 143 fixed-test samples using a random 200-sample training background selected with seed 2026.

### 3. Maximum/mean/median PCE aggregation sensitivity

```bash
python 06_aggregation_sensitivity.py \
  --raw-da-csv data/D-A.csv \
  --excluded-pairs-csv baseline/excluded_from_model_development.csv \
  --output-dir aggregation_sensitivity \
  --split-seed 12 \
  --model-seed 12
```

All target-aggregation rules use the same pair identities and target-blind split. Model identity is selected by validation RMSE only; the test set is evaluated after selection.

### 4. Radius-2 preparation and leakage/collision audits

```bash
python 00_prepare_datasets.py \
  --inhouse-csv data/D-A.csv \
  --excluded-pairs-csv baseline/excluded_from_model_development.csv \
  --gao-fd data/gao_fd_fp.npy \
  --gao-fa data/gao_fa_fp.npy \
  --gao-y data/gao_fp_Y.npy \
  --output-dir prepared_data \
  --max-len 200
```

This step generates aligned radius-2 arrays, source metadata, cross-source overlap audits, near-similarity diagnostics, complete-representation collision groups, FPtand truncation audits, and conservative split-group IDs.

### 5. Fixed role-aware 101-model campaign

```bash
python 01_run_101_seeds.py \
  --config config_fixed_structure_only_roleaware.json \
  --max-workers 1
```

Model seeds 0-100 use one common target-blind structure-only group-aware split.

### 6. Fixed legacy-encoding control

```bash
python 01_run_101_seeds.py \
  --config config_fixed_structure_only_legacy.json \
  --max-workers 1
```

Role-aware and legacy controls use matched branch data, safe groups, split seed, and model-seed set.

### 7. Nested split/model uncertainty campaign

```bash
python 01_run_101_seeds.py \
  --config config_nested_roleaware.json \
  --max-workers 1
```

The supplied campaign uses **10 split seeds × 5 model seeds** per branch. Predictions are ensembled only within the same split.

### 8. Summarize role-aware and legacy results

```bash
python 02_summarize_results.py \
  --fixed-config config_fixed_structure_only_roleaware.json \
  --nested-config config_nested_roleaware.json \
  --output-dir results_roleaware

python 02_summarize_results.py \
  --fixed-config config_fixed_structure_only_legacy.json \
  --output-dir results_legacy
```

### 9. Export exact fixed and nested split assignments

```bash
python 05_export_split_indices.py \
  --repository-root . \
  --output-dir split_indices
```

This creates the human-readable and compressed index archive described above and verifies the fixed TextCNN test indices against the completed-run ensemble prediction tables.

### 10. Generate diagnostic figures

```bash
python 03_make_submission_figures.py \
  --summary-dir results_roleaware \
  --output-dir figures_roleaware
```

### 11. Generate graphical abstract / TOC image

```bash
python make_graphical_abstract.py
```

The corrected graphical abstract is archived in PNG, PDF, and SVG form under `graphical_abstract/`. It deliberately avoids the unsupported phrase **"Tests generalization"** and instead distinguishes **internal prediction** from **limited scaffold extrapolation**.

### One-command workflows

Linux/macOS:

```bash
bash run_revision_workflow.sh
```

Windows PowerShell:

```powershell
./run_revision_workflow.ps1
```

## Key output directories

### `baseline/`

Contains parent/development/scaffold tables, exact baseline split indices, conventional-model selection, target-blind fixed predictions, strict scaffold-hold-out predictions, bootstrap intervals, ranking/calibration diagnostics, and nested baseline variance summaries.

### `prepared_data/`

Contains the aligned radius-2 arrays, source metadata, safe representation groups, exact and near cross-source overlap audits, complete representation-collision groups, and FPtand active-bit/truncation audits.

### `results_roleaware/`

Contains fixed role-aware model-seed metrics and prediction matrices, best-single and ensemble predictions, source-specific joint-test metrics, ranking/calibration/error diagnostics, and nested split/model summaries.

### `interpretation/`

Contains the complete 2,048-feature SHAP importance archive, all test-sample SHAP values, selected top features, correlation matrix, mapped atomic environments, and interpretation manifest.

## Interpretation and generalization boundary

The repository intentionally preserves the following distinctions:

- **Scaffold-disjoint hold-out:** a chemical extrapolation diagnostic on structure-resolved in-house data.
- **Fixed and nested TextCNN tests:** target-blind internal prediction evaluations under exact-representation grouping.
- **Gao integration:** fingerprint-only prediction-oriented data integration, **not** independent external validation.
- **SHAP/mapped environments:** candidate hashed-bit structural signals from the structure-resolved branch, **not** unique causal fragment assignments.

## Reproducibility status and remaining archival limitation

The repository package includes the complete code, exact source inputs used here, exact fixed and nested partition assignments, fixed-model prediction matrices, nested individual-run metrics, split-ensemble metrics, variance decompositions, SHAP archives, environment record, and figure source data.

One limitation of the uploaded completed-run archive remains: the original raw `runs/nested_roleaware/...` folders containing every nested model's full test-prediction vector were not included. The exact nested partitions and all manuscript-reported nested metrics are archived, but those raw per-model nested prediction vectors cannot be reconstructed from summary metrics alone. If the original `runs/nested_roleaware/` directory still exists, add it to the permanent release or accompanying Zenodo/Figshare archive.

See `REPRODUCIBILITY_STATUS.md` for details.

## Recommended release procedure

Before manuscript submission/publication:

1. Synchronize the contents of this package to `luoyang0901/Interpretable-fingerprint`.
2. Verify `CHECKSUMS.sha256` and the split-specific checksum file.
3. Create a GitHub tag/release named **`v2.0-review-revision`**.
4. Use `RELEASE_NOTES_v2.0-review-revision.md` as the release description.
5. If possible, archive the same release on Zenodo/Figshare and add the permanent DOI to this README and the manuscript Data Availability Statement.

## Citation

A `CITATION.cff` file is included. Please cite the associated manuscript when using this workflow or data.
