# Reproducibility status

## Included and verified in this revision package

The repository package now contains the complete revised Python workflow and JSON configurations, the exact input arrays/data used by the completed run, the exact baseline fixed split, deterministic fixed TextCNN partitions, deterministic nested TextCNN partitions, collision/leakage audits, fixed-run prediction matrices, nested run metrics, SHAP archives, figure source data, and the recorded software environment.

### Exact train/validation/test partition archive

The previous missing-index issue has been addressed with `05_export_split_indices.py` and the `split_indices/` archive.

- `baseline_parent_split_assignments.csv` assigns all 836 structure-resolved parent pairs to development train, development validation, development fixed test, strict scaffold-disjoint hold-out, or scaffold-overlap quarantine.
- `baseline_fixed_split_indices.npz` preserves the baseline index arrays from the completed-run archive.
- `gao_only_fixed_split_assignments.csv`, `inhouse_r2_fixed_split_assignments.csv`, and `joint_inhouse_gao_fixed_split_assignments.csv` provide row-level fixed TextCNN assignments.
- Matching `*_fixed_split_indices.npz` files provide the same indices in compressed NumPy form.
- `*_nested_split_assignments.csv` and `*_nested_split_indices.npz` provide exact row assignments for split seeds 0-9 for every TextCNN branch.
- `exact_split_manifest.json` records split methods, seeds, per-split counts, group-overlap audits, and fixed-test verification.
- `CHECKSUMS.sha256` protects the split archive against accidental changes.

Fixed TextCNN counts are 375/53/107 for Gao-only, 501/71/143 for in-house radius-2, and 875/125/250 for the joint branch. All train-validation, train-test, and validation-test representation-group overlaps are zero. The regenerated fixed test indices for all three branches match the archived completed-run ensemble-prediction tables exactly.

The nested assignments are reconstructed using the exact `random_group_split` implementation, safe representation-group identifiers, split seeds 0-9, and the branch-specific archived inputs. Counts can vary slightly by seed because a conservative representation group can contain multiple rows; no representation group crosses partition boundaries.

### Other included reviewer-requested materials

- Exact 1,076-record structure-available in-house input and the 836-pair aggregated parent set.
- Exact 535-row Gao donor/acceptor fingerprint arrays and labels.
- Strict 20-pair donor- and acceptor-scaffold-disjoint hold-out and 101-pair scaffold-overlap quarantine.
- Cross-source exact/near-overlap and representation-collision audits.
- FPtand per-sample active-bit/truncation audit and donor/acceptor/total active-bit distribution summary.
- Fixed role-aware 101-seed metrics, prediction matrices, best/ensemble predictions, ranking metrics, source-specific joint-test metrics, and PCE-stratified errors.
- Nested role-aware individual-run metrics, split-ensemble metrics, empirical summaries, and variance decompositions.
- Complete structure-resolved SHAP archives, mapped environments, selected-feature tables, and model archive.
- Exact completed-run software versions in `RUN_ENVIRONMENT.json`, `environment.yml`, and `requirements-lock.txt`.
- Corrected graphical abstract / TOC image in PNG, PDF, and SVG form, together with its generation script.

## Remaining archival limitation

The uploaded completed-run result archive does not contain the original raw `runs/nested_roleaware/...` per-model folders with every nested model's full test-prediction vector. The exact nested partitions and all manuscript-reported nested per-run/split metrics are now archived, so this no longer prevents reproduction of the partition design or reported statistics. However, the historical per-model nested prediction vectors cannot be recovered exactly from summary metrics alone.

If the original `runs/nested_roleaware/` folder still exists on the experiment computer, it is worth adding it to a permanent GitHub Release/Zenodo archive. If it does not exist, do not claim that the historical raw per-model nested prediction vectors are archived; instead state that the exact partitions, code, seeds, and reported nested metrics are archived and that a fresh end-to-end rerun can regenerate prediction vectors in the recorded environment.
