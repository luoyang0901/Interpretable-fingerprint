# Data and output description

This repository archives the inputs, exact partitions, analysis outputs, and interpretation files used for the major-revision manuscript.

## `data/`

- `D-A.csv`: 1,076 structure-available literature records before pair-level aggregation.
- `gao_fd_fp.npy`, `gao_fa_fp.npy`, `gao_fp_Y.npy`: 535 retained Gao fingerprint-only donor fingerprints, acceptor fingerprints, and PCE labels.

## `baseline/`

Radius-3 structure-resolved branch. Key files include the 836-pair parent table, the 20-pair scaffold-disjoint hold-out, 101 quarantine pairs, the 715-pair development set, exact fixed split archive, conventional-model comparison, scaffold-hold-out predictions, bootstrap intervals, and the pre-specified Random Forest interpretation pipeline.

## `prepared_data/`

Radius-2 in-house/Gao alignment plus exact overlap, near-similarity, representation-collision, and FPtand encoding audits. `representation_overlap_and_collision_audit_summary.csv` gives the five requested cross-source exact-overlap counts together with the 49-group/111-sample duplicate-label summary. The `*_safe_any_encoding_group_ids.npy` arrays are the conservative groups used for TextCNN splitting.

## `split_indices/`

Explicit fixed and nested train/validation/test archives added for reviewer-level reproducibility.

- `baseline_parent_split_assignments.csv`: all 836 parent pairs assigned to development train, validation, fixed test, strict scaffold hold-out, or scaffold-overlap quarantine.
- `*_fixed_split_assignments.csv`: row-level fixed TextCNN assignments for Gao-only, in-house radius-2, and joint branches.
- `*_nested_split_assignments.csv`: row-level assignments for every split seed 0-9.
- matching `.npz` files: compact integer-index versions of the same partitions.
- `exact_split_manifest.json`: split methods, seeds, counts, group-overlap audits, and fixed-test verification status.
- `CHECKSUMS.sha256`: checksums for the exact split archive.

The baseline indices are copied from the completed-run `baseline/split_indices.npz`. TextCNN assignments are deterministically reconstructed from the archived fingerprint arrays, safe representation-group identifiers, the exact split functions in `common_utils.py`, and the manuscript seeds. The regenerated fixed test indices match the completed-run ensemble prediction tables exactly for all three branches.

## `aggregation_sensitivity/`

Maximum-, mean-, and median-PCE aggregation analyses using the same pair identities, common target-blind split, validation-only model selection, and untouched test evaluation.

## `results_roleaware/`

Fixed role-aware TextCNN summaries for model seeds 0-100 and nested 10 split × 5 model-seed uncertainty analyses. Fixed branches also archive the all-model prediction matrices on the common test set.

## `results_legacy/`

Matched fixed-split legacy-encoding control summaries and paired role-aware-versus-legacy comparisons.

## `interpretation/`

Complete leakage-controlled SHAP archive for the pre-specified radius-3 Random Forest, including all 2,048 feature importance values, test-sample SHAP values, pre-defined top ten features, correlations, mapped molecular environments, and the interpretation manifest. Fingerprint bit indices are zero based throughout these tables.

## `figures_roleaware/`

Publication diagnostic figures and their source table.

## `graphical_abstract/`

Corrected graphical abstract/TOC image. It deliberately distinguishes internal prediction from scaffold-level extrapolation and does not use the former unsupported phrase "Tests generalization".
