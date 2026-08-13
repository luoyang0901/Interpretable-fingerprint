#!/usr/bin/env bash
set -euo pipefail

python Baseline.py \
  --da-csv data/D-A.csv \
  --output-dir baseline \
  --aggregation-rule maximum \
  --scaffold-test-size 20 \
  --scaffold-seed 42 \
  --fixed-split-seed 12 \
  --bootstrap-repeats 2000

python 04_shap_interpretation.py \
  --pipeline baseline/interpretation_pipeline.joblib \
  --output-dir interpretation \
  --top-k 10 \
  --background-size 200 \
  --background-seed 2026 \
  --explain-scope test

python 06_aggregation_sensitivity.py \
  --raw-da-csv data/D-A.csv \
  --excluded-pairs-csv baseline/excluded_from_model_development.csv \
  --output-dir aggregation_sensitivity

python 00_prepare_datasets.py \
  --inhouse-csv data/D-A.csv \
  --excluded-pairs-csv baseline/excluded_from_model_development.csv \
  --gao-fd data/gao_fd_fp.npy \
  --gao-fa data/gao_fa_fp.npy \
  --gao-y data/gao_fp_Y.npy \
  --output-dir prepared_data \
  --max-len 200

python 01_run_101_seeds.py --config config_fixed_structure_only_roleaware.json --max-workers 1
python 01_run_101_seeds.py --config config_fixed_structure_only_legacy.json --max-workers 1
python 01_run_101_seeds.py --config config_nested_roleaware.json --max-workers 1

python 02_summarize_results.py \
  --fixed-config config_fixed_structure_only_roleaware.json \
  --nested-config config_nested_roleaware.json \
  --output-dir results_roleaware

python 02_summarize_results.py \
  --fixed-config config_fixed_structure_only_legacy.json \
  --output-dir results_legacy

python 03_make_submission_figures.py \
  --summary-dir results_roleaware \
  --output-dir figures_roleaware
