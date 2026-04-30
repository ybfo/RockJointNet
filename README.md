# RockJointNet

RockJointNet is the working repository for the rock-joint shear-strength study built around the GeoSPIN model family. This repo keeps the paper-facing assets in one place: benchmark summaries, figure scripts, table builders, cached predictions, and the small checkpoints needed for reproducibility checks.

The current layout is meant for day-to-day research use rather than for a polished software release. The emphasis is on making the benchmark pipeline easy to rerun and the final figures and tables easy to update when experiments change.

## Quick start

From the repository root:

```powershell
python .\build_all.py
```

This regenerates the result figures and the result tables from the cached experiment outputs already stored in the repo. It does not retrain the full benchmark suite.

If you only want a subset:

```powershell
python .\scripts\build_figures.py
python .\scripts\build_tables.py
```

## Repository layout

```text
RockJointNet/
|-- build_all.py
|-- manuscript_tnnls.tex
|-- requirements.txt
|-- checkpoints/
|-- data/
|-- outputs/
|-- scripts/
`-- source_inputs/
```

`data/` stores the cached benchmark summaries, counterfactual sweep exports, local direct-shear curve caches, and table sources. `scripts/` contains the figure and table generators used by the rebuild step. `outputs/` holds the rendered result figures and result tables. `source_inputs/` keeps the original imported tables used to build the cleaned benchmark caches.

## Main generated files

The build step writes the result figures and result tables to `outputs/`.

Figures:

- `outputs/figures/figure_03_prediction_and_error_field.png`
- `outputs/figures/figure_04_feature_response_maps.png`
- `outputs/figures/figure_05_slope_and_curvature_maps.png`
- `outputs/figures/figure_06_stress_path_profiles.png`
- `outputs/figures/figure_09_prediction_panels.png`
- `outputs/figures/figure_10_curve_reconstruction.png`
Tables:

- `outputs/tables/table_01_benchmark_results.csv`
- `outputs/tables/table_02_transform_ablation.csv`
- `outputs/tables/table_03_constraint_checks.csv`
- `outputs/tables/table_04_curve_reconstruction.csv`
- `outputs/tables/table_05_noise_robustness.csv`
- `outputs/tables/table_06_inverse_recovery.csv`

## Data notes

The default scripts read from cached CSV files rather than from live training logs. The most important ones are:

- `data/experiment_results/counterfactual_feature_sweeps.csv`
- `data/experiment_results/prediction_cases.csv`
- `data/experiment_results/fixed_context_constraint_checks.csv`
- `data/experiment_results/transform_ablation_results.csv`
- `data/experiment_results/noise_robustness_results.csv`
- `data/experiment_results/inverse_recovery_results.csv`
- `data/recent_methods/benchmark_summary.csv`
- `data/local_curve/curve_reconstruction_predictions.csv`
- `data/local_curve/curve_reconstruction_metrics.csv`

That makes the figure and table build deterministic and fast, which is helpful when the manuscript is still moving.

## Checkpoints

This repo is not checkpoint-heavy, but it does keep a few model files for traceability and smoke testing. The canonical checkpoint index is `checkpoints/checkpoint_manifest.csv`.

The main file used for a quick implementation sanity check is:

- `checkpoints/rockjointnet_smoke_checkpoint.pt`

The archived baseline checkpoints are kept only so the manuscript comparisons remain auditable.
