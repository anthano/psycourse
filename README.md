# Plasma Lipid Alterations and Polygenic Risk Track Multidimensional Psychosis Severity Across Classical Diagnostic Boundaries

______________________________________________________________________

Analysis code accompanying the manuscript.

## Overview

This repository contains all analysis code to reproduce the results, figures, and
supplementary tables. The pipeline identifies a severe psychosis subtype using an SVM
classifier, then characterises it using plasma lipidomics and polygenic risk scores
(PRS) via univariate regression, lipid class enrichment (permutation-based), canonical
correlation analysis (CCA), incremental R², and mediation analysis.

**Input data are not included** (patient-level data; available on reasonable request
subject to the PsyCourse data access agreement). The pipeline requires the raw data to
be placed in `src/psycourse/data/` before running.

______________________________________________________________________

## Repository Structure

```
psycourse/
├── src/psycourse/
│   ├── config.py                     # Paths and shared plot constants
│   ├── data_management/              # Data ingestion and preparation
│   │   ├── data_cleaning.py          # Clean phenotypic, lipidomic, PRS, and cluster label data
│   │   ├── data_encoding.py          # Encode categorical variables
│   │   ├── data_prep_analysis.py     # Merge modalities for regression analyses
│   │   ├── data_prep_classifier.py   # Prepare sparse feature matrix for SVM
│   │   ├── data_prep_integrated_analysis.py  # Integrated PRS + lipid dataset
│   │   ├── data_prep_multimodal_lipid_subset.py  # Subset with all three modalities
│   │   └── task_*.py                 # pytask entry points for each prep step
│   ├── ml_pipeline/                  # SVM subtype classifier
│   │   ├── train_model.py            # Nested CV training, ROC, learning curve
│   │   ├── apply_model_to_new_data.py # Apply trained model to held-out data
│   │   ├── concat_full_predicted_probability_df.py
│   │   ├── impute.py                 # KNN imputation helper
│   │   └── task_*.py                 # pytask entry points
│   ├── data_analysis/                # Statistical analyses
│   │   ├── univariate_analysis.py    # Lipid and PRS regressions ~ subtype probability
│   │   ├── univariate_analysis_panss.py  # Lipid/PRS regressions ~ PANSS subscales
│   │   ├── univariate_binarized_analysis.py  # Regressions with binarised outcome
│   │   ├── lipid_permutation_analysis.py  # Permutation-based lipid class enrichment
│   │   ├── incremental_r2.py         # Block-level incremental R² (permutation p-values)
│   │   ├── mediation_analysis.py     # Mediation: PRS → lipid class score → subtype probability
│   │   ├── cca_regression.py         # CCA: PRS block × lipid class block
│   │   ├── baseline_models.py        # Null/baseline model fits
│   │   ├── elastic_net.py            # Elastic-net regression helper
│   │   ├── multivariate_analysis.py  # Multivariate model fits
│   │   └── task_*.py                 # pytask entry points
│   ├── descriptive_stats/            # Sample description and supplementary export
│   │   ├── descriptive_tables.py     # N-per-analysis tables, demographic summaries
│   │   ├── task_descriptive_tables.py
│   │   └── task_export_supplementary.py  # Compile all results into supplementary .xlsx + TOC .docx
│   └── plots/                        # Figures
│       ├── descriptive_plots.py      # Demographic and classifier performance figures
│       ├── univariate_plots.py       # Forest/dot plots for lipid and PRS associations
│       ├── lipid_enrichment_plot.py  # Lipid class enrichment heatmap/bar plots
│       ├── incremental_r2_plot.py    # R² decomposition figures
│       ├── mediation_plots.py        # Mediation path diagrams
│       ├── cca_plot.py               # CCA biplot / loading plots
│       ├── svm_plots.py              # ROC curve, confusion matrix, learning curve
│       └── task_*.py                 # pytask entry points
├── tests/                            # Unit tests (pytest)
├── hpc_runs/                         # Slurm / HPC submission scripts
├── pyproject.toml                    # Package metadata and pixi environment definition
└── pixi.lock                         # Locked dependency versions
```

______________________________________________________________________

## Analysis Pipeline

The pipeline runs in the following logical order (pytask resolves dependencies
automatically):

| Stage | Module              | Description                                                                                                                               |
| ----- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | `data_management`   | Clean and merge phenotypic, lipidomic, PRS, and cluster-label data                                                                        |
| 2     | `ml_pipeline`       | Train SVM classifier (nested CV); generate predicted subtype probabilities                                                                |
| 3     | `data_analysis`     | Univariate regressions (lipids and PRS ~ subtype probability; sensitivity models); lipid class enrichment; CCA; incremental R²; mediation |
| 4     | `descriptive_stats` | Sample-size tables; export all results to supplementary Excel workbook                                                                    |
| 5     | `plots`             | Generate all manuscript and supplementary figures                                                                                         |

______________________________________________________________________

## Requirements

- [pixi](https://prefix.dev/) — manages the conda/pip environment (Python 3.12, R 4.4+)
- [pytask](https://pytask-dev.readthedocs.io/) — build system (installed via pixi)

Key Python dependencies (see `pyproject.toml` for pinned versions): `numpy`, `pandas`,
`scikit-learn`, `statsmodels`, `scipy`, `pingouin`, `matplotlib`, `seaborn`, `openpyxl`,
`python-docx`, `pytask`, `pytask-parallel`

______________________________________________________________________

## How to Run

**1. Install the environment**

```bash
pixi install
```

**2. Place input data**

Copy the required raw data files into `src/psycourse/data/`:

- `230614_v6.0/230614_v6.0_psycourse_wd.csv` — PsyCourse phenotypic data
- `lipidomics/lipid_intensities.csv` — plasma lipidomics intensities
- `lipidomics/sample_description.csv` — lipidomics sample metadata
- PRS files and cluster label files (see `data_cleaning.py` for expected names)

**3. Run the full pipeline**

```bash
pixi run pytask
```

To run in parallel (recommended for the permutation-heavy enrichment steps):

```bash
pixi run pytask --n-workers 4
```

**4. Output**

All intermediate and final outputs are written to `bld/`:

```
bld/
├── data/        # Cleaned and merged datasets
├── models/      # Trained SVM classifier
├── results/     # Analysis results (.pkl files), figures, and supplementary tables
└── ...
```

The main deliverables are:

- `bld/results/supplementary_tables/supplementary_tables.xlsx` — all supplementary
  result tables
- `bld/results/supplementary_tables/supplementary_toc.docx` — table of contents document
- `bld/results/figures/` — all manuscript figures

______________________________________________________________________

## Notes

- **Data availability**: Patient-level data cannot be shared publicly. The PsyCourse
  study data are available on request; see
  [https://www.psycourse.de](https://www.psycourse.de).
- **Reproducibility**: The `pixi.lock` file pins all dependency versions. Run
  `pixi install` to reproduce the exact environment.
- **HPC**: Scripts for running permutation analyses on a Slurm cluster are in
  `hpc_runs/`.

______________________________________________________________________

## License

MIT — see `pyproject.toml`.
