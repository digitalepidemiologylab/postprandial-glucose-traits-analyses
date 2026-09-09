# Characterizing glucose-response traits from free-living meals

Analysis code accompanying **Short-term postprandial glucose monitoring reveals stable traits from noisy free-living meals**, by M. Toumi and M. Salathé, Digital Epidemiology Lab, EPFL, Lausanne, Switzerland.

## Repository contents

- [Code/](Code/): analysis notebooks, shared utilities and bootstrap summary tables.
- [Figures/](Figures/): figures from the analyses.
- [Results/](Results/): exported analysis results.
- [Data/](Data/): analysis inputs and dataset descriptions.

## Analysis notebooks

The numbered notebooks cover the analyses below. Most load the prepared inputs independently. Shared functions are in [Code/utils.py](Code/utils.py), and input paths are defined in [Code/data_paths.py](Code/data_paths.py).

| Order | Notebook | Purpose | Outputs |
| --- | --- | --- | --- |
| 01 | [Descriptive statistics](Code/01_descriptive_statistics.ipynb) | Cohort summaries, outcome PCA, variance components and repeatability | Saved notebook outputs and figures |
| 02 | [Within-person temporal CV](Code/02_within_person_temporal_cv.ipynb) | Response prediction in held-out periods of observed participants | `Results/model_comparison_within_participant_CV.csv` |
| 03 | [Held-out participant CV](Code/03_held_out_participant_cv.ipynb) | Response prediction for participants absent from training | `Results/LOPO_CV_results.csv` |
| 04 | [Full model and characterization](Code/04_full_model_and_characterization.ipynb) | Covariance geometry and associations with participant characteristics and challenges | CGM correlations, adiposity and challenge tables |
| 05 | [Split-half reliability](Code/05_split_half_reliability.ipynb) | Reliability of scores and covariance geometry across meal subsets | Split-half reliability and covariance-spectrum tables |
| 06 | [Monitoring length](Code/06_monitoring_length_analyses.ipynb) | Prediction and score agreement across monitoring windows | Learning curves and PC-score monitoring tables |
| 07 | [Out-of-sample validation](Code/07_out_of_sample_validation.ipynb) | Frozen-model score inference and reliability in held-out participants | `Results/OOS_PC_estimation/` |
| 08 | [Outcome sensitivity](Code/08_outcome_selection_sensitivity.ipynb) | Absolute versus baseline-relative response definitions | `Results/sensitivity_analyses/` and saved notebook outputs |

The [MMER-XGBoost hyperparameter search notebook](Code/hyperparameter_search/MMER_XGBoost_hyperparameter_search.ipynb) and its [Optuna database](Code/hyperparameter_search/mmer_optuna.db) are in `Code/hyperparameter_search/`.

The [MMER-XGBoost bootstrap notebook](Code/MMER_XGBoost_bootstraps.ipynb), its [saved bootstrap archive](Code/bootstraps/parametric_bootstrap_geometry_matrices.npz) and the companion [point-estimate matrix](Code/bootstraps/G_point.npy) are also included.

## Data

This repository does not contain the raw data, which are being prepared for public release. The preprocessing pipeline
in `Code/Preprocessing/` documents the steps used to prepare the analysis data
files. The analyses use three prepared inputs:

| Location | Contents |
| --- | --- |
| `Data/metadata.csv` | Participant covariates |
| `Data/meal_ppgr.csv` | Prepared meal-level outcomes, nutrients, timing and history features |
| `Data/cgm_metrics.csv` | Participant CGM summaries used in notebook 04 |

The meal table is distributed as [Data/meal_ppgr.zip](Data/meal_ppgr.zip). The notebooks read it directly when `Data/meal_ppgr.csv` is absent. The separate data-description generator requires the extracted CSV.

Variable descriptions and aggregate summary tables are in [Data/data_description/](Data/data_description/README.md).

## Contacts

Marouane Toumi: marouane.toumi@epfl.ch 

[LinkedIn](https://www.linkedin.com/in/marouane-toumi)
