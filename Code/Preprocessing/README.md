# Preprocessing and data availability

The raw data are being prepared for public release. The notebooks in this
folder document the preprocessing steps used to prepare the analysis data files.

## Preprocessing steps

1. [Meal merging](processing_01_meal_merging.ipynb): combines food records into
   meals and aligns participant records.
2. [Meal history and PPGR outcomes](processing_02_past_features_and_ppgr_outcomes.ipynb):
   derives meal-history features and postprandial glucose response (PPGR)
   outcomes, identifies standardized meals, and exports the prepared meal table.
3. [CGM metrics](processing_03_CGM_metrics.ipynb): calculates participant-level
   continuous glucose monitoring summaries, excluding standardized-meal
   windows, and produces `Data/cgm_metrics.csv`.

The source files are stored in `Data/raw data/`. The prepared analysis inputs
are `Data/meal_ppgr.csv`, `Data/metadata.csv` and `Data/cgm_metrics.csv`.
Participant metadata is prepared separately from the three notebooks above.

## Running the preprocessing notebooks

Run notebooks 01, 02 and 03 in order. The source files `metadata.csv`,
`cgm_data.csv` and `mfr_food_and_you.csv` must be available in `Data/raw data/`.
These source files are not distributed in this repository. Project paths are
defined in [Code/data_paths.py](../data_paths.py).

In addition to the analysis environment, notebook 02 uses
[tqdm-joblib](https://pypi.org/project/tqdm-joblib/) and notebook 03 uses
[cgmquantify](https://pypi.org/project/cgmquantify/). Install these packages in
the notebook environment with:

```bash
python -m pip install tqdm-joblib cgmquantify
```

## Exported files

| Notebook | File | Contents |
| --- | --- | --- |
| 01 | `Data/raw data/meal_data.csv` | Merged meal records used by notebook 02. |
| 02 | `Data/meal_ppgr.csv` or `Data/meal_ppgr.zip` | Meal-history features, PPGR outcomes and traces, timing variables and standardized-meal labels. |
| 03 | `Data/cgm_metrics.csv` | Participant-level CGM summaries after excluding standardized-meal windows. |

Notebook 02 writes to `MEAL_DATA_PATH`: the ZIP archive is selected when it
exists and the uncompressed CSV is absent. The final analysis eligibility
filters are applied later in `Code/utils.py`.

Variable descriptions and summary tables are in
[Data/data_description/](../../Data/data_description/README.md).
