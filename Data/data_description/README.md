# Dataset descriptions and article tables

The notebooks use three prepared inputs from the repository's `Data/` folder:

| Path | Role |
| --- | --- |
| `Data/meal_ppgr.csv` | Prepared meal-level PPGR outcomes and covariates |
| `Data/metadata.csv` | Prepared participant covariates, including `age` |
| `Data/cgm_metrics.csv` | Derived participant-level CGM summaries |

The meal table is distributed as `Data/meal_ppgr.zip`. Notebooks read the ZIP
directly when the uncompressed CSV is absent. Input paths are defined in
`code/data_paths.py` and work from the repository root or its subfolders.

## Summary tables

| File | Contents |
| --- | --- |
| `analysis_features_summary.csv` | The 65 model predictors, using free-living meals only |
| `glycemic_metrics_summary.csv` | The 28 participant-level CGM and premeal-glucose summaries |
| `ppgr_outcomes_by_meal_type.csv` | The six PPGR outcomes, with free-living and separate standardized-meal columns |
| `generate_descriptions.py` | Generates the three summary tables from the prepared inputs |

To regenerate the tables, extract `Data/meal_ppgr.zip` into `Data/` if the CSV
is absent, then run this command from the repository root:

```bash
python Data/data_description/generate_descriptions.py
```

The script uses the shared data filters and the analysis settings in notebooks
04 and 08. Values are summarized before model scaling and imputation, using
available observations and sample SD (`ddof=1`). CSVs use UTF-8 with a byte-order
mark; displayed values have two decimal places, with scientific notation for
very small values and micronutrient amounts expressed in grams.

## Participant metadata: `metadata.csv`

Participant metadata contains the following variables:

| Variable | Meaning / unit |
| --- | --- |
| `subject_key` | Participant linkage key shared with the two processed analysis tables |
| `id` | Numeric participant key currently required by the loader's metadata mapping; not a model predictor |
| `gender` | Recorded category, `female` or `male`; the analysis meal table maps these to 0 and 1 |
| `age` | Age in years |
| `weight` | Body weight, kg |
| `height` | Height, cm |
| `waist` | Waist circumference, cm |
| `hip` | Hip circumference, cm |
| `cohort` | Food & You subcohort, `cohort_b` or `cohort_c` |

The loader calculates BMI as weight divided by height in metres squared,
`WaistHipRatio` as waist/hip, and `WaistHeightRatio` as waist/height. Missing
anthropometric values remain missing; analyses requiring these covariates use
complete cases. Metadata rows are not all necessarily represented in the final
meal-analysis cohort.

The full source questionnaire and raw dietary/CGM files are kept in the
excluded `Data/raw data/` folder. The metadata loader uses `id` and
`subject_key` to link participant records across the analysis inputs.

## Meal-level responses: `meal_ppgr.csv`

This table is produced by
`code/Preprocessing/processing_02_past_features_and_ppgr_outcomes.ipynb`, using
merged dietary records and CGM measurements. Each row represents a meal with
derived response measurements and contextual features. The working table is
upstream of the final analysis filters: its row count is therefore larger than
the sample reported in the manuscript.

| Important variable(s) | Meaning / unit |
| --- | --- |
| `subject_key` | Participant linkage key, shared with metadata and CGM metrics |
| `eaten_at` | Logged meal timestamp; used to order meals, count observed dates and construct validation windows |
| `is_standardized_meal` | 1 for a retained standardized challenge label, 0 for other meals |
| `standardized_meal_id` | `A`: glucose drink; `B`: white bread; `C`: white bread with butter; `other`: other meals |
| `max_glucose` | Highest glucose in the response window, mg/dL |
| `end_glucose` | Glucose at the end of the 120-minute response window, mg/dL |
| `positive_iAUC` | Area above the premeal baseline, min·mg/dL |
| `peak_duration` | Time above the premeal baseline within the response window, minutes; not time to peak |
| `premeal_glucose` | Baseline glucose used for response derivation, mg/dL |
| `delta_max_glucose`, `delta_end_glucose` | Peak and endpoint glucose minus baseline, mg/dL; used in sensitivity analyses |
| `carb_eaten`, `fat_eaten`, `protein_eaten`, `fiber_eaten` | Macronutrient amounts, g |
| `energy_kcal_eaten`, `eaten_quantity_in_gram` | Meal energy, kcal, and mass, g |
| Six food-group columns | Meal mass in each food group, g: `grains_potatoes_pulses`, `sweets_salty_snacks_alcohol`, `non_alcoholic_beverages`, `dairy_products_meat_fish_eggs_tofu`, `vegetables_fruits`, `oils_fats_nuts` |
| `prev_{1,2,3,6}hr_*` | Previous-window nutrient totals, in the same units as the underlying nutrient; historical energy totals are kcal |
| `trend_glu_{1,2,4,6}` | Least-squares glucose trends over the corresponding preceding windows, mg/dL/hour |
| `time_since_last_meal` | Previous-meal interval supplied by preprocessing, hours |
| `time_to_next_meal` | Forward interval supplied by preprocessing, hours; used for eligibility, not as a prediction feature |
| `hours_numeric` | Meal time of day expressed in hours |

All nutrient amounts, including vitamins and minerals, were converted to grams
during preprocessing. The summary tables retain these values in grams; energy
remains in kcal.

`load_and_prepare_data` in `code/utils.py` joins metadata, checks response
quality, clips negative nutrient values, excludes meals with energy at least
2,000 kcal, applies monitoring-date and next-meal-interval filters, and finally
retains participants with more than 15 eligible responses. That final count
includes both free-living and standardized meals. Model fitting subsequently
excludes standardized challenges. Covariate missingness is handled separately
by the relevant analyses.

The table also contains `ppgr_array` and `ppgr_time_array`, which retain
meal-response glucose readings and their elapsed times, plus food/dish
identifiers and descriptions. The loader uses `ppgr_array` to exclude missing
or constant response traces before constructing the analysis cohort.

## Participant-level summaries: `cgm_metrics.csv`

This table is produced by
`code/Preprocessing/processing_03_CGM_metrics.ipynb`. Each row summarizes a
participant's CGM record after the preprocessing exclusion of standardized-meal
windows. Notebook 04 joins this table to the phenotype coordinates and metadata.

| Important variable(s) | Meaning / unit |
| --- | --- |
| `subject_key` | Participant linkage key |
| `n_observations`, `n_days` | Retained CGM reading count and number of observed dates |
| `mean_glucose`, `median_glucose`, `min_glucose`, `max_glucose`, `q1_glucose`, `q3_glucose` | Glucose-level summaries, mg/dL |
| `interdaysd`, `interdaycv` | Overall glucose SD, mg/dL, and coefficient of variation, % |
| `intradaysd_mean`, `intradaysd_median`, `intradaysd_sd` | Summaries of within-day glucose SDs, mg/dL |
| `intradaycv_mean`, `intradaycv_median`, `intradaycv_sd` | Summaries of within-day glucose CVs, % |
| `TIR`, `TOR` | Time inside/outside 70–180 mg/dL, minutes, calculated from reading counts at a nominal 15-minute interval |
| `PIR`, `POR` | Percentage of retained readings inside/outside 70–180 mg/dL |
| `MAGE`, `MODD`, `CONGA24` | Glucose-variability measures, mg/dL; computed with the documented preprocessing functions |
| `J_index`, `LBGI`, `HBGI`, `ADRR` | Derived glycemic variability/risk indices |
| `GMI`, `eA1c` | Glucose-derived estimates, %; these are not laboratory HbA1c measurements |

The table also contains record-boundary timestamps and diagnostic error
columns. Missing metric values represent unavailable measurements, not zero.

All project paths above are relative to the repository root. The preprocessing
notebooks require private source records; the public analysis entry points are
the numbered notebooks under `code/`.

## Aggregate article tables

`Data/data_description/analysis_features_summary.csv` describes all 65 predictors used by the
main model: meal composition, recent dietary intake, premeal glycemic state,
and meal timing. It includes readable labels, exact dataset variable names,
units, mean ± sample SD, and the number of available observations (`N`).
Demographic covariates are excluded because the main model removes that group.

`Data/data_description/ppgr_outcomes_by_meal_type.csv` describes the four primary PPGR outcomes
and two baseline-relative sensitivity outcomes. Each outcome has separate
mean ± sample SD columns for free-living meals and every observed standardized
meal type. Column headers give the number of meals; every outcome in that column
has that number of available observations:

| Meal group | Definition | Meals | Participants |
| --- | --- | --- | --- |
| Free-living | `is_standardized_meal == False` | 50,463 | 992 |
| A: glucose drink | `is_standardized_meal == True` and `standardized_meal_id == "A"` | 1,766 | 885 |
| B: white bread | `is_standardized_meal == True` and `standardized_meal_id == "B"` | 1,465 | 759 |
| C: white bread with butter | `is_standardized_meal == True` and `standardized_meal_id == "C"` | 1,293 | 694 |

Each meal contributes one observation within its group. Summaries are not
averages of participant means. Participants can contribute to multiple groups.

`Data/data_description/glycemic_metrics_summary.csv` contains only participant-level summaries:
the 26 CGM metrics selected in notebook 04 and its two derived premeal-glucose
summaries. The dataset name `max_glucose` denotes the meal-response peak in the
PPGR table and the overall participant maximum in the CGM table.

All tables use the shared `load_and_prepare_data` eligibility filters. The
predictor table explicitly includes only `is_standardized_meal == False`, giving
50,463 free-living meals from 992 participants. The PPGR table partitions all 54,987 eligible meals into the four groups above.
CGM summaries use one record per participant in the same 992-person cohort.
The derived participant premeal-glucose mean and SD use all eligible meals,
including standardized challenges, matching notebook 04; their table mean and
SD are then computed across participants. Sensitivity outcomes are recomputed
as peak or endpoint glucose minus premeal glucose on the common complete-case
outcome set within each meal group, following notebook 08's definitions.

SD uses `ddof=1`. Summaries use available observations before model scaling and
imputation. `N` therefore varies for glucose trends and time since the previous
meal. TIR and TOR are total accumulated minutes over the participant record,
not percentages. PPGR `peak_duration` is time above premeal glucose, not time
to peak. Estimated sugar fraction follows the loader's
`max(sugar_eaten / (carb_eaten + 1), 0)` definition.

All nutrient amounts are reported in grams, matching the conversion already applied during preprocessing. The generator applies no further nutrient unit conversion. Energy remains in kcal, and glucose measurements retain their stated glucose units. Scientific notation preserves small nutrient values.
