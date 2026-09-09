"""Shared utilities for the MMER phenotyping analysis notebooks.

This module contains data preparation and functions reused across analysis
notebooks.  Analysis-specific plotting and statistical models remain beside
their results in the notebooks.
"""

from __future__ import annotations

import ast as _ast
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import BaseCrossValidator, GroupKFold
from sklearn.utils import check_random_state


SUBJECT_COL = "subject_key"
DATE_COL = "eaten_date"
RANDOM_SLOPES = ["carb_eaten", "fat_eaten", "protein_eaten", "fiber_eaten"]
OUTCOMES = ["max_glucose", "peak_duration", "end_glucose", "positive_iAUC"]
PREDICTION_SUFFIXES = [
    "true_z", "pred_z", "true", "pred", "pred_fixed_z", "pred_fixed"
]

POSITIVE_FOOD_COLUMNS = [
    "carb_eaten", "fat_eaten", "protein_eaten", "fiber_eaten",
    "eaten_quantity_in_gram", "energy_kcal_eaten",
    "dairy_products_meat_fish_eggs_tofu", "vegetables_fruits",
    "sweets_salty_snacks_alcohol", "non_alcoholic_beverages",
    "grains_potatoes_pulses", "oils_fats_nuts",
    "alcohol_eaten", "beta_carotene_eaten", "calcium_eaten", "cholesterol_eaten",
    "fatty_acids_monounsaturated_eaten", "fatty_acids_polyunsaturated_eaten",
    "fatty_acids_saturated_eaten", "folate_eaten", "iron_eaten", "magnesium_eaten",
    "niacin_eaten", "pantothenic_acid_eaten", "phosphorus_eaten", "potassium_eaten",
    "salt_eaten", "sodium_eaten", "sugar_eaten", "vitamin_b1_eaten",
    "vitamin_b12_eaten", "vitamin_b2_eaten", "vitamin_b6_eaten", "vitamin_c_eaten",
    "vitamin_d_eaten", "zinc_eaten",
    "prev_1hr_carb_eaten", "prev_2hr_carb_eaten", "prev_3hr_carb_eaten", "prev_6hr_carb_eaten",
    "prev_1hr_fat_eaten", "prev_2hr_fat_eaten", "prev_3hr_fat_eaten", "prev_6hr_fat_eaten",
    "prev_1hr_protein_eaten", "prev_2hr_protein_eaten", "prev_3hr_protein_eaten", "prev_6hr_protein_eaten",
    "prev_1hr_fiber_eaten", "prev_2hr_fiber_eaten", "prev_3hr_fiber_eaten", "prev_6hr_fiber_eaten",
    "prev_1hr_energy_kcal_eaten", "prev_2hr_energy_kcal_eaten",
    "prev_3hr_energy_kcal_eaten", "prev_6hr_energy_kcal_eaten",
]


def _require_input_columns(frame: pd.DataFrame, columns: Iterable[str], frame_name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"{frame_name} is missing required columns: {missing}")


def load_and_prepare_data(
    metadata_path,
    meal_data_path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.Series]:
    """Load and fully prepare the common dataset used by every analysis.

    Returns ``metadata``, the prepared meal table, the metadata-ID mapping,
    and a subject-cluster series aligned row-for-row with the meal table.
    """
    metadata = pd.read_csv(metadata_path)
    if "subject_app_key" in metadata.columns:
        metadata = metadata.rename(columns={"subject_app_key": SUBJECT_COL})

    _require_input_columns(
        metadata,
        [SUBJECT_COL, "weight", "height", "waist", "hip", "gender"],"metadata",)

    if "id" in metadata.columns:
        id_to_subject_key = dict(zip(metadata["id"], metadata[SUBJECT_COL]))
        metadata = metadata.rename(columns={"id": "fay-id"})
    else:
        _require_input_columns(metadata, ["fay-id"], "metadata")
        id_to_subject_key = dict(zip(metadata["fay-id"], metadata[SUBJECT_COL]))

    # Derived anthropometric measures.
    metadata["bmi"] = metadata["weight"] / ((metadata["height"] / 100) ** 2)
    metadata["WaistHipRatio"] = metadata["waist"] / metadata["hip"]
    metadata["WaistHeightRatio"] = metadata["waist"] / metadata["height"]

    meals = pd.read_csv(meal_data_path, index_col=0)
    covariates = [
        "age", "bmi", "gender", "waist", "weight", "hip", "height",
        "WaistHipRatio", "WaistHeightRatio",
    ]
    _require_input_columns(
        meals,
        [
            SUBJECT_COL,
            "ppgr_array",
            "eaten_at",
            "time_since_last_meal",
            "time_to_next_meal",
            *POSITIVE_FOOD_COLUMNS,
        ],
        "meal data",
    )
    meals = meals.merge(metadata[[*covariates, SUBJECT_COL]], on=SUBJECT_COL, how="inner")
    meals["gender"] = meals["gender"].map({"female": 0, "male": 1})

    meals = meals.loc[meals["ppgr_array"].notna()].copy()
    meals["ppgr_array_ast"] = meals["ppgr_array"].apply(
        lambda value: _ast.literal_eval(value) if isinstance(value, str) else value
    )
    meals["all_equal"] = meals["ppgr_array_ast"].apply(lambda values: len(set(values)) == 1)
    meals = meals.loc[~meals["all_equal"]].copy()

    meals["estimated_net_carb_eaten"] = (
        meals["carb_eaten"] - meals["fiber_eaten"]
    ).clip(lower=0)
    meals["estimated_sugar_fraction"] = (
        meals["sugar_eaten"] / (meals["carb_eaten"] + 1)
    ).clip(lower=0)

    meals["eaten_at"] = pd.to_datetime(meals["eaten_at"])
    meals[DATE_COL] = meals["eaten_at"].dt.date
    meals = meals.loc[
        meals.groupby(SUBJECT_COL)[DATE_COL].transform("nunique").ge(7)
    ].copy()

    meals[POSITIVE_FOOD_COLUMNS] = meals[POSITIVE_FOOD_COLUMNS].clip(lower=0)
    meals = meals.loc[meals["energy_kcal_eaten"] < 2000].copy()
    meals["time_since_last_meal"] = meals["time_since_last_meal"].where(
        meals["time_since_last_meal"] <= 24
    )
    meals["time_to_next_meal"] = meals["time_to_next_meal"].where(
        meals["time_to_next_meal"] <= 24
    )
    meals = meals.loc[
        (meals["time_to_next_meal"] >= 2)
        | meals["time_to_next_meal"].isna()
    ].reset_index(drop=True)

    # Retain participants with more than 15 eligible meal responses.
    meal_counts = meals.groupby(SUBJECT_COL).size()
    eligible = meal_counts.loc[meal_counts > 15].index
    meals = meals.loc[meals[SUBJECT_COL].isin(eligible)].copy().reset_index()
    clusters = meals[SUBJECT_COL].copy()
    print(f"data shape : {meals.shape}")
    return metadata, meals, id_to_subject_key, clusters


def _impute_missing(X_tr, X_va, strategy, subj_tr, subj_va):
    """
    Fill NA in feature frames using TRAINING-only statistics.

    strategy:
        None                                        -> no-op
        'population_mean'  / 'population_median'     -> one value per column
        'participant_mean' / 'participant_median'   -> per-subject value per column,
                                                       with population fallback for
                                                       subjects missing/unseen in train.
    Returns (X_tr_filled, X_va_filled).
    """
    if strategy is None:
        return X_tr, X_va

    agg = "median" if strategy.endswith("median") else "mean"
    pop_fill = getattr(X_tr, agg)()                       # Series: training-only, per column

    if strategy.startswith("population"):
        return X_tr.fillna(pop_fill), X_va.fillna(pop_fill)

    # participant_* : per-subject stat from training, population fallback
    subj_fill = X_tr.groupby(subj_tr).agg(agg)            # index=subject, cols=features
    subj_fill = subj_fill.fillna(pop_fill)               # subjects with NaN stat -> population

    def _apply(X, subj):
        fills = subj_fill.reindex(subj.values)           # one row per X row, in order
        fills.index = X.index
        fills = fills.fillna(pop_fill)                   # subjects unseen in train -> population
        return X.fillna(fills)

    return _apply(X_tr, subj_tr), _apply(X_va, subj_va)


def to_32bit(x):
    arr = np.asarray(x)

    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32, copy=False)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int32, copy=False)

    return x


def build_design_mats(data, train_idx=None, val_idx=None, target_outcome=None,
                      apply_mundlak=True,
                      use_interactions=[False],
                      population_z_scoring_features=True,
                      remove_features=None,
                      remove_std_meals_from_training=False,
                      random_slopes=None,
                      population_z_scoring_outcome=False,
                      full_train=False,
                      imputation=None,
                      fixed_scalers=None,
                     ):
    """
    Build design matrices for mixed-effects modeling.

    Parameters
    ----------
    data : pd.DataFrame
    train_idx, val_idx : array-like
    target_outcome : str or list[str]
    apply_mundlak : bool
        If True: use within-person deviations + add subject means as fixed effects.
        If False: use raw intakes.
    use_interactions : list of str
        e.g., ['cho_fat'] -> interaction between carb and fat.
    population_z_scoring_features : bool
        If True: apply (X - mu) / sigma using training-set stats, unless
        fixed_scalers is provided.
    remove_features : list of str or None
        Feature groups to drop. Valid groups:
        'nutritional', 'food_group', 'past_glucose', 'baseline_glucose',
        'past_food', 'context', 'demographics'.
    remove_std_meals_from_training : bool
    random_slopes : list of str or None
        Variables to use as random slopes in Z. Default: macros.
    imputation : {None, 'participant_mean', 'participant_median',
                  'population_mean', 'population_median'}
        How to fill NA in the feature/covariate matrices.
    fixed_scalers : dict or None
        If provided, reuse original-fit scaling constants instead of recomputing
        them from the current train/bootstrap data.

        Expected keys when relevant:
            - "z_score_mean"
            - "z_score_std"
            - "other_mean"
            - "other_std"
            - "outcome_mean"
            - "outcome_std"

        This is useful for bootstrap refits when all replicates should be
        measured in the same original-fit units.

    Returns
    -------
    X_train, X_val, Z_train, Z_val, y_train, y_val,
    clusters_train, clusters_val, cluster_mapping, info,
    train_dataframe, test_dataframe
    """

    macros = ["carb_eaten", "fat_eaten", "protein_eaten", "fiber_eaten"]

    # Align fixed scalers to the requested columns.
    def _get_fixed_series(name, columns):
        if fixed_scalers is None or name not in fixed_scalers:
            raise KeyError(
                f"fixed_scalers must contain key '{name}' when fixed scalers are used."
            )

        s = fixed_scalers[name]

        if isinstance(s, pd.Series):
            s = s.copy()
        else:
            s = pd.Series(np.asarray(s), index=columns)

        s = s.reindex(columns)

        missing = s[s.isna()].index.tolist()
        if missing:
            raise ValueError(
                f"fixed_scalers['{name}'] is missing values for columns: {missing}"
            )

        return s

    # Feature groups available for model comparisons.
    FEATURE_GROUPS = {
        "nutritional": [
            "energy_kcal_eaten", "eaten_quantity_in_gram", "estimated_net_carb_eaten", "estimated_sugar_fraction",
            'alcohol_eaten', 'beta_carotene_eaten', 'calcium_eaten', 'cholesterol_eaten',
            'fatty_acids_monounsaturated_eaten', 'fatty_acids_polyunsaturated_eaten',
            'fatty_acids_saturated_eaten', 'folate_eaten', 'iron_eaten', 'magnesium_eaten',
            'niacin_eaten', 'pantothenic_acid_eaten', 'phosphorus_eaten', 'potassium_eaten',
            'salt_eaten', 'sodium_eaten', 'sugar_eaten', 'vitamin_b1_eaten',
            'vitamin_b12_eaten', 'vitamin_b2_eaten', 'vitamin_b6_eaten', 'vitamin_c_eaten',
            'vitamin_d_eaten', 'zinc_eaten',
        ],
        "food_group": [
            'grains_potatoes_pulses', 'sweets_salty_snacks_alcohol',
            'non_alcoholic_beverages', 'dairy_products_meat_fish_eggs_tofu',
            'vegetables_fruits', 'oils_fats_nuts',
        ],
        "baseline_glucose": [
            "premeal_glucose",
        ],
        "past_glucose": [
            "trend_glu_1", "trend_glu_2", "trend_glu_4", "trend_glu_6",
        ],
        "past_food": [
            "prev_1hr_carb_eaten", "prev_2hr_carb_eaten", "prev_3hr_carb_eaten", "prev_6hr_carb_eaten",
            "prev_1hr_fat_eaten", "prev_2hr_fat_eaten", "prev_3hr_fat_eaten", "prev_6hr_fat_eaten",
            "prev_1hr_protein_eaten", "prev_2hr_protein_eaten", "prev_3hr_protein_eaten", "prev_6hr_protein_eaten",
            "prev_1hr_fiber_eaten", "prev_2hr_fiber_eaten", "prev_3hr_fiber_eaten", "prev_6hr_fiber_eaten",
            "prev_1hr_energy_kcal_eaten", "prev_2hr_energy_kcal_eaten",
            "prev_3hr_energy_kcal_eaten", "prev_6hr_energy_kcal_eaten",
        ],
        "context": [
            "time_since_last_meal", "hours_numeric",
        ],
        "demographics": [
            "age", "bmi", "gender", "height", "waist", "weight", "hip",
        ],
    }

    MEAL_LEVEL_GROUPS = [
        "nutritional", "food_group", "baseline_glucose",
        "past_glucose", "past_food",
    ]
    COVARIATE_GROUPS = ["context", "demographics"]

    VALID_IMPUTATION = {
        None,
        "participant_mean", "participant_median",
        "population_mean", "population_median",
    }

    if imputation not in VALID_IMPUTATION:
        raise ValueError(
            f"Unknown imputation '{imputation}'. "
            f"Valid: {sorted(v for v in VALID_IMPUTATION if v is not None)}"
        )

    remove_set = set(remove_features) if remove_features else set()
    unknown = remove_set - set(FEATURE_GROUPS.keys())

    if unknown:
        raise ValueError(
            f"Unknown feature groups in remove_features: {unknown}. "
            f"Available: {list(FEATURE_GROUPS.keys())}"
        )

    features = [
        c
        for g in MEAL_LEVEL_GROUPS
        if g not in remove_set
        for c in FEATURE_GROUPS[g]
    ]

    other_covariates = [
        c
        for g in COVARIATE_GROUPS
        if g not in remove_set
        for c in FEATURE_GROUPS[g]
    ]

    # Default random slopes = macros
    if random_slopes is None:
        random_slopes = list(macros)
    else:
        random_slopes = list(random_slopes)

    # Full-data mode uses every row for training and leaves validation empty.
    if full_train:
        train_idx = np.arange(len(data))
        val_idx = np.array([], dtype=int)
    elif train_idx is None or val_idx is None:
        raise ValueError("Provide train_idx and val_idx, or set full_train=True.")

    # Split rows before applying the standardized-meal filter.
    df_tr = data.iloc[train_idx].copy()
    df_va = data.iloc[val_idx].copy()

    if remove_std_meals_from_training:
        df_tr = df_tr[df_tr["is_standardized_meal"] == 0]

    # Parse interactions
    interaction_terms = []

    if use_interactions and use_interactions != [False]:
        for it in use_interactions:
            if not it.startswith("cho_"):
                raise ValueError(f"Interaction must be 'cho_<macro>', got: {it}")

            macro = f"{it[4:]}_eaten"

            if macro not in macros:
                raise ValueError(f"Macro '{macro}' not in macros list.")

            interaction_terms.append(it)

    # Validate random slopes against active feature set
    all_features = macros + features

    missing_rs = [v for v in random_slopes if v not in all_features]

    if missing_rs:
        raise ValueError(
            f"random_slopes contains variables not in the active feature set "
            f"(macros + features after removals): {missing_rs}."
        )

    info = {
        "apply_mundlak": apply_mundlak,
        "interaction_terms": interaction_terms,
        "population_z_scoring_features": population_z_scoring_features,
        "population_z_scoring_outcome": population_z_scoring_outcome,
        "using_fixed_scalers": fixed_scalers is not None,
        "macros": macros,
        "random_slopes": random_slopes,
        "removed_groups": sorted(remove_set),
        "active_meal_features": features,
        "active_covariates": other_covariates,
        "fixed_covariates": other_covariates,
        "other_covariates": other_covariates,
        "imputation": imputation,
    }

    X_tr_macros = df_tr[all_features].copy()
    X_va_macros = df_va[all_features].copy()

    # Impute meal-level features before z-scoring.
    X_tr_macros, X_va_macros = _impute_missing(
        X_tr_macros,
        X_va_macros,
        imputation,
        df_tr["subject_key"],
        df_va["subject_key"],
    )

    # Scale meal-level features using training or fixed statistics.
    if population_z_scoring_features:
        if fixed_scalers is None:
            feat_mean = X_tr_macros.mean()
            feat_std = X_tr_macros.std(ddof=1).replace(0, 1.0)
        else:
            feat_mean = _get_fixed_series("z_score_mean", X_tr_macros.columns)
            feat_std = _get_fixed_series("z_score_std", X_tr_macros.columns).replace(0, 1.0)

        info["z_score_mean"] = feat_mean
        info["z_score_std"] = feat_std

        X_tr_macros = (X_tr_macros - feat_mean) / feat_std
        X_va_macros = (X_va_macros - feat_mean) / feat_std

    if other_covariates:
        X_tr_other = df_tr[other_covariates].copy()
        X_va_other = df_va[other_covariates].copy()

        # Impute covariates before z-scoring.
        X_tr_other, X_va_other = _impute_missing(
            X_tr_other,
            X_va_other,
            imputation,
            df_tr["subject_key"],
            df_va["subject_key"],
        )

        if population_z_scoring_features:
            if fixed_scalers is None:
                other_mean = X_tr_other.mean()
                other_std = X_tr_other.std(ddof=1).replace(0, 1.0)
            else:
                other_mean = _get_fixed_series("other_mean", X_tr_other.columns)
                other_std = _get_fixed_series("other_std", X_tr_other.columns).replace(0, 1.0)

            info["other_mean"] = other_mean
            info["other_std"] = other_std

            X_tr_other = (X_tr_other - other_mean) / other_std
            X_va_other = (X_va_other - other_mean) / other_std
    else:
        X_tr_other = pd.DataFrame(index=df_tr.index)
        X_va_other = pd.DataFrame(index=df_va.index)

    # Combine meal-level features and covariates for the fixed effects.
    X_tr_list = [X_tr_macros, X_tr_other]
    X_va_list = [X_va_macros, X_va_other]

    X_train_df = pd.concat(X_tr_list, axis=1)
    X_val_df = pd.concat(X_va_list, axis=1)

    info["input_features"] = X_train_df.columns.tolist()

    X_train = X_train_df.values
    X_val = X_val_df.values

    # Random-effects design: intercept and selected slopes.
    Z_tr_base = X_tr_macros[random_slopes].copy()
    Z_va_base = X_va_macros[random_slopes].copy()

    Z_train = np.column_stack([np.ones(len(Z_tr_base)), Z_tr_base.values])
    Z_val = np.column_stack([np.ones(len(Z_va_base)), Z_va_base.values])

    info["random_effects_columns"] = ["intercept"] + list(random_slopes)

    outcome_cols = (
        [target_outcome]
        if isinstance(target_outcome, str)
        else list(target_outcome)
    )

    y_train = df_tr[outcome_cols].values
    y_val = df_va[outcome_cols].values

    # Scale outcomes using training or fixed statistics.
    if population_z_scoring_outcome:
        if fixed_scalers is None:
            y_mean = y_train.mean(axis=0)
            y_std = y_train.std(axis=0, ddof=1)
            y_std = np.where(y_std == 0, 1.0, y_std)

            y_mean = pd.Series(y_mean, index=outcome_cols)
            y_std = pd.Series(y_std, index=outcome_cols)
        else:
            y_mean = _get_fixed_series("outcome_mean", outcome_cols)
            y_std = _get_fixed_series("outcome_std", outcome_cols).replace(0, 1.0)

        info["outcome_mean"] = y_mean
        info["outcome_std"] = y_std

        y_train = (y_train - y_mean.values) / y_std.values
        y_val = (y_val - y_mean.values) / y_std.values

    subjects = df_tr["subject_key"].unique()
    subject_to_cluster = {s: i for i, s in enumerate(subjects)}

    clusters_train = df_tr["subject_key"].map(subject_to_cluster)
    clusters_val = df_va["subject_key"].map(subject_to_cluster)

    cluster_mapping = pd.DataFrame({
        "cluster": clusters_train.values,
        "subject_key": df_tr["subject_key"].values,
    }).drop_duplicates().reset_index(drop=True)

    # Preserve row alignment when combining identifiers, features and outcomes.
    y_train_df = pd.DataFrame(
        np.asarray(y_train).reshape(len(df_tr), len(outcome_cols)),
        columns=outcome_cols,
        index=df_tr.index,
    )

    y_val_df = pd.DataFrame(
        np.asarray(y_val).reshape(len(df_va), len(outcome_cols)),
        columns=outcome_cols,
        index=df_va.index,
    )

    id_tr = pd.DataFrame(
        {
            "subject_key": df_tr["subject_key"].values,
            "cluster": clusters_train.values,
        },
        index=df_tr.index,
    )

    id_va = pd.DataFrame(
        {
            "subject_key": df_va["subject_key"].values,
            "cluster": clusters_val.values,
        },
        index=df_va.index,
    )

    train_dataframe = pd.concat([id_tr, X_train_df, y_train_df], axis=1)
    test_dataframe = pd.concat([id_va, X_val_df, y_val_df], axis=1)

    # Retain row indices after filtering.
    info["train_idx_used"] = df_tr.index.values
    info["val_idx_used"] = df_va.index.values

    return (
        X_train, X_val,
        Z_train, Z_val,
        y_train, y_val,
        clusters_train, clusters_val,
        cluster_mapping,
        info,
        train_dataframe,
        test_dataframe,
    )


class LeavePercentTimeOut(BaseCrossValidator):
    """
    Leave p% (or k days if p>=1) of each subject's timeline out as a contiguous block per split.
    Requires X to be a pandas DataFrame with a datetime column `time_col`.
    Applies an optional time-based embargo around the validation block within each subject.
    """
    def __init__(self, p=0.2, n_splits=5, time_col="eaten_at",
                 embargo=pd.Timedelta("0h"), shuffle_starts=False, random_state=None):
        self.p = float(p)
        self.n_splits = int(n_splits)
        self.time_col = time_col
        self.embargo = embargo
        self.shuffle_starts = shuffle_starts
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups must be provided")
        if not isinstance(X, pd.DataFrame) or self.time_col not in X.columns:
            raise ValueError("X must be a DataFrame with a datetime column named time_col")

        rng = check_random_state(self.random_state)
        groups = np.asarray(groups)
        unique = np.unique(groups)

        # Precompute per-row day for speed
        times = pd.to_datetime(X[self.time_col].values)
        days = pd.to_datetime(pd.Series(times)).dt.normalize().values  # midnight-normalized dates

        # For each split, choose a start-day index per subject
        for split_id in range(self.n_splits):
            train_idx_all, val_idx_all = [], []

            for g in unique:
                rows = np.where(groups == g)[0]
                # Sort this subject's rows by time
                r_times = times[rows]
                order = np.argsort(r_times)
                rows = rows[order]

                # Unique days and mapping row->day index
                r_days = pd.to_datetime(pd.Series(r_times)).dt.normalize().values
                u_days, inv = np.unique(r_days, return_inverse=True)
                n_days = len(u_days)

                # How many days to hold out
                k_days = int(self.p) if self.p >= 1 else int(np.ceil(self.p * n_days))
                k_days = max(1, min(k_days, n_days - 1))

                # Pick start of contiguous block
                max_start = n_days - k_days
                if self.shuffle_starts:
                    start = int(round(max_start * split_id / max(1, self.n_splits - 1)))

                else:
                    # Evenly sweep start positions across splits
                    start = split_id * k_days
                   
                val_day_idxs = np.arange(start, start + k_days)
                val_mask = np.isin(inv, val_day_idxs)
                val_rows = rows[val_mask]
                train_rows = rows[~val_mask]

                # Optional embargo: drop train samples within +/- embargo of val window
                if self.embargo > pd.Timedelta(0):
                    val_min_t = r_times[val_mask].min()
                    val_max_t = r_times[val_mask].max()
                    left = val_min_t - self.embargo
                    right = val_max_t + self.embargo
                    t_train = r_times[~val_mask]
                    keep = (t_train < left) | (t_train > right)
                    train_rows = train_rows[keep]

                val_idx_all.append(val_rows)
                train_idx_all.append(train_rows)

            yield (np.concatenate(train_idx_all), np.concatenate(val_idx_all))

    def get_validation_windows(self, X, groups):
        """
        Return a DataFrame with validation time windows per subject per split.
        
        Parameters
        ----------
        X : pd.DataFrame
            Must contain self.time_col (datetime).
        groups : array-like
            Subject IDs (same as passed to .split()).
        
        Returns
        -------
        pd.DataFrame with columns:
            split_id, subject_id, val_start_raw, val_end_raw,
            val_start_embargoed, val_end_embargoed
        """
        if not isinstance(X, pd.DataFrame) or self.time_col not in X.columns:
            raise ValueError(f"X must be a DataFrame with datetime column '{self.time_col}'")
        
        rng = check_random_state(self.random_state)
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        times = pd.to_datetime(X[self.time_col].values)
    
        records = []
    
        for split_id in range(self.n_splits):
            for g in unique_groups:
                # Get subject's rows and sort by time
                mask = groups == g
                subj_rows = np.where(mask)[0]
                subj_times = times[mask]
                order = np.argsort(subj_times)
                subj_rows = subj_rows[order]
                subj_times = subj_times[order]
    
                # Get unique days for block selection
                subj_days = pd.to_datetime(subj_times).normalize().values
                unique_days, day_indices = np.unique(subj_days, return_inverse=True)
                n_days = len(unique_days)
    
                if n_days < 2:
                    continue  # Can't split
    
                # Determine holdout size
                k_days = int(self.p) if self.p >= 1 else int(np.ceil(self.p * n_days))
                k_days = max(1, min(k_days, n_days - 1))
                max_start = n_days - k_days
    
                # Choose start index
                if self.shuffle_starts:
                    start_idx = rng.randint(0, max_start + 1)
                else:
                    start_idx = int(round(max_start * split_id / max(1, self.n_splits - 1)))
    
                # Identify validation days and times
                val_day_mask = np.isin(day_indices, np.arange(start_idx, start_idx + k_days))
                val_times_raw = subj_times[val_day_mask]
    
                if len(val_times_raw) == 0:
                    continue
    
                val_start_raw = val_times_raw.min()
                val_end_raw = val_times_raw.max()
    
                # Apply embargo for reporting (optional)
                val_start_embargoed = val_start_raw - self.embargo
                val_end_embargoed = val_end_raw + self.embargo

                validation_window_duration = val_end_raw - val_start_raw
    
                records.append({
                    "split_id": split_id,
                    "subject_key": g,
                    "val_start_raw": val_start_raw,
                    "val_end_raw": val_end_raw,
                    "val_start_embargoed": val_start_embargoed,
                    "val_end_embargoed": val_end_embargoed, 
                    "validation_window_duration": validation_window_duration
                })
    
        return pd.DataFrame(records)
    def get_training_windows(self, X, groups):
        # Return each participant's training intervals for every split.
        if not isinstance(X, pd.DataFrame) or self.time_col not in X.columns:
            raise ValueError(f"X must be a DataFrame with datetime column '{self.time_col}'")
        
        # First get validation windows
        val_windows = self.get_validation_windows(X, groups)
        
        # Get overall date range for each subject
        subject_ranges = X.groupby(groups)[self.time_col].agg(['min', 'max']).reset_index()
        subject_ranges.columns = ['subject_key', 'overall_start', 'overall_end']
        
        # Merge with validation windows
        merged = val_windows.merge(subject_ranges, on='subject_key', how='left')
        
        # Calculate training windows (everything except embargoed validation period)
        records = []
        for _, row in merged.iterrows():
            # Training window is split into two parts around the embargoed validation period
            train_windows = []
            
            # Part 1: Before validation embargo
            if row['overall_start'] < row['val_start_embargoed']:
                train_windows.append({
                    'train_start': row['overall_start'],
                    'train_end': row['val_start_embargoed']
                })
            
            # Part 2: After validation embargo  
            if row['val_end_embargoed'] < row['overall_end']:
                train_windows.append({
                    'train_start': row['val_end_embargoed'], 
                    'train_end': row['overall_end']
                })
            
            # Fall back to the full observed range when no training interval remains.
            if not train_windows:
                train_windows.append({
                    'train_start': row['overall_start'],
                    'train_end': row['overall_end']
                })
            
            # Create record for each training window part
            for i, window in enumerate(train_windows):
                duration = window['train_end'] - window['train_start']
                records.append({
                    "split_id": row['split_id'],
                    "subject_key": row['subject_key'],
                    "train_window_part": i,
                    "train_start": window['train_start'],
                    "train_end": window['train_end'],
                    "train_duration": duration,
                    "val_start_embargoed": row['val_start_embargoed'],
                    "val_end_embargoed": row['val_end_embargoed']
                })
        
        return pd.DataFrame(records)


def make_xgb(njobs=6):
    import xgboost as xgb
    return xgb.XGBRegressor(
        booster="gbtree",
        n_estimators=1200,
        learning_rate=0.02258140698292982,
        max_depth=6,
        subsample=0.7743798042397052,
        colsample_bytree=0.7166838990816966,
        n_jobs=njobs,
        random_state=42,
        reg_alpha=2.3759453153584325,
        reg_lambda=0.031006288561511164,
        min_child_weight=8,
        gamma=0.6335343145218796,
        multi_strategy="one_output_per_tree",
        tree_method="hist",
        enable_categorical=True,
    )    


def make_tag(model_name, std_mode=True, interactions=(False,), mundlak_corr=False):
    """Return the stable output namespace used by the model-comparison tables."""
    return model_name


def cols_for(tag, metrics, suffixes=PREDICTION_SUFFIXES):
    return {suffix: [f"{tag}__{metric}__{suffix}" for metric in metrics] for suffix in suffixes}


def ensure_cols(frame, columns_by_suffix, tag):
    needed = [column for columns in columns_by_suffix.values() for column in columns]
    needed.append(f"{tag}__fold")
    missing = [column for column in needed if column not in frame.columns]
    if not missing:
        return frame
    additions = pd.DataFrame(np.nan, index=frame.index, columns=missing)
    return pd.concat([frame, additions], axis=1)


def evaluate_models(data, models, metrics, std_col="is_standardized_meal"):
    """Evaluate outcome-specific and uniform-average multivariate R-squared."""
    subsets = {"all meals": data}
    if std_col in data.columns:
        subsets["standardized"] = data.loc[data[std_col] == 1]
    rows = []
    for model in models:
        for subset, frame in subsets.items():
            available = [
                metric for metric in metrics
                if f"{model}__{metric}__true" in frame.columns
                and f"{model}__{metric}__pred" in frame.columns
            ]
            for metric in available:
                true_col = f"{model}__{metric}__true"
                pred_col = f"{model}__{metric}__pred"
                paired = frame[[true_col, pred_col]].dropna()
                if len(paired) < 3:
                    continue
                r_value, p_value = pearsonr(paired[true_col], paired[pred_col])
                rows.append({
                    "subset": subset, "Model": model, "Metric": metric,
                    "n": len(paired), "r2": r2_score(paired[true_col], paired[pred_col]),
                    "r": r_value, "p": p_value,
                })
            if len(available) > 1:
                true_cols = [f"{model}__{metric}__true" for metric in available]
                pred_cols = [f"{model}__{metric}__pred" for metric in available]
                paired = frame[true_cols + pred_cols].dropna()
                if len(paired) >= 3:
                    rows.append({
                        "subset": subset, "Model": model,
                        "Metric": "Multivariate (uniform R²)", "n": len(paired),
                        "r2": r2_score(
                            paired[true_cols].to_numpy(), paired[pred_cols].to_numpy(),
                            multioutput="uniform_average",
                        ),
                        "r": None, "p": None,
                    })
    return pd.DataFrame(rows)


def show_performance(performance, metrics):
    metric_order = list(metrics) + ["Multivariate (uniform R²)"]
    tables = {}
    for subset in performance["subset"].unique():
        selected = performance.loc[performance["subset"] == subset].copy()
        selected["cell"] = selected.apply(
            lambda row: f"{row['r2']:.2f}" if pd.isna(row["r"])
            else f"{row['r2']:.2f} | {row['r']:.2f}",
            axis=1,
        )
        table = selected.pivot(index="Metric", columns="Model", values="cell").reindex(metric_order)
        table.insert(0, "n", selected.groupby("Metric")["n"].max().reindex(metric_order))
        print(f"\n── {subset} (R² | Pearson r) ──")
        print(table.fillna("—").to_string())
        tables[subset] = table
    return tables


def save_performance(performance, metric_order, path="model_performance.csv"):
    selected = performance.copy()
    selected["cell"] = selected.apply(
        lambda row: f"{row['r2']:.2f}" if pd.isna(row["r"])
        else f"{row['r2']:.2f} | {row['r']:.2f}",
        axis=1,
    )
    table = selected.pivot_table(index=["subset", "Metric"], columns="Model", values="cell", aggfunc="first")
    order = pd.MultiIndex.from_product([selected["subset"].unique(), metric_order])
    table = table.reindex(order)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.fillna("—").to_csv(path)
    print(f"Saved performance table: {path}")
    return table


def split_half_within_subject(
    data,
    split_mode="even_odd",
    subject_col=SUBJECT_COL,
    date_col=DATE_COL,
    min_meals_per_half=1,
):
    """Split each participant into odd/even or chronological meal halves."""
    if split_mode not in {"even_odd", "temporal"}:
        raise ValueError("split_mode must be 'even_odd' or 'temporal'")
    if min_meals_per_half < 1:
        raise ValueError("min_meals_per_half must be at least 1")

    frame = data.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    half_a, half_b = [], []
    for _, participant in frame.groupby(subject_col, sort=False):
        participant = participant.sort_values(date_col).reset_index(drop=True)
        if split_mode == "even_odd":
            subset_a = participant.iloc[::2].copy()
            subset_b = participant.iloc[1::2].copy()
        else:
            midpoint = len(participant) // 2
            subset_a = participant.iloc[:midpoint].copy()
            subset_b = participant.iloc[midpoint:].copy()
        if len(subset_a) < min_meals_per_half or len(subset_b) < min_meals_per_half:
            continue
        half_a.append(subset_a)
        half_b.append(subset_b)

    if not half_a or not half_b:
        raise ValueError("No participants satisfied the split-half meal threshold")
    half_a = pd.concat(half_a, ignore_index=True)
    half_b = pd.concat(half_b, ignore_index=True)

    def span_for(subset, label):
        span = subset.groupby(subject_col)[date_col].agg(
            first_meal="min", last_meal="max", n_meals="count"
        ).reset_index()
        span["half"] = label
        span["span_days"] = (
            span["last_meal"] - span["first_meal"]
        ).dt.total_seconds() / 86400.0
        return span

    span = pd.concat([span_for(half_a, "A"), span_for(half_b, "B")], ignore_index=True)
    span = span[[subject_col, "half", "first_meal", "last_meal", "n_meals", "span_days"]]
    return half_a, half_b, span.sort_values([subject_col, "half"]).reset_index(drop=True)


def _to_long(scores, pc, subject_col=SUBJECT_COL):
    """Convert paired PC columns into the long layout required by pingouin ICC."""
    return pd.concat(
        [
            scores[[subject_col, f"{pc}_A"]].rename(columns={f"{pc}_A": "rating"}).assign(half="A"),
            scores[[subject_col, f"{pc}_B"]].rename(columns={f"{pc}_B": "rating"}).assign(half="B"),
        ],
        ignore_index=True,
    )


def fit_training_fold(train_data, n_jobs_model=4):
    """
    Fit preprocessing and MMER using training participants only.

    Returns
    -------
    fitted_result
        Fitted MixedModel result returned by MixedEffectEstimator.fit.
    scalers
        Fold-specific preprocessing statistics.
    train_info
        Useful metadata for reproducibility.
    """
    (  X_train,  _,  Z_train,  _,  y_train,  _,  clusters_train,  _,  cluster_mapping,  info_,  _,validation_dataframe,) = build_design_mats(
        train_data,
        full_train=True,
        target_outcome=OUTCOMES,
        apply_mundlak=False,
        use_interactions=[False],
        population_z_scoring_features=True,
        remove_std_meals_from_training=True,
        remove_features=["demographics"],
        random_slopes=RANDOM_SLOPES,
        population_z_scoring_outcome=True,
    )

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    if y_train.ndim == 1:
        y_train = y_train[:, None]

    groups_train = (
        np.asarray(clusters_train)
        .reshape(-1, 1)
    )

    xgb_fe = make_xgb(njobs=n_jobs_model)

    from mmer import MixedEffectEstimator

    estimator = MixedEffectEstimator(
        xgb_fe,
        max_iter=50,
        slq_steps=40,
        tol=1e-7,
        n_jobs=n_jobs_model,
    )

    fitted_result = estimator.fit(
        X_train,
        y_train,
        groups_train,
        random_slopes=([0, 1, 2, 3],),
    )

    scalers = {
        "z_score_mean": info_["z_score_mean"],
        "z_score_std": info_["z_score_std"],
        "other_mean": info_.get("other_mean"),
        "other_std": info_.get("other_std"),
        "outcome_mean": info_.get("outcome_mean"),
        "outcome_std": info_.get("outcome_std"),
    }

    train_info = {
        "n_meals": len(X_train),
        "n_subjects": train_data[SUBJECT_COL].nunique(),
        "cluster_mapping": cluster_mapping,
        "design_info": info_,
        "converged": fitted_result.is_converged,
        "best_log_likelihood": fitted_result.best_log_likelihood,
    }

    return fitted_result, scalers, train_info


def transform_validation_data(validation_data, fold_scalers):
    """
    Build held-out design matrices using preprocessing frozen in training.
    All held-out rows go through as 'train' so the whole half is transformed
    with the fixed scalers.
    """
    (
        X_validation, _, Z_validation, _, y_validation, _,
        clusters_validation, _, cluster_mapping, info_, _, validation_dataframe,
    ) = build_design_mats(
        validation_data,
        full_train=True,
        target_outcome=OUTCOMES,
        apply_mundlak=False,
        use_interactions=[False],
        population_z_scoring_features=True,
        remove_std_meals_from_training=True,
        remove_features=["demographics"],
        random_slopes=RANDOM_SLOPES,
        population_z_scoring_outcome=True,
        fixed_scalers=fold_scalers,           # frozen training scalers
    )

    X_validation = np.asarray(X_validation)
    y_validation = np.asarray(y_validation)
    if y_validation.ndim == 1:
        y_validation = y_validation[:, None]

    groups_validation = np.asarray(clusters_validation).reshape(-1, 1)

    return (
        X_validation,
        y_validation,
        groups_validation,
        validation_dataframe,
        cluster_mapping,
    )


def infer_blups_from_frozen_model(fitted_result, validation_data, fold_scalers):
    """
    Estimate validation-participant BLUPs without refitting MMER, indexed by
    real subject_key (not the local integer cluster ids).
    """
    (
        X_validation,
        y_validation,
        groups_validation,
        validation_dataframe,
        cluster_mapping,
    ) = transform_validation_data(validation_data, fold_scalers)

    blups = fitted_result.blups(
        X_validation,
        y_validation,
        groups_validation,
        group_idx=0,
    ).copy()

    # Map local integer cluster ids -> real subject_key so that half A / half B
    # align on the same participant and downstream merges use real keys.
    cluster_to_subject = (
        cluster_mapping
        .drop_duplicates("cluster")
        .set_index("cluster")["subject_key"]
    )
    blups.index = pd.Index(
        pd.Series(blups.index).map(cluster_to_subject).values,
        name=SUBJECT_COL,
    )

    if blups.index.isna().any():
        raise RuntimeError(
            "Validation BLUP rows could not be mapped back to subject_key; "
            "check that groups passed to .blups() match cluster_mapping."
        )

    return blups


def derive_pc_basis(fitted_result, group_idx=0):
    """
    Eigendecompose the training-fold random-effect covariance.
    """
    G = np.asarray(fitted_result.G[group_idx])

    eigenvalues, eigenvectors = np.linalg.eigh(G)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    q = fitted_result._G[group_idx].n_effects
    m = fitted_result.n_responses

    if q != 5:
        raise ValueError(
            f"Expected 5 random effects per response, found {q}."
        )

    # PC1 orientation:
    # positive values mean higher general response tone.
    intercept_indices = np.array([
        response_index * q
        for response_index in range(m)
    ])

    if eigenvectors[intercept_indices, 0].sum() < 0:
        eigenvectors[:, 0] *= -1

    # PC2 orientation:
    # positive values mean greater carbohydrate responsiveness.
    carbohydrate_indices = np.array([
        response_index * q + 1
        for response_index in range(m)
    ])

    if eigenvectors[carbohydrate_indices, 1].sum() < 0:
        eigenvectors[:, 1] *= -1

    return eigenvalues, eigenvectors


def project_blups_to_pcs(
    blups,
    eigenvalues,
    eigenvectors,
    n_components=2,
    standardize=True,
):
    """
    Project BLUP vectors onto training-derived PC axes.
    """
    blup_matrix = blups.to_numpy()

    scores = (
        blup_matrix
        @ eigenvectors[:, :n_components]
    )

    if standardize:
        pc_sd = np.sqrt(
            np.maximum(
                eigenvalues[:n_components],
                1e-12,
            )
        )

        scores = scores / pc_sd

    score_columns = [
        f"PC{i + 1}"
        for i in range(n_components)
    ]

    return pd.DataFrame(
        scores,
        index=blups.index,
        columns=score_columns,
    )


def make_subject_folds(data, n_splits=5):
    subject_ids = (
        data[SUBJECT_COL]
        .drop_duplicates()
        .to_numpy()
    )

    splitter = GroupKFold(n_splits=n_splits)
    dummy_X = np.zeros((len(subject_ids), 1))

    folds = []

    for fold_id, (train_idx, test_idx) in enumerate(
        splitter.split(dummy_X, groups=subject_ids),
        start=1,
    ):
        folds.append({
            "fold_id": fold_id,
            "train_subjects": subject_ids[train_idx],
            "test_subjects": subject_ids[test_idx],
        })

    return folds


def _fmt_ci(ci, dp=2):
    """2-element CI array -> '[0.41, 0.85]'."""
    if ci is None or (isinstance(ci, float) and np.isnan(ci)):
        return ""
    return f"[{ci[0]:.{dp}f}, {ci[1]:.{dp}f}]"


def icc_split_half(scores, pcs=("PC1", "PC2"),
                   variants=("ICC(A,1)", "ICC(C,1)"), dp=2):
    """Per-fold and pooled split-half reliability summaries."""
    import pingouin as pg
    summary_rows, full_rows = [], []
    scopes = [(f, g) for f, g in scores.groupby("fold")] + [("pooled", scores)]

    for scope, g in scopes:
        for pc in pcs:
            long = _to_long(g, pc)
            n = long[SUBJECT_COL].nunique()
            row = {"fold": scope, "pc": pc, "n": n}

            # wide pairs for correlations (complete cases)
            a, b = f"{pc}_A", f"{pc}_B"
            pair = g[[a, b]].dropna()

            if n < 3:
                for v in variants:
                    row[v] = np.nan
                    row[f"{v} CI"] = ""
                row["r"] = np.nan;   row["r CI"] = ""
                row["rho"] = np.nan; row["rho CI"] = ""
                summary_rows.append(row)
                continue

            tbl = pg.intraclass_corr(
                data=long, targets=SUBJECT_COL, raters="half",
                ratings="rating", nan_policy="omit",
            ).set_index("Type")

            tbl.insert(0, "pc", pc)
            tbl.insert(0, "fold", scope)
            full_rows.append(tbl.reset_index())

            for v in variants:
                row[v] = round(tbl.loc[v, "ICC"], dp)
                row[f"{v} CI"] = _fmt_ci(tbl.loc[v, "CI95"], dp)
            # A,1 and C,1 use the same ANOVA F statistic and p-value.
            row["pval"] = round(tbl.loc[variants[0], "pval"], 4)

            # Pearson + Spearman on the same complete pairs
            if len(pair) >= 4:
                pr = pg.corr(pair[a], pair[b], method="pearson")
                sr = pg.corr(pair[a], pair[b], method="spearman")
                ci_col = "CI95%" if "CI95%" in pr.columns else "CI95"
                row["r"]   = round(pr["r"].iloc[0], dp)
                row["r CI"] = _fmt_ci(pr[ci_col].iloc[0], dp)
                row["rho"] = round(sr["r"].iloc[0], dp)
                row["rho CI"] = _fmt_ci(sr[ci_col].iloc[0], dp)
            else:
                row["r"] = np.nan;   row["r CI"] = ""
                row["rho"] = np.nan; row["rho CI"] = ""

            summary_rows.append(row)

    return pd.DataFrame(summary_rows), pd.concat(full_rows, ignore_index=True)
