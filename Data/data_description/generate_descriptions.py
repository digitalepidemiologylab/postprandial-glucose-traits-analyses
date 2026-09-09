#!/usr/bin/env python3
"""Generate the three aggregate article tables from the prepared analysis inputs.

Run from the repository root:
    python Data/data_description/generate_descriptions.py

See README.md in this directory for the cohort definitions and units.
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import math
from pathlib import Path
import statistics
from typing import Iterable

import numpy as np
import pandas as pd

SOURCE_FILES = (
    'Data/meal_ppgr.csv', 'Data/metadata.csv', 'Data/cgm_metrics.csv',
    'Code/utils.py', 'Code/04_full_model_and_characterization.ipynb',
    'Code/08_outcome_selection_sensitivity.ipynb',
)


def build_tables(project_root: Path) -> dict:
    """Apply the shared eligibility filters and calculate unscaled sample means/SDs.

    The loader definitions are taken directly from Code/utils.py by name, so its
    filtering logic is not duplicated. Only those definitions are executed; this
    avoids importing scipy/sklearn or any of the modeling functions. Notebook
    settings and feature lists are read as literals, never executed. A changed
    or unsupported analysis configuration fails explicitly rather than silently
    producing tables for a different model.
    """
    ROOT = project_root
    def load_analysis_data():
        # Execute the unmodified shared loader without importing model dependencies.
        tree = ast.parse((ROOT / 'Code/utils.py').read_text())
        names = {'SUBJECT_COL', 'DATE_COL', 'POSITIVE_FOOD_COLUMNS', 'RANDOM_SLOPES', 'OUTCOMES'}
        nodes = [n for n in tree.body if
                 (isinstance(n, ast.FunctionDef) and n.name in {'_require_input_columns', 'load_and_prepare_data'}) or
                 (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
        env = {'pd': pd, 'np': np, '_ast': ast, 'Iterable': Iterable}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<shared input loader>', 'exec'), env)
        metadata, data, _, _ = env['load_and_prepare_data'](ROOT / 'Data/metadata.csv', ROOT / 'Data/meal_ppgr.csv')
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_design_mats')
        feature_groups = next(ast.literal_eval(n.value) for n in ast.walk(func)
                              if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FEATURE_GROUPS' for t in n.targets))
        return metadata, data, feature_groups, env['RANDOM_SLOPES'], env['OUTCOMES']


    def notebook_code(notebook):
        cells = json.loads((ROOT / 'Code' / notebook).read_text(encoding='utf-8'))['cells']
        return [ast.parse(''.join(cell['source'])) for cell in cells if cell['cell_type'] == 'code']


    def notebook_assignment(notebook, name):
        """Read the named setting without executing a notebook or relying on cell numbers."""
        values = [ast.literal_eval(node.value)
                  for tree in notebook_code(notebook) for node in ast.walk(tree)
                  if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
                                                         for target in node.targets)]
        if not values or any(value != values[0] for value in values[1:]):
            raise ValueError(f'Missing or conflicting {name} definitions in {notebook}')
        return values[0]


    def main_model_call():
        """Locate the first full-data fit in the main characterization notebook."""
        for tree in notebook_code('04_full_model_and_characterization.ipynb'):
            for statement in tree.body:
                if not isinstance(statement, (ast.Assign, ast.Expr)):
                    continue
                for node in ast.walk(statement):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                            and node.func.id == 'build_design_mats'
                            and any(key.arg == 'full_train' and isinstance(key.value, ast.Constant)
                                    and key.value.value is True for key in node.keywords)):
                        return node
        raise ValueError('Cannot find the full-data model configuration in notebook 04')


    metadata, data, groups, macros, outcomes = load_analysis_data()
    assert data['is_standardized_meal'].isin([False, True]).all()
    free = data.loc[data['is_standardized_meal'].eq(False)].copy()
    standardized = data.loc[data['is_standardized_meal'].eq(True)].copy()
    assert not free['is_standardized_meal'].any()
    assert len(free) + len(standardized) == len(data)
    assert standardized['standardized_meal_id'].notna().all()
    assert metadata['subject_key'].is_unique
    assert free['subject_key'].nunique() == data['subject_key'].nunique()
    participant_ids = free['subject_key'].unique()
    cgm = pd.read_csv(ROOT / 'Data/cgm_metrics.csv', index_col=0)
    assert cgm['subject_key'].is_unique
    cgm = cgm.loc[cgm['subject_key'].isin(participant_ids)].copy()
    assert len(cgm) == len(participant_ids)

    call = main_model_call()
    assert next(ast.literal_eval(k.value) for k in call.keywords if k.arg == 'remove_features') == ['demographics']
    assert next(ast.literal_eval(k.value) for k in call.keywords if k.arg == 'remove_std_meals_from_training') is True

    # Metadata dictionaries contain display labels only; active variables come from code.
    labels = {
        'carb_eaten': 'Carbohydrate', 'fat_eaten': 'Fat', 'protein_eaten': 'Protein', 'fiber_eaten': 'Dietary fiber',
        'energy_kcal_eaten': 'Meal energy', 'eaten_quantity_in_gram': 'Meal mass',
        'estimated_net_carb_eaten': 'Estimated net carbohydrate',
        'estimated_sugar_fraction': 'Estimated sugar fraction',
        'alcohol_eaten': 'Alcohol', 'beta_carotene_eaten': 'Beta-carotene',
        'calcium_eaten': 'Calcium', 'cholesterol_eaten': 'Cholesterol',
        'fatty_acids_monounsaturated_eaten': 'Monounsaturated fatty acids',
        'fatty_acids_polyunsaturated_eaten': 'Polyunsaturated fatty acids',
        'fatty_acids_saturated_eaten': 'Saturated fatty acids',
        'folate_eaten': 'Folate', 'iron_eaten': 'Iron', 'magnesium_eaten': 'Magnesium',
        'niacin_eaten': 'Niacin', 'pantothenic_acid_eaten': 'Pantothenic acid',
        'phosphorus_eaten': 'Phosphorus', 'potassium_eaten': 'Potassium',
        'salt_eaten': 'Salt', 'sodium_eaten': 'Sodium', 'sugar_eaten': 'Sugar',
        'vitamin_b1_eaten': 'Vitamin B1 (thiamine)', 'vitamin_b12_eaten': 'Vitamin B12 (cobalamin)',
        'vitamin_b2_eaten': 'Vitamin B2 (riboflavin)', 'vitamin_b6_eaten': 'Vitamin B6',
        'vitamin_c_eaten': 'Vitamin C', 'vitamin_d_eaten': 'Vitamin D', 'zinc_eaten': 'Zinc',
        'grains_potatoes_pulses': 'Grains, potatoes and pulses',
        'sweets_salty_snacks_alcohol': 'Sweets, salty snacks and alcoholic beverages',
        'non_alcoholic_beverages': 'Non-alcoholic beverages',
        'dairy_products_meat_fish_eggs_tofu': 'Dairy products, meat, fish, eggs and tofu',
        'vegetables_fruits': 'Vegetables and fruit', 'oils_fats_nuts': 'Oils, fats and nuts',
        'premeal_glucose': 'Premeal glucose',
        'time_since_last_meal': 'Time since previous meal', 'hours_numeric': 'Meal time of day',
    }
    for h in [1, 2, 4, 6]:
        labels[f'trend_glu_{h}'] = f'Glucose trend over the previous {h} h'
    for name in groups['past_food']:
        _, window, *nutrient = name.split('_')
        nutrient_name = '_'.join(nutrient)
        label = 'Energy' if nutrient_name == 'energy_kcal_eaten' else labels[nutrient_name]
        labels[name] = f'{label} intake in the previous {window[:-2]} h'

    categories = {
        'macros': 'Meal composition', 'nutritional': 'Meal composition', 'food_group': 'Meal composition',
        'past_food': 'Recent dietary intake', 'baseline_glucose': 'Premeal glycemic state',
        'past_glucose': 'Premeal glycemic state', 'context': 'Meal timing',
    }
    subcategories = {'macros': 'Macronutrients', 'nutritional': 'Energy and nutrient composition',
                     'food_group': 'Food-group composition', 'past_food': 'Prior nutrient and energy intake',
                     'baseline_glucose': 'Baseline glucose', 'past_glucose': 'Prior glucose trends', 'context': 'Timing'}

    micronutrients = {'beta_carotene_eaten', 'folate_eaten', 'vitamin_b12_eaten', 'vitamin_d_eaten',
                      'calcium_eaten', 'cholesterol_eaten', 'iron_eaten', 'magnesium_eaten', 'niacin_eaten',
                      'pantothenic_acid_eaten', 'phosphorus_eaten', 'potassium_eaten', 'sodium_eaten',
                      'vitamin_b1_eaten', 'vitamin_b2_eaten', 'vitamin_b6_eaten', 'vitamin_c_eaten', 'zinc_eaten'}
    grams_confirmed = True  # Nutrient amounts have already been converted to grams.


    def nutrient_unit(name):
        if name == 'estimated_sugar_fraction':
            return 'Ratio', 1.0
        if 'energy_kcal' in name:
            return 'kcal', 1.0
        if name == 'premeal_glucose':
            return 'mg/dL', 1.0
        if name.startswith('trend_glu_'):
            return 'mg/dL/h', 1.0
        if name in groups['context']:
            return 'h', 1.0
        return 'g', 1.0


    def summary(series, factor=1.0):
        numeric = pd.to_numeric(series, errors='raise')
        assert not np.isinf(numeric.dropna()).any(), series.name
        valid = numeric.dropna() * factor
        assert len(valid) > 1, series.name
        mean, sd = float(valid.mean()), float(valid.std(ddof=1))
        # Independent arithmetic check against Python's statistics implementation.
        assert math.isclose(mean, statistics.mean(valid), rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(sd, statistics.stdev(valid), rel_tol=1e-12, abs_tol=1e-12)
        return {'mean': mean, 'sd': sd, 'n': len(valid), 'missing': int(numeric.isna().sum())}


    feature_rows = []
    ordered = [('macros', macros)] + [(g, groups[g]) for g in ['nutritional', 'food_group', 'past_food', 'baseline_glucose', 'past_glucose', 'context']]
    for group, variables in ordered:
        for name in variables:
            unit, factor = nutrient_unit(name)
            feature_rows.append({'category': categories[group], 'subcategory': subcategories[group],
                                 'label': labels[name], 'variable': name, 'unit': unit,
                                 'level': 'Meal', 'source': 'meal_ppgr.csv',
                                 'scientific_notation': name in micronutrients,
                                 'display_factor': factor, **summary(free[name], factor)})
    expected = set(macros + [name for group, names in groups.items() if group != 'demographics' for name in names])
    assert len(feature_rows) == len(expected)
    assert {row['variable'] for row in feature_rows} == expected

    metric_labels = {
        'max_glucose': ('Peak postprandial glucose', 'mg/dL'),
        'end_glucose': ('Glucose at 120 minutes', 'mg/dL'),
        'peak_duration': ('Time above premeal glucose', 'min'),
        'positive_iAUC': ('Positive incremental area under the glucose curve', 'mg·min/dL'),
        'delta_max_glucose': ('Peak glucose increase above baseline', 'mg/dL'),
        'delta_end_glucose': ('Glucose at 120 minutes minus baseline', 'mg/dL'),
    }
    metric_rows = []
    for name in outcomes:
        label, unit = metric_labels[name]
        metric_rows.append({'category': 'Postprandial glucose response', 'label': label, 'variable': name,
                           'unit': unit, 'level': 'Meal', 'source': 'meal_ppgr.csv', **summary(free[name])})

    absolute = notebook_assignment('08_outcome_selection_sensitivity.ipynb', 'ABSOLUTE_OUTCOMES')
    delta_names = notebook_assignment('08_outcome_selection_sensitivity.ipynb', 'DELTA_OUTCOMES')
    complete = free.dropna(subset=[*absolute, 'premeal_glucose']).copy()
    for name, absolute_name in [('delta_max_glucose', 'max_glucose'), ('delta_end_glucose', 'end_glucose')]:
        assert name in delta_names
        values = complete[absolute_name] - complete['premeal_glucose']
        values.name = name
        label, unit = metric_labels[name]
        metric_rows.append({'category': 'PPGR sensitivity outcomes', 'label': label, 'variable': name,
                           'unit': unit, 'level': 'Meal', 'source': 'meal_ppgr.csv (derived as in notebook 08)', **summary(values)})

    cgm_labels = {
        'mean_glucose': ('Mean glucose', 'mg/dL', 'Glucose distribution'),
        'median_glucose': ('Median glucose', 'mg/dL', 'Glucose distribution'),
        'min_glucose': ('Minimum glucose', 'mg/dL', 'Glucose distribution'),
        'max_glucose': ('Maximum glucose', 'mg/dL', 'Glucose distribution'),
        'q1_glucose': ('25th percentile of glucose', 'mg/dL', 'Glucose distribution'),
        'q3_glucose': ('75th percentile of glucose', 'mg/dL', 'Glucose distribution'),
        'interdaysd': ('Overall glucose SD', 'mg/dL', 'Glucose variability'),
        'interdaycv': ('Overall glucose coefficient of variation', '%', 'Glucose variability'),
        'intradaysd_mean': ('Mean daily glucose SD', 'mg/dL', 'Glucose variability'),
        'intradaysd_median': ('Median daily glucose SD', 'mg/dL', 'Glucose variability'),
        'intradaysd_sd': ('SD of daily glucose SDs', 'mg/dL', 'Glucose variability'),
        'intradaycv_mean': ('Mean daily glucose coefficient of variation', '%', 'Glucose variability'),
        'intradaycv_median': ('Median daily glucose coefficient of variation', '%', 'Glucose variability'),
        'intradaycv_sd': ('SD of daily glucose coefficients of variation', 'percentage points', 'Glucose variability'),
        'TOR': ('Total time outside 70–180 mg/dL', 'min', 'Time in range'),
        'TIR': ('Total time within 70–180 mg/dL', 'min', 'Time in range'),
        'POR': ('Percentage of readings outside 70–180 mg/dL', '%', 'Time in range'),
        'MAGE': ('Mean amplitude of glycemic excursions', 'mg/dL', 'Glucose variability'),
        'J_index': ('J-index', 'Index', 'Glycemic indices'),
        'LBGI': ('Low blood glucose index', 'Index', 'Glycemic indices'),
        'HBGI': ('High blood glucose index', 'Index', 'Glycemic indices'),
        'ADRR': ('Average daily risk range', 'Index', 'Glycemic indices'),
        'MODD': ('Mean of daily differences', 'mg/dL', 'Glucose variability'),
        'CONGA24': ('Continuous overall net glycemic action over 24 h', 'mg/dL', 'Glucose variability'),
        'GMI': ('Glucose management indicator', '%', 'Estimated glycated hemoglobin'),
        'eA1c': ('Estimated HbA1c', '%', 'Estimated glycated hemoglobin'),
    }
    cgm_names = notebook_assignment('04_full_model_and_characterization.ipynb', 'cgm_metrics_indices')
    for name in cgm_names:
        label, unit, category = cgm_labels[name]
        metric_rows.append({'category': f'CGM: {category.lower()}', 'label': label, 'variable': name,
                           'unit': unit, 'level': 'Participant', 'source': 'cgm_metrics.csv', **summary(cgm[name])})

    category_order = ['Postprandial glucose response', 'PPGR sensitivity outcomes',
                      'CGM: glucose distribution', 'CGM: glucose variability',
                      'CGM: time in range', 'CGM: glycemic indices', 'CGM: estimated glycated hemoglobin']
    metric_rows.sort(key=lambda row: category_order.index(row['category']))
    for row in metric_rows:
        if row['category'] == 'CGM: estimated glycated hemoglobin':
            row['category'] = 'CGM: HbA1c estimates'

    # Notebook 04 derives these two participant summaries from all eligible meals.
    premeal = data.groupby('subject_key')['premeal_glucose'].agg(mean_premeal_glucose='mean', std_premeal_glucose='std')
    for name, label in [('mean_premeal_glucose', 'Participant mean premeal glucose'),
                        ('std_premeal_glucose', 'Participant SD of premeal glucose')]:
        metric_rows.append({'category': 'Participant premeal glucose summaries', 'label': label,
                           'variable': name, 'unit': 'mg/dL', 'level': 'Participant',
                           'source': 'meal_ppgr.csv (all eligible meals; derived as in notebook 04)', **summary(premeal[name])})

    # Give PPGR outcomes their own table, with one column per observed meal type.
    standard_labels = {'A': 'glucose drink', 'B': 'white bread', 'C': 'white bread with butter'}
    meal_groups = [('free_living', 'Free-living meals', free)]
    for meal_id in sorted(standardized['standardized_meal_id'].unique()):
        label = f'Standardized {meal_id}'
        if meal_id in standard_labels:
            label += f': {standard_labels[meal_id]}'
        meal_groups.append((str(meal_id), label,
                            standardized.loc[standardized['standardized_meal_id'].eq(meal_id)].copy()))
    assert sum(len(frame) for _, _, frame in meal_groups) == len(data)
    ppgr_rows = []
    for original in [r for r in metric_rows if r['level'] == 'Meal']:
        row = {key: original[key] for key in ['category', 'label', 'variable', 'unit']}
        row['groups'] = {}
        for key, _, frame in meal_groups:
            if row['variable'] in ['delta_max_glucose', 'delta_end_glucose']:
                selected = frame.dropna(subset=[*absolute, 'premeal_glucose'])
                absolute_name = 'max_glucose' if row['variable'] == 'delta_max_glucose' else 'end_glucose'
                values = selected[absolute_name] - selected['premeal_glucose']
            else:
                values = frame[row['variable']]
            values.name = row['variable']
            row['groups'][key] = summary(values)
            # Counts in column headers apply to every displayed outcome in this group.
            assert row['groups'][key]['n'] == len(frame), (row['variable'], key)
        for name in ['mean', 'sd', 'n']:
            assert row['groups']['free_living'][name] == original[name], row['variable']
        if row['variable'] in outcomes:
            overall = summary(data[row['variable']])
            combined_mean = sum(g['n'] * g['mean'] for g in row['groups'].values()) / overall['n']
            combined_variance = sum((g['n'] - 1) * g['sd'] ** 2 + g['n'] * (g['mean'] - combined_mean) ** 2
                                    for g in row['groups'].values()) / (overall['n'] - 1)
            assert math.isclose(combined_mean, overall['mean'], rel_tol=1e-12)
            assert math.isclose(math.sqrt(combined_variance), overall['sd'], rel_tol=1e-12)
        ppgr_rows.append(row)
    metric_rows = [row for row in metric_rows if row['level'] == 'Participant']
    group_details = [{'key': key, 'label': label, 'meals': len(frame), 'participants': int(frame['subject_key'].nunique())}
                     for key, label, frame in meal_groups]

    notes = {
        'eligible_meals': len(data), 'free_living_meals': len(free), 'participants': len(participant_ids),
        'cgm_participants': len(cgm), 'feature_count': len(feature_rows), 'glycemic_metric_count': len(metric_rows),
        'ppgr_outcome_count': len(ppgr_rows), 'ppgr_meal_groups': group_details,
        'cgm_metric_count': len(cgm_names), 'nutrients_in_grams_confirmed': grams_confirmed,
        'method': 'Sample SD (ddof=1); available observations; before model scaling and imputation.',
        'cohort': 'Shared load_and_prepare_data filters. Predictors use only is_standardized_meal == False; PPGR outcomes are split into free-living meals and each standardized_meal_id among True rows.',
        'derived_participant_glucose': 'Mean and sample SD of premeal glucose within each participant over all eligible meals, matching notebook 04; table mean and SD then computed across participants.',
        'demographics': 'Demographic covariates are excluded from the main prediction model and from the predictor table.',
        'estimated_sugar_fraction': 'max(sugar_eaten / (carb_eaten + 1), 0), matching the loader.',
    }
    return {'features': feature_rows, 'glycemic_metrics': metric_rows, 'ppgr_outcomes': ppgr_rows, 'notes': notes}


def format_number(value: float, scientific: bool = False) -> str:
    if scientific:
        mantissa, exponent = f'{value:.2e}'.split('e')
        return f'{mantissa}e{int(exponent):+d}'
    return f'{value:,.2f}'


def mean_sd(row: dict) -> str:
    scientific = row.get('scientific_notation', False) or any(
        value != 0 and abs(value) < 0.01 for value in (row['mean'], row['sd']))
    return f"{format_number(row['mean'], scientific)} ± {format_number(row['sd'], scientific)}"


def csv_bytes(headers: list, rows: list) -> bytes:
    """Use UTF-8 BOM, CSV quoting, and CRLF line endings for spreadsheet readers."""
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer, lineterminator='\r\n')
    writer.writerow(headers)
    writer.writerows(rows)
    return buffer.getvalue().encode('utf-8-sig')


def render_tables(tables: dict) -> dict[str, bytes]:
    """Convert verified full-precision summaries to the article's display format."""
    groups = tables['notes']['ppgr_meal_groups']
    return {
        'analysis_features_summary.csv': csv_bytes(
            ['Category', 'Feature group', 'Feature', 'Dataset variable', 'Unit', 'Mean ± SD', 'N'],
            [[r['category'], r['subcategory'], r['label'], r['variable'], r['unit'], mean_sd(r), r['n']]
             for r in tables['features']]),
        'glycemic_metrics_summary.csv': csv_bytes(
            ['Category', 'Metric', 'Dataset variable', 'Unit', 'Mean ± SD', 'N', 'Observation level'],
            [[r['category'], r['label'], r['variable'], r['unit'], mean_sd(r), r['n'], r['level']]
             for r in tables['glycemic_metrics']]),
        'ppgr_outcomes_by_meal_type.csv': csv_bytes(
            ['Category', 'PPGR outcome', 'Dataset variable', 'Unit',
             *[f"{g['label']} (n={g['meals']:,}): Mean ± SD" for g in groups]],
            [[r['category'], r['label'], r['variable'], r['unit'],
              *[mean_sd({**r['groups'][g['key']], 'unit': r['unit']}) for g in groups]]
             for r in tables['ppgr_outcomes']]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2],
                        help='Repository root (defaults to the location of this script).')
    parser.add_argument('--output-dir', type=Path,
                        help='Destination directory (defaults to Data/data_description/).')
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = (args.output_dir or project_root / 'Data' / 'data_description').resolve()
    missing = [name for name in SOURCE_FILES if not (project_root / name).is_file()]
    if missing:
        parser.error('Missing required inputs/source files: ' + ', '.join(missing))

    tables = build_tables(project_root)
    payloads = render_tables(tables)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        (output_dir / name).write_bytes(data)
    print(f'Generated three CSVs in {output_dir}')
    print(f"Predictors: {len(tables['features'])}; participant metrics: {len(tables['glycemic_metrics'])}; "
          f"PPGR outcomes: {len(tables['ppgr_outcomes'])}.")


if __name__ == '__main__':
    main()
