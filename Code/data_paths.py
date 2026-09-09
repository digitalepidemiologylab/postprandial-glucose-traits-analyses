"""Canonical input paths for the MMER notebooks.

Analyses use the three prepared tables in Data/, with meal_ppgr.zip supported
when meal_ppgr.csv is absent. Preprocessing additionally uses private source
files and intermediates in Data/raw data/ (Git-ignored).
Importing this module does not read data or create any directories.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "Code"
DATA_DIR = PROJECT_ROOT / "Data"

METADATA_PATH = DATA_DIR / "metadata.csv"
MEAL_DATA_PATH = DATA_DIR / "meal_ppgr.csv"
if not MEAL_DATA_PATH.is_file() and (DATA_DIR / "meal_ppgr.zip").is_file():
    MEAL_DATA_PATH = DATA_DIR / "meal_ppgr.zip"
CGM_METRICS_PATH = DATA_DIR / "cgm_metrics.csv"

RAW_DATA_DIR = DATA_DIR / "raw data"
RAW_METADATA_PATH = RAW_DATA_DIR / "metadata.csv"
RAW_CGM_PATH = RAW_DATA_DIR / "cgm_data.csv"
RAW_FOOD_PATH = RAW_DATA_DIR / "mfr_food_and_you.csv"
MERGED_MEALS_PATH = RAW_DATA_DIR / "meal_data.csv"
CLEANED_FOOD_PATH = RAW_DATA_DIR / "mfr_food_and_you_cleaned.csv"
