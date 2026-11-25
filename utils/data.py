"""Loading of the project's existing data and result files (read-only).

Everything shown in the app comes from files written by src/crop_yield_pipeline.py:
  data/crop_yield_clean.csv, outputs/results.json, outputs/tables/*.csv, outputs/figures/*.png
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN = ROOT / "data" / "crop_yield_clean.csv"
DATA_RAW = ROOT / "data" / "raw" / "df_yield_merged.csv"
RESULTS = ROOT / "outputs" / "results.json"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

NUM = ["avg_rainfall_mm", "log_pesticides", "avg_temp_c", "year"]
CAT = ["crop", "country"]
ALL_FEATURES = NUM + CAT
TARGET = "yield_t_ha"
SEED = 42

LABELS = {
    "avg_rainfall_mm": "Rainfall (mm/year)",
    "pesticides_tonnes": "Pesticides (tonnes)",
    "log_pesticides": "log(1 + pesticides)",
    "avg_temp_c": "Avg. temperature (°C)",
    "year": "Year",
    "yield_t_ha": "Yield (t/ha)",
    "crop": "Crop",
    "country": "Country",
}

DATASET_URL = "https://zenodo.org/records/18761663"


def missing_outputs():
    """Return the list of required pipeline outputs that are not present."""
    need = [DATA_CLEAN, RESULTS, TABLES / "model_comparison.csv"]
    return [str(p.relative_to(ROOT)) for p in need if not p.exists()]


@st.cache_data(show_spinner=False)
def load_clean():
    return pd.read_csv(DATA_CLEAN)


@st.cache_data(show_spinner=False)
def load_raw():
    return pd.read_csv(DATA_RAW)


@st.cache_data(show_spinner=False)
def load_results():
    with open(RESULTS, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_table(name, index_col=None):
    return pd.read_csv(TABLES / f"{name}.csv", index_col=index_col)


def figure_path(name):
    p = FIGURES / f"{name}.png"
    return p if p.exists() else None
