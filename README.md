# Crop Yield Prediction Using Statistical Modeling and Machine Learning

AAT project for **Statistical Modeling (24AM5PCSML)**, Department of Machine Learning, B.M.S. College of Engineering, 2026–27.
SDG 2: Zero Hunger.

## Contents

| Path | What it is |
|---|---|
| `REPORT.md` | Full academic report (sections 1–21 of the AAT structure) |
| `PRESENTATION.md` | Content for the 12-slide presentation, with the figure to use on each slide |
| `src/crop_yield_pipeline.py` | Integrated, executable pipeline (Phase 1 + Phase 2) |
| `src/build_notebook.py` | Turns the pipeline into `Crop_Yield_Prediction.ipynb` |
| `Crop_Yield_Prediction.ipynb` | Notebook version, executed, with outputs |
| `data/raw/` | Original files from Zenodo record 18761663 (unmodified) |
| `data/crop_yield_clean.csv` | Cleaned analytical dataset written by the pipeline |
| `outputs/run_log.txt` | Full console output of the last run (every number in the report comes from here) |
| `outputs/results.json` | All results in machine-readable form |
| `outputs/figures/` | 21 figures (EDA, diagnostics, bootstrap, tuning, comparison) |
| `outputs/tables/` | CSV tables (descriptive stats, coefficients, VIF, CV folds, comparison, …) |

## Dataset

Laktionov, I., Semenov, S., & Diachenko, G. (2026). *Dataset for Crop Yield Prediction* [Data set]. Zenodo.
https://zenodo.org/records/18761663 — licence CC-BY-4.0.
Yield and pesticide data come from FAOSTAT (QCL and RP domains). Rainfall and temperature come from the Kaggle
*Crop Yield Prediction Dataset* (patelris).

The files can be downloaded again with:

```bash
for f in df_yield_merged.csv yield.csv rainfall.csv pesticides.csv temp.csv; do
  curl -L -o data/raw/$f "https://zenodo.org/api/records/18761663/files/$f/content"; done
```

## How to run

```bash
pip install -r requirements.txt
python src/crop_yield_pipeline.py          # about 8–10 min on a laptop; writes outputs/
python src/build_notebook.py               # optional: build the notebook
jupyter nbconvert --to notebook --execute --inplace Crop_Yield_Prediction.ipynb
```

The random seed is 42 everywhere. Set `N_JOBS=1` to limit the run to one CPU core.
