"""Rebuild the project's final trained models for the dashboard.

The pipeline (src/crop_yield_pipeline.py) does not save fitted models. This module
re-creates them with the SAME train/test split (80/20, seed 42), the SAME preprocessing
(ColumnTransformer definitions copied from the pipeline) and the SAME tuned
hyperparameters, read from outputs/results.json. No re-tuning is done.

`verify()` recomputes the test-set metrics and compares them with
outputs/tables/model_comparison.csv, so the app can show that the rebuilt models
reproduce the published results.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from utils.data import ALL_FEATURES, CAT, NUM, ROOT, SEED, TARGET, load_clean, load_results, load_table

MODEL_CACHE = ROOT / "models" / "trained_models.joblib"

MODEL_ORDER = [
    "Simple Linear Regression", "Multiple Linear Regression", "MLR (Phase 2 features, OLS)",
    "Ridge Regression", "LASSO Regression", "LASSO (1-SE rule)",
    "Decision Tree Regressor", "Random Forest Regressor", "Gradient Boosting (HistGB)",
]
MAIN_MODELS = ["Simple Linear Regression", "Multiple Linear Regression", "Ridge Regression",
               "LASSO Regression", "Decision Tree Regressor", "Random Forest Regressor"]


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"R2": r2_score(y_true, y_pred), "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mse, "RMSE": float(np.sqrt(mse))}


@st.cache_data(show_spinner=False)
def get_split():
    """Identical to section 3 of the pipeline."""
    df = load_clean()
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=SEED)
    return train_df.copy(), test_df.copy()


def make_prep(num, cat, scale=True):
    """Copied from the pipeline (section 9)."""
    parts = []
    if num:
        parts.append(("num", StandardScaler() if scale else "passthrough", num))
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
    return ColumnTransformer(parts)


def build_estimators(res):
    """Unfitted estimators with the tuned hyperparameters stored in results.json."""
    slr_x = res["slr"]["x"]
    sel = res["feature_selection"]["selected"]
    sel_num = [f for f in NUM if f in sel]
    sel_cat = [f for f in CAT if f in sel]
    reg, tun = res["regularisation"], res["ml_tuning"]
    dt, rf, gb = (tun["Decision Tree"]["best_params"], tun["Random Forest"]["best_params"],
                  tun["Gradient Boosting"]["best_params"])
    return {
        "Simple Linear Regression": Pipeline([
            ("sel", ColumnTransformer([("x", "passthrough", [slr_x])])), ("lr", LinearRegression())]),
        "Multiple Linear Regression": Pipeline([
            ("prep", ColumnTransformer([
                ("num", "passthrough", NUM),
                ("crop", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), ["crop"])])),
            ("lr", LinearRegression())]),
        "MLR (Phase 2 features, OLS)": Pipeline([
            ("prep", ColumnTransformer([
                ("num", StandardScaler(), sel_num),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), sel_cat)])),
            ("lr", LinearRegression())]),
        "Ridge Regression": Pipeline([("prep", make_prep(sel_num, sel_cat)),
                                      ("model", Ridge(alpha=reg["ridge_alpha"]))]),
        "LASSO Regression": Pipeline([("prep", make_prep(sel_num, sel_cat)),
                                      ("model", Lasso(alpha=reg["lasso_alpha"], max_iter=50_000, random_state=SEED))]),
        "LASSO (1-SE rule)": Pipeline([("prep", make_prep(sel_num, sel_cat)),
                                       ("model", Lasso(alpha=reg["lasso_1se_alpha"], max_iter=50_000, random_state=SEED))]),
        "Decision Tree Regressor": Pipeline([
            ("prep", make_prep(sel_num, sel_cat, scale=False)),
            ("model", DecisionTreeRegressor(random_state=SEED, max_depth=dt["max_depth"],
                                            min_samples_leaf=dt["min_samples_leaf"]))]),
        "Random Forest Regressor": Pipeline([
            ("prep", make_prep(sel_num, sel_cat, scale=False)),
            ("model", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1,
                                            max_depth=rf["max_depth"], max_features=rf["max_features"],
                                            min_samples_leaf=rf["min_samples_leaf"]))]),
        "Gradient Boosting (HistGB)": Pipeline([
            ("prep", make_prep(sel_num, sel_cat, scale=False)),
            ("model", HistGradientBoostingRegressor(random_state=SEED, learning_rate=gb["learning_rate"],
                                                    max_iter=gb["max_iter"], max_leaf_nodes=gb["max_leaf_nodes"],
                                                    min_samples_leaf=gb["min_samples_leaf"]))]),
    }


def _fingerprint(res):
    return {"reg": res["regularisation"].get("ridge_alpha"), "tun": res["ml_tuning"],
            "n_train": res["split"]["train"], "sel": res["feature_selection"]["selected"]}


@st.cache_resource(show_spinner="Loading trained models (first run trains them once, ~1 minute)…")
def get_models():
    """Fitted models keyed by name. Cached in memory and on disk (models/, git-ignored)."""
    res = load_results()
    fp = _fingerprint(res)
    if MODEL_CACHE.exists():
        try:
            saved = joblib.load(MODEL_CACHE)
            if saved.get("fingerprint") == fp:
                return saved["models"]
        except Exception:
            pass  # corrupt / incompatible cache -> retrain
    train_df, _ = get_split()
    X, y = train_df[ALL_FEATURES], train_df[TARGET]
    models = {name: est.fit(X, y) for name, est in build_estimators(res).items()}
    MODEL_CACHE.parent.mkdir(exist_ok=True)
    joblib.dump({"fingerprint": fp, "models": models}, MODEL_CACHE, compress=3)
    return models


@st.cache_data(show_spinner=False)
def test_predictions():
    """Test-set predictions of every model (DataFrame, one column per model)."""
    _, test_df = get_split()
    models = get_models()
    out = test_df[["country", "crop", "year", TARGET]].copy()
    for name in MODEL_ORDER:
        out[name] = models[name].predict(test_df[ALL_FEATURES])
    return out


@st.cache_data(show_spinner=False)
def verify():
    """Recomputed test metrics vs the pipeline's model_comparison.csv."""
    preds = test_predictions()
    published = load_table("model_comparison", index_col="Model")
    rows = []
    for name in MODEL_ORDER:
        m = metrics(preds[TARGET], preds[name])
        rows.append({"Model": name, "R2 (app)": m["R2"], "R2 (pipeline)": published.loc[name, "R2"],
                     "RMSE (app)": m["RMSE"], "RMSE (pipeline)": published.loc[name, "RMSE"],
                     "abs. diff RMSE": abs(m["RMSE"] - published.loc[name, "RMSE"])})
    return pd.DataFrame(rows).set_index("Model")


@st.cache_resource(show_spinner=False)
def get_statsmodels():
    """The Phase 1 statsmodels fits (identical formulas to the pipeline)."""
    res = load_results()
    train_df, test_df = get_split()
    slr = smf.ols(f"{TARGET} ~ {res['slr']['x']}", data=train_df).fit()
    mlr = smf.ols(res["mlr"]["formula"], data=train_df).fit()
    mlr_log = smf.ols(res["mlr"]["formula"].replace(TARGET, f"np.log({TARGET})"), data=train_df).fit()
    return slr, mlr, mlr_log


def model_inputs(name, res):
    """Which raw inputs a model actually uses (for the prediction page)."""
    if name == "Simple Linear Regression":
        return [res["slr"]["x"]]
    if name == "Multiple Linear Regression":
        return NUM + ["crop"]
    return res["feature_selection"]["selected"]
