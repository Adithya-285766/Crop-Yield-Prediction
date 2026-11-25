import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import LABELS, figure_path, load_results, load_table
from utils.models import get_models
from utils.ui import GREEN, cards, note, page_header, section, show_png

page_header("Phase 2: Advanced Modeling", "Feature selection, Ridge and LASSO regularisation, and tree-based ML models")

res = load_results()
comp = load_table("model_comparison", index_col="Model")
FS = res["feature_selection"]
REG = res["regularisation"]
TUN = res["ml_tuning"]


def grp(name):
    n = name.split("__", 1)[1]
    return n.split("_")[0] if name.startswith("cat__") else n


def metric_cards(model):
    r = comp.loc[model]
    cards([("Test R²", f"{r['R2']:.4f}", f"train {r['Train_R2']:.4f}"),
           ("Test MAE", f"{r['MAE']:.4f}", "t/ha"),
           ("Test MSE", f"{r['MSE']:.4f}", "(t/ha)²"),
           ("Test RMSE", f"{r['RMSE']:.4f}", f"CV RMSE {r['CV_RMSE_mean']:.3f} ± {r['CV_RMSE_std']:.3f}")])


tab_fs, tab_reg, tab_dt, tab_rf, tab_gb = st.tabs(["Feature selection", "Ridge & LASSO", "Decision Tree",
                                                   "Random Forest", "Gradient Boosting"])

with tab_fs:
    st.markdown("All selection steps use **training data only**. Three techniques were applied:")
    t = pd.DataFrame(FS["table"]).set_index("feature")
    num_cols = [c for c in t.columns if c not in ("type", "selected")]
    t[num_cols] = t[num_cols].apply(pd.to_numeric, errors="coerce")
    t.index = [LABELS.get(i, i) for i in t.index]
    c1, c2 = st.columns([1.1, 1])
    with c1:
        section("1 · Correlation / association")
        st.dataframe(t[["type", "pearson_r", "spearman_rho", "eta_squared"]].style.format(
            {"pearson_r": "{:.4f}", "spearman_rho": "{:.4f}", "eta_squared": "{:.4f}"}, na_rep="—"), width="stretch")
        section("2 · Recursive Feature Elimination (RFECV)")
        st.markdown(f"A linear model on the full one-hot design kept **{FS['rfecv_n']} of {FS['rfecv_total']}** encoded columns "
                    "(5-fold CV RMSE). Removed: " + ", ".join(f"`{c.split('__')[1]}`" for c in FS["rfecv_dropped"]) + ".")
        st.caption("With a dummy for every country, national pesticide use adds little new information, so RFE drops it.")
    with c2:
        section("3 · Model-based importance")
        pi = t["perm_importance_mean"].sort_values()
        fig = go.Figure(go.Bar(x=pi.values, y=pi.index, orientation="h", marker_color=GREEN,
                               error_x=dict(type="data", array=t.loc[pi.index, "perm_importance_std"], color="#5b6b5b"),
                               hovertemplate="%{y}: ΔR² = %{x:.4f}<extra></extra>"))
        fig.update_layout(title="Permutation importance (drop in validation R²)", height=330, xaxis_title="Mean ΔR²")
        st.plotly_chart(fig, width="stretch")
        ri = t["rf_impurity_importance"].sort_values()
        fig = go.Figure(go.Bar(x=ri.values, y=ri.index, orientation="h", marker_color="#81c784"))
        fig.update_layout(title="Random Forest impurity importance (grouped)", height=300, xaxis_title="Share")
        st.plotly_chart(fig, width="stretch")
    note(f"<b>Rule (fixed in advance):</b> keep a feature if shuffling it lowers validation R² by more than 0.001. "
         f"<b>All six passed</b>: {', '.join(LABELS[f] for f in FS['selected'])}. Rainfall's high tree importance reflects that "
         "it is constant per country, so it acts as a country identifier, not as a weather signal.")

with tab_reg:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Ridge (L2)")
        st.latex(r"\min_\beta \sum_i (y_i - x_i^\top\beta)^2 + \alpha\sum_j \beta_j^2")
        st.markdown("Shrinks all coefficients smoothly towards 0; none become exactly 0. Stabilises correlated predictors.")
    with c2:
        st.markdown("#### LASSO (L1)")
        st.latex(r"\min_\beta \tfrac{1}{2n}\sum_i (y_i - x_i^\top\beta)^2 + \alpha\sum_j |\beta_j|")
        st.markdown("Can set coefficients **exactly to 0**, so it also performs feature selection.")
    cards([("Ridge best α", f"{REG['ridge_alpha']:.4f}", f"CV RMSE {REG['ridge_cv_rmse']:.4f}"),
           ("LASSO best α", f"{REG['lasso_alpha']:.6f}", f"CV RMSE {REG['lasso_cv_rmse']:.4f}"),
           ("LASSO zeros (best α)", f"{len(REG['lasso_zeroed'])} / {REG['n_coef']}", "coefficients = 0"),
           ("LASSO zeros (1-SE α)", f"{len(REG['lasso_1se_zeroed'])} / {REG['n_coef']}",
            f"α = {REG['lasso_1se_alpha']:.5f}, CV RMSE {REG['lasso_1se_cv_rmse']:.3f}")], accent=True)
    section("Test-set performance")
    rows = ["Ridge Regression", "LASSO Regression", "LASSO (1-SE rule)", "MLR (Phase 2 features, OLS)"]
    st.dataframe(comp.loc[rows, ["R2", "MAE", "MSE", "RMSE", "CV_R2_mean", "CV_RMSE_mean"]].style.format("{:.4f}"),
                 width="stretch")
    section("Coefficients")
    coef = load_table("ridge_lasso_coefficients", index_col=0)
    group = st.radio("Coefficient group", ["Numeric features", "Crops", "Countries"], horizontal=True)
    if group == "Numeric features":
        sub = coef[coef.index.str.startswith("num__")]
    elif group == "Crops":
        sub = coef[coef.index.str.startswith("cat__crop_")]
    else:
        sub = coef[coef.index.str.startswith("cat__country_")].sort_values("ridge")
    labels = [LABELS.get(i.split("__", 1)[1], i.split("__", 1)[1].replace("crop_", "").replace("country_", "")) for i in sub.index]
    fig = go.Figure()
    for col, name, colr in [("ridge", "Ridge", "#81c784"), ("lasso", "LASSO", GREEN), ("lasso_1se", "LASSO (1-SE)", "#f59e0b")]:
        fig.add_trace(go.Bar(name=name, x=labels, y=sub[col], marker_color=colr))
    fig.update_layout(barmode="group", height=420 if group != "Countries" else 520,
                      title=f"{group}: coefficients (t/ha; numeric features per 1 SD)", xaxis_tickangle=-45 if group == "Countries" else 0)
    st.plotly_chart(fig, width="stretch")
    with st.expander("Coefficients set exactly to zero"):
        st.markdown(f"**Best-α LASSO ({len(REG['lasso_zeroed'])}):** " + ", ".join(z.split("_", 3)[-1] for z in REG["lasso_zeroed"]))
        st.markdown(f"**1-SE LASSO ({len(REG['lasso_1se_zeroed'])}):** " + ", ".join(z.split("country_")[-1] for z in REG["lasso_1se_zeroed"]))
        st.caption("All zeroed coefficients are country dummies; no crop or numeric feature was removed.")
    show_png(figure_path("15_ridge_lasso_alpha_tuning"), "α tuning: 5-fold CV RMSE vs α (pipeline figure)")
    show_png(figure_path("16_lasso_path"), "LASSO coefficient path (pipeline figure)")
    note("<b>Ridge vs LASSO:</b> with 10,504 rows and 115 columns there is little variance to remove, so the optimal α is small "
         "and Ridge, LASSO and plain OLS perform the same (R² ≈ 0.735). LASSO's value is a <b>simpler model</b>: the 1-SE "
         "version drops 26 country dummies for −0.005 R², and it generalises better to unseen countries.")

with tab_dt:
    p = TUN["Decision Tree"]["best_params"]
    st.markdown("**Algorithm:** recursively split on the feature and threshold that most reduces the sum of squared errors; "
                "each leaf predicts the mean yield of its training rows. It captures interactions (crop × country) and "
                "thresholds automatically, with no distributional assumptions.")
    st.markdown(f"**Tuned** (GridSearchCV, 5-fold, RMSE): `max_depth = {p['max_depth']}`, `min_samples_leaf = {p['min_samples_leaf']}`; "
                f"CV RMSE = {TUN['Decision Tree']['cv_rmse']:.4f}")
    metric_cards("Decision Tree Regressor")
    note(f"Train R² {comp.loc['Decision Tree Regressor', 'Train_R2']:.4f} vs test {comp.loc['Decision Tree Regressor', 'R2']:.4f}: "
         "the largest overfitting gap of all models (a single deep tree memorises noise).", "warn")
    show_png(figure_path("17_dt_tuning_heatmap"), "Decision Tree tuning grid: CV RMSE (pipeline figure)")

with tab_rf:
    p = TUN["Random Forest"]["best_params"]
    st.markdown("**Algorithm:** an average of 300 decision trees, each trained on a bootstrap sample and using a random subset "
                "of features at each split. Averaging de-correlated deep trees reduces variance while keeping bias low.")
    st.markdown(f"**Tuned:** `max_features = {p['max_features']}`, `min_samples_leaf = {p['min_samples_leaf']}`, "
                f"`max_depth = {p['max_depth']}`, `n_estimators = 300`; CV RMSE = {TUN['Random Forest']['cv_rmse']:.4f}")
    metric_cards("Random Forest Regressor")
    rf = get_models()["Random Forest Regressor"]
    names = rf.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names)
    g = imp.groupby([grp(n) for n in imp.index]).sum().sort_values()
    c1, c2 = st.columns(2)
    fig = go.Figure(go.Bar(x=g.values, y=[LABELS.get(i, i) for i in g.index], orientation="h", marker_color=GREEN))
    fig.update_layout(title="Final RF: impurity importance by feature", height=340, xaxis_title="Share of importance")
    c1.plotly_chart(fig, width="stretch")
    top = imp.sort_values(ascending=False).head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=top.values, y=[n.split("__", 1)[1] for n in top.index], orientation="h", marker_color="#81c784"))
    fig.update_layout(title="Top 15 encoded columns", height=340)
    c2.plotly_chart(fig, width="stretch")

with tab_gb:
    p = TUN["Gradient Boosting"]["best_params"]
    st.markdown("**Algorithm (HistGradientBoosting, optional model):** builds shallow trees sequentially, each fitting the "
                "residuals of the current ensemble: F<sub>m</sub>(x) = F<sub>m−1</sub>(x) + η·h<sub>m</sub>(x).",
                unsafe_allow_html=True)
    st.markdown(f"**Tuned:** learning rate = {p['learning_rate']}, max_iter = {p['max_iter']}, max_leaf_nodes = {p['max_leaf_nodes']}, "
                f"min_samples_leaf = {p['min_samples_leaf']}; CV RMSE = {TUN['Gradient Boosting']['cv_rmse']:.4f}")
    metric_cards("Gradient Boosting (HistGB)")
