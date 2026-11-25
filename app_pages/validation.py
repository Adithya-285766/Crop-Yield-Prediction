import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import LABELS, figure_path, load_results, load_table
from utils.ui import GREEN, cards, note, page_header, section, show_png

page_header("Estimator Validation", "5-fold cross-validation and bootstrap resampling of the Phase 1 models")

res = load_results()
V = res["validation"]
cvs = V["cv_summary"]

st.markdown("""
**Why validate?** A coefficient or R² computed once, on one sample, could be a lucky draw. Validation shows
(1) how well the model predicts new data and (2) how stable the estimated coefficients are when the sample changes.
""")

tab_cv, tab_boot = st.tabs(["K-fold cross-validation", "Bootstrap sampling"])

with tab_cv:
    st.markdown("The training set is split into **K = 5** folds. Each fold is held out once while the model is trained on "
                "the other four. The same `KFold(5, shuffle=True, random_state=42)` is used for every model in the project.")
    cards([("SLR CV R²", f"{cvs['SLR']['R2']['mean']:.4f}", f"± {cvs['SLR']['R2']['std']:.4f} (SD)"),
           ("SLR CV RMSE", f"{cvs['SLR']['RMSE']['mean']:.3f}", f"± {cvs['SLR']['RMSE']['std']:.3f} t/ha"),
           ("MLR CV R²", f"{cvs['MLR']['R2']['mean']:.4f}", f"± {cvs['MLR']['R2']['std']:.4f} (SD)"),
           ("MLR CV RMSE", f"{cvs['MLR']['RMSE']['mean']:.3f}", f"± {cvs['MLR']['RMSE']['std']:.3f} t/ha")], accent=True)
    folds = load_table("phase1_cv_folds")
    section("Scores per fold")
    c1, c2 = st.columns([1, 1.2])
    metric = c2.radio("Metric", ["R2", "RMSE", "MAE", "MSE"], horizontal=True)
    fig = go.Figure()
    for m, colr in [("SLR", "#a5d6a7"), ("MLR", GREEN)]:
        d = folds[folds.model == m]
        fig.add_trace(go.Bar(name=m, x=[f"Fold {k}" for k in d.fold], y=d[metric], marker_color=colr,
                             text=d[metric].round(3), textposition="outside"))
    fig.update_layout(barmode="group", title=f"Validation {metric} per fold", height=400)
    c2.plotly_chart(fig, width="stretch")
    piv = folds.pivot(index="fold", columns="model", values=["R2", "RMSE"]).round(4)
    piv.columns = [f"{b} {a}" for a, b in piv.columns]
    c1.dataframe(piv, width="stretch")
    section("Coefficient stability across folds (MLR)")
    fc = pd.DataFrame(V["mlr_fold_coefs"]).set_index("fold")
    fc.columns = [LABELS[c.replace("num__", "")] for c in fc.columns]
    st.dataframe(fc.style.format("{:.6f}"), width="stretch")
    note("Coefficient signs never change across folds, and values move only slightly (pesticides 0.554–0.580, temperature "
         "−0.148 to −0.158). The fold R² values span a narrow range (0.587–0.619), and the CV R² (0.605) agrees with the test "
         "R² (0.615). <b>The estimator is stable.</b>")

with tab_boot:
    st.markdown(f"""
**How it works:** draw n rows *with replacement* from the n training rows, refit the model, record the coefficients,
and score the **out-of-bag** rows (those not drawn, ≈ 36.8 %). Repeat **B = {V['bootstrap_B']:,}** times. The SD of the
replicates is the bootstrap standard error; the 2.5th and 97.5th percentiles give a 95 % CI, without assuming normality.

Because residuals are correlated within countries, a **cluster bootstrap** (resampling whole countries,
B = {V['cluster_bootstrap_B']}) was also run.
""")
    bt = load_table("bootstrap_coefficients").set_index("coefficient")
    bt.index = [LABELS.get(i, i) for i in bt.index]
    section("Coefficient distribution")
    st.dataframe(bt.style.format("{:.6f}", na_rep="—"), width="stretch")
    sel = st.selectbox("Compare 95 % intervals for", [i for i in bt.index if pd.notna(bt.loc[i, "cluster_ci2.5"])])
    r = bt.loc[sel]
    fig = go.Figure()
    for k, (lo, hi, lab, colr) in enumerate([(r["ci2.5"], r["ci97.5"], "Row bootstrap", "#a5d6a7"),
                                              (r["cluster_ci2.5"], r["cluster_ci97.5"], "Country-cluster bootstrap", GREEN)]):
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lab, lab], mode="lines+markers", line=dict(color=colr, width=8),
                                 marker=dict(size=12), name=lab, hovertemplate="%{x:.6f}<extra></extra>"))
    fig.add_vline(x=r["OLS_estimate"], line=dict(color="#f59e0b"), annotation_text="OLS estimate")
    fig.add_vline(x=0, line=dict(color="#5b6b5b", dash="dot"), annotation_text="0", annotation_position="bottom")
    fig.update_layout(title=f"95 % bootstrap confidence intervals: {sel}", height=280, showlegend=False)
    st.plotly_chart(fig, width="stretch")
    note("Bootstrap means ≈ OLS estimates (no visible bias). Row-bootstrap SEs match the classical SEs. The <b>cluster</b> SEs are "
         "4–5× larger; with them the <b>rainfall interval includes 0</b>, while pesticides (+), temperature (−) and year (+) "
         "remain clearly non-zero.")
    section("Variation in model performance (out-of-bag)")
    perf = load_table("bootstrap_oob_performance", index_col=0)
    st.dataframe(perf.style.format("{:.4f}"), width="stretch")
    note("The MLR and SLR out-of-bag intervals do not overlap (R² 0.590–0.620 vs 0.025–0.040), so the MLR is better beyond "
         "sampling noise.")
    section("Bootstrap distributions (pipeline figure)")
    show_png(figure_path("11_bootstrap_distributions"), "Bootstrap distributions, B = 1,000; dashed = 95 % percentile interval")
