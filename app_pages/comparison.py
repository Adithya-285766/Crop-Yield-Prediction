import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.data import TARGET, load_results, load_table
from utils.models import MAIN_MODELS, MODEL_ORDER, metrics, test_predictions, verify
from utils.ui import GREEN, SEQ, cards, note, page_header, section

page_header("Model Comparison", "All models on the same 20 % test set and the same 5 CV folds")

res = load_results()
comp = load_table("model_comparison", index_col="Model")
rob = load_table("robustness_temporal_and_unseen_country", index_col="Model")
best = comp["RMSE"].idxmin()

show_all = st.toggle("Include supplementary models (Gradient Boosting, LASSO 1-SE, OLS with Phase 2 features)", value=True)
models = MODEL_ORDER if show_all else MAIN_MODELS

b = comp.loc[best]
cards([("🏆 Best model", best.replace(" Regressor", ""), "lowest test RMSE and MAE"),
       ("Test R²", f"{b['R2']:.4f}", f"CV {b['CV_R2_mean']:.4f} ± {b['CV_R2_std']:.4f}"),
       ("Test RMSE", f"{b['RMSE']:.4f} t/ha", f"vs MLR {comp.loc['Multiple Linear Regression', 'RMSE']:.3f}"),
       ("Test MAE", f"{b['MAE']:.4f} t/ha", f"vs MLR {comp.loc['Multiple Linear Regression', 'MAE']:.3f}")], accent=True)

section("Comparison table (test set)")
t = comp.loc[models, ["R2", "MAE", "MSE", "RMSE", "Train_R2", "CV_R2_mean", "CV_R2_std", "CV_RMSE_mean", "Overfit_gap_R2"]]
st.dataframe(t.style.format("{:.4f}").highlight_max(subset=["R2", "CV_R2_mean"], color="#c8e6c9")
             .highlight_min(subset=["MAE", "MSE", "RMSE", "CV_RMSE_mean"], color="#c8e6c9"), width="stretch")
st.caption("Overfit gap = train R² − test R². CV columns: 5-fold CV on the training set (SD over folds, ddof = 0).")

section("Performance charts")
metric = st.radio("Metric", ["R2", "MAE", "MSE", "RMSE"], horizontal=True, format_func=lambda m: "R²" if m == "R2" else m)
d = comp.loc[models, metric]
fig = go.Figure(go.Bar(x=d.index, y=d.values, marker_color=[GREEN if m == best else "#a5d6a7" for m in d.index],
                       text=[f"{v:.3f}" for v in d.values], textposition="outside",
                       hovertemplate="%{x}<br>%{y:.4f}<extra></extra>"))
fig.update_layout(title=f"Test {metric}" + (" (higher is better)" if metric == "R2" else " (lower is better)"),
                  height=440, xaxis_tickangle=-20, yaxis=dict(range=[0, d.max() * 1.15]))
st.plotly_chart(fig, width="stretch")

section("Actual vs predicted")
preds = test_predictions()
c1, c2 = st.columns([1, 3])
chosen = c1.multiselect("Models", models, default=["Multiple Linear Regression", "LASSO Regression", best])
crop_f = c1.selectbox("Crop filter", ["All crops"] + sorted(preds.crop.unique()))
pp = preds if crop_f == "All crops" else preds[preds.crop == crop_f]
if not chosen:
    c2.info("Select at least one model.")
else:
    fig = go.Figure()
    for i, m in enumerate(chosen):
        mm = metrics(pp[TARGET], pp[m])
        fig.add_trace(go.Scattergl(x=pp[TARGET], y=pp[m], mode="markers", name=f"{m} (R² {mm['R2']:.3f})",
                                   marker=dict(size=5, opacity=.45, color=SEQ[(i + 1) % len(SEQ)] if m != best else GREEN),
                                   customdata=np.stack([pp.country, pp.crop, pp.year], axis=1),
                                   hovertemplate="%{customdata[0]} · %{customdata[1]} · %{customdata[2]}<br>actual %{x:.2f}"
                                                 " · predicted %{y:.2f}<extra></extra>"))
    lim = [0, float(max(pp[TARGET].max(), pp[chosen].max().max()) * 1.03)]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="#1f2a1f", dash="dash"), name="perfect prediction"))
    fig.update_layout(title=f"Test set: actual vs predicted ({crop_f})", xaxis_title="Actual yield (t/ha)",
                      yaxis_title="Predicted yield (t/ha)", height=520, legend=dict(orientation="h", y=-.18))
    c2.plotly_chart(fig, width="stretch")

section("Generalisation: three evaluation schemes")
st.markdown("A random split puts different years of the same country-crop series in both train and test, which is optimistic "
            "for panel data. Two stricter tests used the tuned settings: **future years** (train 1990–2009, test 2010–2013) and "
            "**unseen countries** (5-fold GroupKFold by country).")
fig = go.Figure()
for col, lab, colr in [(comp.loc[models, "R2"], "Random 80/20 test", GREEN),
                       (rob.loc[models, "Temporal_R2"], "Future years (2010–13)", "#f59e0b"),
                       (rob.loc[models, "UnseenCountry_R2"], "Unseen countries", "#2563eb")]:
    fig.add_trace(go.Bar(name=lab, x=models, y=col, marker_color=colr, hovertemplate="%{x}<br>R² %{y:.3f}<extra></extra>"))
fig.update_layout(barmode="group", height=460, yaxis=dict(range=[0, 1], title="R²"), xaxis_tickangle=-20,
                  legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, width="stretch")
st.dataframe(rob.loc[models].style.format("{:.4f}"), width="stretch")
note("The Random Forest is best on the random split and on future years (R² 0.935). For <b>unseen countries</b> all models drop "
     "sharply; RF (0.627) and Gradient Boosting (0.630) tie within one SD. Country-dummy linear models do worse there than the "
     "crop-only MLR (0.589), because an unseen country's dummy carries no information.")

section("Why statistical and ML models differ")
st.markdown("""
1. **Functional form:** OLS forces one additive slope per feature for every crop; trees learn crop × country interactions and thresholds.
2. **Purpose:** the regression models are built for **inference** (coefficients, CIs, tests); ML models minimise **prediction error**.
3. **Assumptions:** OLS assumptions are violated, which affects its inference; trees make no distributional assumptions.
4. **Panel memorisation:** part of the ML advantage comes from recognising country-crop combinations seen in other years. The unseen-country test removes this, and the gap narrows sharply.
""")

with st.expander("✅ Reproducibility check: dashboard models vs pipeline results"):
    v = verify()
    st.dataframe(v.style.format("{:.6f}"), width="stretch")
    ok = (v["abs. diff RMSE"] < 1e-9).all()
    (st.success if ok else st.warning)(
        "The models loaded by this app reproduce the pipeline's published test metrics exactly." if ok else
        "Some rebuilt models differ from the published metrics. Re-run the pipeline and delete models/ to retrain.")
