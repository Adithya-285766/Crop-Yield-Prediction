import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.data import LABELS, TARGET, load_results, load_table
from utils.models import get_split, get_statsmodels, test_predictions
from utils.ui import GREEN, cards, note, page_header, section

page_header("Phase 1: Statistical Modeling", "Simple and Multiple Linear Regression (OLS), fitted on the "
            "training set (n = 10,504) and evaluated on the 20 % test set (n = 2,626)")

res = load_results()
slr, mlr = res["slr"], res["mlr"]
preds = test_predictions()
train_df, test_df = get_split()


def avp_and_resid(name, key):
    y, p = preds[TARGET], preds[name]
    c1, c2 = st.columns(2)
    lim = [0, float(max(y.max(), p.max()) * 1.03)]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=y, y=p, mode="markers", marker=dict(size=5, color=GREEN, opacity=.35),
                               customdata=np.stack([preds.country, preds.crop, preds.year], axis=1),
                               hovertemplate="%{customdata[0]} · %{customdata[1]} · %{customdata[2]}<br>"
                                             "actual %{x:.2f} · predicted %{y:.2f}<extra></extra>", name="test rows"))
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="#f59e0b", dash="dash"), name="perfect prediction"))
    fig.update_layout(title="Actual vs predicted (test set)", xaxis_title="Actual yield (t/ha)",
                      yaxis_title="Predicted yield (t/ha)", height=420, legend=dict(orientation="h", y=-.2))
    c1.plotly_chart(fig, width="stretch", key=f"avp_{key}")
    r = y - p
    fig = go.Figure(go.Scattergl(x=p, y=r, mode="markers", marker=dict(size=5, color=GREEN, opacity=.35),
                                 hovertemplate="predicted %{x:.2f}<br>residual %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#f59e0b"))
    fig.update_layout(title="Residuals vs predicted (test set)", xaxis_title="Predicted yield (t/ha)",
                      yaxis_title="Residual = actual − predicted (t/ha)", height=420)
    c2.plotly_chart(fig, width="stretch", key=f"res_{key}")


tab_slr, tab_mlr, tab_cmp = st.tabs(["Simple Linear Regression", "Multiple Linear Regression", "SLR vs MLR"])

with tab_slr:
    section("Model and equation")
    st.latex(r"Y = \beta_0 + \beta_1 X + \varepsilon")
    st.markdown(f"**X = {LABELS[slr['x']]}**, chosen as the numeric feature with the largest |Pearson r| with "
                "yield **in the training set** (rainfall 0.017, log-pesticides 0.183, temperature −0.140, year 0.088).")
    st.latex(rf"\widehat{{\text{{Yield}}}} = {slr['b0']:.4f} + {slr['b1']:.4f}\,\ln(1+\text{{pesticides}})")
    cards([("Test R²", f"{slr['test']['R2']:.4f}", "variance explained"),
           ("Test MAE", f"{slr['test']['MAE']:.4f}", "t/ha"),
           ("Test MSE", f"{slr['test']['MSE']:.4f}", "(t/ha)²"),
           ("Test RMSE", f"{slr['test']['RMSE']:.4f}", "t/ha")])
    section("Coefficients (training data)")
    c1, c2 = st.columns([1.2, 1])
    slr_sm = get_statsmodels()[0]
    ci = slr_sm.conf_int()
    c1.dataframe({"Term": ["β₀ (intercept)", f"β₁ ({LABELS[slr['x']]})"],
                  "Estimate": [f"{v:.4f}" for v in slr_sm.params],
                  "Std. error": [f"{v:.4f}" for v in slr_sm.bse],
                  "t": [f"{v:.2f}" for v in slr_sm.tvalues],
                  "p-value": [f"{v:.2e}" for v in slr_sm.pvalues],
                  "95 % CI": [f"[{a:.3f}, {b:.3f}]" for a, b in ci.values]},
                 hide_index=True, width="stretch")
    c2.markdown(f"Training R² = **{slr['r2_train']:.4f}** · F = {slr['F']:.1f} (p = {slr['F_p']:.1e})")
    note(f"Doubling pesticide use is associated with <b>+{slr['b1'] * np.log(2):.2f} t/ha</b> (β₁ × ln 2). The slope is highly "
         "<i>significant</i> but explains only ~3–4 % of yield variance: <b>significant does not mean predictive</b>.")

    section("Regression line (training data)")
    xs = np.linspace(train_df[slr["x"]].min(), train_df[slr["x"]].max(), 100)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=train_df[slr["x"]], y=train_df[TARGET], mode="markers",
                               marker=dict(size=4, color=GREEN, opacity=.25), name="training observations"))
    fig.add_trace(go.Scatter(x=xs, y=slr["b0"] + slr["b1"] * xs, mode="lines", line=dict(color="#f59e0b", width=3),
                             name=f"Ŷ = {slr['b0']:.3f} + {slr['b1']:.3f}·X"))
    fig.update_layout(xaxis_title=LABELS[slr["x"]], yaxis_title="Yield (t/ha)", height=430,
                      legend=dict(orientation="h", y=-.2))
    st.plotly_chart(fig, width="stretch")
    section("Test-set predictions")
    avp_and_resid("Simple Linear Regression", "slr")

with tab_mlr:
    section("Model and equation")
    st.latex(r"Y = \beta_0 + \beta_1\,\text{Rain} + \beta_2\ln(1+\text{Pest}) + \beta_3\,\text{Temp} + "
             r"\beta_4\,\text{Year} + \sum_{k=1}^{9}\gamma_k D_k + \varepsilon")
    st.markdown("D₁…D₉ are crop dummy variables (reference = **Cassava**). Country is not included: rainfall has a "
                "single value per country, so it would be perfectly collinear with country dummies.")
    p = mlr["coef"]
    st.latex(rf"\widehat{{Y}} = {p['Intercept']['coef']:.2f} {p['avg_rainfall_mm']['coef']:+.6f}\,\text{{Rain}} "
             rf"{p['log_pesticides']['coef']:+.4f}\ln(1+\text{{Pest}}) {p['avg_temp_c']['coef']:+.4f}\,\text{{Temp}} "
             rf"{p['year']['coef']:+.4f}\,\text{{Year}} + \text{{crop effect}}")
    cards([("Test R²", f"{mlr['test']['R2']:.4f}", f"train R² {mlr['r2_train']:.3f} (adj. {mlr['adj_r2_train']:.3f})"),
           ("Test MAE", f"{mlr['test']['MAE']:.4f}", "t/ha"),
           ("Test MSE", f"{mlr['test']['MSE']:.4f}", "(t/ha)²"),
           ("Test RMSE", f"{mlr['test']['RMSE']:.4f}", "t/ha")])
    section("Coefficients (training data)")
    ct = load_table("mlr_coefficients", index_col=0)
    ct.index = [i.replace("C(crop)[T.", "Crop: ").replace("]", "") for i in ct.index]
    c1, c2 = st.columns([1.3, 1])
    c1.dataframe(ct.style.format({"coef": "{:.6f}", "std_err": "{:.6f}", "t": "{:.2f}", "p_value": "{:.4g}",
                                  "ci_low": "{:.6f}", "ci_high": "{:.6f}"}), width="stretch", height=520)
    crop_rows = ct[ct.index.str.startswith("Crop:")]
    fig = go.Figure(go.Bar(x=crop_rows["coef"], y=crop_rows.index.str.replace("Crop: ", ""), orientation="h",
                           marker_color=[GREEN if v > 0 else "#dc2626" for v in crop_rows["coef"]],
                           error_x=dict(type="data", array=crop_rows["ci_high"] - crop_rows["coef"], color="#5b6b5b"),
                           hovertemplate="%{y}: %{x:+.3f} t/ha vs cassava<extra></extra>"))
    fig.update_layout(title="Crop effects vs Cassava (t/ha, 95 % CI)", height=420, xaxis_title="t/ha")
    c2.plotly_chart(fig, width="stretch")
    section("Interpretation (holding other variables constant)")
    se = mlr["std_effect"]
    st.markdown(f"""
- **Temperature:** each +1 °C goes with **{p['avg_temp_c']['coef']:.3f} t/ha**, consistent with heat stress (an association, not a causal effect).
- **Pesticides:** doubling national pesticide use goes with **+{p['log_pesticides']['coef'] * np.log(2):.2f} t/ha**.
- **Year:** **+{p['year']['coef']:.3f} t/ha per year** (≈ {p['year']['coef'] * 10:.2f} t/ha per decade), the technology trend.
- **Rainfall:** {p['avg_rainfall_mm']['coef'] * 1000:.3f} t/ha per 1,000 mm. Small, and **not significant** once errors are
  clustered by country (see Diagnostics).
- **Effect of a 1-SD increase:** log-pesticides {se['log_pesticides']:+.2f}, temperature {se['avg_temp_c']:+.2f},
  year {se['year']:+.2f}, rainfall {se['avg_rainfall_mm']:+.2f} t/ha.
""")
    section("Test-set predictions")
    avp_and_resid("Multiple Linear Regression", "mlr")

with tab_cmp:
    rows = {"Simple Linear Regression": slr["test"], "Multiple Linear Regression": mlr["test"],
            "MLR on log(yield) (corrective model)": res["assumptions"]["log_model"]["test"]}
    import pandas as pd
    t = pd.DataFrame(rows).T[["R2", "MAE", "MSE", "RMSE"]]
    st.dataframe(t.style.format("{:.4f}").highlight_max(subset=["R2"], color="#c8e6c9")
                 .highlight_min(subset=["MAE", "MSE", "RMSE"], color="#c8e6c9"), width="stretch")
    fig = go.Figure()
    for m, colr in zip(["MAE", "RMSE"], [GREEN, "#a5d6a7"]):
        fig.add_trace(go.Bar(name=m, x=t.index, y=t[m], marker_color=colr, text=t[m].round(3), textposition="outside"))
    fig.update_layout(barmode="group", title="Test error (t/ha), lower is better", height=380, yaxis_title="t/ha")
    st.plotly_chart(fig, width="stretch")
    note("Adding crop type and the other variables raises test R² from 0.037 to 0.615. The log-target model (a correction "
         "for skewed, heteroscedastic residuals, back-transformed with Duan's smearing factor "
         f"{res['assumptions']['log_model']['smear']:.3f}) improves further, to R² {res['assumptions']['log_model']['test']['R2']:.3f}.")
