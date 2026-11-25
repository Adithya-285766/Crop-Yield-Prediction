import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from utils.data import NUM, LABELS, figure_path, load_results, load_table
from utils.models import get_split, get_statsmodels
from utils.ui import GREEN, note, page_header, pill, section, show_png

page_header("Regression Diagnostics", "Assumption tests for the Multiple Linear Regression (training residuals, α = 0.05)")

res = load_results()
A = res["assumptions"]["raw"]
L = res["assumptions"]["log"]
_, mlr, mlr_log = get_statsmodels()
train_df, _ = get_split()

section("Summary")
summary = [
    ("Linearity", "Rainbow test; residual plots", f"F = {A['rainbow_F']:.3f}, p = {A['rainbow_p']:.3f}", "Partly", "part"),
    ("Normality", "Shapiro–Wilk; Jarque–Bera; Q-Q", f"W = {A['shapiro_W']:.3f}, p = {A['shapiro_p']:.1e}; JB = {A['jb']:,.0f}", "No", "no"),
    ("Homoscedasticity", "Breusch–Pagan", f"LM = {A['bp_LM']:,.1f}, p < 0.001", "No", "no"),
    ("Independence", "Durbin–Watson (panel order)", f"DW = {A['dw_panel_order']:.3f} (random order {A['dw_random_order']:.3f})", "No", "no"),
    ("Multicollinearity", "VIF", f"max VIF = {max(res['assumptions']['vif'].values()):.2f}", "Yes", "ok"),
]
html = ("<table style='width:100%;border-collapse:collapse;font-size:.93rem'><tr style='text-align:left;"
        "border-bottom:2px solid #e8f5e9'><th>Assumption</th><th>Test</th><th>Result</th><th>Satisfied?</th></tr>")
for a, t, r, v, k in summary:
    html += (f"<tr style='border-bottom:1px solid #eef2ec'><td style='padding:.5rem 0'><b>{a}</b></td><td>{t}</td>"
             f"<td>{r}</td><td>{pill(v, k)}</td></tr>")
st.markdown(html + "</table>", unsafe_allow_html=True)

resid, fitted = mlr.resid, mlr.fittedvalues
tabs = st.tabs(["Linearity", "Normality", "Homoscedasticity", "Independence", "Multicollinearity (VIF)",
                "Corrective measures"])

with tabs[0]:
    st.markdown("**Purpose:** check that E[Y|X] is linear in the predictors. **H₀: the relationship is linear** (Utts' Rainbow test).")
    c1, c2 = st.columns(2)
    fig = go.Figure(go.Scattergl(x=fitted, y=resid, mode="markers", marker=dict(size=4, color=GREEN, opacity=.25)))
    fig.add_hline(y=0, line=dict(color="#f59e0b"))
    fig.update_layout(title="Residuals vs fitted", xaxis_title="Fitted yield (t/ha)", yaxis_title="Residual (t/ha)", height=420)
    c1.plotly_chart(fig, width="stretch")
    xf = c2.selectbox("Residuals vs predictor", NUM, format_func=lambda c: LABELS[c])
    fig = go.Figure(go.Scattergl(x=train_df[xf], y=resid, mode="markers", marker=dict(size=4, color=GREEN, opacity=.25)))
    fig.add_hline(y=0, line=dict(color="#f59e0b"))
    fig.update_layout(title=f"Residuals vs {LABELS[xf]}", xaxis_title=LABELS[xf], yaxis_title="Residual", height=360)
    c2.plotly_chart(fig, width="stretch")
    note(f"Rainbow F = {A['rainbow_F']:.3f}, <b>p = {A['rainbow_p']:.3f}</b>, so H₀ is not rejected: no overall curvature. But the "
         "fan and diagonal stripes in the plot show that one additive slope is forced on crops whose yields differ about "
         "tenfold (crop × feature interactions). <b>Verdict: partly satisfied.</b>")

with tabs[1]:
    st.markdown("**Purpose:** t-tests and confidence intervals assume normal errors. **H₀: residuals are normally distributed.**")
    c1, c2 = st.columns(2)
    (osm, osr), (slope, icpt, _) = stats.probplot(resid, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=osm, y=osr, mode="markers", marker=dict(size=4, color=GREEN, opacity=.4), name="residuals"))
    fig.add_trace(go.Scatter(x=[osm.min(), osm.max()], y=[slope * osm.min() + icpt, slope * osm.max() + icpt],
                             mode="lines", line=dict(color="#f59e0b"), name="normal reference"))
    fig.update_layout(title="Normal Q-Q plot", xaxis_title="Theoretical quantiles", yaxis_title="Ordered residuals",
                      height=420, showlegend=False)
    c1.plotly_chart(fig, width="stretch")
    fig = go.Figure(go.Histogram(x=resid, nbinsx=70, marker_color=GREEN))
    fig.update_layout(title="Histogram of residuals", xaxis_title="Residual (t/ha)", yaxis_title="Count", height=420, bargap=.03)
    c2.plotly_chart(fig, width="stretch")
    note(f"Shapiro–Wilk (random 5,000 residuals, the SciPy maximum): W = {A['shapiro_W']:.3f}, <b>p = {A['shapiro_p']:.1e}</b>. "
         f"Jarque–Bera (all residuals): JB = {A['jb']:,.0f}, p ≈ 0. Skewness {A['resid_skew']:.2f}, kurtosis {A['resid_kurtosis']:.2f} "
         "(normal: 0 and 3). The Q-Q points bend far above the line in the right tail. <b>Verdict: not satisfied.</b> With "
         "n &gt; 10,000 the coefficients remain valid (CLT), but exact intervals are affected.")

with tabs[2]:
    st.markdown("**Purpose:** OLS standard errors assume constant error variance. **H₀: homoscedastic residuals** (Breusch–Pagan).")
    fig = go.Figure(go.Scattergl(x=fitted, y=np.sqrt(np.abs(resid / resid.std())), mode="markers",
                                 marker=dict(size=4, color=GREEN, opacity=.25)))
    fig.update_layout(title="Scale-location plot", xaxis_title="Fitted yield (t/ha)",
                      yaxis_title="√|standardised residual|", height=420)
    st.plotly_chart(fig, width="stretch")
    note(f"Breusch–Pagan LM = {A['bp_LM']:,.1f} (p ≈ 0); F = {A['bp_F']:.1f} (p ≈ 0). The spread grows with the fitted value: "
         "high-yield crops have much larger errors. <b>Verdict: not satisfied (heteroscedasticity).</b>")

with tabs[3]:
    st.markdown("**Purpose:** errors should be uncorrelated. **H₀: no first-order autocorrelation** (Durbin–Watson ≈ 2).")
    c1, c2 = st.columns(2)
    c1.metric("DW, random (training) order", f"{A['dw_random_order']:.3f}")
    c2.metric("DW, panel order (country → crop → year)", f"{A['dw_panel_order']:.3f}")
    ordered = resid.loc[train_df.sort_values(["country", "crop", "year"]).index]
    fig = go.Figure(go.Scattergl(x=ordered.values[:-1], y=ordered.values[1:], mode="markers",
                                 marker=dict(size=4, color=GREEN, opacity=.25)))
    fig.update_layout(title="Lag plot: residual(t) vs residual(t+1) in panel order", xaxis_title="eₜ",
                      yaxis_title="eₜ₊₁", height=420)
    st.plotly_chart(fig, width="stretch")
    note("DW depends on row order. In shuffled order it is ≈ 2 by construction. In the meaningful panel order it is "
         f"<b>{A['dw_panel_order']:.3f}</b>, showing strong positive autocorrelation: a country that over-yields one year tends to "
         "over-yield the next. <b>Verdict: not satisfied</b>, so classical standard errors are too small.")

with tabs[4]:
    st.latex(r"\text{VIF}_j = \frac{1}{1-R_j^2}")
    vif = load_table("vif")
    vif["feature"] = vif["feature"].str.replace("C(crop)[T.", "Crop: ", regex=False).str.replace("]", "", regex=False)
    c1, c2 = st.columns([1, 1.3])
    c1.dataframe(vif.round(3), hide_index=True, width="stretch", height=480)
    fig = go.Figure(go.Bar(x=vif["VIF"], y=vif["feature"], orientation="h", marker_color=GREEN))
    fig.add_vline(x=5, line=dict(color="#f59e0b", dash="dash"), annotation_text="5 (moderate)")
    fig.add_vline(x=10, line=dict(color="#dc2626", dash="dash"), annotation_text="10 (serious)")
    fig.update_layout(title="Variance Inflation Factors", height=480, xaxis=dict(range=[0, 11]))
    c2.plotly_chart(fig, width="stretch")
    note("Every VIF is below 3. <b>Verdict: satisfied.</b> (The large condition number statsmodels reports is a scaling artefact "
         "of raw year values near 2000, not real collinearity.)")

with tabs[5]:
    st.markdown("#### (a) Log-transform the target")
    comp = pd.DataFrame({
        "Raw yield model": [A["rainbow_p"], A["shapiro_W"], A["resid_skew"], A["resid_kurtosis"], A["bp_LM"], A["dw_panel_order"],
                            res["mlr"]["test"]["R2"], res["mlr"]["test"]["RMSE"]],
        "log(yield) model": [L["rainbow_p"], L["shapiro_W"], L["resid_skew"], L["resid_kurtosis"], L["bp_LM"], L["dw_panel_order"],
                             res["assumptions"]["log_model"]["test"]["R2"], res["assumptions"]["log_model"]["test"]["RMSE"]]},
        index=["Rainbow p", "Shapiro–Wilk W", "Residual skew", "Residual kurtosis", "Breusch–Pagan LM",
               "Durbin–Watson (panel)", "Test R² (t/ha scale)", "Test RMSE (t/ha)"])
    st.dataframe(comp.style.format("{:.4f}"), width="stretch")
    note("The log transform reduces skewness, kurtosis and heteroscedasticity, and improves test accuracy, but does not fully "
         "remove the violations.")
    show_png(figure_path("10_mlr_diagnostics_log"), "Diagnostics of the log(yield) model (pipeline figure)")
    st.markdown("#### (b) Robust standard errors")
    se = load_table("mlr_robust_standard_errors", index_col=0).loc[NUM]
    se.index = [LABELS[i] for i in se.index]
    st.dataframe(se.style.format("{:.6f}"), width="stretch")
    note("Country-clustered SEs, which allow heteroscedasticity and within-country correlation, are 4–5× the classical ones. "
         "<b>Rainfall loses significance (p = 0.62)</b>; pesticides, temperature and year stay significant.")
