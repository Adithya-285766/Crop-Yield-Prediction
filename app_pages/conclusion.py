import streamlit as st

from utils.data import load_results, load_table
from utils.models import MAIN_MODELS
from utils.ui import card, cards, note, page_header, section

page_header("Conclusion & SDG 2", "Final results, findings, limitations and contribution to Zero Hunger")

res = load_results()
comp = load_table("model_comparison", index_col="Model")
rob = load_table("robustness_temporal_and_unseen_country", index_col="Model")
best = res["best_model"]

section("Final results")
t = comp.loc[MAIN_MODELS, ["R2", "MAE", "MSE", "RMSE", "CV_R2_mean"]].join(rob.loc[MAIN_MODELS, ["Temporal_R2", "UnseenCountry_R2"]])
t.columns = ["Test R²", "Test MAE", "Test MSE", "Test RMSE", "5-fold CV R²", "Future-years R²", "Unseen-country R²"]
st.dataframe(t.style.format("{:.4f}").highlight_max(subset=["Test R²", "5-fold CV R²", "Future-years R²", "Unseen-country R²"],
                                                    color="#c8e6c9").highlight_min(subset=["Test MAE", "Test MSE", "Test RMSE"],
                                                                                   color="#c8e6c9"), width="stretch")

c1, c2 = st.columns([1, 2])
with c1:
    card("Final model", "Random Forest", f"R² {comp.loc[best, 'R2']:.3f} · RMSE {comp.loc[best, 'RMSE']:.2f} t/ha", accent=True)
with c2:
    note("<b>Why Random Forest:</b> lowest MAE, MSE and RMSE; highest CV R² (0.965); best when forecasting 2010–2013 "
         "(R² 0.935); smaller overfitting gap than the single tree. It is less interpretable, so the <b>MLR</b> (with "
         "cluster-robust SEs) is kept as the explanatory model. <i>Statistical models explain; ML models predict.</i>")

section("Important findings")
st.markdown("""
1. **Crop type dominates** yield (η² = 0.54), followed by **country** (η² = 0.21).
2. With robust inference, **higher temperature is associated with lower yield** (−0.154 t/ha per °C), while pesticide
   intensity and the time trend (+0.084 t/ha per year) are associated with higher yield. **Rainfall, as recorded, shows no reliable association.**
3. OLS **normality, homoscedasticity and independence** assumptions are violated. A log transform and country-clustered SEs
   partly correct this, and CV and bootstrap show the estimates are stable.
4. **Regularisation** brings no accuracy gain at this sample size, but LASSO yields a sparser model that transfers better to new countries.
5. **Tree ensembles** capture crop × country interactions: Random Forest R² = 0.973 vs 0.615 for the MLR.
6. **Transfer to unseen countries** is the main weakness (best R² ≈ 0.63).
""")

c1, c2 = st.columns(2)
with c1:
    section("Limitations")
    st.markdown("""
- **Climate variability:** rainfall is a fixed long-term average; there is no seasonal weather, drought or heatwave data.
- **Regional differences:** national averages hide sub-national variation.
- **Data quality:** temperature stations had to be merged; pesticides are national totals, not per-crop rates.
- **Missing variables:** soil, fertiliser, irrigation, humidity, crop variety and management.
- **Unseen conditions:** tree models cannot extrapolate beyond 1990–2013 conditions.
- **Association, not causation.**
""")
with c2:
    section("Future enhancements")
    st.markdown("""
- Sub-national (district or state) yield data with **seasonal weather** (monthly rainfall, heat days).
- Add **soil, fertiliser rate, irrigation** and satellite indices (NDVI).
- **Mixed-effects / panel models** for the country-year structure.
- **Time-series validation** and prediction intervals (quantile forests or conformal prediction).
- Explainability with **SHAP**, and deployment as a regional forecasting service.
""")

section("SDG 2: Zero Hunger contribution")
items = [
    ("1 · Agricultural productivity (2.3)", "Identified crop and country as the dominant drivers and a +0.084 t/ha/year trend.",
     "Targeting extension work at the biggest yield gaps."),
    ("2 · Food security (2.1, 2.c)", "Future-year forecasts for known countries reach R² 0.935 (RMSE 2.23 t/ha).",
     "Early warning of production shortfalls; import and buffer-stock planning."),
    ("3 · Informed decisions", "Crop-specific predictions with known error (e.g. MAE 0.32 t/ha for wheat).",
     "Farm-level tools, once local soil and weather data are added."),
    ("4 · Resource utilisation (2.4)", "Positive pesticide–yield association, largely shared with country effects.",
     "With per-hectare input rates: find where extra inputs stop paying off."),
    ("5 · Sustainable planning (2.4)", "Robust negative temperature association (cluster-robust p = 0.0012).",
     "Prioritising heat-tolerant varieties and irrigation in warming regions."),
]
for title, shown, potential in items:
    c1, c2, c3 = st.columns([1.1, 1.5, 1.5])
    c1.markdown(f"**{title}**")
    c2.markdown(f"✅ *Demonstrated:* {shown}")
    c3.markdown(f"🔭 *Potential:* {potential}")
note("<b>Demonstrated vs potential:</b> this project shows that national crop yields can be described and predicted from public "
     "data with measured accuracy. It does <b>not</b> demonstrate real-world gains in productivity or food security; those "
     "depend on better data, deployment and field validation.", "warn")

cards([("Records", f"{res['clean']['rows']:,}", "101 countries · 10 crops"),
       ("Models compared", "9", "6 required + 3 supplementary"),
       ("Best test R²", f"{comp.loc[best, 'R2']:.3f}", "Random Forest"),
       ("Future-years R²", f"{rob.loc[best, 'Temporal_R2']:.3f}", "2010–2013 forecast")])
