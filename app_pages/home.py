import plotly.graph_objects as go
import streamlit as st

from utils.data import DATASET_URL, load_results, load_table
from utils.models import MAIN_MODELS
from utils.ui import GREEN, SEQ, card, cards, hero, note, section

res = load_results()
comp = load_table("model_comparison", index_col="Model")
best = res["best_model"]
b = comp.loc[best]

hero("Crop Yield Prediction",
     "Predicting crop yield (tonnes per hectare) for 101 countries and 10 crops using statistical "
     "regression and machine learning: from classical OLS with full assumption testing to tuned "
     "tree ensembles.",
     tags=["Statistical Modeling · 24AM5PCSML", "BMSCE · AAT 2026–27", "SDG 2 · Zero Hunger"])

cards([
    ("Records (clean)", f"{res['clean']['rows']:,}", f"from {res['raw']['rows']:,} raw rows"),
    ("Model features", "6", "crop, country, rainfall, pesticides, temperature, year"),
    ("Crops · Countries", f"{res['clean']['crops']} · {res['clean']['countries']}",
     f"years {res['clean']['year_min']}–{res['clean']['year_max']}"),
    ("Best model", "Random Forest" if "Random Forest" in best else best, "lowest test RMSE"),
], accent=True)
st.write("")
cards([
    ("Test R²", f"{b['R2']:.3f}", "share of yield variance explained"),
    ("Test RMSE", f"{b['RMSE']:.2f} t/ha", "typical prediction error"),
    ("Test MAE", f"{b['MAE']:.2f} t/ha", "average absolute error"),
    ("5-fold CV R²", f"{b['CV_R2_mean']:.3f}", f"± {b['CV_R2_std']:.3f} across folds"),
])

section("Model performance summary (20 % hold-out test set)")
left, right = st.columns([1.35, 1])
with left:
    d = comp.loc[MAIN_MODELS, "R2"]
    fig = go.Figure(go.Bar(x=d.values, y=d.index, orientation="h",
                           marker_color=[GREEN if m == best else "#a5d6a7" for m in d.index],
                           text=[f"{v:.3f}" for v in d.values], textposition="outside",
                           hovertemplate="%{y}<br>R² = %{x:.4f}<extra></extra>"))
    fig.update_layout(title="Test R² by model", height=330, xaxis=dict(range=[0, 1.08], title="R²"),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
with right:
    t = comp.loc[MAIN_MODELS, ["R2", "MAE", "RMSE"]].round(3)
    st.dataframe(t, width="stretch", height=260)
    st.caption("Full comparison, including Gradient Boosting and robustness tests, is on the "
               "**Model Comparison** page.")

section("Dataset")
c1, c2 = st.columns([1.2, 1])
with c1:
    st.markdown(f"""
**Dataset for Crop Yield Prediction** (Laktionov, Semenov & Diachenko, 2026), Zenodo, CC-BY-4.0
· [{DATASET_URL}]({DATASET_URL})

- Yield and pesticide data come from **FAOSTAT**; rainfall and temperature come from the Kaggle *Crop Yield
  Prediction Dataset*.
- **Target:** yield in t/ha (FAO hg/ha ÷ 10,000).
- **Cleaning:** {res['raw']['key_duplicates']:,} rows repeated the same country-year-crop key because of
  multiple temperature stations, so temperature was averaged per key, giving {res['clean']['rows']:,} records.
- **Not in the dataset:** humidity, soil, fertiliser, cultivated area and production.
""")
with c2:
    crop_counts = res["clean"]["crop_counts"]
    fig = go.Figure(go.Pie(labels=list(crop_counts), values=list(crop_counts.values()), hole=.55,
                           marker=dict(colors=SEQ), textinfo="none",
                           hovertemplate="%{label}: %{value} records<extra></extra>"))
    fig.update_layout(title="Records per crop", height=280, showlegend=True,
                      legend=dict(font=dict(size=11)))
    st.plotly_chart(fig, width="stretch")

section("Project workflow")
steps = [("Data", "Zenodo/FAOSTAT download, audit, de-duplication"),
         ("EDA", "Statistics, distributions, correlations"),
         ("SLR & MLR", "OLS models with crop effects"),
         ("Diagnostics", "Shapiro-Wilk, Breusch-Pagan, Durbin-Watson, VIF"),
         ("Validation", "5-fold CV, bootstrap, cluster bootstrap"),
         ("Features", "Correlation, RFECV, permutation importance"),
         ("Ridge/LASSO", "L2/L1 regularisation, α tuning"),
         ("Trees", "Decision Tree, Random Forest, Gradient Boosting"),
         ("Compare", "Test, future-year and unseen-country splits")]
cols = st.columns(len(steps))
for i, (col, (t, d)) in enumerate(zip(cols, steps), start=1):
    col.markdown(f'<div class="step"><div class="n">{i}</div><div class="t">{t}</div><div class="d">{d}</div></div>',
                 unsafe_allow_html=True)

section("SDG 2: Zero Hunger")
c1, c2 = st.columns([1, 2])
with c1:
    card("Aligned targets", "2.3 · 2.4 · 2.c", "productivity · resilient systems · market information", accent=True)
with c2:
    note("This project shows that national crop yields can be predicted from public data with measured "
         f"accuracy (R² {b['R2']:.3f} on held-out data, and R² "
         f"{[r for r in res['robustness'] if r['Model'] == best][0]['Temporal_R2']:.3f} when forecasting the "
         "later years 2010–2013). Such forecasts can support food-security planning. Real-world impact would "
         "need local data and deployment; see the <b>Conclusion & SDG 2</b> page.")
