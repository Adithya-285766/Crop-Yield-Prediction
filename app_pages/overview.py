import streamlit as st

from utils.ui import cards, note, page_header, section

page_header("Project Overview", "Problem, motivation, objectives, scope and applications")

section("Introduction")
st.markdown("""
Crop yield, the amount of crop harvested per hectare, varies widely between crops, countries and years.
It depends on the crop grown, the climate (temperature and rainfall), inputs such as pesticides, and
technology and management, which improve over time. Governments, food agencies and farmers have to plan
planting, imports, storage, input supply and insurance **before** the harvest is known, so reliable yield
estimates are a key input to food planning.
""")

section("Problem statement")
note("Using publicly available country-level data on <b>crop type, rainfall, temperature, pesticide use and "
     "year</b>, build and compare statistical regression models and machine-learning models that predict "
     "<b>crop yield (t/ha)</b>. Test the regression assumptions and the stability of the estimates with "
     "statistical methods, and identify the model that predicts best on unseen data.")

section("Motivation")
c1, c2 = st.columns(2)
with c1:
    st.markdown("""
- **Food security:** national food balances and import plans depend on expected production (area × yield).
- **Farm income:** expected yield drives crop choice, input investment and insurance.
""")
with c2:
    st.markdown("""
- **Climate adaptation:** knowing how yield responds to temperature helps target adaptation measures.
- **Policy:** subsidies, procurement and buffer stocks all need yield forecasts.
""")

section("Objectives")
st.markdown("""
1. Obtain, audit and clean a real, citable crop-yield dataset, and state its limitations.
2. Describe the data with descriptive statistics and visualisations (EDA).
3. Build **Simple** and **Multiple Linear Regression** models and interpret their coefficients.
4. Test the regression assumptions (linearity, normality, homoscedasticity, independence,
   multicollinearity) and apply corrections.
5. Validate the estimators with **K-fold cross-validation** and **bootstrap** resampling.
6. Select features and apply **Ridge** and **LASSO** regularisation.
7. Build tuned **Decision Tree**, **Random Forest** and **Gradient Boosting** regressors.
8. Compare all models fairly on identical splits and metrics, and select the final model.
9. Relate the work to **SDG 2: Zero Hunger**.
""")

section("Scope")
cards([
    ("Geography", "101 countries", "national averages"),
    ("Crops", "10 crops", "cereals, tubers, soybeans, plantains"),
    ("Period", "1990–2013", "annual records"),
    ("Target", "t/ha", "FAO yield ÷ 10,000"),
])
st.write("")
note("<b>Out of scope (not in the dataset):</b> humidity, soil properties, fertiliser use, cultivated "
     "area, production and sub-national data. Rainfall is a single long-term average per country, and "
     "pesticide use is a national total. These variables are not imputed or simulated.", "warn")

section("Real-world applications")
apps = {
    "🌱 Crop planning": "Compare expected yields of candidate crops before sowing.",
    "🏛️ Resource allocation": "Direct extension staff, seed and credit to crop-region pairs predicted to under-perform.",
    "🧪 Input optimisation": "With per-hectare input data, find where extra pesticide or fertiliser stops adding yield.",
    "💧 Irrigation planning": "The negative temperature association points to heat stress, where irrigation and tolerant varieties help.",
    "📦 Production forecasting": "Predicted yield × planned area gives expected production for food-balance planning.",
    "⚠️ Risk management": "Large gaps between predicted and actual yield flag shocks; error levels help price insurance.",
}
cols = st.columns(3)
for i, (k, v) in enumerate(apps.items()):
    with cols[i % 3]:
        st.markdown(f'<div class="card"><div style="font-weight:700;margin-bottom:.3rem">{k}</div>'
                    f'<div style="color:#5b6b5b;font-size:.9rem">{v}</div></div>', unsafe_allow_html=True)
        st.write("")
