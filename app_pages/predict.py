import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data import ALL_FEATURES, LABELS, TARGET, load_clean, load_results, load_table
from utils.models import MODEL_ORDER, get_models, get_split, model_inputs
from utils.ui import GREEN, note, page_header, section

page_header("Crop Yield Prediction", "Live predictions from the project's trained models, using the original preprocessing")

df = load_clean()
res = load_results()
comp = load_table("model_comparison", index_col="Model")
models = get_models()
train_df, test_df = get_split()
best = res["best_model"]

RANGES = {c: (float(df[c].min()), float(df[c].max())) for c in ["avg_rainfall_mm", "pesticides_tonnes", "avg_temp_c", "year"]}

section("1 · Choose model and location")
c1, c2, c3, c4 = st.columns([1.6, 1.3, 1.2, 0.9])
model_name = c1.selectbox("Model", MODEL_ORDER, index=MODEL_ORDER.index(best),
                          help="Test R² of each model is shown below the selector")
countries = sorted(df.country.unique())
country = c2.selectbox("Country", countries, index=countries.index("India") if "India" in countries else 0)
crops_here = sorted(df[df.country == country].crop.unique())
all_crops = sorted(df.crop.unique())
default_crop = "Wheat" if "Wheat" in crops_here else (crops_here[0] if crops_here else all_crops[0])
crop = c3.selectbox("Crop", all_crops, index=all_crops.index(default_crop))
year = c4.number_input("Year", min_value=1980, max_value=2040, value=2013, step=1)

used = model_inputs(model_name, res)
st.caption(f"**{model_name}**: test R² {comp.loc[model_name, 'R2']:.3f}, RMSE {comp.loc[model_name, 'RMSE']:.3f} t/ha. "
           "It uses: " + ", ".join(LABELS.get(f, f) for f in used)
           + ". Other inputs are shown but do not affect its prediction.")
if crop not in crops_here:
    note(f"<b>{crop}</b> was never recorded for <b>{country}</b> in the dataset. The model can still predict, but this "
         "country-crop combination is outside the training data.", "warn")

# Sensible defaults from the dataset for the chosen country (nearest available year)
cdf = df[df.country == country]
row_y = cdf.iloc[(cdf.year - year).abs().argsort()[:1]]
def_rain = float(cdf.avg_rainfall_mm.iloc[0])
def_pest = float(row_y.pesticides_tonnes.iloc[0])
def_temp = float(row_y.avg_temp_c.iloc[0])
def_year = int(row_y.year.iloc[0])

section("2 · Environmental and input features")
st.caption(f"Defaults are {country}'s recorded values (rainfall: country normal; pesticides and temperature: year {def_year}). "
           "Edit them to explore scenarios.")
with st.form("predict_form"):
    k = f"{country}_{year}"
    f1, f2, f3 = st.columns(3)
    rain = f1.number_input("Average rainfall (mm/year)", min_value=0.0, max_value=10000.0, value=def_rain, step=10.0,
                           key=f"rain_{k}", help=f"Dataset range {RANGES['avg_rainfall_mm'][0]:.0f}–{RANGES['avg_rainfall_mm'][1]:.0f} mm")
    pest = f2.number_input("Pesticide use (tonnes, national total)", min_value=0.0, max_value=2_000_000.0, value=def_pest,
                           step=100.0, key=f"pest_{k}", format="%.2f",
                           help=f"Dataset range {RANGES['pesticides_tonnes'][0]:.2f}–{RANGES['pesticides_tonnes'][1]:,.0f} t")
    temp = f3.number_input("Average temperature (°C)", min_value=-20.0, max_value=50.0, value=def_temp, step=0.1,
                           key=f"temp_{k}", format="%.2f",
                           help=f"Dataset range {RANGES['avg_temp_c'][0]:.1f}–{RANGES['avg_temp_c'][1]:.1f} °C")
    submitted = st.form_submit_button("🌾 Predict yield", type="primary", width="stretch")

if not submitted:
    st.info("Set the inputs and click **Predict yield**.")
    st.stop()

# ---- validation ------------------------------------------------------------------------------
errors, warnings = [], []
for name, v in [("Rainfall", rain), ("Pesticides", pest), ("Temperature", temp), ("Year", year)]:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        errors.append(f"{name} is missing or not a number.")
if rain is not None and rain < 0:
    errors.append("Rainfall cannot be negative.")
if pest is not None and pest < 0:
    errors.append("Pesticide use cannot be negative.")
if errors:
    for e in errors:
        st.error(e)
    st.stop()

for col, v, lab in [("avg_rainfall_mm", rain, "Rainfall"), ("pesticides_tonnes", pest, "Pesticide use"),
                    ("avg_temp_c", temp, "Temperature"), ("year", year, "Year")]:
    lo, hi = RANGES[col]
    if not lo <= v <= hi:
        warnings.append(f"{lab} = {v:,.2f} is outside the training range ({lo:,.2f} – {hi:,.2f}). "
                        "Tree models cannot extrapolate, and linear models extrapolate linearly; treat the result with caution.")
if abs(rain - def_rain) > 1e-9:
    warnings.append(f"In the dataset, rainfall is a fixed long-term value per country ({country}: {def_rain:.0f} mm). "
                    "Changing it creates a combination the models never saw.")

# ---- preprocessing: identical to the pipeline ------------------------------------------------
X = pd.DataFrame([{"avg_rainfall_mm": float(rain), "log_pesticides": float(np.log1p(pest)),
                   "avg_temp_c": float(temp), "year": int(year), "crop": crop, "country": country}])[ALL_FEATURES]
all_preds = {m: float(models[m].predict(X)[0]) for m in MODEL_ORDER}
pred = all_preds[model_name]

section("3 · Prediction")
cmain, cside = st.columns([1.15, 1])
with cmain:
    rmse = comp.loc[model_name, "RMSE"]
    st.markdown(f"""
<div class="result">
  <div class="cap">Predicted yield · {crop} · {country} · {int(year)}</div>
  <div class="big">{pred:,.3f} <span class="unit">t/ha</span></div>
  <div class="cap">= {pred * 1000:,.0f} kg/ha · model: <b>{model_name}</b> · typical error (test RMSE) ± {rmse:.2f} t/ha</div>
</div>""", unsafe_allow_html=True)
    if pred < 0:
        st.warning("The model returned a negative yield, which is physically impossible. This happens when a linear model is "
                   "pushed outside its data. It is shown unaltered so that the model's real behaviour is visible.")
    for w in warnings:
        st.warning(w)
with cside:
    st.markdown("**Input details (after preprocessing)**")
    shown = pd.DataFrame({
        "Feature": ["Country", "Crop", "Year", "Rainfall (mm/yr)", "Pesticides (t)", "→ log(1 + pesticides)", "Temperature (°C)"],
        "Value": [country, crop, str(int(year)), f"{rain:,.1f}", f"{pest:,.2f}", f"{np.log1p(pest):.4f}", f"{temp:.2f}"],
        "Used by model": ["✓" if "country" in used else "—", "✓" if "crop" in used else "—", "✓" if "year" in used else "—",
                          "✓" if "avg_rainfall_mm" in used else "—", "", "✓" if "log_pesticides" in used else "—",
                          "✓" if "avg_temp_c" in used else "—"]})
    st.dataframe(shown, hide_index=True, width="stretch")

# ---- context ---------------------------------------------------------------------------------
section("4 · Context")
obs = df[(df.country == country) & (df.crop == crop) & (df.year == year)]
k1, k2, k3 = st.columns(3)
if not obs.empty:
    o = obs.iloc[0]
    split = "training" if obs.index[0] in train_df.index else "test (unseen during training)"
    k1.metric("Recorded FAO yield for this case", f"{o[TARGET]:.3f} t/ha", f"{pred - o[TARGET]:+.3f} prediction error",
              delta_color="off")
    k1.caption(f"This row is in the **{split}** set. Recorded inputs: rainfall {o.avg_rainfall_mm:.0f} mm, "
               f"pesticides {o.pesticides_tonnes:,.2f} t, temperature {o.avg_temp_c:.2f} °C.")
else:
    k1.metric("Recorded FAO yield", "not in dataset", help="No record for this country, crop and year")
crop_all = df[df.crop == crop][TARGET]
k2.metric(f"{crop}: global mean yield", f"{crop_all.mean():.2f} t/ha", f"median {crop_all.median():.2f}", delta_color="off")
hist = df[(df.country == country) & (df.crop == crop)]
k3.metric(f"{crop} in {country}: mean 1990–2013", f"{hist[TARGET].mean():.2f} t/ha" if not hist.empty else "no records")

c1, c2 = st.columns(2)
with c1:
    s = pd.Series(all_preds).loc[MODEL_ORDER]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                           marker_color=[GREEN if m == model_name else "#a5d6a7" for m in s.index],
                           text=[f"{v:.2f}" for v in s.values], textposition="outside"))
    if not obs.empty:
        fig.add_vline(x=float(obs.iloc[0][TARGET]), line=dict(color="#f59e0b", dash="dash"), annotation_text="recorded")
    fig.update_layout(title="Same inputs, every model (t/ha)", height=400, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
with c2:
    if not hist.empty:
        fig = go.Figure(go.Scatter(x=hist.year, y=hist[TARGET], mode="lines+markers", line=dict(color=GREEN),
                                   name="recorded"))
        fig.add_trace(go.Scatter(x=[year], y=[pred], mode="markers", marker=dict(size=14, color="#f59e0b", symbol="star"),
                                 name="this prediction"))
        fig.update_layout(title=f"Recorded {crop} yield in {country}", height=400, xaxis_title="Year",
                          yaxis_title="t/ha", legend=dict(orientation="h", y=-.2))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info(f"No recorded {crop} yields for {country} to compare with.")

note("Predictions describe statistical associations learned from national 1990–2013 data. They are not causal estimates "
     "of what would happen if an input were changed, and they are less reliable for countries or conditions unlike the "
     "training data (unseen-country R² ≈ 0.63).")
