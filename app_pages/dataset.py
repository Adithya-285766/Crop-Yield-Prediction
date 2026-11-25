import pandas as pd
import streamlit as st

from utils.data import DATASET_URL, LABELS, load_clean, load_raw, load_results, load_table
from utils.ui import cards, note, page_header, section

page_header("Dataset Explorer", "The actual cleaned dataset used for every model, with filters and download")

df = load_clean()
res = load_results()
raw = load_raw()

cards([
    ("Raw file", f"{res['raw']['rows']:,} × {res['raw']['cols']}", "df_yield_merged.csv"),
    ("Clean dataset", f"{len(df):,} × {df.shape[1]}", "after de-duplication + engineered columns"),
    ("Missing values", f"{res['raw']['missing_total']}", "in every column"),
    ("Exact duplicates", f"{res['raw']['exact_duplicates']}", f"{res['raw']['key_duplicates']:,} repeated keys merged"),
])
st.caption(f"Source: [{DATASET_URL}]({DATASET_URL}). Yield and pesticides from FAOSTAT; rainfall and temperature from the Kaggle "
           "Crop Yield Prediction Dataset (patelris).")

tab_data, tab_cols, tab_quality, tab_stats = st.tabs(
    ["🔎 Explore data", "📋 Column information", "🧹 Data quality", "📐 Descriptive statistics"])

with tab_data:
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.2, 1.4, 1.2])
        crops = c1.multiselect("Crop", sorted(df.crop.unique()), placeholder="All crops")
        countries = c2.multiselect("Country", sorted(df.country.unique()), placeholder="All countries")
        yr = c3.slider("Year range", int(df.year.min()), int(df.year.max()),
                       (int(df.year.min()), int(df.year.max())))
        c4, c5 = st.columns(2)
        ylo, yhi = float(df.yield_t_ha.min()), float(df.yield_t_ha.max())
        yrng = c4.slider("Yield range (t/ha)", ylo, yhi, (ylo, yhi))
        tmin, tmax = float(df.avg_temp_c.min()), float(df.avg_temp_c.max())
        trng = c5.slider("Temperature range (°C)", tmin, tmax, (tmin, tmax))
    f = df[df.year.between(*yr) & df.yield_t_ha.between(*yrng) & df.avg_temp_c.between(*trng)]
    if crops:
        f = f[f.crop.isin(crops)]
    if countries:
        f = f[f.country.isin(countries)]
    st.markdown(f"**{len(f):,} of {len(df):,} records match the filters.**")
    if f.empty:
        st.warning("No records match these filters. Widen the ranges or clear a selection.")
    else:
        st.dataframe(f, width="stretch", height=420, hide_index=True,
                     column_config={"yield_t_ha": st.column_config.NumberColumn("yield_t_ha", format="%.3f"),
                                    "log_pesticides": st.column_config.NumberColumn(format="%.3f"),
                                    "avg_temp_c": st.column_config.NumberColumn(format="%.2f")})
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download filtered data (CSV)", f.to_csv(index=False).encode("utf-8"),
                       "crop_yield_filtered.csv", "text/csv", disabled=f.empty, width="stretch")
    d2.download_button("⬇️ Download full clean dataset (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "crop_yield_clean.csv", "text/csv", width="stretch")

with tab_cols:
    desc = {
        "country": ("Country (FAO area)", "categorical", "Independent: 101 levels"),
        "year": ("Calendar year", "integer", "Independent"),
        "crop": ("Crop (FAO item)", "categorical", "Independent: 10 levels"),
        "avg_rainfall_mm": ("Long-term average annual rainfall; one value per country", "float, mm/year", "Independent"),
        "pesticides_tonnes": ("National pesticide use, all crops", "float, tonnes", "Source of log_pesticides"),
        "avg_temp_c": ("Average annual temperature (mean over stations)", "float, °C", "Independent"),
        "value_hg_ha": ("Original FAO yield", "integer, hg/ha", "Source of target"),
        "yield_t_ha": ("Crop yield = value_hg_ha ÷ 10,000", "float, t/ha", "**Target (dependent)**"),
        "log_pesticides": ("ln(1 + pesticides_tonnes), reduces skew 5.53 → −0.39", "float", "Independent (model input)"),
    }
    info = pd.DataFrame([{"Column": c, "Pandas dtype": str(df[c].dtype), "Description": desc[c][0],
                          "Type / unit": desc[c][1], "Role": desc[c][2], "Non-null": int(df[c].notna().sum()),
                          "Unique values": int(df[c].nunique())} for c in df.columns])
    st.dataframe(info, width="stretch", hide_index=True)
    note("<b>Model inputs (6):</b> crop, country, rainfall, log(1+pesticides), temperature, year. "
         "<b>Target:</b> yield_t_ha.")

with tab_quality:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Missing values per column (raw file)**")
        st.dataframe(raw.isna().sum().rename("missing").to_frame(), width="stretch")
    with c2:
        st.markdown("**Why rows were merged**")
        st.markdown(f"""
- {res['raw']['key_duplicates']:,} raw rows repeat a (country, year, crop) key.
- Within a key, **only temperature differs** (up to {res['raw']['max_temp_values_per_key']} values per key), because
  several weather stations per country were merged in.
- Keeping them would overweight some countries and leak identical yields into both train and test,
  so temperature was **averaged per key**, giving {res['clean']['rows']:,} rows.
- Rainfall has **{res['raw']['rain_values_per_country']} value per country** across all years (a long-term average).
""")
    st.markdown("**Outlier screening (1.5 × IQR rule): reported, not removed**")
    st.dataframe(load_table("outlier_screening").round(3), width="stretch", hide_index=True)
    st.caption(f"Applying the IQR rule within each crop flags {res['clean']['target_outliers_within_crop']} yield "
               "values. Most pooled 'outliers' are genuinely high-yield tuber crops (potatoes, cassava), so they were kept.")

with tab_stats:
    ds = load_table("descriptive_statistics", index_col=0)
    ds.index = [LABELS.get(i, i) for i in ds.index]
    st.dataframe(ds.round(3), width="stretch")
    st.markdown("**Yield by crop (t/ha)**")
    st.dataframe(load_table("yield_by_crop", index_col=0).round(3), width="stretch")
