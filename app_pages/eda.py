import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data import LABELS, figure_path, load_clean, load_results
from utils.ui import DIVERGING, GREEN, SEQ, note, page_header, section, show_png

page_header("Exploratory Data Analysis", "Interactive charts built from the cleaned dataset "
            "(the original report figures are in the last tab)")

df = load_clean()
res = load_results()
CROP_ORDER = df.groupby("crop").yield_t_ha.mean().sort_values(ascending=False).index.tolist()
NUMS = ["avg_rainfall_mm", "pesticides_tonnes", "log_pesticides", "avg_temp_c", "year", "yield_t_ha"]

t1, t2, t3, t4, t5, t6, t7 = st.tabs(["Yield distribution", "Crop-wise", "Country-wise", "Year-wise trends",
                                      "Correlation", "Scatter plots", "Original report figures"])

with t1:
    c1, c2 = st.columns([1, 3])
    feat = c1.selectbox("Variable", NUMS, index=5, format_func=lambda c: LABELS[c])
    log_x = c1.checkbox("Log scale (ln)", value=False,
                        help="Shows ln(value); useful for the right-skewed yield and pesticide variables")
    bins = c1.slider("Bins", 10, 100, 40)
    crop_split = c1.checkbox("Split by crop (overlay)", value=False)
    x = np.log(df[feat].clip(lower=1e-3)) if log_x else df[feat]
    plot_df = df.assign(_x=x)
    fig = px.histogram(plot_df, x="_x", nbins=bins, color="crop" if crop_split else None,
                       category_orders={"crop": CROP_ORDER}, opacity=.75 if crop_split else 1,
                       color_discrete_sequence=SEQ if crop_split else [GREEN])
    fig.update_layout(title=f"Distribution of {LABELS[feat]}" + (" (ln)" if log_x else ""), bargap=.03,
                      xaxis_title=LABELS[feat] + (" (ln)" if log_x else ""), yaxis_title="Records", height=430,
                      barmode="overlay")
    c2.plotly_chart(fig, width="stretch")
    s = df[feat]
    c2.caption(f"Mean {s.mean():,.3f} · median {s.median():,.3f} · mode {s.mode().iloc[0]:,.3f} · "
               f"variance {s.var():,.3f} · SD {s.std():,.3f} · skewness {s.skew():.3f} · kurtosis {s.kurt():.3f}")
    note("Yield is strongly <b>right-skewed</b> (skew 2.08, mean 7.10 > median 3.95): most records are "
         "low-yield cereals and pulses, and tuber crops form a long right tail. log(yield) is almost symmetric.")

with t2:
    kind = st.radio("Chart", ["Box plot", "Violin", "Mean ± SD bar"], horizontal=True)
    if kind == "Mean ± SD bar":
        g = df.groupby("crop").yield_t_ha.agg(["mean", "std", "count"]).loc[CROP_ORDER]
        fig = go.Figure(go.Bar(x=g.index, y=g["mean"], error_y=dict(type="data", array=g["std"], color="#5b6b5b"),
                               marker_color=GREEN, customdata=g["count"],
                               hovertemplate="%{x}<br>mean %{y:.2f} t/ha<br>n = %{customdata}<extra></extra>"))
    elif kind == "Box plot":
        fig = px.box(df, x="crop", y="yield_t_ha", category_orders={"crop": CROP_ORDER},
                     color_discrete_sequence=[GREEN], points="outliers", hover_data=["country", "year"])
    else:
        fig = px.violin(df, x="crop", y="yield_t_ha", category_orders={"crop": CROP_ORDER}, box=True,
                        color_discrete_sequence=[GREEN])
    fig.update_layout(title="Yield by crop (sorted by mean)", xaxis_title="Crop", yaxis_title="Yield (t/ha)", height=480)
    st.plotly_chart(fig, width="stretch")
    note(f"Crop type alone explains <b>η² = {res['eta_squared']['crop']:.1%}</b> of yield variance "
         f"(one-way ANOVA F = {res['anova_crop']['F']:.1f}, p &lt; 0.001). Potatoes average 18.3 t/ha; soybeans 1.7 t/ha.")

with t3:
    c1, c2, c3 = st.columns([1.3, 1, 1])
    crop_c = c1.selectbox("Crop", CROP_ORDER, index=CROP_ORDER.index("Wheat") if "Wheat" in CROP_ORDER else 0)
    topn = c2.slider("Number of countries", 5, 40, 15)
    which = c3.radio("Show", ["Highest", "Lowest"], horizontal=True)
    g = df[df.crop == crop_c].groupby("country").yield_t_ha.agg(["mean", "count"])
    g = g.sort_values("mean", ascending=(which == "Lowest")).head(topn).iloc[::-1]
    fig = go.Figure(go.Bar(x=g["mean"], y=g.index, orientation="h", marker_color=GREEN, customdata=g["count"],
                           hovertemplate="%{y}<br>mean %{x:.2f} t/ha<br>%{customdata} years<extra></extra>"))
    fig.update_layout(title=f"{which} mean {crop_c} yield by country (1990–2013)", xaxis_title="Mean yield (t/ha)",
                      height=max(350, 22 * len(g) + 120))
    st.plotly_chart(fig, width="stretch")
    note(f"Country explains <b>η² = {res['eta_squared']['country']:.1%}</b> of yield variance; it stands in for soil, "
         "irrigation, technology and management, which the dataset does not record.")

with t4:
    c1, c2 = st.columns([1, 3])
    sel = c1.multiselect("Crops", CROP_ORDER, default=CROP_ORDER[:4] + ["Wheat", "Maize"] if "Wheat" in CROP_ORDER else CROP_ORDER[:5])
    country_t = c1.selectbox("Country (optional)", ["All countries (mean)"] + sorted(df.country.unique()))
    sub = df if country_t.startswith("All") else df[df.country == country_t]
    sub = sub[sub.crop.isin(sel)]
    if sub.empty:
        c2.warning("No data for this selection.")
    else:
        tr = sub.groupby(["year", "crop"]).yield_t_ha.mean().reset_index()
        fig = px.line(tr, x="year", y="yield_t_ha", color="crop", markers=True,
                      category_orders={"crop": CROP_ORDER}, color_discrete_sequence=SEQ)
        fig.update_layout(title=f"Mean yield by year: {country_t}", yaxis_title="Yield (t/ha)", xaxis_title="Year",
                          height=460)
        c2.plotly_chart(fig, width="stretch")
    note("Most crops show a slow upward trend, consistent with technological progress; the multiple "
         "regression estimates +0.084 t/ha per year.")

with t5:
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True, format_func=str.title)
    cols = ["avg_rainfall_mm", "pesticides_tonnes", "log_pesticides", "avg_temp_c", "year", "yield_t_ha"]
    cm = df[cols].corr(method=method)
    labels = [LABELS[c] for c in cols]
    fig = go.Figure(go.Heatmap(z=cm.values, x=labels, y=labels, zmin=-1, zmax=1, colorscale=DIVERGING,
                               text=np.round(cm.values, 2), texttemplate="%{text}", xgap=3, ygap=3,
                               hovertemplate="%{y} vs %{x}<br>r = %{z:.3f}<extra></extra>"))
    fig.update_layout(title=f"{method.title()} correlation matrix", height=520, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    note("Every numeric variable is only <b>weakly</b> correlated with yield (|r| &lt; 0.2). Yield depends mainly "
         "on <i>which crop</i> is grown and <i>where</i>. No predictor pair is strongly correlated (max |r| ≈ 0.34 "
         "excluding pesticides vs its own log).")

with t6:
    c1, c2 = st.columns([1, 3])
    xf = c1.selectbox("X variable", ["log_pesticides", "avg_temp_c", "avg_rainfall_mm", "year"],
                      format_func=lambda c: LABELS[c])
    color_by = c1.checkbox("Colour by crop", value=True)
    crops_s = c1.multiselect("Crops", CROP_ORDER, default=CROP_ORDER)
    trend = c1.checkbox("OLS trend line", value=False)
    s = df[df.crop.isin(crops_s)]
    if s.empty:
        c2.warning("Select at least one crop.")
    else:
        fig = px.scatter(s, x=xf, y="yield_t_ha", color="crop" if color_by else None, opacity=.45,
                         category_orders={"crop": CROP_ORDER}, hover_data=["country", "year"],
                         color_discrete_sequence=SEQ if color_by else [GREEN], render_mode="webgl",
                         trendline="ols" if trend else None,
                         trendline_scope="overall" if trend else None,
                         trendline_color_override="#1f2a1f" if trend else None)
        fig.update_traces(marker=dict(size=5))
        r = s[[xf, "yield_t_ha"]].corr().iloc[0, 1]
        fig.update_layout(title=f"{LABELS[xf]} vs yield (r = {r:.3f})", xaxis_title=LABELS[xf],
                          yaxis_title="Yield (t/ha)", height=500)
        c2.plotly_chart(fig, width="stretch")
    note("Yield forms horizontal bands (the crops). Rainfall forms vertical stripes because each country has a "
         "single fixed rainfall value.")

with t7:
    st.caption("PNG figures generated by `src/crop_yield_pipeline.py` and used in the report.")
    for name, cap in [("01_histograms", "Distributions of numeric variables"),
                      ("02_boxplot_yield_by_crop", "Yield by crop"),
                      ("03_boxplots_numeric", "Box plots of numeric variables"),
                      ("04_scatter_features_vs_yield", "Numeric features vs yield"),
                      ("05_correlation_heatmap", "Correlation heatmaps"),
                      ("06_yield_trend_by_crop", "Yield trend by crop")]:
        show_png(figure_path(name), cap)
