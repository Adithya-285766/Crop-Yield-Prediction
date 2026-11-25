"""Shared look-and-feel: CSS, metric cards, Plotly template."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

GREEN = "#2e7d32"
GREEN_DARK = "#1b5e20"
GREEN_LIGHT = "#e8f5e9"
INK = "#1f2a1f"
INK2 = "#5b6b5b"
GRID = "#e6ebe4"
# Categorical sequence (green first, then distinguishable hues)
SEQ = ["#2e7d32", "#f59e0b", "#2563eb", "#db2777", "#0d9488", "#7c3aed", "#dc2626", "#65a30d",
       "#0891b2", "#a16207"]
DIVERGING = [[0, "#2563eb"], [0.5, "#f3f4f1"], [1, "#dc2626"]]
SEQUENTIAL = [[0, "#e8f5e9"], [1, "#1b5e20"]]

pio.templates["agri"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, Segoe UI, sans-serif", color=INK, size=13),
        paper_bgcolor="white", plot_bgcolor="white", colorway=SEQ,
        xaxis=dict(gridcolor=GRID, zeroline=False, linecolor="#c9d3c6", ticks="outside", tickcolor="#c9d3c6"),
        yaxis=dict(gridcolor=GRID, zeroline=False, linecolor="#c9d3c6"),
        title=dict(font=dict(size=16, color=INK), x=0, xanchor="left"),
        legend=dict(bgcolor="rgba(255,255,255,0)"),
        margin=dict(l=10, r=10, t=50, b=10),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
)
pio.templates.default = "agri"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {GREEN_LIGHT} 0%, #ffffff 100%); }}
.hero {{ background: linear-gradient(120deg, {GREEN_DARK} 0%, {GREEN} 60%, #66bb6a 100%);
        color: white; padding: 2rem 2.2rem; border-radius: 18px; margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(27,94,32,.18); }}
.hero h1 {{ color: white; margin: 0 0 .4rem 0; font-size: 2.2rem; }}
.hero p {{ color: #e8f5e9; margin: 0; font-size: 1.05rem; line-height: 1.55; }}
.hero .tag {{ display:inline-block; background: rgba(255,255,255,.18); padding: .25rem .7rem;
             border-radius: 999px; font-size: .8rem; margin-right: .4rem; margin-bottom: .7rem; }}
.card {{ background: white; border: 1px solid #e3e9e1; border-radius: 14px; padding: 1rem 1.1rem;
        box-shadow: 0 2px 8px rgba(31,42,31,.05); height: 100%; min-height: 128px; }}
.card .label {{ color: {INK2}; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }}
.card .value {{ color: {GREEN_DARK}; font-size: 1.75rem; font-weight: 700; line-height: 1.2; margin-top: .25rem; }}
.card .sub {{ color: {INK2}; font-size: .82rem; margin-top: .2rem; }}
.card.accent {{ border-left: 5px solid {GREEN}; }}
.section-title {{ font-size: 1.25rem; font-weight: 700; color: {INK}; margin: 1.6rem 0 .6rem 0;
                 padding-bottom: .35rem; border-bottom: 2px solid {GREEN_LIGHT}; }}
.note {{ background: {GREEN_LIGHT}; border-left: 4px solid {GREEN}; padding: .8rem 1rem; border-radius: 8px;
        color: {INK}; font-size: .93rem; margin: .5rem 0 1rem 0; }}
.warn {{ background: #fff7e6; border-left: 4px solid #f59e0b; padding: .8rem 1rem; border-radius: 8px;
        color: {INK}; font-size: .93rem; margin: .5rem 0 1rem 0; }}
.step {{ background: white; border: 1px solid #e3e9e1; border-radius: 12px; padding: .8rem .7rem; text-align: center;
        height: 100%; }}
.step .n {{ display:inline-block; width: 28px; height: 28px; line-height: 28px; border-radius: 50%;
           background: {GREEN}; color: white; font-weight: 700; font-size: .85rem; margin-bottom: .35rem; }}
.step .t {{ font-weight: 600; font-size: .9rem; color: {INK}; }}
.step .d {{ font-size: .78rem; color: {INK2}; margin-top: .2rem; }}
.result {{ background: linear-gradient(120deg, {GREEN_DARK}, {GREEN}); color: white; border-radius: 18px;
          padding: 1.6rem 1.8rem; box-shadow: 0 8px 24px rgba(27,94,32,.2); }}
.result .big {{ font-size: 3rem; font-weight: 800; line-height: 1.1; }}
.result .unit {{ font-size: 1.2rem; font-weight: 500; opacity: .9; }}
.result .cap {{ opacity: .85; font-size: .9rem; }}
.pill {{ display:inline-block; padding: .15rem .6rem; border-radius: 999px; font-size: .78rem; font-weight: 600; }}
.pill.ok {{ background: #dcfce7; color: #166534; }}
.pill.no {{ background: #fee2e2; color: #991b1b; }}
.pill.part {{ background: #fef3c7; color: #92400e; }}
div[data-testid="stMetric"] {{ background: white; border: 1px solid #e3e9e1; border-radius: 12px; padding: .7rem .9rem; }}
</style>
"""


def apply_style():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, text, tags=()):
    tag_html = "".join(f'<span class="tag">{t}</span>' for t in tags)
    st.markdown(f'<div class="hero">{tag_html}<h1>{title}</h1><p>{text}</p></div>', unsafe_allow_html=True)


def card(label, value, sub="", accent=False):
    cls = "card accent" if accent else "card"
    st.markdown(f'<div class="{cls}"><div class="label">{label}</div><div class="value">{value}</div>'
                f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def cards(items, accent=False):
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        with col:
            card(*it, accent=accent)


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def note(text, kind="note"):
    st.markdown(f'<div class="{kind}">{text}</div>', unsafe_allow_html=True)


def pill(text, kind):
    return f'<span class="pill {kind}">{text}</span>'


def page_header(title, subtitle):
    st.markdown(f"## {title}")
    st.caption(subtitle)


def show_png(path, caption):
    if path is not None:
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Figure not found ({caption}). Run `python src/crop_yield_pipeline.py` to generate it.")
