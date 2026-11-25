"""Crop Yield Prediction - Streamlit dashboard.

Run from the project folder:
    pip install -r requirements.txt
    streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="wide",
                   initial_sidebar_state="expanded")

from utils.data import missing_outputs  # noqa: E402
from utils.ui import apply_style  # noqa: E402

apply_style()

missing = missing_outputs()
if missing:
    st.error("The dashboard reads the results produced by the analysis pipeline, but these files are "
             "missing:\n\n" + "\n".join(f"- `{m}`" for m in missing) +
             "\n\nRun `python src/crop_yield_pipeline.py` from the project folder first (about 8–10 minutes).")
    st.stop()

P = "app_pages"
pages = {
    "Project": [
        st.Page(f"{P}/home.py", title="Home Dashboard", icon="🏠", default=True),
        st.Page(f"{P}/overview.py", title="Project Overview", icon="📘"),
        st.Page(f"{P}/dataset.py", title="Dataset Explorer", icon="🗂️"),
        st.Page(f"{P}/eda.py", title="Exploratory Data Analysis", icon="📊"),
    ],
    "Phase 1 · Statistical Modeling": [
        st.Page(f"{P}/phase1.py", title="Linear Regression Models", icon="📈"),
        st.Page(f"{P}/diagnostics.py", title="Regression Diagnostics", icon="🩺"),
        st.Page(f"{P}/validation.py", title="Estimator Validation", icon="🔁"),
    ],
    "Phase 2 · Machine Learning": [
        st.Page(f"{P}/phase2.py", title="Advanced Modeling", icon="🌲"),
        st.Page(f"{P}/comparison.py", title="Model Comparison", icon="🏆"),
        st.Page(f"{P}/predict.py", title="Crop Yield Prediction", icon="🌾"),
    ],
    "Wrap-up": [
        st.Page(f"{P}/conclusion.py", title="Conclusion & SDG 2", icon="🎯"),
    ],
}

with st.sidebar:
    st.markdown("### 🌾 Crop Yield Prediction")
    st.caption("Statistical Modeling (24AM5PCSML) · AAT 2026–27\nB.M.S. College of Engineering")

nav = st.navigation(pages)

with st.sidebar:
    st.divider()
    st.caption("Data: FAOSTAT-based dataset, Zenodo record 18761663 (CC-BY-4.0). "
               "All numbers come from `outputs/` written by `src/crop_yield_pipeline.py`.")

nav.run()
