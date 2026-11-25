# Presentation Content (12 slides)
**Crop Yield Prediction Using Statistical Modeling and Machine Learning**
Statistical Modeling (24AM5PCSML) AAT · B.M.S. College of Engineering · 2026–27

Each slide lists its on-slide text, the figure or table to place on it, and short speaker notes. All
numbers come from `outputs/run_log.txt`.

---

## Slide 1: Title and introduction
**On slide**
- Crop Yield Prediction Using Statistical Modeling and Machine Learning
- SDG 2: Zero Hunger
- _Name · USN · Department of Machine Learning, BMSCE · Faculty: ____
- Data: FAO-based crop-yield dataset, 101 countries, 10 crops, 1990–2013

**Speaker notes:** We predict crop yield in tonnes per hectare from crop, country, climate and pesticide
data. We compare classical regression (Phase 1) with regularised and tree-based machine learning
(Phase 2).

---

## Slide 2: Problem statement and objectives
**On slide**
- **Problem:** food planning needs yield estimates *before* harvest; yields vary hugely by crop,
  country and year.
- **Statement:** predict yield (t/ha) from crop, country, rainfall, temperature, pesticide use and
  year; test the regression assumptions; find the best model on unseen data.
- **Objectives:**
  1. EDA.
  2. SLR and MLR.
  3. Assumption tests.
  4. Cross-validation and bootstrap.
  5. Feature selection.
  6. Ridge and LASSO.
  7. Decision Tree, Random Forest and Gradient Boosting.
  8. Fair comparison.
  9. SDG 2 mapping.

---

## Slide 3: Dataset
**On slide** (table)

| | |
|---|---|
| Source | Zenodo record 18761663 (CC-BY-4.0); FAOSTAT yield and pesticides, plus rainfall and temperature |
| Raw | 25,932 rows × 7 columns, 0 missing, 0 exact duplicates |
| Issue found | 12,802 rows repeat (country, year, crop) because of multiple temperature stations → averaged |
| Clean | **13,130 rows**: 101 countries, 10 crops, 1990–2013 |
| Target | yield t/ha = hg/ha ÷ 10,000 (mean 7.10, median 3.95, skew 2.08) |
| Features | crop, country, rainfall (mm), log(1+pesticides t), temperature (°C), year |
| **Not available** | humidity, soil, fertiliser, area, production |

**Speaker notes:** Rainfall has a single value per country (a long-term average). Pesticide use is a
national total. We say so openly and do not invent the missing variables. The split is 80/20 with
seed 42 (10,504 training rows / 2,626 test rows), made before any model is fitted.

---

## Slide 4: Exploratory data analysis
**Figures:** `02_boxplot_yield_by_crop.png` (main) and `05_correlation_heatmap.png` (small)

**On slide**
- Crop explains **54 %** of yield variance (η² = 0.537); ANOVA F = 1693, p < 0.001.
- Country explains **21 %** (η² = 0.209).
- Numeric features are only weakly correlated with yield: log-pesticides r = 0.19,
  temperature r = −0.14, year r = 0.09, rainfall r = 0.01.
- Yield is right-skewed (tuber crops reach up to 50 t/ha). Outliers were kept because they are real.

---

## Slide 5: Phase 1: Simple and multiple linear regression
**Figure:** `07_slr_regression_line.png`

**On slide**
- **SLR:** Ŷ = 2.958 + 0.560·ln(1+pesticides). The slope is significant (p ≈ 10⁻⁸⁰), but **test
  R² = 0.037**, so significant does not mean predictive.
- **MLR** (rainfall, log-pesticides, temperature, year, 9 crop dummies): **test R² = 0.615**,
  RMSE = 4.98 t/ha.
  - Temperature: **−0.154 t/ha per °C**.
  - Year: **+0.084 t/ha per year**.
  - Doubling pesticide use: **+0.40 t/ha**.
  - Potatoes +6.26 and soybeans −10.54 t/ha relative to cassava.

---

## Slide 6: Regression diagnostics
**Figure:** `08_mlr_diagnostics_raw.png`

**On slide** (table)

| Assumption | Test | Result | OK? |
|---|---|---|---|
| Linearity | Rainbow | p = 0.21 (plots show crop-specific patterns) | Partly |
| Normality | Shapiro–Wilk, JB, Q-Q | W = 0.882, p ≈ 0 | ✗ |
| Homoscedasticity | Breusch–Pagan | LM = 1797, p ≈ 0 | ✗ |
| Independence | Durbin–Watson (panel order) | 0.195 | ✗ |
| Multicollinearity | VIF | all < 3 | ✓ |

**Fixes applied:**
- log(yield): test R² rises to 0.647, residual skew falls from 1.51 to −0.78.
- HC3 and country-clustered standard errors: rainfall becomes **not significant** (p = 0.62).

---

## Slide 7: Estimator validation (cross-validation and bootstrap)
**Figure:** `11_bootstrap_distributions.png`

**On slide**
- 5-fold CV: MLR R² = **0.605 ± 0.012**; SLR 0.033 ± 0.007. Coefficient signs are the same in every
  fold.
- Bootstrap (B = 1000): the means match OLS (no bias); OOB R² 95 % CI = [0.590, 0.620].
- Country-cluster bootstrap: SEs are 4–5× larger. The rainfall CI [−0.00098, 0.00068] includes 0;
  pesticides, temperature and year stay significant.

**Speaker notes:** Rows from the same country are correlated, so resampling whole countries gives
honest uncertainty.

---

## Slide 8: Phase 2: Feature selection and regularisation
**Figures:** `14_feature_importance.png` and `16_lasso_path.png`

**On slide**
- Permutation importance (ΔR²): crop 1.49 ≫ temperature 0.28 ≈ rainfall* 0.27 > pesticides 0.20 >
  country 0.17 > year 0.05. **All 6 features kept.**
  (*Rainfall acts as a country identifier, because it is constant within each country.)
- RFECV keeps 110 of 115 encoded columns.
- Ridge (α = 0.178) and LASSO (α = 0.00015) both reach test R² = **0.735**, the same as OLS. With a
  large n there is little variance to remove.
- LASSO with the 1-SE rule (α = 0.0049) sets **26 of 115** coefficients to 0 (all country dummies) for
  only −0.005 R².

---

## Slide 9: Machine learning models
**On slide**
- **Decision Tree:** recursive SSE-minimising splits. Tuned: depth 25, min leaf 1. Train R² 0.9998
  vs test 0.960, so it overfits.
- **Random Forest:** 300 bootstrapped trees with random feature subsets. Tuned: max_features 0.6,
  min leaf 1.
- **Gradient Boosting (HistGB):** sequential trees fit to residuals. Tuned: learning rate 0.1,
  800 iterations, 63 leaves.
- All tuned with GridSearchCV on the same 5 folds (RMSE); pipelines prevent leakage.

---

## Slide 10: Performance comparison
**Figure:** `18_model_comparison_bars.png`

**On slide** (20 % hold-out test set, n = 2,626)

| Model | R² | MAE | MSE | RMSE |
|---|---|---|---|---|
| Simple Linear Regression | 0.0374 | 5.729 | 61.921 | 7.869 |
| Multiple Linear Regression | 0.6151 | 3.181 | 24.762 | 4.976 |
| Ridge Regression | 0.7349 | 2.862 | 17.053 | 4.130 |
| LASSO Regression | 0.7350 | 2.858 | 17.045 | 4.129 |
| Decision Tree Regressor | 0.9598 | 0.787 | 2.586 | 1.608 |
| **Random Forest Regressor** | **0.9730** | **0.656** | **1.734** | **1.317** |

*Supplementary:* Gradient Boosting R² = 0.9637, RMSE = 1.527.

---

## Slide 11: Best model and generalisation
**Figure:** `21_generalisation_r2.png`

**On slide**
- **Final model: Random Forest.** It has the lowest error on every metric and the best CV R²
  (0.965 ± 0.004).
- Future years (train 1990–2009, test 2010–13): RF R² = **0.935**, still the best.
- Unseen countries (GroupKFold): RF R² = **0.627**, tied with Gradient Boosting (0.630). This is the
  main limitation.
- The MLR is kept for interpretation; the Random Forest is used for prediction.

**Speaker notes:** Random splits are optimistic for panel data. The strict tests show where the model
can and cannot be trusted.

---

## Slide 12: SDG 2 contribution and conclusion
**On slide**
- **Demonstrated:**
  - Yield is predictable from public data (R² 0.973; 0.935 for future years).
  - There is a robust negative temperature association.
  - Crop and country dominate yield.
- **Potential (not demonstrated):**
  - Early warning of production shortfalls (Target 2.c).
  - Targeting productivity gaps (2.3).
  - Climate adaptation and input efficiency (2.4).
- **Limitations:**
  - National averages, so local detail is lost.
  - Fixed rainfall.
  - No soil, fertiliser or humidity data.
  - Data end in 2013.
  - Tree models cannot extrapolate.
- **Conclusion:** statistical models explain, machine-learning models predict. Use both, and validate
  with strict splits before deployment.

**Thank you · Questions?**
