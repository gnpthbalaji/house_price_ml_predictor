# Predicting Housing Prices in King County

- [Predicting Housing Prices in King County](#predicting-housing-prices-in-king-county)
  - [Notebooks](#notebooks)
  - [Dataset](#dataset)
  - [Project Goal](#project-goal)
  - [Key Findings](#key-findings)
    - [1. Exploratory Data Analysis](#1-exploratory-data-analysis)
    - [2. Model Building](#2-model-building)
    - [3. Model Evaluation](#3-model-evaluation)
      - [Performance Comparison](#performance-comparison)
      - [Top 5 Most Important Features (by coefficient magnitude)](#top-5-most-important-features-by-coefficient-magnitude)
      - [Residual Analysis](#residual-analysis)
  - [Recommendations](#recommendations)
    - [Why All Models Performed the Same](#why-all-models-performed-the-same)
    - [Improving Model Performance (R² ≈ 0.69 → Higher)](#improving-model-performance-r--069--higher)
  - [Key Takeaway](#key-takeaway)

## Notebooks

- [Exploratory Data Analysis](notebooks/01_exploratory_data_analysis.ipynb)
- [Model Building](notebooks/02_model_building.ipynb)
- [Model Evaluation](notebooks/03_model_eval.ipynb)

---

## Dataset

The **King County House Sales dataset** from [Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) contains **21,613 house sales** in King County, Washington (includes Seattle).

**Features (21 columns):** square footage (living, lot, above, basement), bedrooms, bathrooms, floors, waterfront, view, condition, grade, year built, year renovated, geographic location (lat, long, zipcode), and neighborhood comparisons (sqft_living15, sqft_lot15).

**Target variable:** Sale Price

---

## Project Goal

Build and compare regression models — **Linear Regression, Ridge Regression, and Lasso Regression** — to predict housing prices, evaluating with R², MAE, and RMSE.

---

## Key Findings

### 1. Exploratory Data Analysis

**Price Distribution:** The target variable is heavily right-skewed (skewness = 4.02). The mean price is \$540,088 while the median is \$450,000, indicating a long tail of luxury homes pulling the average up. Nearly 89% of homes fall within one standard deviation (\$172,961 – \$907,215).

**Sale Price Distribution**
![Sale Price Distribution](/output/figures/sale_price_analysis.png)

**Average Price By Decade**
![Average Price By Decade](/output/figures/avg_price_by_decade.png)

**House Price by Year Built**
![Sale Price Distribution](/output/figures/house_price_by_year_built.png)
**Top Correlated Features with Price:**

| Strength | Feature       | Correlation (r) |
| -------- | ------------- | --------------- |
| Strong   | sqft_living   | 0.702           |
| Strong   | grade         | 0.667           |
| Strong   | sqft_above    | 0.606           |
| Strong   | sqft_living15 | 0.585           |
| Strong   | bathrooms     | 0.525           |
| Moderate | view          | 0.397           |
| Moderate | sqft_basement | 0.324           |
| Moderate | bedrooms      | 0.308           |
| Moderate | lat           | 0.307           |

Size and grade (quality) are the strongest price drivers. Notably, bathrooms correlate more strongly with price than bedrooms.

![Feature Correlations](/output/figures/feature_correlation.png)

![Price vs Key Features](/output/figures/price_vs_features.png)

**Multicollinearity:** `sqft_above` and `sqft_living` are highly correlated (r = 0.877), which makes sense since `sqft_living = sqft_above + sqft_basement`. The `sqft_above` feature was dropped to avoid coefficient instability.

**Feature Distributions:** Most homes have 3–4 bedrooms, 2–3 bathrooms, 1,000–3,000 sqft living space, and 1–2 floors. Year built is fairly uniformly distributed (1900–2015), so a derived `house_age` feature was created.

![Feature Distributions](/output/figures/feature_distribution.png)

**Outliers:** 406 price outliers detected via z-score (|z| > 3), representing 1.88% of data. Eleven homes exceeded \$4M; 11 had more than 8 bedrooms. One notable anomaly: a home listed with 33 bedrooms at only 1,620 sqft.

![Outliers Analysis](/output/figures/house_outliers.png)
![Anomaly Analysis](/output/figures/anomaly_price_overcrowded_expensive.png)

---

### 2. Model Building

**Selected Features (11):** bedrooms, bathrooms, sqft_living, floors, waterfront, view, condition, grade, house_age, lat, long

**Train/Test Split:** 80/20 (17,290 train / 4,323 test), features standardized with `StandardScaler`.

**Alpha Tuning:**

- Ridge: best alpha = 0.1 (performance degrades noticeably only at alpha ≥ 1000)
- Lasso: best alpha = 0.1 (retains all 12 features even at alpha = 100)

---

### 3. Model Evaluation

#### Performance Comparison

| Model             | R² (Train) | R² (Test) | MAE (Test) | RMSE (Test) |
| ----------------- | ---------- | --------- | ---------- | ----------- |
| Linear Regression | 0.6925     | 0.6928    | \$128,388  | \$215,494   |
| Ridge Regression  | 0.6925     | 0.6928    | \$128,388  | \$215,494   |
| Lasso Regression  | 0.6925     | 0.6928    | \$128,388  | \$215,494   |

All three models perform virtually identically. The negligible gap between train and test R² (0.0003) confirms **no overfitting**.

![Model Performance Comparison](/output/figures/model__perf_comparison.png)

![Predicted vs Actual](/output/figures/model_prediction_comparison.png)

#### Top 5 Most Important Features (by coefficient magnitude)

| Rank | Feature        | Coefficient |
| ---- | -------------- | ----------- |
| 1    | sqft_living    | +\$159,735  |
| 2    | grade          | +\$121,913  |
| 3    | house_age      | +\$78,689   |
| 4    | lat (latitude) | +\$75,825   |
| 5    | waterfront     | +\$48,323   |

Feature importance is consistent across all three models, confirming that the feature set has low multicollinearity and all features contribute meaningfully.

![Feature Importance](/output/figures/feature_importance_comparison.png)

![Coefficient Comparison](/output/figures/model_coefficient_comparison.png)

#### Residual Analysis

Residuals are approximately normally distributed and centered near zero, but show heteroscedasticity — the models underpredict high-value homes, as seen in the residual scatter plots.

![Residual Analysis](/output/figures/models_residual_plot_comparison.png)

![Error Distribution](/output/figures/models_error_distribution.png)

---

## Recommendations

### Why All Models Performed the Same

After removing `sqft_above` (the primary multicollinearity source), the remaining 11 features had low correlation with each other. With no redundant or weak predictors, regularization (Ridge/Lasso) had nothing to penalize — hence identical results.

### Improving Model Performance (R² ≈ 0.69 → Higher)

The current R² of ~0.69 means the models explain about 69% of price variance. To improve:

1. **Feature Engineering:**
   - Log-transform `price` and `sqft_living` to address right skew and heteroscedasticity
   - Create interaction terms (e.g., `sqft_living × grade`)
   - Encode `zipcode` as a categorical feature (neighborhood-level pricing effects are significant)
   - Add `is_renovated` binary feature and `renovation_age`

2. **Additional Data Sources:**
   - School district quality scores
   - Proximity to amenities (transit, grocery, parks)
   - Crime/safety ratings
   - Days on market before sale

3. **Advanced Models:**
   - Random Forest or Gradient Boosting (XGBoost/LightGBM) to capture non-linear relationships
   - These would likely improve performance substantially, especially for high-value homes where linear models underperform

4. **Outlier Strategy:**
   - Consider capping or removing extreme outliers (33-bedroom home, homes > \$4M) that may distort coefficients
   - Or train separate models for standard vs. luxury segments

---

## Key Takeaway

All three regression models achieved identical performance (R² ≈ 0.69, MAE ≈ \$128K), demonstrating that when features are well-selected with low multicollinearity, regularization provides no additional benefit. The biggest gains would come from feature engineering (especially log transforms and zipcode encoding) and non-linear models rather than from tuning linear approaches. Square footage and build grade are by far the strongest predictors of King County home prices.
