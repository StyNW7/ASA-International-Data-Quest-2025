# =============================================
# THE PRICE OF PROGRESS — FULL DATA PIPELINE
# From time series to cleaned analytical dataset
# =============================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import statsmodels.formula.api as smf

# -----------------------
# STEP 1. Load dataset
# -----------------------
path = "./final_unclean.csv"
df = pd.read_csv(path)

print("✅ Loaded dataset with shape:", df.shape)
print(df.head())

# Ensure consistent naming and sorting
df.columns = df.columns.str.strip()
df = df.sort_values(["ISO3", "Year"]).reset_index(drop=True)

# -----------------------
# STEP 2. Define imputation function
# -----------------------
def impute_numeric_panel(df_panel, var, country_col='ISO3', year_col='Year', region_col='continent'):
    """
    Ethical imputation preserving country and region integrity.
    Returns df with imputed var and metadata flags.
    """
    dfp = df_panel.copy()
    flag_col = f"{var}_imputed"
    method_col = f"{var}_imputed_method"
    dfp[flag_col] = False
    dfp[method_col] = "original"
    orig_na = dfp[var].isna()

    # --- Per-country linear interpolation ---
    dfp = dfp.sort_values([country_col, year_col])
    interp = (
        dfp.groupby(country_col, group_keys=False)
        .apply(lambda g: g.assign(**{var: g[var].interpolate(method='linear', limit_direction='both')}))
    )
    interp_mask = orig_na & interp[var].notna()
    interp.loc[interp_mask, flag_col] = True
    interp.loc[interp_mask, method_col] = "interpolated"

    # --- Per-country ffill/bfill ---
    before_na = interp[var].isna()
    interp[var] = interp.groupby(country_col)[var].transform(lambda s: s.ffill().bfill())
    ffill_mask = before_na & interp[var].notna()
    interp.loc[ffill_mask, flag_col] = True
    interp.loc[ffill_mask, method_col] = "ffill_bfill"

    # --- Region-year median ---
    if region_col in interp.columns:
        before_na = interp[var].isna()
        region_median = interp.groupby([region_col, year_col])[var].transform('median')
        interp[var] = interp[var].fillna(region_median)
        region_mask = before_na & interp[var].notna()
        interp.loc[region_mask, flag_col] = True
        interp.loc[region_mask, method_col] = "region_year_median"

    # --- Global median fallback ---
    before_na = interp[var].isna()
    global_median = interp[var].median(skipna=True)
    interp[var] = interp[var].fillna(global_median)
    global_mask = before_na & interp[var].notna()
    interp.loc[global_mask, flag_col] = True
    interp.loc[global_mask, method_col] = "global_median"

    # --- Any remaining NA (rare) ---
    still_na = interp[var].isna()
    if still_na.any():
        interp.loc[still_na, flag_col] = True
        interp.loc[still_na, method_col] = "unfilled"

    return interp[[country_col, year_col, var, flag_col, method_col]]


# -----------------------
# STEP 3. Impute variables
# -----------------------
vars_to_impute = ['HDI', 'Suicide_rate', 'GDP_per_capita']
df_out = df.copy()

for var in vars_to_impute:
    print(f"🧩 Imputing {var} ...")
    res = impute_numeric_panel(df_out, var, country_col='ISO3', year_col='Year', region_col='continent')
    df_out = df_out.drop(columns=[var], errors='ignore').merge(res, on=['ISO3', 'Year'], how='left')

# -----------------------
# STEP 4. Feature engineering
# -----------------------
df_out['log_GDP_per_capita'] = np.log(df_out['GDP_per_capita'].where(df_out['GDP_per_capita'] > 0))
df_out['HDI_sq'] = df_out['HDI'] ** 2
df_out['HDI_growth'] = df_out.groupby('ISO3')['HDI'].diff()
df_out['Suicide_change'] = df_out.groupby('ISO3')['Suicide_rate'].diff()
df_out['Suicide_per_HDI'] = df_out['Suicide_rate'] / df_out['HDI']

# -----------------------
# STEP 5. Exploratory Data Analysis
# -----------------------
eda_text = []

eda_text.append("=== EDA REPORT: THE PRICE OF PROGRESS ===\n")
eda_text.append(f"Rows: {len(df_out)}, Columns: {len(df_out.columns)}\n")
eda_text.append("Variables:\n" + ", ".join(df_out.columns) + "\n")

eda_text.append("\n--- Missing Values Summary ---\n")
eda_text.append(str(df_out.isna().sum()))

# Correlation
numeric_cols = ['HDI','GDP_per_capita','Suicide_rate','log_GDP_per_capita']
corr = df_out[numeric_cols].corr()
eda_text.append("\n\n--- Correlation Matrix ---\n")
eda_text.append(str(corr))

# Plot correlation heatmap
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix — HDI, GDP, Suicide Rate")
plt.tight_layout()
plt.savefig("corr_matrix.png", dpi=300)
plt.close()

# Scatter
plt.figure(figsize=(6,5))
sns.scatterplot(data=df_out, x='HDI', y='Suicide_rate', hue='continent')
plt.title("HDI vs Suicide Rate")
plt.savefig("hdi_vs_suicide.png", dpi=300)
plt.close()

# -----------------------
# STEP 6. Regression models
# -----------------------
df_model = df_out.dropna(subset=['HDI','Suicide_rate','log_GDP_per_capita'])
model = smf.ols("Suicide_rate ~ HDI + HDI_sq + log_GDP_per_capita", data=df_model).fit(cov_type='HC1')
eda_text.append("\n\n--- Regression Summary ---\n")
eda_text.append(str(model.summary()))

# Compute tipping point
params = model.params
if 'HDI_sq' in params and params['HDI_sq'] != 0:
    tipping_point = -params['HDI'] / (2 * params['HDI_sq'])
    eda_text.append(f"\nEstimated HDI tipping point: {tipping_point:.3f}")

# -----------------------
# STEP 7. Export cleaned data and report
# -----------------------
df_out.to_csv("final_dataset.csv", index=False)
with open("EDA_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(eda_text))

print("✅ Saved final_dataset.csv and EDA_report.txt successfully!")
