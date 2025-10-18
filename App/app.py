# app.py
# Streamlit dashboard: "The Price of Progress"
# Requirements: see requirements.txt
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import base64
import time

st.set_page_config(page_title="The Price of Progress", layout="wide", initial_sidebar_state="expanded")
sns.set_style("whitegrid")

# -----------------------
# Helpers
# -----------------------
@st.cache_data
def load_data(path=None):
    if path:
        df = pd.read_csv(path)
    else:
        # If user doesn't supply file, fallback to try local Output path
        try:
            df = pd.read_csv("./Output/merged_clean_panel.csv")
        except Exception:
            df = None
    return df

def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def fit_quadratic(df, robust=True):
    """Fit Suicide_rate ~ HDI_2023 + HDI_sq + log_GDP_per_capita"""
    dfm = df.dropna(subset=['Suicide_rate','HDI_2023','HDI_sq'])
    if 'log_GDP_per_capita' not in dfm.columns and 'GDP_per_capita' in dfm.columns:
        dfm['log_GDP_per_capita'] = np.log(dfm['GDP_per_capita'].where(dfm['GDP_per_capita']>0, np.nan))
    formula = "Suicide_rate ~ HDI_2023 + HDI_sq + log_GDP_per_capita"
    try:
        model = smf.ols(formula=formula, data=dfm).fit(cov_type='HC1' if robust else None)
        return model, dfm
    except Exception as e:
        return None, dfm

def tipping_point_from_model(model):
    p = model.params
    if 'HDI_2023' in p and 'HDI_sq' in p and p['HDI_sq'] != 0:
        tp = -p['HDI_2023']/(2*p['HDI_sq'])
        return float(tp)
    return None

def fig_to_bytes(fig):
    """Save matplotlib or plotly figure to PNG bytes for download."""
    if hasattr(fig, "to_image"):  # plotly
        return fig.to_image(format="png", width=1000, height=600)
    else:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        return buf.read()

# -----------------------
# Sidebar: Data / Filters
# -----------------------
st.sidebar.title("Settings & Data")
uploaded = st.sidebar.file_uploader("Upload cleaned CSV (merged_clean_panel.csv) — optional", type=["csv"])

df = load_data(uploaded)

if df is None:
    st.sidebar.error("No dataset found. Upload `merged_clean_panel.csv` or place it at ./Output/merged_clean_panel.csv.")
    st.stop()

# ensure key numerics
df = safe_numeric(df, ['HDI_2023','HDI_sq','GDP_per_capita','log_GDP_per_capita','Suicide_rate','coverage_frac'])

# optional compute log GDP if missing
if 'log_GDP_per_capita' not in df.columns and 'GDP_per_capita' in df.columns:
    df['log_GDP_per_capita'] = np.log(df['GDP_per_capita'].where(df['GDP_per_capita']>0, np.nan))

# sidebar filters
continents = sorted(df['continent'].dropna().unique())
income_groups = sorted(df['income_group_auto'].dropna().unique())
data_quality_opts = sorted(df['Low_data_quality_flag'].dropna().unique())

sel_continent = st.sidebar.multiselect("Filter: Continent", options=continents, default=continents)
sel_income = st.sidebar.multiselect("Filter: Income group", options=income_groups, default=income_groups)
sel_quality = st.sidebar.multiselect("Filter: Data quality", options=data_quality_opts, default=data_quality_opts)

# apply filters
mask = df['continent'].isin(sel_continent) if 'continent' in df.columns else pd.Series(True, index=df.index)
mask &= df['income_group_auto'].isin(sel_income) if 'income_group_auto' in df.columns else mask
mask &= df['Low_data_quality_flag'].isin(sel_quality) if 'Low_data_quality_flag' in df.columns else mask
df_f = df[mask].copy()

st.sidebar.markdown(f"**Observations:** {len(df_f)} countries selected")

# -----------------------
# Layout: Header & Key metrics
# -----------------------
st.title("The Price of Progress — interactive exploration")
st.markdown("Exploring relationships between development (HDI, GDP) and suicide rates across countries. \
            Use the sidebar to filter continents, income groups, or data quality flags.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Countries (selected)", f"{df_f['Country Name'].nunique()}")
col2.metric("Avg HDI", f"{df_f['HDI_2023'].mean():.3f}")
col3.metric("Avg Suicide rate", f"{df_f['Suicide_rate'].mean():.2f} per 100k")
col4.metric("Avg GDP per capita", f"${df_f['GDP_per_capita'].median():.0f}")

# -----------------------
# Section: Correlation & Heatmap
# -----------------------
st.header("Correlation & Distributions")
num_cols = ['HDI_2023','GDP_per_capita','Suicide_rate','log_GDP_per_capita']
corr = df_f[num_cols].corr()

st.subheader("Correlation matrix")
fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
fig_corr.update_layout(height=420, margin=dict(l=40,r=40,t=40,b=40))
st.plotly_chart(fig_corr, use_container_width=True)
# download corr png
buf = fig_to_bytes(fig_corr)
st.download_button("Download correlation (PNG)", data=buf, file_name="corr_heatmap.png", mime="image/png")

# -----------------------
# Section: Scatter HDI vs Suicide
# -----------------------
st.header("HDI vs Suicide Rate")
left, right = st.columns([3,1])
with left:
    fig = px.scatter(df_f, x='HDI_2023', y='Suicide_rate', color='continent' if 'continent' in df_f.columns else None,
                     hover_name='Country Name', size_max=12, hover_data=['GDP_per_capita','income_group_auto'])
    # add lowess-like trend by fitting quadratic on selected sample
    model, model_df = fit_quadratic(df_f)
    if model is not None:
        # predicted curve
        hdix = np.linspace(df_f['HDI_2023'].min(), df_f['HDI_2023'].max(), 200)
        params = model.params
        yhat = params.get('Intercept',0) + params.get('HDI_2023',0)*hdix + params.get('HDI_sq',0)*(hdix**2) + params.get('log_GDP_per_capita',0)*(model_df['log_GDP_per_capita'].mean() if 'log_GDP_per_capita' in model_df else 0)
        fig.add_traces(px.line(x=hdix, y=yhat, labels={'x':'HDI','y':'pred'}).data)
        st.plotly_chart(fig, use_container_width=True)
        # show sample pearson
        try:
            m = model_df.dropna(subset=['HDI_2023','Suicide_rate'])
            r, p = pearsonr(m['HDI_2023'], m['Suicide_rate'])
            st.markdown(f"**Pearson correlation (HDI vs Suicide):** r = {r:.3f}, p = {p:.4f} (n={len(m)})")
        except Exception:
            pass
    else:
        st.plotly_chart(fig, use_container_width=True)
with right:
    st.markdown("**Model & Tipping point**")
    if model is not None:
        st.text(model.summary().as_text())
        tp = tipping_point_from_model(model)
        if tp is not None:
            st.success(f"Estimated tipping point (HDI*): {tp:.3f}")
        else:
            st.info("Tipping point not computable.")
        # allow model report download
        model_report = model.summary().as_text()
        st.download_button("Download model summary (.txt)", model_report, file_name="model_summary.txt", mime="text/plain")
    else:
        st.info("Not enough data to fit quadratic model on selected subset.")

# -----------------------
# Section: GDP vs Suicide
# -----------------------
st.header("GDP (log) vs Suicide Rate")
fig2 = px.scatter(df_f, x='log_GDP_per_capita', y='Suicide_rate', color='continent' if 'continent' in df_f.columns else None,
                  hover_name='Country Name', hover_data=['HDI_2023','income_group_auto'])
fig2.update_layout(height=500)
st.plotly_chart(fig2, use_container_width=True)
st.download_button("Download GDP vs Suicide (PNG)", data=fig_to_bytes(fig2), file_name="gdp_vs_suicide.png", mime="image/png")

# -----------------------
# Boxplot by income group
# -----------------------
st.header("Suicide Rate by Income Group")
if 'income_group_auto' in df_f.columns:
    fig_box = px.box(df_f, x='income_group_auto', y='Suicide_rate', points="all",
                     category_orders={"income_group_auto": ['Low','Lower-Middle','Upper-Middle','High']})
    st.plotly_chart(fig_box, use_container_width=True)
    st.download_button("Download boxplot (PNG)", data=fig_to_bytes(fig_box), file_name="boxplot_income.png", mime="image/png")
else:
    st.info("No 'income_group_auto' column available.")

# -----------------------
# Choropleth map
# -----------------------
st.header("Global Map: Suicide Rate")
if 'ISO3' in df_f.columns:
    fig_map = px.choropleth(df_f, locations="ISO3", color="Suicide_rate", hover_name="Country Name",
                            hover_data=['HDI_2023','GDP_per_capita','income_group_auto'],
                            color_continuous_scale="Reds")
    fig_map.update_layout(height=550)
    st.plotly_chart(fig_map, use_container_width=True)
    # save interactive
    html = fig_map.to_html(full_html=False, include_plotlyjs='cdn')
    st.download_button("Download interactive map (HTML)", html, file_name="choropleth_suicide.html", mime="text/html")
else:
    st.info("Missing ISO3 codes for mapping. Add 'ISO3' column.")

# -----------------------
# Residuals and diagnostics
# -----------------------
st.header("Diagnostics & Robustness")
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("VIF & Heteroskedasticity")
    try:
        # compute VIF for HDI, HDI_sq, log GDP
        X = df_f[['HDI_2023','HDI_sq','log_GDP_per_capita']].dropna()
        Xc = sm.add_constant(X)
        vif = pd.DataFrame({
            "variable": Xc.columns,
            "VIF": [sm.stats.outliers_influence.variance_inflation_factor(Xc.values, i) if Xc.shape[0]>0 else np.nan for i in range(Xc.shape[1])]
        })
        st.dataframe(vif.style.format({"VIF":"{:.2f}"}))
    except Exception as e:
        st.write("VIF computation error:", e)

with col_b:
    st.subheader("Top Influential Countries (Cook's D)")
    model2, model_df2 = fit_quadratic(df_f)
    if model2 is not None:
        infl = model2.get_influence()
        cooks = infl.cooks_distance[0]
        tmp = model_df2[['Country Name']].copy()
        tmp['cooks_d'] = cooks
        top = tmp.sort_values('cooks_d', ascending=False).head(10)
        st.table(top)
    else:
        st.info("No model to compute diagnostics.")

# -----------------------
# Download cleaned CSV & quick report
# -----------------------
st.header("Export & Report")
csv_bytes = df_f.to_csv(index=False).encode('utf-8')
st.download_button("Download filtered CSV", csv_bytes, file_name="filtered_merged_clean_panel.csv", mime="text/csv")

# Quick auto-report generation (TXT)
report_lines = []
report_lines.append("Automated EDA + Model summary")
report_lines.append(f"Selection: continents={sel_continent}, income={sel_income}, quality={sel_quality}")
report_lines.append(f"Observations: {len(df_f)}")
report_lines.append("")
report_lines.append("Descriptive stats (HDI, GDP, Suicide):")
report_lines.append(df_f[['HDI_2023','GDP_per_capita','Suicide_rate']].describe().to_string())
if model is not None:
    report_lines.append("\nQuadratic model coefficients (robust SE):")
    report_lines.append(model.summary().tables[1].as_text())
if tipping_point_from_model(model):
    report_lines.append(f"\nEstimated tipping point (HDI*): {tipping_point_from_model(model):.3f}")
report_txt = "\n".join(report_lines)
st.download_button("Download quick report (.txt)", report_txt, file_name="quick_report.txt", mime="text/plain")

# -----------------------
# Footer / Notes
# -----------------------
st.write("---")
st.markdown("""
**Notes & ethics:**  
- This dashboard uses cross-sectional associations — *correlation is not causation*.  
- Data quality varies across countries; use the Low data quality flag to limit analyses.  
- Consider reporting biases and cultural differences in suicide reporting.
""")
