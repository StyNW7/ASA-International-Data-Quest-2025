# app_enhanced.py
# Streamlit dashboard: "The Price of Progress - Enhanced Edition"
# Comprehensive analysis with time series data and predictive modeling

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf
import statsmodels.api as sm
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="The Price of Progress - Enhanced", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .section-header {
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #1f77b4;
        font-size: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: black;
    }
    .prediction-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .info-bubble {
        background-color: #e7f3ff;
        border: 1px solid #b3d9ff;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: black;
    }
    .download-btn {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

sns.set_style("whitegrid")

# -----------------------
# Enhanced Helpers
# -----------------------
@st.cache_data
def load_data(path="./Final/final_clean_dataset.csv"):
    """Load data with enhanced error handling"""
    if path:
        try:
            df = pd.read_csv(path)
            st.sidebar.success(f"✅ Data loaded successfully: {len(df)} rows")
            return df
        except Exception as e:
            st.sidebar.error(f"❌ Error loading uploaded file: {e}")
            return None
    else:
        # Try multiple possible file locations
        possible_paths = [
            "./Final/final_clean_dataset.csv"
        ]
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                st.sidebar.success(f"✅ Data loaded from {path}: {len(df)} rows")
                return df
            except Exception:
                continue
        st.sidebar.error("❌ No dataset found. Please upload your dataset.")
        return None

def safe_numeric(df, cols):
    """Convert columns to numeric with better error reporting"""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def fit_quadratic(df, year=None, robust=True):
    """Fit quadratic model for HDI vs Suicide Rate"""
    if year:
        df_subset = df[df['Year'] == year].copy()
    else:
        df_subset = df.copy()
    
    # Use HDI_2023 if available, otherwise use HDI
    hdi_col = 'HDI_2023' if 'HDI_2023' in df_subset.columns else 'HDI'
    suicide_col = 'Suicide_rate'
    
    required_cols = [suicide_col, hdi_col, 'HDI_sq']
    missing_cols = [col for col in required_cols if col not in df_subset.columns]
    
    if missing_cols:
        return None, df_subset
    
    dfm = df_subset.dropna(subset=required_cols)
    
    if len(dfm) < 10:
        return None, dfm
    
    # Ensure log GDP exists
    if 'log_GDP_per_capita' not in dfm.columns and 'GDP_per_capita' in dfm.columns:
        dfm['log_GDP_per_capita'] = np.log(dfm['GDP_per_capita'].where(dfm['GDP_per_capita'] > 1, np.nan))
    
    formula = f"{suicide_col} ~ {hdi_col} + HDI_sq + log_GDP_per_capita"
    
    try:
        model = smf.ols(formula=formula, data=dfm).fit(cov_type='HC1' if robust else None)
        return model, dfm
    except Exception as e:
        return None, dfm

def tipping_point_from_model(model):
    """Calculate tipping point with validation"""
    try:
        p = model.params
        # Check for both possible HDI column names
        hdi_coeff = None
        if 'HDI_2023' in p:
            hdi_coeff = 'HDI_2023'
        elif 'HDI' in p:
            hdi_coeff = 'HDI'
        
        if hdi_coeff and 'HDI_sq' in p and p['HDI_sq'] != 0:
            tp = -p[hdi_coeff]/(2*p['HDI_sq'])
            if 0.3 <= tp <= 1.0:
                return float(tp)
        return None
    except Exception:
        return None

def fig_to_bytes(fig):
    """Save matplotlib or plotly figure to PNG bytes for download."""
    try:
        if hasattr(fig, "to_image"):  # plotly
            return fig.to_image(format="png", width=1000, height=600, scale=2)
        else:
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            return buf.read()
    except Exception as e:
        st.error(f"Error converting figure to bytes: {e}")
        return None

def create_summary_statistics(df_f):
    """Create comprehensive summary statistics"""
    # Use available columns
    stats_data = {'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max']}
    
    if 'HDI' in df_f.columns or 'HDI_2023' in df_f.columns:
        hdi_col = 'HDI_2023' if 'HDI_2023' in df_f.columns else 'HDI'
        stats_data['HDI'] = df_f[hdi_col].describe().round(3).tolist()
    
    if 'Suicide_rate' in df_f.columns:
        stats_data['Suicide Rate'] = df_f['Suicide_rate'].describe().round(2).tolist()
    
    if 'GDP_per_capita' in df_f.columns:
        stats_data['GDP per capita'] = df_f['GDP_per_capita'].describe().round(0).tolist()
    
    return pd.DataFrame(stats_data)

# -----------------------
# Sidebar: Data / Filters
# -----------------------
st.sidebar.title("🔧 Settings & Data")
st.sidebar.markdown("---")

# Data upload
uploaded = st.sidebar.file_uploader(
    "Upload Enhanced Dataset", 
    type=["csv"],
    help="Upload your comprehensive dataset or use the default one"
)

df = load_data(uploaded)

if df is None:
    st.sidebar.error("""
    No dataset found. Please upload your enhanced dataset containing:
    - Country Name, ISO3, Year
    - HDI, Suicide_rate, GDP_per_capita
    - Additional development indicators
    """)
    st.stop()

# Ensure key numeric columns
numeric_cols = ['HDI', 'HDI_2023', 'HDI_sq', 'GDP_per_capita', 'log_GDP_per_capita', 
               'Suicide_rate', 'HDI_growth', 'Suicide_change', 'HDI_lag1', 'Suicide_rate_lag1']
df = safe_numeric(df, [col for col in numeric_cols if col in df.columns])

# Compute derived columns if missing
if 'log_GDP_per_capita' not in df.columns and 'GDP_per_capita' in df.columns:
    df['log_GDP_per_capita'] = np.log(df['GDP_per_capita'].where(df['GDP_per_capita'] > 0, np.nan))

if 'HDI_sq' not in df.columns:
    hdi_col = 'HDI_2023' if 'HDI_2023' in df.columns else 'HDI'
    if hdi_col in df.columns:
        df['HDI_sq'] = df[hdi_col] ** 2

# Sidebar filters
st.sidebar.subheader("🎯 Data Filters")

# Year range filter (if Year column exists)
if 'Year' in df.columns:
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
else:
    year_range = (2019, 2023)
    # If no Year column, assume it's cross-sectional data
    df['Year'] = 2023  # Default year for compatibility

# Available filters
continents = sorted(df['continent'].dropna().unique()) if 'continent' in df.columns else []
income_groups = sorted(df['income_group_auto'].dropna().unique()) if 'income_group_auto' in df.columns else []
data_quality_opts = sorted(df['Low_data_quality_flag'].dropna().unique()) if 'Low_data_quality_flag' in df.columns else []

# Filter widgets
if continents:
    sel_continent = st.sidebar.multiselect(
        "Continent", 
        options=continents, 
        default=continents
    )
else:
    sel_continent = []

if income_groups:
    sel_income = st.sidebar.multiselect(
        "Income Group", 
        options=income_groups, 
        default=income_groups
    )
else:
    sel_income = []

# HDI range filter
hdi_col = 'HDI_2023' if 'HDI_2023' in df.columns else 'HDI'
if hdi_col in df.columns:
    min_hdi, max_hdi = st.sidebar.slider(
        "HDI Range",
        min_value=float(df[hdi_col].min()),
        max_value=float(df[hdi_col].max()),
        value=(float(df[hdi_col].min()), float(df[hdi_col].max())),
        step=0.01
    )

# Suicide rate range filter
if 'Suicide_rate' in df.columns:
    min_suicide, max_suicide = st.sidebar.slider(
        "Suicide Rate Range",
        min_value=float(df['Suicide_rate'].min()),
        max_value=float(df['Suicide_rate'].max()),
        value=(float(df['Suicide_rate'].min()), float(df['Suicide_rate'].max())),
        step=0.1
    )

# Apply filters
mask = pd.Series(True, index=df.index)

if 'Year' in df.columns:
    mask &= (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])
if continents:
    mask &= df['continent'].isin(sel_continent)
if income_groups:
    mask &= df['income_group_auto'].isin(sel_income)
if hdi_col in df.columns:
    mask &= (df[hdi_col] >= min_hdi) & (df[hdi_col] <= max_hdi)
if 'Suicide_rate' in df.columns:
    mask &= (df['Suicide_rate'] >= min_suicide) & (df['Suicide_rate'] <= max_suicide)

df_f = df[mask].copy()

# Display filter summary
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Filter Summary")
st.sidebar.markdown(f"**Countries:** {df_f['Country Name'].nunique()}")
st.sidebar.markdown(f"**Observations:** {len(df_f)}")
st.sidebar.markdown(f"**Years:** {year_range[0]} - {year_range[1]}")
if hdi_col in df_f.columns:
    st.sidebar.markdown(f"**HDI Range:** {df_f[hdi_col].min():.3f} - {df_f[hdi_col].max():.3f}")

# -----------------------
# Main Content
# -----------------------
st.markdown('<h1 class="main-header">THE PRICE OF PROGRESS</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>
    Advanced Analysis of Human Development and Mental Health Outcomes<br>
    <em>Exploring the complex relationship between development progress and suicide rates across nations and time</em>
</div>
""", unsafe_allow_html=True)

# Key metrics with enhanced styling
st.markdown("### 🌍 Global Overview Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Countries", f"{df_f['Country Name'].nunique()}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    hdi_col = 'HDI_2023' if 'HDI_2023' in df_f.columns else 'HDI'
    avg_hdi = df_f[hdi_col].mean() if hdi_col in df_f.columns else 0
    st.metric("Average HDI", f"{avg_hdi:.3f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    suicide_rate = df_f['Suicide_rate'].mean() if 'Suicide_rate' in df_f.columns else 0
    st.metric("Avg Suicide Rate", f"{suicide_rate:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    median_gdp = df_f['GDP_per_capita'].median() if 'GDP_per_capita' in df_f.columns else 0
    st.metric("Median GDP", f"${median_gdp:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    years_span = f"{year_range[0]}-{year_range[1]}"
    st.metric("Time Span", years_span)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# Section: Data Overview & Summary Statistics
# -----------------------
st.markdown("---")
st.markdown('<h2 class="section-header">📋 Data Overview & Summary Statistics</h2>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Summary Statistics", "Data Sample", "Missing Data", "Data Quality"])

with tab1:
    st.subheader("Descriptive Statistics")
    
    # Create summary statistics
    stats_df = create_summary_statistics(df_f)
    st.dataframe(stats_df, use_container_width=True)
    
    # Additional insights in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hdi_col = 'HDI_2023' if 'HDI_2023' in df_f.columns else 'HDI'
        if hdi_col in df_f.columns and 'Suicide_rate' in df_f.columns:
            hdi_corr = df_f[[hdi_col, 'Suicide_rate']].corr().iloc[0,1]
            st.metric("HDI-Suicide Correlation", f"{hdi_corr:.3f}")
    
    with col2:
        if 'GDP_per_capita' in df_f.columns and 'Suicide_rate' in df_f.columns:
            gdp_corr = df_f[['GDP_per_capita', 'Suicide_rate']].corr().iloc[0,1]
            st.metric("GDP-Suicide Correlation", f"{gdp_corr:.3f}")
    
    with col3:
        complete_cols = [hdi_col, 'Suicide_rate', 'GDP_per_capita'] if hdi_col in df_f.columns else ['Suicide_rate', 'GDP_per_capita']
        complete_cases = df_f[complete_cols].notna().all(axis=1).sum()
        st.metric("Complete Cases", f"{complete_cases}")
    
    # Download button for summary statistics
    if st.button("📥 Download Summary Statistics", key="download_summary"):
        csv_stats = stats_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            data=csv_stats,
            file_name="summary_statistics.csv",
            mime="text/csv"
        )

with tab2:
    st.subheader("Data Sample")
    st.dataframe(df_f.head(10), use_container_width=True)
    st.info(f"Showing 10 of {len(df_f)} rows. Dataset has {len(df_f.columns)} columns.")
    
    # Show column information
    with st.expander("📊 Column Information"):
        col_info = pd.DataFrame({
            'Column': df_f.columns,
            'Data Type': df_f.dtypes.values,
            'Non-Null Count': df_f.notna().sum().values,
            'Null Count': df_f.isna().sum().values
        })
        st.dataframe(col_info, use_container_width=True)

with tab3:
    st.subheader("Missing Data Analysis")
    missing_data = df_f.isnull().sum()
    missing_pct = (missing_data / len(df_f)) * 100
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing %': missing_pct.values
    }).sort_values('Missing Count', ascending=False)
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    if len(missing_df) > 0:
        st.dataframe(missing_df.style.format({"Missing %": "{:.1f}%"}), use_container_width=True)
        
        # Missing data visualization
        fig_missing = px.bar(missing_df.head(20), x='Missing %', y='Column', 
                            title='Missing Data by Column (Top 20)',
                            color='Missing %', color_continuous_scale='Reds',
                            orientation='h')
        fig_missing.update_layout(height=500)
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("🎉 No missing data in the filtered dataset!")

with tab4:
    st.subheader("Data Quality Assessment")
    
    # Data quality indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_countries = df_f['Country Name'].nunique()
        st.metric("Total Countries", total_countries)
    
    with col2:
        if 'Low_data_quality_flag' in df_f.columns:
            low_quality = df_f['Low_data_quality_flag'].str.contains('⚠️|Low', na=False).sum()
            st.metric("Low Quality Data", low_quality)
        else:
            st.metric("Quality Flags", "Not Available")
    
    with col3:
        if 'Year' in df_f.columns:
            year_coverage = df_f['Year'].max() - df_f['Year'].min() + 1
            st.metric("Years Coverage", year_coverage)
    
    # Data quality visualization
    if 'Low_data_quality_flag' in df_f.columns:
        quality_counts = df_f['Low_data_quality_flag'].value_counts()
        fig_quality = px.pie(values=quality_counts.values, names=quality_counts.index,
                            title="Data Quality Distribution")
        st.plotly_chart(fig_quality, use_container_width=True)

# -----------------------
# Section: Enhanced HDI vs Suicide Rate Analysis
# -----------------------
st.markdown("---")
st.markdown('<h2 class="section-header">📊 HDI vs Suicide Rate Analysis</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="info-bubble">
💡 <strong>Understanding the Relationship:</strong> This section explores the complex relationship between 
Human Development Index (HDI) and suicide rates. We analyze whether higher development comes with 
mental health costs, and identify potential tipping points where development might start benefiting mental health.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    # Enhanced scatter plot with multiple customization options
    st.subheader("Interactive Scatter Analysis")
    
    # Customization options
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        size_by = st.selectbox("Size by", 
                              ["None", "GDP_per_capita", "Population"] if "Population" in df_f.columns else ["None", "GDP_per_capita"],
                              index=1 if "GDP_per_capita" in df_f.columns else 0)
    
    with col_b:
        color_by = st.selectbox("Color by", 
                               ["continent", "income_group_auto", "HDI_Tier"] if all(x in df_f.columns for x in ["continent", "income_group_auto"]) else 
                               ["continent"] if "continent" in df_f.columns else ["None"])
    
    with col_c:
        trendline_type = st.selectbox("Trendline", ["None", "Linear", "Quadratic", "Lowess"])
    
    # Create the enhanced scatter plot
    hdi_col = 'HDI_2023' if 'HDI_2023' in df_f.columns else 'HDI'
    
    if hdi_col in df_f.columns and 'Suicide_rate' in df_f.columns:
        fig_scatter = px.scatter(
            df_f, 
            x=hdi_col, 
            y='Suicide_rate',
            color=color_by if color_by != "None" else None,
            size=size_by if size_by != "None" else None,
            hover_name='Country Name',
            hover_data=['GDP_per_capita', 'income_group_auto'] if 'income_group_auto' in df_f.columns else ['GDP_per_capita'],
            title="HDI vs Suicide Rate: The Development-Mental Health Relationship",
            size_max=20,
            opacity=0.7
        )
        
        # Add trendline if selected
        if trendline_type != "None":
            if trendline_type == "Linear":
                fig_scatter.update_traces(
                    selector=dict(mode='markers'),
                    line=dict(dash='dash', color='red', width=2)
                )
            elif trendline_type == "Lowess":
                # Add LOWESS trendline
                fig_scatter.update_traces(
                    line_shape='spline'
                )
        
        # Fit and add quadratic trend for model analysis
        model, model_df = fit_quadratic(df_f)
        if model is not None and trendline_type == "Quadratic":
            hdi_min = df_f[hdi_col].min()
            hdi_max = df_f[hdi_col].max()
            hdi_range = np.linspace(hdi_min, hdi_max, 100)
            
            params = model.params
            gdp_mean = model_df['log_GDP_per_capita'].mean() if 'log_GDP_per_capita' in model_df else 0
            
            # Use correct HDI coefficient name
            hdi_coeff = 'HDI_2023' if 'HDI_2023' in params else 'HDI'
            
            y_pred = (params.get('Intercept', 0) + 
                     params.get(hdi_coeff, 0) * hdi_range + 
                     params.get('HDI_sq', 0) * (hdi_range ** 2) + 
                     params.get('log_GDP_per_capita', 0) * gdp_mean)
            
            fig_scatter.add_trace(
                go.Scatter(
                    x=hdi_range, 
                    y=y_pred, 
                    mode='lines', 
                    name='Quadratic Trend',
                    line=dict(color='red', width=3, dash='dash')
                )
            )
        
        fig_scatter.update_layout(height=600)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Download button for the scatter plot
        if st.button("📥 Download Scatter Plot", key="download_scatter"):
            buf = fig_to_bytes(fig_scatter)
            if buf:
                st.download_button(
                    "Download PNG", 
                    data=buf, 
                    file_name="hdi_vs_suicide_scatter.png", 
                    mime="image/png"
                )
        
        # Correlation metrics
        clean_data = df_f[[hdi_col, 'Suicide_rate']].dropna()
        if len(clean_data) > 2:
            pearson_r, pearson_p = pearsonr(clean_data[hdi_col], clean_data['Suicide_rate'])
            spearman_r, spearman_p = spearmanr(clean_data[hdi_col], clean_data['Suicide_rate'])
            
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.metric("Pearson Correlation", f"{pearson_r:.3f}", 
                         delta=f"p-value: {pearson_p:.4f}")
            with col_c2:
                st.metric("Spearman Correlation", f"{spearman_r:.3f}", 
                         delta=f"p-value: {spearman_p:.4f}")

with col2:
    st.subheader("Model Analysis")
    
    if model is not None:
        # Model summary
        st.success("✅ Quadratic Model Fitted")
        
        # Key coefficients
        hdi_coeff = 'HDI_2023' if 'HDI_2023' in model.params else 'HDI'
        st.metric("HDI Coefficient", f"{model.params.get(hdi_coeff, 0):.3f}")
        st.metric("HDI² Coefficient", f"{model.params.get('HDI_sq', 0):.3f}")
        st.metric("R-squared", f"{model.rsquared:.3f}")
        
        # Tipping point
        tp = tipping_point_from_model(model)
        if tp:
            st.success(f"🎯 Tipping Point: HDI = {tp:.3f}")
            st.info("""
            **Interpretation:** 
            - Below HDI {tp:.3f}: Development may correlate with increasing suicide rates
            - Above HDI {tp:.3f}: Further development may correlate with decreasing suicide rates
            """.format(tp=tp))
        
        # Model diagnostics
        st.subheader("Model Diagnostics")
        st.metric("Observations", f"{model.nobs}")
        st.metric("F-statistic", f"{model.fvalue:.2f}")
        st.metric("Prob (F-statistic)", f"{model.f_pvalue:.4f}")
        
        # Download model summary
        if st.button("📥 Download Model Summary", key="download_model"):
            model_report = model.summary().as_text()
            st.download_button(
                "Download .txt", 
                data=model_report, 
                file_name="quadratic_model_summary.txt", 
                mime="text/plain"
            )
    else:
        st.error("❌ Model could not be fitted")
        st.info("""
        This could be due to:
        - Insufficient data points
        - Missing required variables
        - High multicollinearity
        - Numerical instability
        """)

# Additional HDI analysis
st.markdown("---")
st.markdown("#### 🔍 Detailed HDI Analysis")

col1, col2 = st.columns(2)

with col1:
    # HDI distribution by continent
    if 'continent' in df_f.columns and hdi_col in df_f.columns:
        fig_hdi_box = px.box(df_f, x='continent', y=hdi_col, 
                            title='HDI Distribution by Continent',
                            color='continent')
        st.plotly_chart(fig_hdi_box, use_container_width=True)

with col2:
    # Suicide rate distribution by continent
    if 'continent' in df_f.columns and 'Suicide_rate' in df_f.columns:
        fig_suicide_box = px.box(df_f, x='continent', y='Suicide_rate',
                                title='Suicide Rate Distribution by Continent',
                                color='continent')
        st.plotly_chart(fig_suicide_box, use_container_width=True)

# -----------------------
# Section: Economic Analysis
# -----------------------
st.markdown("---")
st.markdown('<h2 class="section-header">💼 Economic Analysis</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="info-bubble">
💡 <strong>Economic Dimensions:</strong> This section explores how economic factors like GDP, income levels, 
and economic development trajectories relate to suicide rates across different countries and regions.
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["GDP vs Suicide Rate", "Income Group Analysis", "Economic Development", "Wealth vs Well-being"])

with tab1:
    st.subheader("GDP per Capita vs Suicide Rate")
    
    # Choose scale: linear or log
    scale_option = st.radio("GDP Scale:", ["Log Scale", "Linear Scale"], horizontal=True, key="gdp_scale")
    
    if scale_option == "Log Scale":
        x_col = 'log_GDP_per_capita'
        x_title = "Log GDP per Capita"
    else:
        x_col = 'GDP_per_capita'
        x_title = "GDP per Capita (USD)"
    
    if x_col in df_f.columns and 'Suicide_rate' in df_f.columns:
        fig_gdp = px.scatter(
            df_f, 
            x=x_col, 
            y='Suicide_rate',
            color='continent' if 'continent' in df_f.columns else None,
            hover_name='Country Name',
            hover_data=['HDI_2023' if 'HDI_2023' in df_f.columns else 'HDI', 'income_group_auto'] if 'income_group_auto' in df_f.columns else ['HDI_2023' if 'HDI_2023' in df_f.columns else 'HDI'],
            title=f"Suicide Rate vs {x_title}",
            trendline="lowess",
            trendline_color_override="red"
        )
        fig_gdp.update_layout(height=500)
        st.plotly_chart(fig_gdp, use_container_width=True)
        
        # Download button
        if st.button("📥 Download GDP Plot", key="download_gdp"):
            buf = fig_to_bytes(fig_gdp)
            if buf:
                st.download_button(
                    "Download PNG", 
                    data=buf, 
                    file_name="gdp_vs_suicide.png", 
                    mime="image/png"
                )
        
        # GDP correlation
        gdp_corr_data = df_f[[x_col, 'Suicide_rate']].dropna()
        if len(gdp_corr_data) > 2:
            gdp_r, gdp_p = pearsonr(gdp_corr_data[x_col], gdp_corr_data['Suicide_rate'])
            st.metric(f"Correlation with {x_title}", f"{gdp_r:.3f}", delta=f"p-value: {gdp_p:.4f}")

with tab2:
    st.subheader("Suicide Rate by Income Group")
    
    if 'income_group_auto' in df_f.columns:
        # Box plot
        income_order = ['Low', 'Lower-Middle', 'Upper-Middle', 'High']
        available_groups = [group for group in income_order if group in df_f['income_group_auto'].unique()]
        
        fig_income = px.box(
            df_f, 
            x='income_group_auto', 
            y='Suicide_rate',
            category_orders={"income_group_auto": available_groups},
            points="all",
            title="Suicide Rate Distribution by Income Group",
            color='income_group_auto'
        )
        fig_income.update_layout(height=500)
        st.plotly_chart(fig_income, use_container_width=True)
        
        # Download button
        if st.button("📥 Download Income Group Plot", key="download_income"):
            buf = fig_to_bytes(fig_income)
            if buf:
                st.download_button(
                    "Download PNG", 
                    data=buf, 
                    file_name="income_group_analysis.png", 
                    mime="image/png"
                )
        
        # Summary statistics by income group
        income_stats = df_f.groupby('income_group_auto').agg({
            'Suicide_rate': ['mean', 'median', 'std', 'count'],
            'HDI_2023' if 'HDI_2023' in df_f.columns else 'HDI': 'mean',
            'GDP_per_capita': 'median'
        }).round(3)
        
        st.subheader("Summary by Income Group")
        st.dataframe(income_stats, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>📈 Insight:</strong> Middle-income countries often show the highest suicide rates, 
        suggesting that the transition phase in development may be particularly challenging for mental health.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Income group data not available in the dataset")

with tab3:
    st.subheader("Economic Development Patterns")
    
    # Scatter plot with both HDI and GDP
    hdi_col = 'HDI_2023' if 'HDI_2023' in df_f.columns else 'HDI'
    if all(col in df_f.columns for col in [hdi_col, 'GDP_per_capita', 'Suicide_rate']):
        fig_development = px.scatter(
            df_f,
            x=hdi_col,
            y='GDP_per_capita',
            size='Suicide_rate',
            color='Suicide_rate',
            hover_name='Country Name',
            title="Development Landscape: HDI vs GDP with Suicide Rate Indicators",
            color_continuous_scale='Viridis',
            size_max=20
        )
        fig_development.update_layout(height=500)
        st.plotly_chart(fig_development, use_container_width=True)
        
        # Download button
        if st.button("📥 Download Development Plot", key="download_development"):
            buf = fig_to_bytes(fig_development)
            if buf:
                st.download_button(
                    "Download PNG", 
                    data=buf, 
                    file_name="development_landscape.png", 
                    mime="image/png"
                )
        
        st.info("""
        **Interpretation:**
        - Bubble size represents suicide rate magnitude
        - Color intensity shows suicide rate values
        - Upper-right quadrant: High HDI & High GDP (Developed nations)
        - Lower-left quadrant: Low HDI & Low GDP (Developing nations)
        - Note clusters of countries with similar development-mental health profiles
        """)

with tab4:
    st.subheader("Wealth vs Well-being Analysis")
    
    # Analyze the relationship between economic wealth and mental health
    if all(col in df_f.columns for col in ['GDP_per_capita', 'Suicide_rate', hdi_col]):
        # Create wealth categories
        df_f['Wealth_Category'] = pd.cut(df_f['GDP_per_capita'], 
                                       bins=[0, 5000, 15000, 30000, float('inf')],
                                       labels=['Low', 'Lower-Middle', 'Upper-Middle', 'High'])
        
        wealth_analysis = df_f.groupby('Wealth_Category').agg({
            'Suicide_rate': ['mean', 'median', 'std'],
            hdi_col: 'mean',
            'GDP_per_capita': 'median'
        }).round(3)
        
        st.dataframe(wealth_analysis, use_container_width=True)
        
        # Visualization
        fig_wealth = px.scatter(
            df_f,
            x='GDP_per_capita',
            y='Suicide_rate',
            color='Wealth_Category',
            hover_name='Country Name',
            title="Wealth vs Mental Health: GDP per Capita vs Suicide Rate",
            log_x=True
        )
        st.plotly_chart(fig_wealth, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>💡 Key Finding:</strong> The relationship between wealth and mental health is not linear. 
        While extreme poverty is associated with various challenges, very high wealth levels don't 
        necessarily guarantee better mental health outcomes, highlighting the importance of 
        non-economic factors in well-being.
        </div>
        """, unsafe_allow_html=True)

# -----------------------
# Section: Advanced Diagnostics (Keep from previous)
# -----------------------
st.markdown("---")
st.markdown('<h2 class="section-header">🔬 Advanced Diagnostics</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Multicollinearity Check (VIF)")
    
    try:
        # Compute VIF for key variables
        vif_vars = [hdi_col, 'HDI_sq', 'log_GDP_per_capita']
        available_vars = [var for var in vif_vars if var in df_f.columns]
        
        if len(available_vars) >= 2:
            X = df_f[available_vars].dropna()
            if len(X) > 0:
                Xc = sm.add_constant(X)
                vif_vals = []
                
                for i in range(Xc.shape[1]):
                    try:
                        vif_i = variance_inflation_factor(Xc.values, i)
                        vif_vals.append(vif_i)
                    except Exception as e:
                        st.warning(f"Could not compute VIF for variable {i}: {e}")
                        vif_vals.append(np.nan)
                
                vif_df = pd.DataFrame({
                    "Variable": Xc.columns,
                    "VIF": vif_vals
                })
                
                # Remove constant for display
                vif_display = vif_df[vif_df['Variable'] != 'const'].copy()
                vif_display['Status'] = vif_display['VIF'].apply(
                    lambda x: '✅ Low' if x < 5 else '⚠️ Moderate' if x < 10 else '❌ High'
                )
                
                st.dataframe(vif_display.style.format({"VIF": "{:.2f}"}))
                
                # Interpretation
                st.info("""
                **VIF Interpretation:**
                - ✅ VIF < 5: Low multicollinearity
                - ⚠️ 5 ≤ VIF < 10: Moderate multicollinearity  
                - ❌ VIF ≥ 10: High multicollinearity (concern)
                """)
            else:
                st.warning("Insufficient data for VIF calculation")
        else:
            st.warning("Need at least 2 variables for VIF calculation")
    
    except Exception as e:
        st.error(f"VIF computation failed: {e}")

with col2:
    st.subheader("Model Diagnostics")
    
    if model is not None:
        # Influential points (Cook's D)
        try:
            infl = model.get_influence()
            cooks_d = infl.cooks_distance[0]
            
            # Top influential countries
            if 'Country Name' in model_df.columns:
                influence_df = model_df[['Country Name']].copy()
                influence_df['Cooks_D'] = cooks_d
                top_influential = influence_df.nlargest(8, 'Cooks_D')
                
                st.write("**Most Influential Countries (Cook's D):**")
                st.dataframe(top_influential.style.format({"Cooks_D": "{:.4f}"}))
            
            # Heteroskedasticity test
            try:
                _, bp_pval, _, _ = sm.stats.diagnostic.het_breuschpagan(model.resid, model.model.exog)
                st.metric("Breusch-Pagan Test (p-value)", f"{bp_pval:.4f}")
                
                if bp_pval < 0.05:
                    st.warning("Evidence of heteroskedasticity (p < 0.05)")
                else:
                    st.success("No significant heteroskedasticity (p ≥ 0.05)")
            except Exception as e:
                st.warning(f"Could not compute heteroskedasticity test: {e}")
            
            # Residuals plot
            fig_resid = px.scatter(
                x=model.fittedvalues,
                y=model.resid,
                labels={'x': 'Fitted Values', 'y': 'Residuals'},
                title="Residuals vs Fitted Values"
            )
            fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_resid, use_container_width=True)
            
        except Exception as e:
            st.error(f"Model diagnostics failed: {e}")
    else:
        st.info("No model available for diagnostics")

# -----------------------
# Section: Export & Reporting
# -----------------------
st.markdown("---")
st.markdown('<h2 class="section-header">📤 Export & Reporting</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Export")
    
    # Filtered data download
    csv_data = df_f.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Filtered Dataset",
        data=csv_data,
        file_name="filtered_development_analysis.csv",
        mime="text/csv",
        help="Download the currently filtered dataset"
    )
    
    # Summary statistics download
    summary_csv = create_summary_statistics(df_f).to_csv(index=False).encode('utf-8')
    st.download_button(
        "📊 Download Summary Statistics",
        data=summary_csv,
        file_name="summary_statistics.csv",
        mime="text/csv"
    )

with col2:
    st.subheader("Generate Report")
    
    if st.button("📋 Generate Comprehensive Report"):
        with st.spinner("Generating detailed analysis report..."):
            # Create comprehensive report
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("THE PRICE OF PROGRESS - COMPREHENSIVE ANALYSIS REPORT")
            report_lines.append("=" * 60)
            report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"Dataset: {len(df_f)} countries (filtered from {len(df)} total)")
            report_lines.append("")
            
            # Filters applied
            report_lines.append("FILTERS APPLIED:")
            report_lines.append(f"  Continents: {sel_continent}")
            report_lines.append(f"  Income Groups: {sel_income}")
            report_lines.append(f"  HDI Range: {min_hdi:.3f} - {max_hdi:.3f}")
            report_lines.append(f"  Year Range: {year_range[0]} - {year_range[1]}")
            report_lines.append("")
            
            # Key statistics
            report_lines.append("KEY STATISTICS:")
            hdi_col = 'HDI_2023' if 'HDI_2023' in df_f.columns else 'HDI'
            report_lines.append(f"  Average HDI: {df_f[hdi_col].mean():.3f}")
            report_lines.append(f"  Average Suicide Rate: {df_f['Suicide_rate'].mean():.2f} per 100k")
            report_lines.append(f"  Median GDP per capita: ${df_f['GDP_per_capita'].median():,.0f}")
            report_lines.append("")
            
            # Model results
            if model is not None:
                report_lines.append("QUADRATIC MODEL RESULTS:")
                report_lines.append(f"  R-squared: {model.rsquared:.3f}")
                report_lines.append(f"  Observations: {model.nobs}")
                report_lines.append("")
                report_lines.append("COEFFICIENTS:")
                for param, value in model.params.items():
                    report_lines.append(f"  {param}: {value:.4f}")
                report_lines.append("")
                
                tp = tipping_point_from_model(model)
                if tp:
                    report_lines.append(f"ESTIMATED TIPPING POINT: HDI = {tp:.3f}")
                    report_lines.append("")
            
            # Correlation summary
            if 'HDI' in df_f.columns and 'Suicide_rate' in df_f.columns:
                corr, _ = pearsonr(df_f['HDI'], df_f['Suicide_rate'])
                report_lines.append(f"HDI-Suicide Correlation: {corr:.3f}")
            
            report_text = "\n".join(report_lines)
            
            st.download_button(
                "📄 Download Full Report (.txt)",
                data=report_text,
                file_name="progress_analysis_report.txt",
                mime="text/plain"
            )

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <h3>🧠 About This Analysis</h3>
    <p><strong>The Price of Progress</strong> explores the complex relationship between human development 
    and mental health outcomes across nations and time.</p>
    
    <div style='display: flex; justify-content: center; gap: 2rem; margin: 1rem 0;'>
        <div>
            <strong>Data Sources</strong><br>
            World Bank • WHO • UNDP
        </div>
        <div>
            <strong>Methodology</strong><br>
            Statistical Analysis • Machine Learning • Data Visualization
        </div>
        <div>
            <strong>Ethical Considerations</strong><br>
            Correlation ≠ Causation • Cultural Context • Data Limitations
        </div>
    </div>
    
    <p><em>Built with Streamlit • Enhanced with Advanced Analytics</em></p>
    
    <div style='margin-top: 1rem; font-size: 0.9rem;'>
        <strong>Important:</strong> This analysis shows correlational patterns only. 
        Always consider local context and consult mental health professionals for policy decisions.
        If you or someone you know is struggling, please seek professional help.
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------
# Session Information
# -----------------------
with st.sidebar.expander("🔧 Technical Details"):
    st.write(f"**Data Shape:** {df_f.shape}")
    st.write(f"**Memory Usage:** {df_f.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    st.write(f"**Complete Cases:** {df_f.notna().all(axis=1).sum()}")
    st.write(f"**Years Covered:** {sorted(df_f['Year'].unique())}")
    
    if 'model' in locals() and model is not None:
        st.write("**Active Model:** ✅ Quadratic model fitted")
    else:
        st.write("**Active Model:** ❌ No model fitted")

print("🚀 Enhanced Streamlit application loaded successfully!")