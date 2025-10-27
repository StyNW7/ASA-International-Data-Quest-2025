# app.py
# Streamlit dashboard: "The Price of Progress"
# Enhanced version with better UI, error handling, and features
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import statsmodels.api as sm
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import base64
import time
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Page configuration
st.set_page_config(
    page_title="The Price of Progress", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

sns.set_style("whitegrid")

# -----------------------
# Enhanced Helpers
# -----------------------
@st.cache_data
def load_data(path=None):
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
        # If user doesn't supply file, fallback to try local Output path
        try:
            df = pd.read_csv("../Output/merged_clean_panel.csv")
            st.sidebar.success(f"✅ Local data loaded: {len(df)} rows")
            return df
        except Exception:
            st.sidebar.error("❌ No dataset found locally.")
            return None

def safe_numeric(df, cols):
    """Convert columns to numeric with better error reporting"""
    conversion_report = []
    for c in cols:
        if c in df.columns:
            original_non_null = df[c].notna().sum()
            df[c] = pd.to_numeric(df[c], errors='coerce')
            new_non_null = df[c].notna().sum()
            lost_values = original_non_null - new_non_null
            if lost_values > 0:
                conversion_report.append(f"  - {c}: lost {lost_values} non-numeric values")
    if conversion_report:
        st.sidebar.warning("Numeric conversion report:\n" + "\n".join(conversion_report))
    return df

def fit_quadratic(df, robust=True):
    """Fit Suicide_rate ~ HDI_2023 + HDI_sq + log_GDP_per_capita with enhanced error handling"""
    required_cols = ['Suicide_rate', 'HDI_2023', 'HDI_sq']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return None, df
    
    dfm = df.dropna(subset=required_cols)
    
    if len(dfm) < 10:  # Minimum sample size
        st.warning(f"Only {len(dfm)} complete observations available. Need at least 10 for reliable modeling.")
        return None, dfm
    
    # Ensure log GDP exists
    if 'log_GDP_per_capita' not in dfm.columns and 'GDP_per_capita' in dfm.columns:
        dfm['log_GDP_per_capita'] = np.log(dfm['GDP_per_capita'].where(dfm['GDP_per_capita'] > 1, np.nan))
    
    formula = "Suicide_rate ~ HDI_2023 + HDI_sq + log_GDP_per_capita"
    
    try:
        model = smf.ols(formula=formula, data=dfm).fit(cov_type='HC1' if robust else None)
        return model, dfm
    except Exception as e:
        st.error(f"Model fitting failed: {e}")
        return None, dfm

def tipping_point_from_model(model):
    """Calculate tipping point with validation"""
    try:
        p = model.params
        if 'HDI_2023' in p and 'HDI_sq' in p and p['HDI_sq'] != 0:
            tp = -p['HDI_2023']/(2*p['HDI_sq'])
            # Validate tipping point is within reasonable HDI range
            if 0.3 <= tp <= 1.0:
                return float(tp)
            else:
                st.warning(f"Calculated tipping point ({tp:.3f}) outside reasonable HDI range [0.3, 1.0]")
                return None
        return None
    except Exception as e:
        st.error(f"Error calculating tipping point: {e}")
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
    stats_data = {
        'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max'],
        'HDI': df_f['HDI_2023'].describe().round(3).tolist(),
        'Suicide Rate': df_f['Suicide_rate'].describe().round(2).tolist(),
        'GDP per capita': df_f['GDP_per_capita'].describe().round(0).tolist()
    }
    return pd.DataFrame(stats_data)

# -----------------------
# Sidebar: Data / Filters
# -----------------------
st.sidebar.title("🔧 Settings & Data")
st.sidebar.markdown("---")

# Data upload
uploaded = st.sidebar.file_uploader(
    "Upload cleaned CSV (merged_clean_panel.csv)", 
    type=["csv"],
    help="Upload your dataset or use the default one"
)

df = load_data(uploaded)

if df is None:
    st.sidebar.error("""
    No dataset found. Please:
    1. Upload `merged_clean_panel.csv` above, OR
    2. Place it at `./Output/merged_clean_panel.csv`
    """)
    st.stop()

# Ensure key numeric columns
numeric_cols = ['HDI_2023', 'HDI_sq', 'GDP_per_capita', 'log_GDP_per_capita', 'Suicide_rate', 'coverage_frac']
df = safe_numeric(df, numeric_cols)

# Compute log GDP if missing
if 'log_GDP_per_capita' not in df.columns and 'GDP_per_capita' in df.columns:
    df['log_GDP_per_capita'] = np.log(df['GDP_per_capita'].where(df['GDP_per_capita'] > 0, np.nan))

# Sidebar filters
st.sidebar.subheader("🎯 Data Filters")

# Available filters
continents = sorted(df['continent'].dropna().unique()) if 'continent' in df.columns else []
income_groups = sorted(df['income_group_auto'].dropna().unique()) if 'income_group_auto' in df.columns else []
data_quality_opts = sorted(df['Low_data_quality_flag'].dropna().unique()) if 'Low_data_quality_flag' in df.columns else []

# Filter widgets
if continents:
    sel_continent = st.sidebar.multiselect(
        "Continent", 
        options=continents, 
        default=continents,
        help="Filter by continent"
    )
else:
    sel_continent = []

if income_groups:
    sel_income = st.sidebar.multiselect(
        "Income group", 
        options=income_groups, 
        default=income_groups,
        help="Filter by World Bank income classification"
    )
else:
    sel_income = []

if data_quality_opts:
    sel_quality = st.sidebar.multiselect(
        "Data quality", 
        options=data_quality_opts, 
        default=data_quality_opts,
        help="Filter by data quality flags"
    )
else:
    sel_quality = []

# HDI range filter
if 'HDI_2023' in df.columns:
    min_hdi, max_hdi = st.sidebar.slider(
        "HDI Range",
        min_value=float(df['HDI_2023'].min()),
        max_value=float(df['HDI_2023'].max()),
        value=(float(df['HDI_2023'].min()), float(df['HDI_2023'].max())),
        step=0.01,
        help="Filter countries by HDI range"
    )
else:
    min_hdi, max_hdi = 0, 1

# Apply filters
mask = pd.Series(True, index=df.index)

if continents:
    mask &= df['continent'].isin(sel_continent)
if income_groups:
    mask &= df['income_group_auto'].isin(sel_income)
if data_quality_opts:
    mask &= df['Low_data_quality_flag'].isin(sel_quality)
if 'HDI_2023' in df.columns:
    mask &= (df['HDI_2023'] >= min_hdi) & (df['HDI_2023'] <= max_hdi)

df_f = df[mask].copy()

# Display filter summary
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Filter Summary")
st.sidebar.markdown(f"**Countries:** {df_f['Country Name'].nunique()}")
st.sidebar.markdown(f"**Observations:** {len(df_f)}")
st.sidebar.markdown(f"**HDI Range:** {df_f['HDI_2023'].min():.3f} - {df_f['HDI_2023'].max():.3f}")

# -----------------------
# Main Content
# -----------------------
st.markdown('<h1 class="main-header">The Price of Progress</h1>', unsafe_allow_html=True)
st.markdown("""
Exploring the complex relationships between human development, economic growth, and suicide rates across countries. 
Use the sidebar to filter data by continent, income group, data quality, or HDI range.
""")

# Key metrics with enhanced styling
st.markdown("### 📈 Overview Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Countries", f"{df_f['Country Name'].nunique()}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    avg_hdi = df_f['HDI_2023'].mean()
    st.metric("Average HDI", f"{avg_hdi:.3f}", 
              delta=f"{(avg_hdi - df['HDI_2023'].mean()):.3f} vs full dataset" if len(df_f) != len(df) else None)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    suicide_rate = df_f['Suicide_rate'].mean()
    st.metric("Avg Suicide Rate", f"{suicide_rate:.2f} per 100k")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    median_gdp = df_f['GDP_per_capita'].median()
    st.metric("Median GDP/capita", f"${median_gdp:,.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# Section: Data Overview & Summary Statistics
# -----------------------
st.markdown("---")
st.markdown('<h2 class="section-header">📋 Data Overview</h2>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Summary Statistics", "Data Sample", "Missing Data"])

with tab1:
    st.subheader("Descriptive Statistics")
    stats_df = create_summary_statistics(df_f)
    st.dataframe(stats_df, use_container_width=True)
    
    # Additional insights
    col1, col2, col3 = st.columns(3)
    with col1:
        hdi_corr = df_f[['HDI_2023', 'Suicide_rate']].corr().iloc[0,1]
        st.metric("HDI-Suicide Correlation", f"{hdi_corr:.3f}")
    
    with col2:
        gdp_corr = df_f[['GDP_per_capita', 'Suicide_rate']].corr().iloc[0,1]
        st.metric("GDP-Suicide Correlation", f"{gdp_corr:.3f}")
    
    with col3:
        complete_cases = df_f[['HDI_2023', 'Suicide_rate', 'GDP_per_capita']].notna().all(axis=1).sum()
        st.metric("Complete Cases", f"{complete_cases}")

with tab2:
    st.subheader("Data Sample")
    st.dataframe(df_f.head(10), use_container_width=True)
    st.info(f"Showing 10 of {len(df_f)} rows. Dataset has {len(df_f.columns)} columns.")

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
        st.dataframe(missing_df, use_container_width=True)
        
        # Missing data visualization
        fig_missing = px.bar(missing_df, x='Column', y='Missing %', 
                            title='Missing Data by Column (%)',
                            color='Missing %', color_continuous_scale='Reds')
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("🎉 No missing data in the filtered dataset!")

# -----------------------
# Section: Correlation Analysis
# -----------------------
st.markdown("---")
st.markdown('<h2 class="section-header">🔍 Correlation Analysis</h2>', unsafe_allow_html=True)

# Select variables for correlation
corr_vars = st.multiselect(
    "Select variables for correlation analysis:",
    options=[col for col in df_f.columns if df_f[col].dtype in ['float64', 'int64']],
    default=['HDI_2023', 'GDP_per_capita', 'Suicide_rate', 'log_GDP_per_capita'],
    help="Choose numerical variables to include in correlation analysis"
)

if len(corr_vars) >= 2:
    corr_data = df_f[corr_vars].corr(method='pearson')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Enhanced correlation heatmap
        fig_corr = px.imshow(
            corr_data, 
            text_auto=True, 
            color_continuous_scale="RdBu_r", 
            zmin=-1, 
            zmax=1,
            title="Correlation Matrix Heatmap",
            aspect="auto"
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Download correlation matrix
        if st.button("📥 Download Correlation Matrix (PNG)"):
            buf = fig_to_bytes(fig_corr)
            if buf:
                st.download_button(
                    "Download PNG", 
                    data=buf, 
                    file_name="correlation_matrix.png", 
                    mime="image/png"
                )
    
    with col2:
        st.subheader("Top Correlations")
        # Get top correlations
        corr_pairs = []
        for i in range(len(corr_data.columns)):
            for j in range(i+1, len(corr_data.columns)):
                corr_pairs.append({
                    'Variables': f"{corr_data.columns[i]} - {corr_data.columns[j]}",
                    'Correlation': corr_data.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
        top_corrs = corr_df.nlargest(10, 'Abs_Correlation')
        
        for _, row in top_corrs.iterrows():
            corr_val = row['Correlation']
            color = "🟢" if corr_val > 0.5 else "🟡" if corr_val > 0.3 else "🔴" if corr_val > 0 else "🔵"
            st.write(f"{color} **{row['Variables']}**: {corr_val:.3f}")

# -----------------------
# Section: HDI vs Suicide Rate Analysis
# -----------------------
st.markdown("---")
st.markdown('<h2 class="section-header">📊 HDI vs Suicide Rate</h2>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    # Enhanced scatter plot
    fig_scatter = px.scatter(
        df_f, 
        x='HDI_2023', 
        y='Suicide_rate',
        color='continent' if 'continent' in df_f.columns else None,
        size='GDP_per_capita' if 'GDP_per_capita' in df_f.columns else None,
        hover_name='Country Name',
        hover_data=['GDP_per_capita', 'income_group_auto'] if 'income_group_auto' in df_f.columns else ['GDP_per_capita'],
        title="HDI vs Suicide Rate by Country",
        size_max=15,
        opacity=0.7
    )
    
    # Fit and add quadratic trend
    model, model_df = fit_quadratic(df_f)
    if model is not None:
        hdi_range = np.linspace(df_f['HDI_2023'].min(), df_f['HDI_2023'].max(), 100)
        params = model.params
        gdp_mean = model_df['log_GDP_per_capita'].mean() if 'log_GDP_per_capita' in model_df else 0
        
        y_pred = (params.get('Intercept', 0) + 
                 params.get('HDI_2023', 0) * hdi_range + 
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
    
    # Correlation metrics
    if 'HDI_2023' in df_f.columns and 'Suicide_rate' in df_f.columns:
        clean_data = df_f[['HDI_2023', 'Suicide_rate']].dropna()
        if len(clean_data) > 2:
            pearson_r, pearson_p = pearsonr(clean_data['HDI_2023'], clean_data['Suicide_rate'])
            spearman_r, spearman_p = spearmanr(clean_data['HDI_2023'], clean_data['Suicide_rate'])
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Pearson Correlation", f"{pearson_r:.3f}", 
                         delta=f"p-value: {pearson_p:.4f}")
            with col_b:
                st.metric("Spearman Correlation", f"{spearman_r:.3f}", 
                         delta=f"p-value: {spearman_p:.4f}")

with col2:
    st.subheader("Model Analysis")
    
    if model is not None:
        # Model summary
        st.success("✅ Quadratic Model Fitted")
        
        # Key coefficients
        st.metric("HDI Coefficient", f"{model.params.get('HDI_2023', 0):.3f}")
        st.metric("HDI² Coefficient", f"{model.params.get('HDI_sq', 0):.3f}")
        st.metric("R-squared", f"{model.rsquared:.3f}")
        
        # Tipping point
        tp = tipping_point_from_model(model)
        if tp:
            st.success(f"🎯 Tipping Point: HDI = {tp:.3f}")
            st.info("""
            **Interpretation:** 
            Below this HDI level, development may be associated with increasing suicide rates.
            Above this level, further development may correlate with decreasing suicide rates.
            """)
        
        # Model diagnostics
        st.subheader("Model Diagnostics")
        st.metric("Observations", f"{model.nobs}")
        st.metric("F-statistic", f"{model.fvalue:.2f}")
        st.metric("Prob (F-statistic)", f"{model.f_pvalue:.4f}")
        
        # Download model summary
        if st.button("📥 Download Model Summary"):
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

# -----------------------
# Section: Economic Analysis
# -----------------------
st.markdown("---")
st.markdown('<h2 class="section-header">💼 Economic Analysis</h2>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["GDP vs Suicide Rate", "Income Group Analysis", "Economic Development"])

with tab1:
    st.subheader("GDP per Capita vs Suicide Rate")
    
    # Choose scale: linear or log
    scale_option = st.radio("GDP Scale:", ["Log Scale", "Linear Scale"], horizontal=True)
    
    if scale_option == "Log Scale":
        x_col = 'log_GDP_per_capita'
        x_title = "Log GDP per Capita"
    else:
        x_col = 'GDP_per_capita'
        x_title = "GDP per Capita (USD)"
    
    if x_col in df_f.columns:
        fig_gdp = px.scatter(
            df_f, 
            x=x_col, 
            y='Suicide_rate',
            color='continent' if 'continent' in df_f.columns else None,
            hover_name='Country Name',
            hover_data=['HDI_2023', 'income_group_auto'] if 'income_group_auto' in df_f.columns else ['HDI_2023'],
            title=f"Suicide Rate vs {x_title}",
            trendline="lowess",
            trendline_color_override="red"
        )
        fig_gdp.update_layout(height=500)
        st.plotly_chart(fig_gdp, use_container_width=True)
        
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
            title="Suicide Rate Distribution by Income Group"
        )
        fig_income.update_layout(height=500)
        st.plotly_chart(fig_income, use_container_width=True)
        
        # Summary statistics by income group
        income_stats = df_f.groupby('income_group_auto').agg({
            'Suicide_rate': ['mean', 'median', 'std', 'count'],
            'HDI_2023': 'mean',
            'GDP_per_capita': 'median'
        }).round(3)
        
        st.subheader("Summary by Income Group")
        st.dataframe(income_stats, use_container_width=True)
    else:
        st.info("Income group data not available")

with tab3:
    st.subheader("Development Trajectories")
    
    # Scatter plot with both HDI and GDP
    if all(col in df_f.columns for col in ['HDI_2023', 'GDP_per_capita', 'Suicide_rate']):
        fig_development = px.scatter(
            df_f,
            x='HDI_2023',
            y='GDP_per_capita',
            size='Suicide_rate',
            color='Suicide_rate',
            hover_name='Country Name',
            title="Development Landscape: HDI vs GDP with Suicide Rate",
            color_continuous_scale='Viridis',
            size_max=20
        )
        fig_development.update_layout(height=500)
        st.plotly_chart(fig_development, use_container_width=True)
        
        st.info("""
        **Interpretation:**
        - Bubble size represents suicide rate
        - Color intensity shows suicide rate magnitude
        - Upper-right quadrant: High HDI & High GDP
        - Lower-left quadrant: Low HDI & Low GDP
        """)

# -----------------------
# Section: Geographic Analysis
# -----------------------
st.markdown("---")
st.markdown('<h2 class="section-header">🌍 Geographic Analysis</h2>', unsafe_allow_html=True)

if 'ISO3' in df_f.columns:
    # Map visualization
    col1, col2 = st.columns([3, 1])
    
    with col1:
        map_variable = st.selectbox(
            "Select variable for map:",
            options=['Suicide_rate', 'HDI_2023', 'GDP_per_capita'],
            format_func=lambda x: {
                'Suicide_rate': 'Suicide Rate',
                'HDI_2023': 'Human Development Index',
                'GDP_per_capita': 'GDP per Capita'
            }[x]
        )
        
        color_scale = st.selectbox(
            "Color scale:",
            options=['Reds', 'Blues', 'Viridis', 'Plasma', 'Inferno'],
            index=0
        )
        
        fig_map = px.choropleth(
            df_f,
            locations="ISO3",
            color=map_variable,
            hover_name="Country Name",
            hover_data=['HDI_2023', 'GDP_per_capita', 'income_group_auto'] if 'income_group_auto' in df_f.columns else ['HDI_2023', 'GDP_per_capita'],
            color_continuous_scale=color_scale,
            title=f"Global Distribution of {map_variable.replace('_', ' ').title()}"
        )
        fig_map.update_layout(height=600)
        st.plotly_chart(fig_map, use_container_width=True)
    
    with col2:
        st.subheader("Regional Summary")
        
        if 'continent' in df_f.columns:
            continent_stats = df_f.groupby('continent').agg({
                'Suicide_rate': 'mean',
                'HDI_2023': 'mean',
                'GDP_per_capita': 'median',
                'Country Name': 'count'
            }).round(3).rename(columns={'Country Name': 'Count'})
            
            st.dataframe(continent_stats)
            
            # Top countries by suicide rate
            st.subheader("Extreme Values")
            top_suicide = df_f.nlargest(5, 'Suicide_rate')[['Country Name', 'Suicide_rate', 'HDI_2023']]
            bottom_suicide = df_f.nsmallest(5, 'Suicide_rate')[['Country Name', 'Suicide_rate', 'HDI_2023']]
            
            st.write("**Highest Suicide Rates:**")
            st.dataframe(top_suicide)
            
            st.write("**Lowest Suicide Rates:**")
            st.dataframe(bottom_suicide)
    
    # Download interactive map
    if st.button("📥 Download Interactive Map"):
        html_map = fig_map.to_html(full_html=False, include_plotlyjs='cdn')
        st.download_button(
            "Download HTML", 
            data=html_map, 
            file_name="global_analysis_map.html", 
            mime="text/html"
        )
else:
    st.info("ISO3 country codes not available for mapping")

# -----------------------
# Section: Advanced Diagnostics
# -----------------------
st.markdown("---")
st.markdown('<h2 class="section-header">🔬 Advanced Diagnostics</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Multicollinearity Check (VIF)")
    
    try:
        # Compute VIF for key variables
        vif_vars = ['HDI_2023', 'HDI_sq', 'log_GDP_per_capita']
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
        "📥 Download Filtered CSV",
        data=csv_data,
        file_name="filtered_development_data.csv",
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
    st.subheader("Automated Report")
    
    # Generate comprehensive report
    if st.button("📋 Generate Full Report"):
        with st.spinner("Generating comprehensive report..."):
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("THE PRICE OF PROGRESS - ANALYSIS REPORT")
            report_lines.append("=" * 60)
            report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"Dataset: {len(df_f)} countries (filtered from {len(df)} total)")
            report_lines.append("")
            
            # Filters applied
            report_lines.append("FILTERS APPLIED:")
            report_lines.append(f"  Continents: {sel_continent}")
            report_lines.append(f"  Income Groups: {sel_income}")
            report_lines.append(f"  Data Quality: {sel_quality}")
            report_lines.append(f"  HDI Range: {min_hdi:.3f} - {max_hdi:.3f}")
            report_lines.append("")
            
            # Key statistics
            report_lines.append("KEY STATISTICS:")
            report_lines.append(f"  Average HDI: {df_f['HDI_2023'].mean():.3f}")
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
            if len(corr_vars) >= 2:
                report_lines.append("CORRELATION SUMMARY:")
                for i in range(len(corr_data.columns)):
                    for j in range(i+1, len(corr_data.columns)):
                        corr_val = corr_data.iloc[i, j]
                        report_lines.append(f"  {corr_data.columns[i]} vs {corr_data.columns[j]}: {corr_val:.3f}")
            
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
<div style='text-align: center; color: #666;'>
    <h3>🧠 Notes & Ethical Considerations</h3>
    <p><strong>Important Disclaimers:</strong></p>
    <ul style='text-align: left;'>
        <li>This analysis shows <em>correlational relationships only</em> — correlation does not imply causation</li>
        <li>Suicide data quality varies significantly across countries due to cultural, reporting, and methodological differences</li>
        <li>Economic development is complex and multidimensional — HDI and GDP capture only certain aspects</li>
        <li>Always consider local context, cultural factors, and data limitations when interpreting results</li>
        <li>If you or someone you know is struggling, please seek help from mental health professionals</li>
    </ul>
    <p><em>Built with Streamlit • Data sources: World Bank, WHO, UNDP</em></p>
</div>
""", unsafe_allow_html=True)

# -----------------------
# Session info (hidden by default)
# -----------------------
with st.sidebar.expander("Session Information"):
    st.write(f"**Data shape:** {df_f.shape}")
    st.write(f"**Memory usage:** {df_f.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    st.write(f"**Columns:** {len(df_f.columns)}")
    st.write(f"**Complete cases:** {df_f.notna().all(axis=1).sum()}")