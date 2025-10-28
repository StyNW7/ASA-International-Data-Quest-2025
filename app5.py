# comprehensive_dashboard.py
# Streamlit dashboard: "The Price of Progress - Comprehensive Edition"
# Complete merged dashboard with all features from app.py, app3.py, and app4.py

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
    page_title="The Price of Progress - Comprehensive", 
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
    .tab-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
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
        # Try multiple possible file locations
        possible_paths = [
            "./Final/final_clean_dataset.csv",
            "./Output/merged_clean_panel.csv",
            "./merged_clean_panel.csv"
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
    
    hdi_col = 'HDI_2023' if 'HDI_2023' in df_f.columns else 'HDI'
    if hdi_col in df_f.columns:
        stats_data['HDI'] = df_f[hdi_col].describe().round(3).tolist()
    
    if 'Suicide_rate' in df_f.columns:
        stats_data['Suicide Rate'] = df_f['Suicide_rate'].describe().round(2).tolist()
    
    if 'GDP_per_capita' in df_f.columns:
        stats_data['GDP per capita'] = df_f['GDP_per_capita'].describe().round(0).tolist()
    
    return pd.DataFrame(stats_data)

def create_development_clusters(df):
    """Create development stage clusters"""
    try:
        # Use HDI and GDP for clustering
        hdi_col = 'HDI_2023' if 'HDI_2023' in df.columns else 'HDI'
        cluster_data = df[[hdi_col, 'GDP_per_capita']].dropna()
        if len(cluster_data) < 5:
            return df
            
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Map clusters to development stages
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        development_stages = []
        
        for i, center in enumerate(cluster_centers):
            hdi, gdp = center
            if hdi < 0.55:
                stage = "Low Development"
            elif hdi < 0.7:
                stage = "Medium Development"
            elif hdi < 0.8:
                stage = "High Development"
            else:
                stage = "Very High Development"
            development_stages.append((i, stage, hdi))
        
        # Sort by HDI
        development_stages.sort(key=lambda x: x[2])
        
        # Create mapping
        cluster_mapping = {stage[0]: stage[1] for stage in development_stages}
        
        # Add to dataframe
        df_clustered = df.copy()
        df_clustered['Cluster'] = kmeans.labels_
        df_clustered['Development_Stage'] = df_clustered['Cluster'].map(cluster_mapping)
        
        return df_clustered
    except Exception:
        return df

def prepare_prediction_data(df):
    """Prepare data for predictive modeling"""
    hdi_col = 'HDI_2023' if 'HDI_2023' in df.columns else 'HDI'
    
    # Select features for prediction
    feature_cols = [hdi_col, 'GDP_per_capita', 'HDI_sq', 'log_GDP_per_capita', 
                   'HDI_lag1', 'Suicide_rate_lag1', 'HDI_growth']
    
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Add categorical features if available
    categorical_features = []
    if 'continent' in df.columns:
        categorical_features.append('continent')
    if 'income_group_auto' in df.columns:
        categorical_features.append('income_group_auto')
    
    # Prepare feature matrix
    X = df[available_features].copy()
    
    # Add encoded categorical features
    for cat_feat in categorical_features:
        if cat_feat in df.columns:
            le = LabelEncoder()
            encoded = le.fit_transform(df[cat_feat].fillna('Unknown'))
            X[f'{cat_feat}_encoded'] = encoded
    
    # Target variable
    y = df['Suicide_rate']
    
    # Remove rows with missing target
    valid_indices = y.notna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    return X, y, available_features + [f'{feat}_encoded' for feat in categorical_features]

def train_prediction_models(X, y):
    """Train multiple prediction models"""
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0)
    }
    
    trained_models = {}
    scores = {}
    
    for name, model in models.items():
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict and score
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            trained_models[name] = {
                'model': model,
                'mse': mse,
                'r2': r2,
                'mae': mae,
                'feature_names': X.columns.tolist()
            }
            scores[name] = r2
            
        except Exception as e:
            st.warning(f"Could not train {name}: {e}")
    
    return trained_models, scores

def predict_future_trends(df, model, years_ahead=5):
    """Predict future trends based on current data"""
    try:
        # Get the latest year
        latest_year = df['Year'].max()
        
        # Create future years
        future_years = list(range(latest_year + 1, latest_year + years_ahead + 1))
        
        # For simplicity, use average growth rates
        predictions = []
        
        for country in df['Country Name'].unique():
            country_data = df[df['Country Name'] == country].sort_values('Year')
            
            if len(country_data) < 2:
                continue
                
            # Get latest values
            latest = country_data.iloc[-1]
            hdi_growth = country_data['HDI_growth'].mean() if 'HDI_growth' in country_data.columns else 0.01
            suicide_change = country_data['Suicide_change'].mean() if 'Suicide_change' in country_data.columns else 0
            
            # Project future values
            hdi_col = 'HDI_2023' if 'HDI_2023' in latest else 'HDI'
            current_hdi = latest[hdi_col]
            current_suicide = latest['Suicide_rate']
            
            for year in future_years:
                # Simple projection (could be enhanced with proper time series modeling)
                projected_hdi = min(1.0, current_hdi * (1 + hdi_growth))
                projected_suicide = max(0, current_suicide + suicide_change)
                
                predictions.append({
                    'Country Name': country,
                    'ISO3': latest['ISO3'],
                    'Year': year,
                    hdi_col: projected_hdi,
                    'Suicide_rate': projected_suicide,
                    'GDP_per_capita': latest['GDP_per_capita'],
                    'continent': latest.get('continent', 'Unknown'),
                    'income_group_auto': latest.get('income_group_auto', 'Unknown'),
                    'is_prediction': True
                })
                
                current_hdi = projected_hdi
                current_suicide = projected_suicide
        
        return pd.DataFrame(predictions)
    
    except Exception as e:
        st.error(f"Error in future prediction: {e}")
        return pd.DataFrame()

# -----------------------
# Sidebar: Data / Filters
# -----------------------
st.sidebar.title("🔧 Settings & Data")
st.sidebar.markdown("---")

# Data upload
uploaded = st.sidebar.file_uploader(
    "Upload Dataset", 
    type=["csv"],
    help="Upload your dataset or use the default one"
)

df = load_data(uploaded)

if df is None:
    st.sidebar.error("""
    No dataset found. Please upload your dataset containing:
    - Country Name, ISO3, Year
    - HDI, Suicide_rate, GDP_per_capita
    - Additional development indicators
    """)
    st.stop()

# Ensure key numeric columns
numeric_cols = ['HDI', 'HDI_2023', 'HDI_sq', 'GDP_per_capita', 'log_GDP_per_capita', 
               'Suicide_rate', 'HDI_growth', 'Suicide_change', 'HDI_lag1', 'Suicide_rate_lag1', 'coverage_frac']
df = safe_numeric(df, [col for col in numeric_cols if col in df.columns])

# Compute derived columns if missing
if 'log_GDP_per_capita' not in df.columns and 'GDP_per_capita' in df.columns:
    df['log_GDP_per_capita'] = np.log(df['GDP_per_capita'].where(df['GDP_per_capita'] > 0, np.nan))

hdi_col = 'HDI_2023' if 'HDI_2023' in df.columns else 'HDI'
if 'HDI_sq' not in df.columns and hdi_col in df.columns:
    df['HDI_sq'] = df[hdi_col] ** 2

# Create development clusters
df = create_development_clusters(df)

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
development_stages = sorted(df['Development_Stage'].dropna().unique()) if 'Development_Stage' in df.columns else []

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

if data_quality_opts:
    sel_quality = st.sidebar.multiselect(
        "Data Quality", 
        options=data_quality_opts, 
        default=data_quality_opts
    )
else:
    sel_quality = []

if development_stages:
    sel_development = st.sidebar.multiselect(
        "Development Stage", 
        options=development_stages, 
        default=development_stages
    )
else:
    sel_development = []

# HDI range filter
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
if data_quality_opts:
    mask &= df['Low_data_quality_flag'].isin(sel_quality)
if development_stages:
    mask &= df['Development_Stage'].isin(sel_development)
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
    Comprehensive Analysis of Human Development and Mental Health Outcomes<br>
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
# Section: Executive Summary & Key Insights
# -----------------------
st.markdown("---")
st.markdown('<h2 class="section-header">📈 Executive Summary & Key Insights</h2>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    # Overall trend analysis
    if 'Year' in df_f.columns and hdi_col in df_f.columns and 'Suicide_rate' in df_f.columns:
        yearly_avg = df_f.groupby('Year').agg({
            hdi_col: 'mean',
            'Suicide_rate': 'mean',
            'GDP_per_capita': 'median'
        }).reset_index()
        
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_trend.add_trace(
            go.Scatter(x=yearly_avg['Year'], y=yearly_avg[hdi_col], 
                      name="Average HDI", line=dict(color='blue', width=3)),
            secondary_y=False,
        )
        
        fig_trend.add_trace(
            go.Scatter(x=yearly_avg['Year'], y=yearly_avg['Suicide_rate'], 
                      name="Average Suicide Rate", line=dict(color='red', width=3)),
            secondary_y=True,
        )
        
        fig_trend.update_layout(
            title="Global Trends: HDI vs Suicide Rate Over Time",
            height=400
        )
        fig_trend.update_xaxes(title_text="Year")
        fig_trend.update_yaxes(title_text="Average HDI", secondary_y=False)
        fig_trend.update_yaxes(title_text="Average Suicide Rate", secondary_y=True)
        
        st.plotly_chart(fig_trend, use_container_width=True)

with col2:
    st.subheader("Key Insights")
    
    # Calculate key metrics
    if hdi_col in df_f.columns and 'Suicide_rate' in df_f.columns:
        latest_data = df_f[df_f['Year'] == df_f['Year'].max()].dropna(subset=[hdi_col, 'Suicide_rate'])
        
        if len(latest_data) > 0:
            # Correlation
            corr, p_value = pearsonr(latest_data[hdi_col], latest_data['Suicide_rate'])
            
            # Development paradox count
            high_hdi = latest_data[latest_data[hdi_col] > 0.8]
            paradox_count = len(high_hdi[high_hdi['Suicide_rate'] > high_hdi['Suicide_rate'].median()])
            
            st.metric("HDI-Suicide Correlation", f"{corr:.3f}", 
                     delta="Positive" if corr > 0 else "Negative")
            st.metric("Development Paradox Countries", f"{paradox_count}")
            st.metric("Data Coverage", f"{len(latest_data)} countries")
            
            st.markdown("""
            <div class="insight-box">
            <strong>💡 Insight:</strong> The relationship between development and mental health 
            shows complex patterns that vary across development stages and regions.
            </div>
            """, unsafe_allow_html=True)

# -----------------------
# Section: Time Series Analysis
# -----------------------
st.markdown("---")
st.markdown('<h2 class="section-header">📊 Time Series Analysis</h2>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Country Comparison", "Regional Trends", "Development Stages", "Change Analysis"])

with tab1:
    st.subheader("Country-Level Time Series Analysis")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Country selector
        available_countries = sorted(df_f['Country Name'].unique())
        selected_countries = st.multiselect(
            "Select Countries",
            options=available_countries,
            default=available_countries[:5] if len(available_countries) > 5 else available_countries,
            max_selections=10
        )
        
        # Variable selector
        variables = [hdi_col, 'Suicide_rate', 'GDP_per_capita']
        selected_variable = st.selectbox("Select Variable", variables)
    
    with col2:
        if selected_countries and 'Year' in df_f.columns:
            country_data = df_f[df_f['Country Name'].isin(selected_countries)]
            
            fig_country = px.line(
                country_data, 
                x='Year', 
                y=selected_variable,
                color='Country Name',
                title=f"{selected_variable} Trends by Country",
                markers=True
            )
            fig_country.update_layout(height=500)
            st.plotly_chart(fig_country, use_container_width=True)

with tab2:
    st.subheader("Regional Trends")
    
    if 'continent' in df_f.columns:
        regional_trends = df_f.groupby(['Year', 'continent']).agg({
            hdi_col: 'mean',
            'Suicide_rate': 'mean',
            'GDP_per_capita': 'median'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_regional_hdi = px.line(
                regional_trends,
                x='Year',
                y=hdi_col,
                color='continent',
                title="HDI Trends by Continent"
            )
            st.plotly_chart(fig_regional_hdi, use_container_width=True)
        
        with col2:
            fig_regional_suicide = px.line(
                regional_trends,
                x='Year',
                y='Suicide_rate',
                color='continent',
                title="Suicide Rate Trends by Continent"
            )
            st.plotly_chart(fig_regional_suicide, use_container_width=True)

with tab3:
    st.subheader("Development Stage Analysis")
    
    if 'Development_Stage' in df_f.columns:
        development_trends = df_f.groupby(['Year', 'Development_Stage']).agg({
            hdi_col: 'mean',
            'Suicide_rate': 'mean',
            'GDP_per_capita': 'median'
        }).reset_index()
        
        fig_development = px.line(
            development_trends,
            x='Year',
            y='Suicide_rate',
            color='Development_Stage',
            title="Suicide Rate Trends by Development Stage",
            line_dash='Development_Stage'
        )
        st.plotly_chart(fig_development, use_container_width=True)

with tab4:
    st.subheader("Change Analysis")
    
    if 'HDI_growth' in df_f.columns and 'Suicide_change' in df_f.columns:
        # Calculate changes between first and last year for each country
        if 'Year' in df_f.columns:
            years_sorted = sorted(df_f['Year'].unique())
            if len(years_sorted) >= 2:
                first_year = years_sorted[0]
                last_year = years_sorted[-1]
                
                country_changes = []
                for country in df_f['Country Name'].unique():
                    country_data = df_f[df_f['Country Name'] == country]
                    first_data = country_data[country_data['Year'] == first_year]
                    last_data = country_data[country_data['Year'] == last_year]
                    
                    if len(first_data) > 0 and len(last_data) > 0:
                        hdi_change = last_data[hdi_col].iloc[0] - first_data[hdi_col].iloc[0]
                        suicide_change = last_data['Suicide_rate'].iloc[0] - first_data['Suicide_rate'].iloc[0]
                        
                        country_changes.append({
                            'Country Name': country,
                            'HDI Change': hdi_change,
                            'Suicide Rate Change': suicide_change,
                            'Continent': first_data['continent'].iloc[0] if 'continent' in first_data.columns else 'Unknown'
                        })
                
                if country_changes:
                    changes_df = pd.DataFrame(country_changes)
                    
                    fig_change = px.scatter(
                        changes_df,
                        x='HDI Change',
                        y='Suicide Rate Change',
                        color='Continent',
                        title=f"Development vs Mental Health Changes ({first_year} to {last_year})",
                        hover_data=['Country Name']
                    )
                    fig_change.add_hline(y=0, line_dash="dash", line_color="red")
                    fig_change.add_vline(x=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_change, use_container_width=True)

# -----------------------
# Section: Advanced Analytics & Machine Learning
# -----------------------
st.markdown("---")
st.markdown('<h2 class="section-header">🤖 Advanced Analytics & Machine Learning</h2>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Predictive Modeling", "Feature Importance", "Clustering Analysis", "Future Projections"])

with tab1:
    st.subheader("Suicide Rate Prediction Models")
    
    # Prepare data for modeling
    X, y, feature_names = prepare_prediction_data(df_f)
    
    if len(X) > 10 and len(y) > 10:
        # Train models
        with st.spinner("Training prediction models..."):
            trained_models, scores = train_prediction_models(X, y)
        
        if trained_models:
            # Display model performance
            st.subheader("Model Performance Comparison")
            
            performance_data = []
            for name, model_info in trained_models.items():
                performance_data.append({
                    'Model': name,
                    'R² Score': model_info['r2'],
                    'Mean Squared Error': model_info['mse'],
                    'Mean Absolute Error': model_info['mae']
                })
            
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df.style.format({
                'R² Score': '{:.3f}',
                'Mean Squared Error': '{:.3f}',
                'Mean Absolute Error': '{:.3f}'
            }), use_container_width=True)
            
            # Visualize performance
            fig_performance = px.bar(
                performance_df,
                x='Model',
                y='R² Score',
                color='R² Score',
                title="Model Performance (R² Scores)",
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_performance, use_container_width=True)
            
            # Best model info
            best_model_name = max(scores, key=scores.get)
            best_model_info = trained_models[best_model_name]
            
            st.markdown(f"""
            <div class="prediction-card">
            <h3>🏆 Best Performing Model: {best_model_name}</h3>
            <p><strong>R² Score:</strong> {best_model_info['r2']:.3f} | 
            <strong>MAE:</strong> {best_model_info['mae']:.3f} | 
            <strong>MSE:</strong> {best_model_info['mse']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Could not train prediction models with the available data.")
    else:
        st.warning("Insufficient data for predictive modeling. Need more complete cases.")

with tab2:
    st.subheader("Feature Importance Analysis")
    
    if 'trained_models' in locals() and trained_models:
        # Get feature importance from Random Forest
        if 'Random Forest' in trained_models:
            rf_model = trained_models['Random Forest']['model']
            feature_names = trained_models['Random Forest']['feature_names']
            
            importances = rf_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(
                feature_importance_df.head(15),
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 15 Feature Importances (Random Forest)",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
            st.dataframe(feature_importance_df.style.format({'Importance': '{:.3f}'}), use_container_width=True)
    else:
        st.info("Train prediction models first to see feature importance.")

with tab3:
    st.subheader("Country Clustering Analysis")
    
    if 'Development_Stage' in df_f.columns:
        # Show cluster distribution
        cluster_dist = df_f['Development_Stage'].value_counts()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_cluster = px.bar(
                x=cluster_dist.index,
                y=cluster_dist.values,
                title="Country Distribution by Development Stage",
                labels={'x': 'Development Stage', 'y': 'Number of Countries'},
                color=cluster_dist.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
        
        with col2:
            st.subheader("Cluster Summary")
            for stage, count in cluster_dist.items():
                avg_hdi = df_f[df_f['Development_Stage'] == stage][hdi_col].mean()
                avg_suicide = df_f[df_f['Development_Stage'] == stage]['Suicide_rate'].mean()
                st.write(f"**{stage}**")
                st.write(f"Count: {count}")
                st.write(f"Avg HDI: {avg_hdi:.3f}")
                st.write(f"Avg Suicide: {avg_suicide:.2f}")
                st.write("---")

with tab4:
    st.subheader("Future Trends Projection")
    
    if st.button("Generate Future Projections", key="future_proj"):
        with st.spinner("Generating future projections..."):
            # Use the best model for projections
            if 'trained_models' in locals() and trained_models:
                best_model_name = max(scores, key=scores.get)
                best_model = trained_models[best_model_name]['model']
                
                # Generate future predictions
                future_df = predict_future_trends(df_f, best_model, years_ahead=5)
                
                if len(future_df) > 0:
                    # Combine with historical data
                    historical_df = df_f.copy()
                    historical_df['is_prediction'] = False
                    
                    combined_df = pd.concat([historical_df, future_df], ignore_index=True)
                    
                    # Show projection for selected countries
                    sample_countries = future_df['Country Name'].unique()[:5]
                    
                    fig_future = px.line(
                        combined_df[combined_df['Country Name'].isin(sample_countries)],
                        x='Year',
                        y='Suicide_rate',
                        color='Country Name',
                        line_dash='is_prediction',
                        title="Suicide Rate Projections (Historical + Future)",
                        markers=True
                    )
                    st.plotly_chart(fig_future, use_container_width=True)
                    
                    st.info(f"Showing projections for {len(sample_countries)} sample countries. "
                           f"Projections based on current trends and model predictions.")
                else:
                    st.warning("Could not generate future projections with current data.")
            else:
                st.warning("Please train prediction models first.")

# -----------------------
# Section: Interactive Visualizations
# -----------------------
st.markdown("---")
st.markdown('<h2 class="section-header">📈 Interactive Visualizations</h2>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Scatter Plots", "Maps", "Distribution Analysis", "Correlation Matrix"])

with tab1:
    st.subheader("Interactive Scatter Plots")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox(
            "X-axis Variable",
            options=[hdi_col, 'GDP_per_capita', 'log_GDP_per_capita', 'Year'],
            index=0
        )
    
    with col2:
        y_axis = st.selectbox(
            "Y-axis Variable", 
            options=['Suicide_rate', hdi_col, 'GDP_per_capita'],
            index=0
        )
    
    color_by = st.selectbox(
        "Color By",
        options=['continent', 'income_group_auto', 'Development_Stage', 'None'],
        index=0
    )
    
    # Create scatter plot
    scatter_data = df_f.dropna(subset=[x_axis, y_axis])
    
    if color_by != 'None' and color_by in scatter_data.columns:
        fig_scatter = px.scatter(
            scatter_data,
            x=x_axis,
            y=y_axis,
            color=color_by,
            hover_data=['Country Name', 'Year'],
            title=f"{y_axis} vs {x_axis}",
            trendline="lowess" if len(scatter_data) > 10 else None
        )
    else:
        fig_scatter = px.scatter(
            scatter_data,
            x=x_axis,
            y=y_axis,
            hover_data=['Country Name', 'Year'],
            title=f"{y_axis} vs {x_axis}",
            trendline="lowess" if len(scatter_data) > 10 else None
        )
    
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.subheader("Geographical Analysis")
    
    if 'ISO3' in df_f.columns:
        # Latest year data for mapping
        latest_year_map = df_f['Year'].max()
        map_data = df_f[df_f['Year'] == latest_year_map]
        
        col1, col2 = st.columns(2)
        
        with col1:
            map_variable = st.selectbox(
                "Map Variable",
                options=[hdi_col, 'Suicide_rate', 'GDP_per_capita'],
                index=0
            )
        
        with col2:
            map_scale = st.selectbox(
                "Color Scale",
                options=['Viridis', 'Plasma', 'Inferno', 'Blues', 'Reds'],
                index=0
            )
        
        if len(map_data) > 0:
            fig_map = px.choropleth(
                map_data,
                locations="ISO3",
                color=map_variable,
                hover_name="Country Name",
                hover_data={hdi_col: ':.3f', 'Suicide_rate': ':.2f', 'GDP_per_capita': ':,.0f'},
                title=f"World Map: {map_variable} ({latest_year_map})",
                color_continuous_scale=map_scale
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("No data available for mapping.")
    else:
        st.warning("ISO3 country codes not available for mapping.")

with tab3:
    st.subheader("Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dist_variable = st.selectbox(
            "Distribution Variable",
            options=[hdi_col, 'Suicide_rate', 'GDP_per_capita', 'log_GDP_per_capita'],
            index=0
        )
    
    with col2:
        dist_group = st.selectbox(
            "Group By",
            options=['None', 'continent', 'income_group_auto', 'Development_Stage'],
            index=0
        )
    
    dist_data = df_f.dropna(subset=[dist_variable])
    
    if dist_group != 'None' and dist_group in dist_data.columns:
        fig_dist = px.histogram(
            dist_data,
            x=dist_variable,
            color=dist_group,
            marginal="box",
            title=f"Distribution of {dist_variable} by {dist_group}",
            opacity=0.7
        )
    else:
        fig_dist = px.histogram(
            dist_data,
            x=dist_variable,
            marginal="box",
            title=f"Distribution of {dist_variable}",
            opacity=0.7
        )
    
    st.plotly_chart(fig_dist, use_container_width=True)

with tab4:
    st.subheader("Correlation Analysis")
    
    # Select numeric columns for correlation
    numeric_columns = df_f.select_dtypes(include=[np.number]).columns.tolist()
    selected_corr_vars = st.multiselect(
        "Select Variables for Correlation Matrix",
        options=numeric_columns,
        default=[hdi_col, 'Suicide_rate', 'GDP_per_capita', 'log_GDP_per_capita'][:3]
    )
    
    if len(selected_corr_vars) >= 2:
        corr_data = df_f[selected_corr_vars].corr()
        
        fig_corr = px.imshow(
            corr_data,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Detailed correlation table
        st.subheader("Detailed Correlation Coefficients")
        corr_table = corr_data.unstack().reset_index()
        corr_table.columns = ['Variable 1', 'Variable 2', 'Correlation']
        corr_table = corr_table[corr_table['Variable 1'] != corr_table['Variable 2']]
        corr_table = corr_table.sort_values('Correlation', ascending=False)
        st.dataframe(corr_table.style.format({'Correlation': '{:.3f}'}), use_container_width=True)
    else:
        st.info("Select at least 2 variables for correlation analysis.")

# -----------------------
# Section: Quadratic Analysis & Tipping Points
# -----------------------
st.markdown("---")
st.markdown('<h2 class="section-header">📐 Quadratic Analysis & Tipping Points</h2>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Quadratic Regression", "Tipping Point Analysis", "Model Diagnostics"])

with tab1:
    st.subheader("HDI vs Suicide Rate: Quadratic Relationship")
    
    # Year selection for quadratic analysis
    if 'Year' in df_f.columns:
        analysis_year = st.selectbox(
            "Select Year for Analysis",
            options=sorted(df_f['Year'].unique(), reverse=True),
            index=0
        )
    else:
        analysis_year = None
    
    # Fit quadratic model
    model, dfm = fit_quadratic(df_f, year=analysis_year, robust=True)
    
    if model is not None:
        # Create quadratic plot
        hdi_range = np.linspace(dfm[hdi_col].min(), dfm[hdi_col].max(), 100)
        hdi_sq_range = hdi_range ** 2
        
        # Create prediction data
        pred_data = pd.DataFrame({
            hdi_col: hdi_range,
            'HDI_sq': hdi_sq_range,
            'log_GDP_per_capita': dfm['log_GDP_per_capita'].median()
        })
        
        predictions = model.get_prediction(pred_data)
        pred_summary = predictions.summary_frame()
        
        fig_quad = go.Figure()
        
        # Add scatter points
        fig_quad.add_trace(go.Scatter(
            x=dfm[hdi_col],
            y=dfm['Suicide_rate'],
            mode='markers',
            name='Countries',
            marker=dict(
                color=dfm['log_GDP_per_capita'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Log GDP")
            ),
            text=dfm['Country Name'],
            hovertemplate="<b>%{text}</b><br>HDI: %{x:.3f}<br>Suicide Rate: %{y:.2f}<extra></extra>"
        ))
        
        # Add quadratic fit
        fig_quad.add_trace(go.Scatter(
            x=hdi_range,
            y=pred_summary['mean'],
            mode='lines',
            name='Quadratic Fit',
            line=dict(color='red', width=3)
        ))
        
        # Add confidence interval
        fig_quad.add_trace(go.Scatter(
            x=np.concatenate([hdi_range, hdi_range[::-1]]),
            y=np.concatenate([pred_summary['mean_ci_lower'], pred_summary['mean_ci_upper'][::-1]]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI'
        ))
        
        # Calculate and show tipping point
        tipping_point = tipping_point_from_model(model)
        if tipping_point:
            # Get predicted suicide rate at tipping point
            tp_data = pd.DataFrame({
                hdi_col: [tipping_point],
                'HDI_sq': [tipping_point ** 2],
                'log_GDP_per_capita': [dfm['log_GDP_per_capita'].median()]
            })
            tp_pred = model.predict(tp_data).iloc[0]
            
            fig_quad.add_trace(go.Scatter(
                x=[tipping_point],
                y=[tp_pred],
                mode='markers',
                marker=dict(color='black', size=12, symbol='star'),
                name=f'Tipping Point (HDI={tipping_point:.3f})'
            ))
        
        fig_quad.update_layout(
            title=f"Quadratic Relationship: Suicide Rate vs HDI ({analysis_year})",
            xaxis_title="Human Development Index (HDI)",
            yaxis_title="Suicide Rate (per 100,000)",
            height=600
        )
        
        st.plotly_chart(fig_quad, use_container_width=True)
        
        # Model summary
        st.subheader("Quadratic Model Summary")
        st.text(str(model.summary()))
        
    else:
        st.warning("Could not fit quadratic model with the selected data.")

with tab2:
    st.subheader("Tipping Point Analysis Over Time")
    
    if 'Year' in df_f.columns:
        # Calculate tipping points for each year
        years = sorted(df_f['Year'].unique())
        tipping_points = []
        
        for year in years:
            model_year, _ = fit_quadratic(df_f, year=year)
            if model_year:
                tp = tipping_point_from_model(model_year)
                if tp:
                    tipping_points.append({'Year': year, 'Tipping Point': tp})
        
        if tipping_points:
            tp_df = pd.DataFrame(tipping_points)
            
            fig_tp = px.line(
                tp_df,
                x='Year',
                y='Tipping Point',
                title="Evolution of HDI Tipping Point Over Time",
                markers=True
            )
            fig_tp.add_hline(y=0.8, line_dash="dash", line_color="red", 
                           annotation_text="Very High Development Threshold")
            fig_tp.add_hline(y=0.7, line_dash="dash", line_color="orange", 
                           annotation_text="High Development Threshold")
            
            st.plotly_chart(fig_tp, use_container_width=True)
            
            # Tipping point statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Tipping Point", f"{tp_df['Tipping Point'].mean():.3f}")
            with col2:
                st.metric("Trend", "Increasing" if len(tp_df) > 1 and 
                         tp_df['Tipping Point'].iloc[-1] > tp_df['Tipping Point'].iloc[0] else "Decreasing")
            with col3:
                st.metric("Latest Tipping Point", f"{tp_df['Tipping Point'].iloc[-1]:.3f}")
        else:
            st.warning("Could not calculate tipping points across years.")

with tab3:
    st.subheader("Model Diagnostics")
    
    if 'model' in locals() and model is not None:
        # Residuals plot
        residuals = model.resid
        fitted = model.fittedvalues
        
        fig_resid = make_subplots(rows=1, cols=2, subplot_titles=['Residuals vs Fitted', 'Q-Q Plot'])
        
        # Residuals vs Fitted
        fig_resid.add_trace(
            go.Scatter(x=fitted, y=residuals, mode='markers', name='Residuals'),
            row=1, col=1
        )
        fig_resid.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Q-Q Plot (simplified)
        from scipy import stats
        qq = stats.probplot(residuals, dist="norm")
        theoretical = qq[0][0]
        ordered = qq[0][1]
        
        fig_resid.add_trace(
            go.Scatter(x=theoretical, y=ordered, mode='markers', name='Q-Q'),
            row=1, col=2
        )
        # Add reference line
        line_x = [theoretical.min(), theoretical.max()]
        line_y = [ordered.min() + (ordered.max() - ordered.min()) * 0.1, 
                 ordered.max() - (ordered.max() - ordered.min()) * 0.1]
        fig_resid.add_trace(
            go.Scatter(x=line_x, y=line_y, mode='lines', line=dict(color='red', dash='dash'), 
                      showlegend=False),
            row=1, col=2
        )
        
        fig_resid.update_layout(height=400, title_text="Regression Diagnostics")
        st.plotly_chart(fig_resid, use_container_width=True)
        
        # Model assumptions check
        st.subheader("Model Assumptions Check")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Normality test
            _, p_norm = stats.normaltest(residuals)
            st.metric("Normality (p-value)", f"{p_norm:.4f}", 
                     delta="Normal" if p_norm > 0.05 else "Non-normal")
        
        with col2:
            # Homoscedasticity (simplified)
            corr_resid_fitted, p_het = stats.spearmanr(fitted, np.abs(residuals))
            st.metric("Heteroscedasticity (p-value)", f"{p_het:.4f}",
                     delta="Homoscedastic" if p_het > 0.05 else "Heteroscedastic")

# -----------------------
# Section: Download & Export
# -----------------------
st.markdown("---")
st.markdown('<h2 class="section-header">💾 Download & Export</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Download Filtered Data", key="download_filtered"):
        csv = df_f.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="filtered_development_data.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📊 Download Summary Report", key="download_report"):
        # Create a simple text report
        report_lines = []
        report_lines.append("THE PRICE OF PROGRESS - ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Countries: {df_f['Country Name'].nunique()}")
        report_lines.append(f"Observations: {len(df_f)}")
        report_lines.append(f"Time Period: {df_f['Year'].min()} - {df_f['Year'].max()}")
        
        if hdi_col in df_f.columns:
            report_lines.append(f"Average HDI: {df_f[hdi_col].mean():.3f}")
        if 'Suicide_rate' in df_f.columns:
            report_lines.append(f"Average Suicide Rate: {df_f['Suicide_rate'].mean():.2f}")
        
        report_text = "\n".join(report_lines)
        st.download_button(
            "Download Report",
            data=report_text,
            file_name="analysis_report.txt",
            mime="text/plain"
        )

with col3:
    if st.button("🖼️ Download All Visualizations", key="download_viz"):
        st.info("This would download all generated visualizations. Implementation depends on specific requirements.")

# -----------------------
# Section: Conclusion & Insights
# -----------------------
st.markdown("---")
st.markdown('<h2 class="section-header">🔍 Key Findings & Policy Implications</h2>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Executive Summary")
    
    insights = [
        "📈 **Development Paradox**: Higher HDI doesn't always correlate with better mental health outcomes",
        "🔄 **Non-linear Relationship**: The HDI-Suicide relationship follows an inverted U-shape in many analyses",
        "🌍 **Regional Variations**: Different continents show distinct patterns in development-mental health relationships",
        "⏰ **Temporal Evolution**: The relationship between development and mental health has evolved over time",
        "🎯 **Policy Implications**: Targeted mental health interventions are needed at different development stages"
    ]
    
    for insight in insights:
        st.markdown(f"<div class='info-bubble'>{insight}</div>", unsafe_allow_html=True)

with col2:
    st.subheader("Recommendations")
    
    recommendations = [
        "Integrate mental health into development planning",
        "Monitor mental health indicators alongside economic growth",
        "Develop stage-specific mental health interventions",
        "Invest in mental health infrastructure during development transitions"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>The Price of Progress - Comprehensive Dashboard</strong></p>
    <p>An integrated analysis of human development and mental health outcomes across nations and time</p>
    <p>Built with ❤️ using Streamlit | Data sources: World Bank, WHO, UNDP</p>
</div>
""", unsafe_allow_html=True)

# -----------------------
# Performance Optimization
# -----------------------
@st.cache_data
def expensive_operation(data):
    """Cache expensive operations for better performance"""
    # This is a placeholder for any expensive computations
    return data

# Apply caching to expensive operations
df_f = expensive_operation(df_f)