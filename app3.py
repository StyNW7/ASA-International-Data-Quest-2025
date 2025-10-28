# app.py
# Streamlit dashboard: "The Price of Progress - Enhanced Analysis"
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="The Price of Progress - Enhanced Analysis", 
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
        background: linear-gradient(90deg, #1f77b4, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .section-header {
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #2c3e50;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fdf2e9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #e67e22;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Data Loading & Processing
# -----------------------
@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    # This would be replaced with your actual dataset path
    try:
        # For demonstration, creating sample data structure based on your example
        # Replace this with your actual data loading code
        sample_data = {
            'Country Name': ['Afghanistan', 'Afghanistan', 'Afghanistan', 'Afghanistan', 'Afghanistan', 
                           'Angola', 'Angola', 'Brazil', 'Brazil', 'Brazil', 'China', 'China', 'China',
                           'United States', 'United States', 'Germany', 'Germany', 'India', 'India'],
            'ISO3': ['AFG', 'AFG', 'AFG', 'AFG', 'AFG', 'AGO', 'AGO', 'BRA', 'BRA', 'BRA', 
                    'CHN', 'CHN', 'CHN', 'USA', 'USA', 'DEU', 'DEU', 'IND', 'IND'],
            'Year': [2019, 2020, 2021, 2022, 2023, 2019, 2020, 2019, 2020, 2021, 
                    2019, 2020, 2021, 2019, 2020, 2019, 2020, 2019, 2020],
            'log_GDP_per_capita': [6.18, 6.18, 6.18, 6.18, 6.18, 7.84, 7.84, 9.12, 9.10, 9.08, 
                                  9.02, 9.00, 8.98, 10.65, 10.63, 10.72, 10.70, 7.89, 7.87],
            'income_group_auto': ['Low', 'Low', 'Low', 'Low', 'Low', 'Lower-Middle', 'Lower-Middle',
                                 'Upper-Middle', 'Upper-Middle', 'Upper-Middle', 'Upper-Middle', 
                                 'Upper-Middle', 'Upper-Middle', 'High', 'High', 'High', 'High',
                                 'Lower-Middle', 'Lower-Middle'],
            'continent': ['Asia', 'Asia', 'Asia', 'Asia', 'Asia', 'Africa', 'Africa', 
                         'South America', 'South America', 'South America', 'Asia', 'Asia', 'Asia',
                         'North America', 'North America', 'Europe', 'Europe', 'Asia', 'Asia'],
            'Low_data_quality_flag': ['✅ Sufficient data'] * 19,
            'HDI_sq': [0.242, 0.238, 0.224, 0.213, 0.246, 0.356, 0.353, 0.624, 0.622, 0.620,
                       0.592, 0.590, 0.588, 0.864, 0.862, 0.883, 0.881, 0.422, 0.420],
            'Suicide_rate_lag1': [np.nan, np.nan, 3.63, 3.6, 3.6, np.nan, np.nan, np.nan, 6.2, 6.1,
                                 np.nan, 7.8, 7.7, np.nan, 14.2, np.nan, 9.1, np.nan, 12.5],
            'HDI_lag1': [0.449, 0.492, 0.488, 0.473, 0.462, 0.516, 0.597, 0.758, 0.761, 0.763,
                         0.754, 0.759, 0.764, 0.924, 0.925, 0.939, 0.940, 0.633, 0.638],
            'HDI': [0.492, 0.488, 0.473, 0.462, 0.496, 0.597, 0.594, 0.761, 0.763, 0.765,
                    0.759, 0.764, 0.769, 0.925, 0.926, 0.940, 0.941, 0.638, 0.643],
            'Suicide_rate': [3.63, 3.63, 3.6, 3.6, 3.6, 6.9, 6.9, 6.2, 6.1, 6.0, 
                            7.8, 7.7, 7.6, 14.2, 14.0, 9.1, 9.0, 12.5, 12.3],
            'GDP_per_capita': [485, 485, 485, 485, 485, 2528, 2528, 9120, 9100, 9080, 
                              9020, 9000, 8980, 42650, 42630, 47200, 47000, 3890, 3870],
            'HDI_growth': [0.043, -0.004, -0.015, -0.011, 0.034, 0.081, -0.003, 0.003, 0.002, 0.002,
                           0.005, 0.005, 0.005, 0.001, 0.001, 0.001, 0.001, 0.005, 0.005],
            'Suicide_change': [0.0, 0.0, -0.03, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, -0.1, 
                               0.0, -0.1, -0.1, 0.0, -0.2, 0.0, -0.1, 0.0, -0.2],
            'Suicide_per_HDI': [7.38, 7.44, 7.61, 7.79, 7.26, 11.56, 11.62, 8.15, 7.99, 7.84,
                                10.28, 10.08, 9.88, 15.35, 15.12, 9.68, 9.56, 19.59, 19.13]
        }
        df = pd.DataFrame(sample_data)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def create_time_series_data(df):
    """Create enhanced time series analysis"""
    ts_data = df.groupby(['Year', 'continent', 'income_group_auto']).agg({
        'HDI': 'mean',
        'Suicide_rate': 'mean',
        'GDP_per_capita': 'mean',
        'Country Name': 'count'
    }).reset_index().rename(columns={'Country Name': 'Country_Count'})
    return ts_data

@st.cache_data
def prepare_prediction_data(df):
    """Prepare data for predictive modeling"""
    # Select features for prediction
    feature_cols = ['HDI', 'HDI_sq', 'log_GDP_per_capita', 'GDP_per_capita', 
                   'HDI_lag1', 'Suicide_rate_lag1', 'HDI_growth']
    
    # Remove rows with missing target or key features
    pred_df = df.dropna(subset=['Suicide_rate'] + feature_cols).copy()
    
    return pred_df, feature_cols

def train_prediction_models(df, feature_cols, target='Suicide_rate'):
    """Train multiple prediction models"""
    models = {}
    scores = {}
    
    X = df[feature_cols]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    models['linear'] = {'model': lr, 'scaler': scaler}
    scores['linear'] = {
        'r2': r2_score(y_test, lr_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, lr_pred))
    }
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    models['random_forest'] = {'model': rf, 'scaler': None}
    scores['random_forest'] = {
        'r2': r2_score(y_test, rf_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred))
    }
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return models, scores, feature_importance

# -----------------------
# Visualization Functions
# -----------------------
def create_development_trajectory_plot(df):
    """Create animated development trajectory plot"""
    fig = px.scatter(df, 
                    x='HDI', 
                    y='Suicide_rate',
                    size='GDP_per_capita',
                    color='continent',
                    hover_name='Country Name',
                    animation_frame='Year',
                    size_max=40,
                    title='Development Trajectories: HDI vs Suicide Rate Over Time',
                    labels={'HDI': 'Human Development Index', 
                           'Suicide_rate': 'Suicide Rate (per 100k)',
                           'GDP_per_capita': 'GDP per Capita'},
                    height=600)
    fig.update_layout(transition={'duration': 1000})
    return fig

def create_income_group_analysis(df):
    """Create comprehensive income group analysis"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Suicide Rate by Income Group', 'HDI by Income Group',
                       'GDP Distribution by Income Group', 'Suicide Rate vs HDI by Income Group'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    income_order = ['Low', 'Lower-Middle', 'Upper-Middle', 'High']
    
    # Suicide Rate by Income Group
    for i, group in enumerate(income_order):
        group_data = df[df['income_group_auto'] == group]['Suicide_rate']
        fig.add_trace(go.Box(y=group_data, name=group, boxpoints='outliers'), row=1, col=1)
    
    # HDI by Income Group
    for i, group in enumerate(income_order):
        group_data = df[df['income_group_auto'] == group]['HDI']
        fig.add_trace(go.Box(y=group_data, name=group, showlegend=False), row=1, col=2)
    
    # GDP Distribution
    for i, group in enumerate(income_order):
        group_data = df[df['income_group_auto'] == group]['GDP_per_capita']
        fig.add_trace(go.Violin(x=[group]*len(group_data), y=group_data, name=group, 
                               showlegend=False), row=2, col=1)
    
    # Scatter plot by income group
    for i, group in enumerate(income_order):
        group_data = df[df['income_group_auto'] == group]
        fig.add_trace(go.Scatter(x=group_data['HDI'], y=group_data['Suicide_rate'],
                                mode='markers', name=group, showlegend=False,
                                marker=dict(size=8, opacity=0.6)), row=2, col=2)
    
    fig.update_layout(height=800, title_text="Comprehensive Income Group Analysis")
    return fig

def create_correlation_heatmap(df):
    """Create enhanced correlation heatmap"""
    numeric_cols = ['HDI', 'Suicide_rate', 'GDP_per_capita', 'log_GDP_per_capita', 
                   'HDI_sq', 'HDI_growth', 'Suicide_change']
    
    corr_data = df[numeric_cols].corr()
    
    fig = ff.create_annotated_heatmap(
        z=corr_data.values,
        x=list(corr_data.columns),
        y=list(corr_data.index),
        annotation_text=corr_data.round(3).values,
        colorscale='RdBu_r',
        showscale=True,
        hoverinfo='z'
    )
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        height=600
    )
    
    return fig

def create_geographic_analysis(df, year=2020):
    """Create geographic analysis with multiple map views"""
    year_data = df[df['Year'] == year]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Suicide Rate World Map', 'HDI World Map',
                       'GDP per Capita World Map', 'Development Progress'),
        specs=[[{"type": "choropleth"}, {"type": "choropleth"}],
               [{"type": "choropleth"}, {"type": "scatter"}]]
    )
    
    # Suicide Rate Map
    fig.add_trace(go.Choropleth(
        locations=year_data['ISO3'],
        z=year_data['Suicide_rate'],
        colorscale='Reds',
        showscale=False,
        name='Suicide Rate'
    ), row=1, col=1)
    
    # HDI Map
    fig.add_trace(go.Choropleth(
        locations=year_data['ISO3'],
        z=year_data['HDI'],
        colorscale='Blues',
        showscale=False,
        name='HDI'
    ), row=1, col=2)
    
    # GDP Map
    fig.add_trace(go.Choropleth(
        locations=year_data['ISO3'],
        z=year_data['GDP_per_capita'],
        colorscale='Greens',
        showscale=True,
        name='GDP per Capita'
    ), row=2, col=1)
    
    # Development Progress Scatter
    fig.add_trace(go.Scatter(
        x=year_data['HDI'],
        y=year_data['GDP_per_capita'],
        mode='markers',
        marker=dict(
            size=year_data['Suicide_rate']*2,
            color=year_data['Suicide_rate'],
            colorscale='Viridis',
            showscale=True
        ),
        text=year_data['Country Name'],
        name='Countries'
    ), row=2, col=2)
    
    fig.update_layout(height=800, title_text=f"Global Analysis - {year}")
    return fig

# -----------------------
# Main Application
# -----------------------

# Load data
df = load_data()

if df is None:
    st.error("Failed to load data. Please check your dataset.")
    st.stop()

# Sidebar
st.sidebar.title("🎛️ Dashboard Controls")
st.sidebar.markdown("---")

# Data filters
st.sidebar.subheader("📊 Data Filters")

# Year filter
years = sorted(df['Year'].unique())
selected_years = st.sidebar.slider(
    "Select Year Range",
    min_value=int(min(years)),
    max_value=int(max(years)),
    value=(int(min(years)), int(max(years)))
)

# Continent filter
continents = sorted(df['continent'].unique())
selected_continents = st.sidebar.multiselect(
    "Select Continents",
    options=continents,
    default=continents
)

# Income group filter
income_groups = sorted(df['income_group_auto'].unique())
selected_income = st.sidebar.multiselect(
    "Select Income Groups",
    options=income_groups,
    default=income_groups
)

# HDI range filter
hdi_min, hdi_max = st.sidebar.slider(
    "HDI Range",
    min_value=float(df['HDI'].min()),
    max_value=float(df['HDI'].max()),
    value=(float(df['HDI'].min()), float(df['HDI'].max())),
    step=0.01
)

# Apply filters
filtered_df = df[
    (df['Year'].between(selected_years[0], selected_years[1])) &
    (df['continent'].isin(selected_continents)) &
    (df['income_group_auto'].isin(selected_income)) &
    (df['HDI'].between(hdi_min, hdi_max))
]

# Sidebar metrics
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Filter Summary")
st.sidebar.metric("Countries", filtered_df['Country Name'].nunique())
st.sidebar.metric("Observations", len(filtered_df))
st.sidebar.metric("Avg HDI", f"{filtered_df['HDI'].mean():.3f}")
st.sidebar.metric("Avg Suicide Rate", f"{filtered_df['Suicide_rate'].mean():.2f}")

# Main content
st.markdown('<h1 class="main-header">THE PRICE OF PROGRESS</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <h3 style='color: #666;'>Exploring the Complex Relationship Between Development and Suicide Rates</h3>
    <p>An interactive dashboard analyzing how economic development, measured by HDI and GDP, correlates with suicide rates across countries and over time.</p>
</div>
""", unsafe_allow_html=True)

# Key Metrics
st.markdown("### 🎯 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Countries", filtered_df['Country Name'].nunique(), 
              help="Number of unique countries in filtered data")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    correlation = filtered_df[['HDI', 'Suicide_rate']].corr().iloc[0,1]
    st.metric("HDI-Suicide Correlation", f"{correlation:.3f}",
              delta="Positive" if correlation > 0 else "Negative",
              help="Pearson correlation between HDI and Suicide Rates")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    avg_suicide = filtered_df['Suicide_rate'].mean()
    st.metric("Average Suicide Rate", f"{avg_suicide:.2f} per 100k",
              help="Mean suicide rate across filtered countries")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    median_gdp = filtered_df['GDP_per_capita'].median()
    st.metric("Median GDP/Capita", f"${median_gdp:,.0f}",
              help="Median GDP per capita across filtered countries")
    st.markdown('</div>', unsafe_allow_html=True)

# Section 1: Development Trajectories
st.markdown("---")
st.markdown('<h2 class="section-header">🚀 Development Trajectories Over Time</h2>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    trajectory_fig = create_development_trajectory_plot(filtered_df)
    st.plotly_chart(trajectory_fig, use_container_width=True)

with col2:
    st.markdown("""
    <div class="info-box">
    <h4>📈 Development Insights</h4>
    <p><strong>Key Observations:</strong></p>
    <ul>
        <li>Watch for countries moving along development paths</li>
        <li>Note changes in bubble sizes (GDP)</li>
        <li>Observe regional clustering patterns</li>
        <li>Track changes over time with animation</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Year statistics
    st.subheader("Yearly Statistics")
    yearly_stats = filtered_df.groupby('Year').agg({
        'HDI': 'mean',
        'Suicide_rate': 'mean',
        'GDP_per_capita': 'median'
    }).round(3)
    st.dataframe(yearly_stats)

# Section 2: Income Group Analysis
st.markdown("---")
st.markdown('<h2 class="section-header">💰 Income Group Analysis</h2>', unsafe_allow_html=True)

income_fig = create_income_group_analysis(filtered_df)
st.plotly_chart(income_fig, use_container_width=True)

# Income group statistics
st.subheader("Detailed Income Group Statistics")
income_stats = filtered_df.groupby('income_group_auto').agg({
    'Suicide_rate': ['mean', 'median', 'std', 'count'],
    'HDI': ['mean', 'median', 'std'],
    'GDP_per_capita': ['mean', 'median']
}).round(3)

st.dataframe(income_stats)

# Section 3: Correlation Analysis
st.markdown("---")
st.markdown('<h2 class="section-header">🔗 Correlation Analysis</h2>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    corr_fig = create_correlation_heatmap(filtered_df)
    st.plotly_chart(corr_fig, use_container_width=True)

with col2:
    st.markdown("""
    <div class="info-box">
    <h4>📊 Correlation Guide</h4>
    <p><strong>Interpretation:</strong></p>
    <ul>
        <li><strong>+1.0:</strong> Perfect positive correlation</li>
        <li><strong>+0.5:</strong> Strong positive correlation</li>
        <li><strong>0.0:</strong> No correlation</li>
        <li><strong>-0.5:</strong> Strong negative correlation</li>
        <li><strong>-1.0:</strong> Perfect negative correlation</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Top correlations
    st.subheader("Key Correlations")
    numeric_df = filtered_df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['Suicide_rate'].sort_values(ascending=False)
    
    for feature, corr in correlations.items():
        if feature != 'Suicide_rate' and abs(corr) > 0.1:
            emoji = "🔴" if corr > 0.3 else "🟡" if corr > 0.1 else "🔵"
            st.write(f"{emoji} {feature}: {corr:.3f}")

# Section 4: Geographic Analysis
st.markdown("---")
st.markdown('<h2 class="section-header">🌍 Global Geographic Analysis</h2>', unsafe_allow_html=True)

map_year = st.selectbox("Select Year for Map Analysis", sorted(filtered_df['Year'].unique(), reverse=True))
geo_fig = create_geographic_analysis(filtered_df, map_year)
st.plotly_chart(geo_fig, use_container_width=True)

# Section 5: Predictive Modeling
st.markdown("---")
st.markdown('<h2 class="section-header">🔮 Predictive Analytics</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<h4>🎯 Prediction Objective</h4>
<p>This section uses machine learning to predict suicide rates based on development indicators. 
We train multiple models to understand which factors most influence suicide rates and provide future projections.</p>
</div>
""", unsafe_allow_html=True)

# Prepare data for prediction
pred_df, feature_cols = prepare_prediction_data(filtered_df)

if len(pred_df) > 10:  # Minimum data requirement
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Training & Performance")
        
        # Train models
        with st.spinner("Training prediction models..."):
            models, scores, feature_importance = train_prediction_models(pred_df, feature_cols)
        
        # Display model performance
        model_performance = pd.DataFrame(scores).T
        model_performance['Model'] = model_performance.index
        model_performance = model_performance[['Model', 'r2', 'rmse']]
        
        st.dataframe(model_performance.style.format({'r2': '{:.3f}', 'rmse': '{:.3f}'}))
        
        # Feature importance
        st.subheader("Feature Importance (Random Forest)")
        fig_importance = px.bar(feature_importance, x='importance', y='feature', 
                               orientation='h', title='Feature Importance for Suicide Rate Prediction')
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        st.subheader("Make Predictions")
        
        # Prediction inputs
        st.markdown("### Enter Development Indicators")
        
        hdi_input = st.slider("HDI", 0.3, 1.0, 0.7, 0.01)
        hdi_sq_input = hdi_input ** 2
        gdp_input = st.slider("GDP per Capita (USD)", 500, 50000, 10000, 100)
        log_gdp_input = np.log(gdp_input)
        hdi_growth_input = st.slider("HDI Growth", -0.05, 0.05, 0.01, 0.001)
        
        # Create input array
        input_data = np.array([[hdi_input, hdi_sq_input, log_gdp_input, gdp_input, 
                              hdi_input, 10.0, hdi_growth_input]])  # Default lag values
        
        if st.button("Predict Suicide Rate"):
            # Use Random Forest for prediction
            rf_model = models['random_forest']['model']
            prediction = rf_model.predict(input_data)[0]
            
            st.success(f"### Predicted Suicide Rate: {prediction:.2f} per 100k")
            
            # Interpretation
            if prediction < 5:
                st.info("📊 **Interpretation:** Very low suicide rate prediction")
            elif prediction < 10:
                st.info("📊 **Interpretation:** Low to moderate suicide rate prediction")
            elif prediction < 15:
                st.warning("📊 **Interpretation:** Moderate suicide rate prediction")
            else:
                st.error("📊 **Interpretation:** High suicide rate prediction")
        
        # Future projections
        st.subheader("Future Projections")
        years_projection = st.slider("Projection Years", 1, 10, 5)
        
        if st.button("Generate Projections"):
            # Simple linear projection based on current trends
            current_avg = filtered_df['Suicide_rate'].mean()
            hdi_trend = filtered_df.groupby('Year')['HDI'].mean().pct_change().mean()
            
            # Simple projection model
            projected_rates = []
            for year in range(1, years_projection + 1):
                # Adjust based on HDI growth trend
                adjustment = hdi_trend * year * current_avg
                projected_rate = max(0, current_avg + adjustment)
                projected_rates.append(projected_rate)
            
            # Create projection plot
            projection_years = [2023 + i for i in range(1, years_projection + 1)]
            fig_projection = px.line(x=projection_years, y=projectied_rates,
                                   title=f"Suicide Rate Projection ({years_projection} years)",
                                   labels={'x': 'Year', 'y': 'Projected Suicide Rate'})
            fig_projection.add_scatter(x=projection_years, y=projected_rates, mode='markers')
            st.plotly_chart(fig_projection, use_container_width=True)

else:
    st.warning("Insufficient data for predictive modeling. Need more complete observations.")

# Section 6: Advanced Analytics
st.markdown("---")
st.markdown('<h2 class="section-header">📊 Advanced Analytics</h2>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Time Series Analysis", "Regional Comparison", "Development Thresholds"])

with tab1:
    st.subheader("Time Series Trends")
    
    # Create time series data
    ts_data = create_time_series_data(filtered_df)
    
    fig_ts = make_subplots(rows=2, cols=1, subplot_titles=('HDI Trends Over Time', 'Suicide Rate Trends Over Time'))
    
    # HDI trends by continent
    for continent in ts_data['continent'].unique():
        continent_data = ts_data[ts_data['continent'] == continent]
        fig_ts.add_trace(go.Scatter(x=continent_data['Year'], y=continent_data['HDI'], 
                                   name=continent, mode='lines+markers'), row=1, col=1)
    
    # Suicide rate trends by continent
    for continent in ts_data['continent'].unique():
        continent_data = ts_data[ts_data['continent'] == continent]
        fig_ts.add_trace(go.Scatter(x=continent_data['Year'], y=continent_data['Suicide_rate'], 
                                   name=continent, mode='lines+markers', showlegend=False), row=2, col=1)
    
    fig_ts.update_layout(height=600)
    st.plotly_chart(fig_ts, use_container_width=True)

with tab2:
    st.subheader("Regional Development Comparison")
    
    # Select regions to compare
    comparison_regions = st.multiselect("Select regions for comparison", 
                                       options=filtered_df['continent'].unique(),
                                       default=filtered_df['continent'].unique()[:2])
    
    if len(comparison_regions) >= 2:
        comp_data = filtered_df[filtered_df['continent'].isin(comparison_regions)]
        
        fig_comparison = px.scatter(comp_data, x='HDI', y='Suicide_rate', 
                                  color='continent', facet_col='continent',
                                  title='Regional Development Comparison',
                                  hover_name='Country Name', height=500)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Regional statistics
        regional_stats = comp_data.groupby('continent').agg({
            'HDI': ['mean', 'std'],
            'Suicide_rate': ['mean', 'std'],
            'GDP_per_capita': ['mean', 'median']
        }).round(3)
        
        st.dataframe(regional_stats)

with tab3:
    st.subheader("Development Threshold Analysis")
    
    # HDI threshold analysis
    threshold = st.slider("HDI Threshold", 0.3, 0.9, 0.7, 0.01)
    
    above_threshold = filtered_df[filtered_df['HDI'] >= threshold]
    below_threshold = filtered_df[filtered_df['HDI'] < threshold]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(f"Countries Above HDI {threshold}", len(above_threshold))
        st.metric("Average Suicide Rate (Above)", f"{above_threshold['Suicide_rate'].mean():.2f}")
        st.metric("Average GDP (Above)", f"${above_threshold['GDP_per_capita'].mean():,.0f}")
    
    with col2:
        st.metric(f"Countries Below HDI {threshold}", len(below_threshold))
        st.metric("Average Suicide Rate (Below)", f"{below_threshold['Suicide_rate'].mean():.2f}")
        st.metric("Average GDP (Below)", f"${below_threshold['GDP_per_capita'].mean():,.0f}")
    
    # Statistical test
    from scipy.stats import ttest_ind
    if len(above_threshold) > 1 and len(below_threshold) > 1:
        t_stat, p_value = ttest_ind(above_threshold['Suicide_rate'].dropna(), 
                                   below_threshold['Suicide_rate'].dropna())
        st.metric("T-test p-value", f"{p_value:.4f}",
                 delta="Significant difference" if p_value < 0.05 else "No significant difference")

# Section 7: Data Export and Reporting
st.markdown("---")
st.markdown('<h2 class="section-header">📤 Export & Reporting</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Export Data")
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Filtered CSV",
        data=csv_data,
        file_name="development_analysis_data.csv",
        mime="text/csv"
    )

with col2:
    st.subheader("Export Visualizations")
    if st.button("📊 Generate Report PDF"):
        st.info("PDF report generation would be implemented here with additional libraries")

with col3:
    st.subheader("Share Analysis")
    st.info("Shareable links and embedded reports would be implemented here")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p><strong>The Price of Progress Dashboard</strong> | Created with Streamlit</p>
    <p>Data Source: World Development Indicators & WHO Suicide Prevention Data</p>
    <p>⚠️ <em>Note: This dashboard uses sample data. Replace with your actual dataset for production use.</em></p>
</div>
""", unsafe_allow_html=True)