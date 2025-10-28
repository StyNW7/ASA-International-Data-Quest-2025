# streamlit_prediction_dashboard.py
# Interactive Streamlit dashboard for world map predictions - FIXED VERSION

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Future Predictions Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌍"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and prepare data"""
    try:
        df = pd.read_csv("./Final/final_clean_dataset.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def prepare_features(df):
    """Prepare features for modeling"""
    df_clean = df.copy()
    
    # Create features
    df_clean = df_clean.sort_values(['Country Name', 'Year'])
    
    # Lag features
    df_clean['Suicide_lag1'] = df_clean.groupby('Country Name')['Suicide_rate'].shift(1)
    df_clean['HDI_lag1'] = df_clean.groupby('Country Name')['HDI'].shift(1)
    df_clean['GDP_lag1'] = df_clean.groupby('Country Name')['GDP_per_capita'].shift(1)
    
    # Growth rates
    df_clean['HDI_growth'] = df_clean.groupby('Country Name')['HDI'].pct_change()
    df_clean['GDP_growth'] = df_clean.groupby('Country Name')['GDP_per_capita'].pct_change()
    
    # Polynomial features
    df_clean['HDI_sq'] = df_clean['HDI'] ** 2
    df_clean['log_GDP'] = np.log(df_clean['GDP_per_capita'] + 1)
    
    # Encode categorical variables
    if 'continent' in df_clean.columns:
        le_continent = LabelEncoder()
        df_clean['continent_encoded'] = le_continent.fit_transform(df_clean['continent'].fillna('Unknown'))
    
    if 'income_group_auto' in df_clean.columns:
        le_income = LabelEncoder()
        df_clean['income_encoded'] = le_income.fit_transform(df_clean['income_group_auto'].fillna('Unknown'))
    
    return df_clean

@st.cache_resource
def train_models(df_clean):
    """Train prediction models for different targets"""
    models = {}
    feature_columns = ['HDI', 'HDI_sq', 'GDP_per_capita', 'log_GDP', 
                      'Suicide_lag1', 'HDI_lag1', 'GDP_lag1', 
                      'HDI_growth', 'GDP_growth']
    
    # Add encoded features if available
    if 'continent_encoded' in df_clean.columns:
        feature_columns.append('continent_encoded')
    if 'income_encoded' in df_clean.columns:
        feature_columns.append('income_encoded')
    
    available_features = [f for f in feature_columns if f in df_clean.columns]
    
    # Train models for different targets
    targets = ['Suicide_rate', 'HDI', 'GDP_per_capita']
    
    for target in targets:
        training_data = df_clean.dropna(subset=available_features + [target])
        if len(training_data) > 50:  # Minimum samples
            X = training_data[available_features]
            y = training_data[target]
            
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            model.fit(X, y)
            models[target] = {
                'model': model,
                'features': available_features,
                'r2': model.score(X, y)  # Simple R² for info
            }
    
    return models

def predict_future(df_clean, models, target_variable, target_year, scenario):
    """Generate predictions for a specific year and scenario"""
    
    # Scenario parameters
    scenario_params = {
        'Pessimistic': {'hdi_growth': 0.001, 'gdp_growth': 0.005, 'stress_factor': 0.01},
        'Baseline': {'hdi_growth': 0.005, 'gdp_growth': 0.02, 'stress_factor': 0.005},
        'Optimistic': {'hdi_growth': 0.01, 'gdp_growth': 0.04, 'stress_factor': 0.002}
    }
    
    params = scenario_params[scenario]
    latest_year = df_clean['Year'].max()
    years_ahead = target_year - latest_year
    
    if years_ahead <= 0:
        st.warning("Target year must be in the future!")
        return df_clean[df_clean['Year'] == latest_year]
    
    predictions = []
    
    for country in df_clean['Country Name'].unique():
        country_data = df_clean[df_clean['Country Name'] == country].sort_values('Year')
        if len(country_data) < 2:
            continue
            
        latest = country_data.iloc[-1]
        
        # Project HDI and GDP based on scenario
        projected_hdi = min(1.0, latest['HDI'] * (1 + params['hdi_growth']) ** years_ahead)
        projected_gdp = latest['GDP_per_capita'] * (1 + params['gdp_growth']) ** years_ahead
        
        # Prepare features for prediction
        pred_features = {}
        for feature in models[target_variable]['features']:
            if feature == 'HDI':
                pred_features[feature] = projected_hdi
            elif feature == 'HDI_sq':
                pred_features[feature] = projected_hdi ** 2
            elif feature == 'GDP_per_capita':
                pred_features[feature] = projected_gdp
            elif feature == 'log_GDP':
                pred_features[feature] = np.log(projected_gdp + 1)
            elif feature == 'Suicide_lag1':
                pred_features[feature] = latest['Suicide_rate']
            elif feature == 'HDI_lag1':
                pred_features[feature] = latest['HDI']
            elif feature == 'GDP_lag1':
                pred_features[feature] = latest['GDP_per_capita']
            elif feature == 'HDI_growth':
                pred_features[feature] = params['hdi_growth']
            elif feature == 'GDP_growth':
                pred_features[feature] = params['gdp_growth']
            else:
                pred_features[feature] = latest.get(feature, 0)
        
        # Create feature array and predict
        feature_array = np.array([[pred_features.get(f, 0) for f in models[target_variable]['features']]])
        
        try:
            if target_variable == 'Suicide_rate':
                predicted_value = models[target_variable]['model'].predict(feature_array)[0]
                # Apply stress factor for suicide rate predictions
                predicted_value = max(0, predicted_value * (1 + params['stress_factor'] * years_ahead))
            else:
                predicted_value = models[target_variable]['model'].predict(feature_array)[0]
        except:
            predicted_value = latest[target_variable]
        
        predictions.append({
            'Country Name': country,
            'ISO3': latest.get('ISO3', ''),
            'Year': target_year,
            'Predicted_Value': predicted_value,
            'Scenario': scenario,
            'Continent': latest.get('continent', 'Unknown'),
            'Income_Group': latest.get('income_group_auto', 'Unknown'),
            'Current_Value': latest[target_variable],
            'Projected_HDI': projected_hdi,
            'Projected_GDP': projected_gdp
        })
    
    return pd.DataFrame(predictions)

def create_world_map(predictions_df, target_variable, target_year, scenario):
    """Create interactive world map visualization"""
    
    if 'ISO3' not in predictions_df.columns or predictions_df['ISO3'].isna().all():
        st.warning("No ISO3 codes available for mapping. Showing data table instead.")
        st.dataframe(predictions_df[['Country Name', 'Predicted_Value', 'Current_Value']].head(10))
        return None
    
    # Determine color scale based on variable
    if target_variable == 'Suicide_rate':
        color_scale = 'Reds'
        title_suffix = 'Suicide Rate (per 100,000)'
    elif target_variable == 'HDI':
        color_scale = 'Viridis'
        title_suffix = 'Human Development Index'
    else:  # GDP_per_capita
        color_scale = 'Greens'
        title_suffix = 'GDP per Capita (USD)'
    
    fig = px.choropleth(
        predictions_df,
        locations="ISO3",
        color="Predicted_Value",
        hover_name="Country Name",
        hover_data={
            'Predicted_Value': ':.3f',
            'Current_Value': ':.3f',
            'Projected_HDI': ':.3f',
            'Projected_GDP': ':,.0f',
            'Continent': True,
            'Income_Group': True
        },
        color_continuous_scale=color_scale,
        title=f"Predicted {title_suffix} in {target_year}<br>{scenario} Scenario",
        projection="natural earth",
        width=1200,
        height=700
    )
    
    fig.update_layout(coloraxis_colorbar=dict(title=title_suffix))
    return fig

def create_comparison_chart(predictions_df, target_variable):
    """Create bar chart comparing current vs predicted values"""
    
    # Calculate changes
    predictions_df['Change'] = predictions_df['Predicted_Value'] - predictions_df['Current_Value']
    predictions_df['Abs_Change'] = predictions_df['Change'].abs()  # Absolute value for sorting
    predictions_df['Percent_Change'] = (predictions_df['Change'] / predictions_df['Current_Value']) * 100
    
    # Get top 20 countries with largest absolute changes - FIXED: use column name instead of calculation
    top_changes = predictions_df.nlargest(20, 'Abs_Change')
    
    fig = go.Figure()
    
    # Add current values
    fig.add_trace(go.Bar(
        name='Current',
        x=top_changes['Country Name'],
        y=top_changes['Current_Value'],
        marker_color='lightblue',
        hovertemplate='<b>%{x}</b><br>Current: %{y:.3f}<extra></extra>'
    ))
    
    # Add predicted values
    fig.add_trace(go.Bar(
        name='Predicted',
        x=top_changes['Country Name'],
        y=top_changes['Predicted_Value'],
        marker_color='coral',
        hovertemplate='<b>%{x}</b><br>Predicted: %{y:.3f}<br>Change: %{customdata:.3f}<extra></extra>',
        customdata=top_changes['Change']
    ))
    
    # Determine y-axis title
    if target_variable == 'Suicide_rate':
        y_title = 'Suicide Rate (per 100,000)'
    elif target_variable == 'HDI':
        y_title = 'Human Development Index'
    else:
        y_title = 'GDP per Capita (USD)'
    
    fig.update_layout(
        title=f"Top 20 Countries with Largest Changes: Current vs Predicted {target_variable.replace('_', ' ').title()}",
        xaxis_title="Country",
        yaxis_title=y_title,
        barmode='group',
        xaxis_tickangle=-45,
        height=500
    )
    
    return fig

def main():
    """Main Streamlit app"""
    
    st.markdown('<h1 class="main-header">🌍 Future Predictions Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("Interactive world map predictions for HDI, Suicide Rates, and GDP")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        if df is None:
            st.stop()
        
        df_clean = prepare_features(df)
        models = train_models(df_clean)
    
    # Sidebar controls
    st.sidebar.title("🎯 Prediction Settings")
    
    # Target variable selection
    target_variable = st.sidebar.selectbox(
        "Select Variable to Predict:",
        options=['Suicide_rate', 'HDI', 'GDP_per_capita'],
        format_func=lambda x: {
            'Suicide_rate': 'Suicide Rate',
            'HDI': 'Human Development Index', 
            'GDP_per_capita': 'GDP per Capita'
        }[x]
    )
    
    # Check if model exists for selected variable
    if target_variable not in models:
        st.error(f"No trained model available for {target_variable}. Please try another variable.")
        st.stop()
    
    # Year selection
    current_year = df_clean['Year'].max()
    target_year = st.sidebar.slider(
        "Target Year for Prediction:",
        min_value=current_year + 1,
        max_value=2100,
        value=2050,
        step=5
    )
    
    # Scenario selection
    scenario = st.sidebar.selectbox(
        "Select Scenario:",
        options=['Pessimistic', 'Baseline', 'Optimistic'],
        help="Pessimistic: Slow growth, high stress | Baseline: Current trends | Optimistic: Rapid growth, low stress"
    )
    
    # Model info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Information")
    if target_variable in models:
        st.sidebar.metric("Model R² Score", f"{models[target_variable]['r2']:.3f}")
    st.sidebar.metric("Countries in Dataset", df_clean['Country Name'].nunique())
    st.sidebar.metric("Latest Data Year", current_year)
    
    # Generate predictions
    with st.spinner(f"Generating {scenario} scenario predictions for {target_year}..."):
        predictions_df = predict_future(df_clean, models, target_variable, target_year, scenario)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_avg = predictions_df['Current_Value'].mean()
        st.metric("Current Global Average", f"{current_avg:.2f}")
    
    with col2:
        predicted_avg = predictions_df['Predicted_Value'].mean()
        st.metric("Predicted Global Average", f"{predicted_avg:.2f}")
    
    with col3:
        change = predicted_avg - current_avg
        st.metric("Change", f"{change:+.2f}")
    
    with col4:
        pct_change = (change / current_avg) * 100 if current_avg != 0 else 0
        st.metric("Percent Change", f"{pct_change:+.1f}%")
    
    # World map visualization
    st.markdown("---")
    st.subheader("🌍 World Map Prediction")
    
    fig_map = create_world_map(predictions_df, target_variable, target_year, scenario)
    if fig_map:
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Displaying data in table format (no geographic coordinates available)")
        st.dataframe(predictions_df[['Country Name', 'Predicted_Value', 'Current_Value', 'Change', 'Percent_Change']].sort_values('Change', ascending=False))
    
    # Comparison chart
    st.markdown("---")
    st.subheader("📊 Country Comparison")
    
    fig_comparison = create_comparison_chart(predictions_df, target_variable)
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Data table - FIXED SECTION
    st.markdown("---")
    st.subheader("📋 Detailed Predictions")

    # Filter options for table - FIXED: Handle NaN values
    col1, col2 = st.columns(2)
    with col1:
        # Get unique continents, handle NaN values
        continent_options = predictions_df['Continent'].dropna().unique()
        show_continent = st.multiselect(
            "Filter by Continent:",
            options=sorted(continent_options),
            default=sorted(continent_options)
        )

    with col2:
        # Get unique income groups, handle NaN values
        income_options = predictions_df['Income_Group'].dropna().unique()
        show_income = st.multiselect(
            "Filter by Income Group:",
            options=sorted(income_options),
            default=sorted(income_options)
        )

    # Apply filters - FIXED: Handle NaN values in filtering
    # Option 1: Include rows with NaN values
    filtered_df = predictions_df[
        (predictions_df['Continent'].isin(show_continent) | predictions_df['Continent'].isna()) & 
        (predictions_df['Income_Group'].isin(show_income) | predictions_df['Income_Group'].isna())
    ]

    # Option 2: If you prefer to exclude rows with NaN values, use this instead:
    # filtered_df = predictions_df[
    #     predictions_df['Continent'].notna() & 
    #     predictions_df['Income_Group'].notna() &
    #     predictions_df['Continent'].isin(show_continent) & 
    #     predictions_df['Income_Group'].isin(show_income)
    # ]
    
    # Display table
    display_columns = ['Country Name', 'Continent', 'Income_Group', 'Current_Value', 'Predicted_Value', 'Change', 'Percent_Change']
    display_df = filtered_df[display_columns].sort_values('Change', ascending=False)
    
    st.dataframe(
        display_df.style.format({
            'Current_Value': '{:.3f}',
            'Predicted_Value': '{:.3f}', 
            'Change': '{:+.3f}',
            'Percent_Change': '{:+.1f}%'
        }),
        use_container_width=True
    )
    
    # Download button
    csv_data = predictions_df.to_csv(index=False)
    st.download_button(
        "📥 Download Predictions CSV",
        data=csv_data,
        file_name=f"predictions_{target_variable}_{target_year}_{scenario}.csv",
        mime="text/csv"
    )
    
    # Scenario explanations
    st.markdown("---")
    st.subheader("📖 Scenario Explanations")
    
    scenario_info = {
        'Pessimistic': {
            'description': 'Slow development progress with high economic stress',
            'assumptions': 'HDI growth: 0.1% annually, GDP growth: 0.5% annually, High mental health stress factors'
        },
        'Baseline': {
            'description': 'Continuation of current trends',
            'assumptions': 'HDI growth: 0.5% annually, GDP growth: 2% annually, Moderate mental health stress factors'
        },
        'Optimistic': {
            'description': 'Accelerated development with mental health focus',
            'assumptions': 'HDI growth: 1% annually, GDP growth: 4% annually, Low mental health stress factors'
        }
    }
    
    for scen, info in scenario_info.items():
        with st.expander(f"{scen} Scenario"):
            st.write(f"**Description**: {info['description']}")
            st.write(f"**Key Assumptions**: {info['assumptions']}")

if __name__ == "__main__":
    main()