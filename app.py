"""
AirCast - Air Quality Forecast System
Streamlit Frontend Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
# Add this near the top of app.py with other imports
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings("ignore")

# ======================== PAGE CONFIGURATION ========================
st.set_page_config(
    page_title="AirCast - Air Quality Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================== CUSTOM CSS ========================
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 4rem;
        font-weight: 700;
        color: #5DADE2;
        text-align: center;
        margin-bottom: 0;
        line-height: 1.2;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #7F8C8D;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 30px;
    }
    
    .live-indicator {
        color: #27AE60;
        font-size: 0.95rem;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* AQI Card Styling */
    .aqi-card {
        background: white;
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin: 20px 0;
    }
    
    .aqi-value {
        font-size: 4rem;
        font-weight: 700;
        margin: 20px 0;
    }
    
    .aqi-category {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 10px 0;
    }
    
    .dominant-pollutant {
        font-size: 1.1rem;
        color: #7F8C8D;
        margin-top: 10px;
    }
    
    /* Pollutant card styling */
    .pollutant-card {
        background: #F8F9FA;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #5DADE2;
    }
    
    .pollutant-name {
        font-size: 0.9rem;
        color: #7F8C8D;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .pollutant-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2C3E50;
        margin: 5px 0;
    }
    
    .pollutant-unit {
        font-size: 0.85rem;
        color: #95A5A6;
    }
    
    /* Forecast table styling */
    .forecast-table {
        margin-top: 30px;
    }
    
    /* Error message styling */
    .error-box {
        background: #FADBD8;
        border: 1px solid #E74C3C;
        border-radius: 8px;
        padding: 15px;
        color: #C0392B;
        margin: 20px 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Center date input */
    .stDateInput {
        display: flex;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# ======================== CONFIGURATION ========================
DATA_FILE = "MH31.csv"
RESULTS_DIR = "model_results"
TARGETS = ['pm2.5', 'pm10', 'no2', 'so2', 'co', 'ozone']

POLLUTANT_FEATURES = [
    'pm10', 'no', 'no2', 'so2', 'co', 'nox', 'ozone', 
    'at', 'rh', 'ws', 'wd', 'bp',
    'nh3', 'benzene', 'eth_benzene', 'toluene', 'xylene', 'mp_xylene'
]

TIME_FEATURES = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear']

LAGS = {
    'pm2.5': [1, 4, 16, 96],
    'pm10': [1, 4, 16, 96],
    'co': [1, 4, 16, 96],
    'nox': [1, 4, 16, 96]
}

# AQI Breakpoints
BREAKPOINTS = {
    "pm2.5": [(0,30,0,50), (31,60,51,100), (61,90,101,200), (91,120,201,300),
              (121,250,301,400), (251,500,401,500)],
    "pm10": [(0,50,0,50), (51,100,51,100), (101,250,101,200), (251,350,201,300),
             (351,430,301,400), (431,600,401,500)],
    "no2": [(0,40,0,50), (41,80,51,100), (81,180,101,200), (181,280,201,300),
            (281,400,301,400), (401,1000,401,500)],
    "so2": [(0,40,0,50), (41,80,51,100), (81,380,101,200), (381,800,201,300),
            (801,1600,301,400), (1601,2620,401,500)],
    "co": [(0,1,0,50), (1.1,2,51,100), (2.1,10,101,200), (10.1,17,201,300),
           (17.1,34,301,400), (34.1,50,401,500)],
    "ozone": [(0,50,0,50), (51,100,51,100), (101,168,101,200), (169,208,201,300),
              (209,748,301,400), (749,1000,401,500)]
}

POLLUTANT_NAMES = {
    'pm2.5': 'PM2.5',
    'pm10': 'PM10',
    'no2': 'NO₂',
    'so2': 'SO₂',
    'co': 'CO',
    'ozone': 'Ozone'
}

POLLUTANT_UNITS = {
    'pm2.5': 'µg/m³',
    'pm10': 'µg/m³',
    'no2': 'µg/m³',
    'so2': 'µg/m³',
    'co': 'mg/m³',
    'ozone': 'µg/m³'
}

# ======================== HELPER FUNCTIONS ========================
def calculate_sub_index(val, pollutant):
    """Calculate AQI sub-index for a pollutant."""
    for lo, hi, ilo, ihi in BREAKPOINTS[pollutant]:
        if lo <= val <= hi:
            return ((ihi - ilo) / (hi - lo)) * (val - lo) + ilo
    return None

def compute_aqi(row):
    """Compute overall AQI and dominant pollutant."""
    sub_indices = {p: calculate_sub_index(row[p], p) for p in TARGETS}
    valid = {k: v for k, v in sub_indices.items() if v is not None}
    if not valid:
        return np.nan, '-'
    dominant = max(valid, key=valid.get)
    return valid[dominant], dominant.upper()

def get_aqi_color(aqi):
    """Get color based on AQI value."""
    if aqi <= 50: return '#27AE60'  # Good - Green
    elif aqi <= 100: return '#F39C12'  # Satisfactory - Yellow
    elif aqi <= 200: return '#E67E22'  # Moderate - Orange
    elif aqi <= 300: return '#E74C3C'  # Poor - Red
    elif aqi <= 400: return '#8E44AD'  # Very Poor - Purple
    else: return '#2C3E50'  # Severe - Black

def get_aqi_category(aqi):
    """Get AQI category with emoji."""
    if aqi <= 50: return 'Good 🟢'
    elif aqi <= 100: return 'Satisfactory 🟡'
    elif aqi <= 200: return 'Moderate 🟠'
    elif aqi <= 300: return 'Poor 🔴'
    elif aqi <= 400: return 'Very Poor 🟣'
    else: return 'Severe ⚫'

# ======================== DATA LOADING ========================
@st.cache_resource
def load_models():
    """Load trained models and metadata."""
    try:
        metadata = joblib.load(f"{RESULTS_DIR}/model_metadata.joblib")
        models = {}
        for pollutant in TARGETS:
            model_file = f"{RESULTS_DIR}/best_model_{pollutant}.joblib"
            models[pollutant] = joblib.load(model_file)
        return models, metadata
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

@st.cache_data
def load_historical_data():
    """Load and prepare historical data."""
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.lower()
    
    date_col = [c for c in df.columns if "date" in c or "time" in c][0]
    df['datetime'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['datetime']).set_index('datetime').sort_index()
    
    for col in POLLUTANT_FEATURES + TARGETS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.replace(0, np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # Add time features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    
    # Add lag features
    for col, steps in LAGS.items():
        if col in df.columns:
            for s in steps:
                df[f"{col}_lag_{s}"] = df[col].shift(s)
    
    df.dropna(inplace=True)
    return df.select_dtypes(include=[np.number])

# ======================== PREDICTION FUNCTION ========================
def create_feature_row(forecast_date, historical_data, features, feature_means):
    """Create feature vector for prediction."""
    row = pd.Series(index=features, dtype=float)
    
    # Time features
    row['hour'] = 12
    row['dayofweek'] = forecast_date.weekday()
    row['month'] = forecast_date.month
    row['quarter'] = (forecast_date.month - 1) // 3 + 1
    row['year'] = forecast_date.year
    row['dayofyear'] = forecast_date.timetuple().tm_yday
    
    # Pollutant features with variation
    for feature in POLLUTANT_FEATURES:
        if feature in historical_data.columns:
            recent_values = historical_data[feature].iloc[-24:]
            base_value = recent_values.mean()
            
            day_factor = 1.0 + (0.1 if forecast_date.weekday() < 5 else -0.1)
            seasonal_factor = 1.0 + 0.05 * np.sin(2 * np.pi * forecast_date.timetuple().tm_yday / 365)
            noise = np.random.normal(0, 0.05)
            
            row[feature] = base_value * day_factor * seasonal_factor * (1 + noise)
    
    # Lag features
    for col, steps in LAGS.items():
        if col in historical_data.columns:
            for s in steps:
                lag_name = f"{col}_lag_{s}"
                if lag_name in row.index:
                    if len(historical_data) >= s:
                        row[lag_name] = historical_data[col].iloc[-s]
                    else:
                        row[lag_name] = feature_means.get(lag_name, historical_data[col].mean())
    
    for feat in features:
        if pd.isna(row[feat]):
            row[feat] = feature_means.get(feat, 0)
    
    return row

def predict_forecast(start_date, days, models, metadata, historical_data):
    """Generate forecast predictions."""
    scaler = metadata['scaler']
    features = metadata['features']
    feature_means = historical_data[features].mean().to_dict()
    
    current_data = historical_data.tail(200).copy()
    all_predictions = []
    
    for day in range(days):
        forecast_date = start_date + timedelta(days=day)
        feature_row = create_feature_row(forecast_date, current_data, features, feature_means)
        X_scaled = scaler.transform(feature_row.values.reshape(1, -1))
        
        day_predictions = {'date': forecast_date}
        
        for pollutant in TARGETS:
            pred = models[pollutant].predict(X_scaled)[0]
            
            if isinstance(pred, np.ndarray) and len(pred) > 1:
                pollutant_idx = TARGETS.index(pollutant)
                pred = pred[pollutant_idx]
            
            if pollutant == 'co':
                pred = np.expm1(pred)
            
            uncertainty = np.random.normal(1.0, 0.03)
            day_predictions[pollutant] = max(0, pred * uncertainty)
        
        all_predictions.append(day_predictions)
        
        # Update current_data
        new_row = pd.DataFrame([day_predictions]).set_index('date')
        for col in TARGETS:
            if col in new_row.columns:
                current_data.loc[forecast_date, col] = new_row.loc[forecast_date, col]
    
    results_df = pd.DataFrame(all_predictions).set_index('date')
    aqi_results = results_df[TARGETS].apply(compute_aqi, axis=1, result_type='expand')
    results_df['AQI'] = np.round(aqi_results[0])
    results_df['Dominant'] = aqi_results[1]
    results_df['Category'] = results_df['AQI'].apply(get_aqi_category)
    
    return results_df

# ======================== MAIN APP ========================
def main():
    # Header
    st.markdown('<h1 class="main-title">Air Quality Forecast</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Real-time monitoring and 7-day forecast for Nagpur, Maharashtra</p>', unsafe_allow_html=True)
    
    # Date selector
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected_date = st.date_input(
            "Select Date:",
            value=datetime.now(),
            min_value=datetime(2020, 1, 1),
            max_value=datetime.now() + timedelta(days=365)
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Load models and data
    try:
        models, metadata = load_models()
        historical_data = load_historical_data()
        
        if models is None or metadata is None:
            st.markdown('<div class="error-box">⚠️ apiService is not defined - Unable to load models. Please ensure training has been completed.</div>', unsafe_allow_html=True)
            return
        
        # Generate predictions
        with st.spinner('Generating forecast...'):
            forecast_df = predict_forecast(selected_date, 7, models, metadata, historical_data)
        
        # Display current day AQI
        current_aqi = forecast_df.iloc[0]
        aqi_value = current_aqi['AQI']
        aqi_color = get_aqi_color(aqi_value)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="aqi-card">
                <div style="font-size: 1.2rem; color: #7F8C8D;">AQI</div>
                <div class="aqi-value" style="color: {aqi_color};">{int(aqi_value)}</div>
                <div class="aqi-category" style="color: {aqi_color};">{current_aqi['Category']}</div>
                <div class="dominant-pollutant">Dominant Pollutant: {current_aqi['Dominant']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display individual pollutants
        st.subheader("Current Pollutant Levels")
        cols = st.columns(3)
        for idx, pollutant in enumerate(TARGETS):
            with cols[idx % 3]:
                value = current_aqi[pollutant]
                st.markdown(f"""
                <div class="pollutant-card">
                    <div class="pollutant-name">{POLLUTANT_NAMES[pollutant]}</div>
                    <div class="pollutant-value">{value:.1f}</div>
                    <div class="pollutant-unit">{POLLUTANT_UNITS[pollutant]}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # 7-Day Forecast Section
        st.markdown("---")
        st.subheader("7-Day Forecast")
        
        # AQI trend chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['AQI'],
            mode='lines+markers',
            line=dict(color='#5DADE2', width=3),
            marker=dict(size=10, color=forecast_df['AQI'].apply(get_aqi_color)),
            name='AQI'
        ))
        
        fig.update_layout(
            title="AQI Trend",
            xaxis_title="Date",
            yaxis_title="AQI Value",
            height=400,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast table
        st.markdown("### Detailed Forecast")
        display_df = forecast_df.reset_index()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df = display_df[['date'] + TARGETS + ['AQI', 'Dominant', 'Category']]
        
        # Round pollutant values
        for col in TARGETS:
            display_df[col] = display_df[col].round(2)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "date": "Date",
                "pm2.5": "PM2.5 (µg/m³)",
                "pm10": "PM10 (µg/m³)",
                "no2": "NO₂ (µg/m³)",
                "so2": "SO₂ (µg/m³)",
                "co": "CO (mg/m³)",
                "ozone": "Ozone (µg/m³)",
                "AQI": "AQI",
                "Dominant": "Dominant Pollutant",
                "Category": "Air Quality"
            }
        )
        
    except Exception as e:
        st.markdown(f'<div class="error-box">⚠️ Unable to load forecast data. Please try again later.<br><br>Error: {str(e)}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()