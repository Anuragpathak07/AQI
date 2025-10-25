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
    /* Light mode color scheme */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f0f7 100%);
    }
    
    /* Main title styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        line-height: 1.2;
    }
    
    .subtitle {
        font-size: 1.15rem;
        color: #64748b;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 30px;
        font-weight: 500;
    }
    
    /* AQI Card Styling */
    .aqi-card {
        background: white;
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        text-align: center;
        margin: 20px 0;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .aqi-value {
        font-size: 5rem;
        font-weight: 800;
        margin: 20px 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .aqi-category {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 10px 0;
    }
    
    .dominant-pollutant {
        font-size: 1.1rem;
        color: #64748b;
        margin-top: 15px;
        font-weight: 500;
    }
    
    /* Pollutant card styling */
    .pollutant-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        border-left: 5px solid #667eea;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .pollutant-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .pollutant-name {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    
    .pollutant-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        margin: 8px 0;
    }
    
    .pollutant-unit {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 500;
    }
    
    /* Forecast card styling */
    .forecast-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }
    
    .forecast-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .forecast-date {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .forecast-aqi {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 10px 0;
    }
    
    .forecast-category {
        font-size: 1rem;
        font-weight: 600;
        margin-top: 8px;
    }
    
    /* Section header */
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin: 40px 0 25px 0;
        text-align: center;
    }
    
    /* Error message styling */
    .error-box {
        background: #fee;
        border: 2px solid #f87171;
        border-radius: 12px;
        padding: 20px;
        color: #b91c1c;
        margin: 20px 0;
        font-weight: 500;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Center date input */
    .stDateInput {
        display: flex;
        justify-content: center;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
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
    if aqi <= 50: return '#10b981'  # Good - Green
    elif aqi <= 100: return '#f59e0b'  # Satisfactory - Amber
    elif aqi <= 200: return '#f97316'  # Moderate - Orange
    elif aqi <= 300: return '#ef4444'  # Poor - Red
    elif aqi <= 400: return '#a855f7'  # Very Poor - Purple
    else: return '#1e293b'  # Severe - Dark

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
    st.markdown('<h1 class="main-title">🌫️ AirCast</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Real-time Air Quality Monitoring & Forecast • Nagpur, Maharashtra</p>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Date selector
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected_date = st.date_input(
            "📅 Select Start Date",
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
            st.markdown('<div class="error-box">⚠️ Unable to load models. Please ensure training has been completed and model files exist.</div>', unsafe_allow_html=True)
            return
        
        # Generate predictions
        with st.spinner('🔄 Generating forecast...'):
            forecast_df = predict_forecast(selected_date, 7, models, metadata, historical_data)
        
        # Display current day AQI
        current_aqi = forecast_df.iloc[0]
        aqi_value = current_aqi['AQI']
        aqi_color = get_aqi_color(aqi_value)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="aqi-card">
                <div style="font-size: 1.1rem; color: #94a3b8; font-weight: 600; letter-spacing: 1px;">CURRENT AIR QUALITY INDEX</div>
                <div class="aqi-value" style="color: {aqi_color};">{int(aqi_value)}</div>
                <div class="aqi-category" style="color: {aqi_color};">{current_aqi['Category']}</div>
                <div class="dominant-pollutant">Primary Pollutant: {current_aqi['Dominant']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display individual pollutants
        st.markdown('<div class="section-header">💨 Current Pollutant Levels</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="section-header">📊 7-Day AQI Forecast</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Forecast cards in a row
        cols = st.columns(7)
        for idx, (date, row) in enumerate(forecast_df.iterrows()):
            with cols[idx]:
                aqi = int(row['AQI'])
                color = get_aqi_color(aqi)
                day_name = date.strftime('%a')
                date_str = date.strftime('%m/%d')
                
                st.markdown(f"""
                <div class="forecast-card">
                    <div class="forecast-date">{day_name}</div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 10px;">{date_str}</div>
                    <div class="forecast-aqi" style="color: {color};">{aqi}</div>
                    <div class="forecast-category" style="color: {color}; font-size: 0.8rem;">{row['Category'].split()[0]}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # AQI trend chart
        fig = go.Figure()
        
        # Add area fill
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['AQI'],
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)',
            line=dict(color='rgba(102, 126, 234, 0)', width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add main line
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['AQI'],
            mode='lines+markers',
            line=dict(color='#667eea', width=4),
            marker=dict(
                size=12, 
                color=forecast_df['AQI'].apply(get_aqi_color),
                line=dict(color='white', width=2)
            ),
            name='AQI',
            hovertemplate='<b>%{x|%B %d, %Y}</b><br>AQI: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': "Air Quality Index Trend",
                'font': {'size': 20, 'color': '#1e293b', 'family': 'Arial, sans-serif'}
            },
            xaxis_title="Date",
            yaxis_title="AQI Value",
            height=450,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", color='#64748b'),
            xaxis=dict(
                showgrid=True,
                gridcolor='#f1f5f9',
                showline=True,
                linecolor='#e2e8f0'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#f1f5f9',
                showline=True,
                linecolor='#e2e8f0'
            ),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.markdown(f'<div class="error-box">⚠️ Unable to load forecast data. Please try again later.<br><br>Error: {str(e)}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()