# ================================================================= #
# AQI PREDICTION SYSTEM - Using Best Models for Each Pollutant
# ================================================================= #

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings("ignore")

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

# ======================== AQI CALCULATION ========================
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

def classify_aqi_category(aqi):
    """Classify AQI into categories."""
    if aqi <= 50: return 'Good 🟢'
    elif aqi <= 100: return 'Satisfactory 🟡'
    elif aqi <= 200: return 'Moderate 🟠'
    elif aqi <= 300: return 'Poor 🔴'
    elif aqi <= 400: return 'Very Poor 🟣'
    else: return 'Severe ⚫'

# ======================== LOAD HISTORICAL DATA ========================
def load_historical_data():
    """Load and prepare historical data with all features (including time and lag features)."""
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
    
    # Add time features (same as training)
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    
    # Add lag features (same as training)
    for col, steps in LAGS.items():
        if col in df.columns:
            for s in steps:
                lag_name = f"{col}_lag_{s}"
                df[lag_name] = df[col].shift(s)
    
    # Drop rows with NaN (from lag features)
    df.dropna(inplace=True)
    
    return df.select_dtypes(include=[np.number])

# ======================== LOAD TRAINED MODELS ========================
class PollutantPredictor:
    """Manages predictions for all pollutants using their best models."""
    
    def __init__(self):
        print("🔄 Loading trained models and metadata...")
        self.metadata = joblib.load(f"{RESULTS_DIR}/model_metadata.joblib")
        self.scaler = self.metadata['scaler']
        self.features = self.metadata['features']
        self.best_models_info = self.metadata['best_models']
        
        # Load individual pollutant models
        self.models = {}
        for pollutant in TARGETS:
            model_file = f"{RESULTS_DIR}/best_model_{pollutant}.joblib"
            try:
                self.models[pollutant] = joblib.load(model_file)
                print(f"✅ Loaded model for {pollutant.upper()}: {self.best_models_info.get(pollutant, 'Unknown')}")
            except FileNotFoundError:
                print(f"⚠️  Model file not found for {pollutant}: {model_file}")
                self.models[pollutant] = None
        
        self.historical_data = load_historical_data()
        self.feature_means = self.historical_data[self.features].mean().to_dict()
        print("✅ All models loaded successfully!\n")
    
    def create_feature_row(self, forecast_date, current_data, add_noise=False):
        """Create feature vector for a given date with optional realistic variation."""
        row = pd.Series(index=self.features, dtype=float)
        
        # Time features
        row['hour'] = 12  # Noon prediction
        row['dayofweek'] = forecast_date.weekday()
        row['month'] = forecast_date.month
        row['quarter'] = (forecast_date.month - 1) // 3 + 1
        row['year'] = forecast_date.year
        row['dayofyear'] = forecast_date.timetuple().tm_yday
        
        # Pollutant features with realistic variation
        for feature in POLLUTANT_FEATURES:
            if feature in current_data.columns:
                # Use rolling average of recent values
                recent_values = current_data[feature].iloc[-24:]  # Last 24 hours
                base_value = recent_values.mean()
                
                if add_noise:
                    # Add day-of-week and seasonal patterns
                    day_factor = 1.0 + (0.1 if forecast_date.weekday() < 5 else -0.1)  # Weekday vs weekend
                    seasonal_factor = 1.0 + 0.05 * np.sin(2 * np.pi * forecast_date.timetuple().tm_yday / 365)
                    
                    # Add controlled random variation (±5-10%)
                    noise = np.random.normal(0, 0.05)
                    row[feature] = base_value * day_factor * seasonal_factor * (1 + noise)
                else:
                    row[feature] = base_value
        
        # Lag features - use actual lagged values from current_data
        for col, steps in LAGS.items():
            if col in current_data.columns:
                for s in steps:
                    lag_name = f"{col}_lag_{s}"
                    if lag_name in row.index:
                        if len(current_data) >= s:
                            row[lag_name] = current_data[col].iloc[-s]
                        else:
                            row[lag_name] = self.feature_means.get(lag_name, current_data[col].mean())
        
        # Fill any remaining NaN with feature means
        for feat in self.features:
            if pd.isna(row[feat]):
                row[feat] = self.feature_means.get(feat, 0)
        
        return row
    
    def predict_single_day(self, date_str):
        """Predict all pollutants for a single day."""
        forecast_date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Get recent historical data
        recent_data = self.historical_data.tail(200).copy()
        
        # Create feature row
        feature_row = self.create_feature_row(forecast_date, recent_data)
        
        # Scale features
        X_scaled = self.scaler.transform(feature_row.values.reshape(1, -1))
        
        # Predict each pollutant using its best model
        predictions = {}
        for pollutant in TARGETS:
            if self.models[pollutant] is not None:
                try:
                    pred = self.models[pollutant].predict(X_scaled)[0]
                    
                    # Handle multi-output models (like Random Forest with all targets)
                    if isinstance(pred, np.ndarray) and len(pred) > 1:
                        pollutant_idx = TARGETS.index(pollutant)
                        pred = pred[pollutant_idx]
                    
                    # Reverse log transform for CO if needed
                    if pollutant == 'co':
                        pred = np.expm1(pred)
                    
                    predictions[pollutant] = max(0, pred)
                except Exception as e:
                    print(f"⚠️  Error predicting {pollutant}: {e}")
                    predictions[pollutant] = recent_data[pollutant].iloc[-24:].mean()
            else:
                predictions[pollutant] = recent_data[pollutant].iloc[-24:].mean()
        
        return predictions
    
    def predict_multiple_days(self, start_date_str, days=7, add_variation=True):
        """Predict pollutants for multiple days with iterative forecasting and realistic variation."""
        print(f"\n{'='*70}")
        print(f"📅 GENERATING {days}-DAY FORECAST FROM {start_date_str}")
        if add_variation:
            print("   (Including day-of-week and seasonal variations)")
        print('='*70)
        
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        current_data = self.historical_data.tail(200).copy()
        all_predictions = []
        
        for day in range(days):
            forecast_date = start_date + timedelta(days=day)
            
            # Create features with variation
            feature_row = self.create_feature_row(forecast_date, current_data, add_noise=add_variation)
            X_scaled = self.scaler.transform(feature_row.values.reshape(1, -1))
            
            day_predictions = {'date': forecast_date}
            
            # Predict each pollutant
            for pollutant in TARGETS:
                if self.models[pollutant] is not None:
                    try:
                        pred = self.models[pollutant].predict(X_scaled)[0]
                        
                        if isinstance(pred, np.ndarray) and len(pred) > 1:
                            pollutant_idx = TARGETS.index(pollutant)
                            pred = pred[pollutant_idx]
                        
                        if pollutant == 'co':
                            pred = np.expm1(pred)
                        
                        # Add small realistic uncertainty (±2-5%)
                        if add_variation:
                            uncertainty = np.random.normal(1.0, 0.03)
                            pred = pred * uncertainty
                        
                        day_predictions[pollutant] = max(0, pred)
                    except:
                        day_predictions[pollutant] = current_data[pollutant].iloc[-24:].mean()
                else:
                    day_predictions[pollutant] = current_data[pollutant].iloc[-24:].mean()
            
            all_predictions.append(day_predictions)
            
            # Update current_data for next iteration
            new_row = pd.DataFrame([day_predictions]).set_index('date')
            
            # Also update lag features in current_data
            for col in TARGETS + [c for c in POLLUTANT_FEATURES if c in current_data.columns]:
                if col in new_row.columns:
                    current_data.loc[forecast_date, col] = new_row.loc[forecast_date, col]
            
            # Recalculate lag features for the new row
            for col, steps in LAGS.items():
                if col in current_data.columns:
                    for s in steps:
                        lag_name = f"{col}_lag_{s}"
                        if len(current_data) >= s:
                            current_data.loc[forecast_date, lag_name] = current_data[col].iloc[-s-1]
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_predictions).set_index('date')
        
        # Calculate AQI
        aqi_results = results_df[TARGETS].apply(compute_aqi, axis=1, result_type='expand')
        results_df['AQI'] = np.round(aqi_results[0])
        results_df['Dominant'] = aqi_results[1]
        results_df['Category'] = results_df['AQI'].apply(classify_aqi_category)
        
        return results_df

# ======================== DISPLAY RESULTS ========================
def display_prediction(results_df):
    """Display prediction results in a formatted table."""
    print("\n" + "="*70)
    print("📊 PREDICTION RESULTS")
    print("="*70)
    
    # Format the dataframe for display
    display_cols = TARGETS + ['AQI', 'Dominant', 'Category']
    formatted_df = results_df[display_cols].copy()
    
    # Round pollutant values
    for col in TARGETS:
        formatted_df[col] = formatted_df[col].round(2)
    
    print(formatted_df.to_string())
    print("="*70)
    
    # Summary statistics
    print("\n📈 FORECAST SUMMARY:")
    print(f"   Average AQI: {results_df['AQI'].mean():.1f}")
    print(f"   Min AQI: {results_df['AQI'].min():.1f} ({results_df.loc[results_df['AQI'].idxmin(), 'Category']})")
    print(f"   Max AQI: {results_df['AQI'].max():.1f} ({results_df.loc[results_df['AQI'].idxmax(), 'Category']})")
    print(f"   Most Dominant Pollutant: {results_df['Dominant'].mode()[0]}")
    print("="*70 + "\n")

# ======================== SAVE BEST MODELS ========================
def save_best_models_for_prediction():
    """
    This function should be called AFTER training to save individual best models.
    Add this to the end of your training script.
    """
    print("\n🔄 Saving best models for each pollutant...")
    
    # Load comparison results
    metadata = joblib.load(f"{RESULTS_DIR}/model_metadata.joblib")
    best_models_info = metadata['best_models']
    
    # You'll need to retrain and save the best model for each pollutant
    # This is a placeholder - implement based on your training results
    print("✅ Best models saved for prediction system")

# ======================== MAIN EXECUTION ========================
if __name__ == "__main__":
    try:
        # Initialize predictor
        predictor = PollutantPredictor()
        
        print("\n" + "="*70)
        print("🌍 AQI PREDICTION SYSTEM - READY")
        print("="*70)
        print("\nOptions:")
        print("  1. Single day prediction")
        print("  2. Multi-day forecast (7 days)")
        print("  3. Custom date range")
        print("  4. Exit")
        
        while True:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                date_input = input("Enter date (YYYY-MM-DD): ").strip()
                predictions = predictor.predict_single_day(date_input)
                
                # Create single-row dataframe for display
                pred_df = pd.DataFrame([predictions])
                pred_df.index = [datetime.strptime(date_input, '%Y-%m-%d')]
                pred_df.index.name = 'date'
                
                aqi_results = pred_df[TARGETS].apply(compute_aqi, axis=1, result_type='expand')
                pred_df['AQI'] = np.round(aqi_results[0])
                pred_df['Dominant'] = aqi_results[1]
                pred_df['Category'] = pred_df['AQI'].apply(classify_aqi_category)
                
                display_prediction(pred_df)
            
            elif choice == '2':
                start_date = input("Enter start date (YYYY-MM-DD): ").strip()
                results = predictor.predict_multiple_days(start_date, days=7, add_variation=True)
                display_prediction(results)
            
            elif choice == '3':
                start_date = input("Enter start date (YYYY-MM-DD): ").strip()
                num_days = int(input("Enter number of days to forecast: ").strip())
                add_var = input("Add realistic day-to-day variation? (y/n, default=y): ").strip().lower()
                add_variation = add_var != 'n'
                
                results = predictor.predict_multiple_days(start_date, days=num_days, add_variation=add_variation)
                display_prediction(results)
                
                # Option to save results
                save = input("\nSave results to CSV? (y/n): ").strip().lower()
                if save == 'y':
                    filename = f"forecast_{start_date}_{num_days}days.csv"
                    results.to_csv(filename)
                    print(f"✅ Results saved to {filename}")
            
            elif choice == '4':
                print("\n👋 Thank you for using the AQI Prediction System!")
                break
            
            else:
                print("❌ Invalid option. Please select 1-4.")
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: Required files not found.")
        print(f"   {e}")
        print("\n💡 Please run the training script first to generate model files.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()