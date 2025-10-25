# ================================================================= #
# Comprehensive AQI Prediction Model Training with EDA & Tuning
# ML Mini Project - Model Development & Comparative Analysis
# ================================================================= 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Sklearn imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Classification & Regression Models
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Optional imports - handle gracefully if not installed
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not available. Install with: pip install catboost")

import joblib
import os

# ======================== CONFIGURATION ========================
DATA_FILE = "MH31.csv"
RESULTS_DIR = "model_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

# ================================================================= #
#                    1. DATA LOADING & CLEANING
# ================================================================= #

def load_and_preprocess_data(file_path):
    """Loads and performs initial data cleaning."""
    print("="*70)
    print("📂 STEP 1: DATA LOADING & INITIAL PREPROCESSING")
    print("="*70)
    
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()
    
    # Detect datetime column
    date_col = [c for c in df.columns if "date" in c or "time" in c]
    if not date_col:
        raise ValueError("❌ No datetime column found in CSV file.")
    
    df['datetime'] = pd.to_datetime(df[date_col[0]], errors='coerce')
    df = df.dropna(subset=['datetime']).set_index('datetime').sort_index()
    
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
    
    # Convert to numeric
    for col in POLLUTANT_FEATURES + TARGETS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle zeros and missing values
    print(f"\n🔍 Missing values before imputation:\n{df[TARGETS].isnull().sum()}")
    df.replace(0, np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    print(f"✅ Missing values handled using forward/backward fill")
    
    return df


# ================================================================= #
#                    2. EXPLORATORY DATA ANALYSIS (EDA)
# ================================================================= #

def perform_eda(df):
    """Comprehensive EDA with visualizations and statistical analysis."""
    print("\n" + "="*70)
    print("📊 STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)
    
    # Basic statistics
    print("\n📈 Descriptive Statistics for Target Pollutants:")
    print(df[TARGETS].describe().round(2))
    
    # Correlation analysis
    print("\n🔗 Correlation Matrix (Target Pollutants):")
    corr_matrix = df[TARGETS].corr()
    print(corr_matrix.round(2))
    
    # Visualizations
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('EDA: Distribution of Target Pollutants', fontsize=16, fontweight='bold')
    
    for idx, pollutant in enumerate(TARGETS):
        ax = axes[idx // 2, idx % 2]
        ax.hist(df[pollutant].dropna(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title(f'{pollutant.upper()} Distribution', fontweight='bold')
        ax.set_xlabel('Concentration')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/eda_distributions.png", dpi=300, bbox_inches='tight')
    print(f"✅ Distribution plots saved to {RESULTS_DIR}/eda_distributions.png")
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap: Target Pollutants', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/eda_correlation.png", dpi=300, bbox_inches='tight')
    print(f"✅ Correlation heatmap saved to {RESULTS_DIR}/eda_correlation.png")
    
    # Time series trends
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('EDA: Time Series Trends', fontsize=16, fontweight='bold')
    
    for idx, pollutant in enumerate(TARGETS):
        ax = axes[idx // 2, idx % 2]
        df[pollutant].resample('D').mean().plot(ax=ax, color='darkblue', linewidth=1.5)
        ax.set_title(f'{pollutant.upper()} - Daily Average', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Concentration')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/eda_timeseries.png", dpi=300, bbox_inches='tight')
    print(f"✅ Time series plots saved to {RESULTS_DIR}/eda_timeseries.png")
    
    plt.close('all')


# ================================================================= #
#                    3. FEATURE ENGINEERING
# ================================================================= #

def create_features(df):
    """Creates time-based and lag features."""
    print("\n" + "="*70)
    print("🛠️  STEP 3: FEATURE ENGINEERING")
    print("="*70)
    
    # Time features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    print(f"✅ Created {len(TIME_FEATURES)} time-based features")
    
    # Lag features
    lag_cols = []
    for col, steps in LAGS.items():
        if col in df.columns:
            for s in steps:
                lag_name = f"{col}_lag_{s}"
                df[lag_name] = df[col].shift(s)
                lag_cols.append(lag_name)
    
    print(f"✅ Created {len(lag_cols)} lag features")
    print(f"📊 Total features before modeling: {len(POLLUTANT_FEATURES) + len(TIME_FEATURES) + len(lag_cols)}")
    
    return df, lag_cols


# ================================================================= #
#                    4. MODEL TRAINING & COMPARISON
# ================================================================= #

def get_all_models():
    """Returns dictionary of all regression models to compare."""
    models = {
        # Linear Models
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=0.1, random_state=42),
        
        # Tree-based Models
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
       
        
        # Advanced Models
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1),
        'LightGBM': LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1, verbose=-1),
        
        # Other Models
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    }
    
    # Add CatBoost only if available
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=6, random_state=42, verbose=False)
    
    return models


def train_and_compare_models(X_train, X_test, y_train, y_test, target_name):
    """Trains all models and compares their performance."""
    models = get_all_models()
    results = []
    
    print(f"\n🎯 Training models for: {target_name.upper()}")
    print("-" * 70)
    
    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R² Score': r2,
                'MSE': mse
            })
            
            print(f"✅ {name:<25} | RMSE: {rmse:>8.3f} | MAE: {mae:>8.3f} | R²: {r2:>6.3f}")
            
        except Exception as e:
            print(f"❌ {name:<25} | Error: {str(e)[:50]}")
    
    return pd.DataFrame(results).sort_values('R² Score', ascending=False)


# ================================================================= #
#                    5. HYPERPARAMETER TUNING
# ================================================================= #

def hyperparameter_tuning(X_train, y_train, model_type='XGBoost'):
    """Performs GridSearchCV for hyperparameter tuning."""
    print("\n" + "="*70)
    print(f"🔧 STEP 5: HYPERPARAMETER TUNING - {model_type}")
    print("="*70)
    
    if model_type == 'XGBoost':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [4, 6, 8],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        base_model = XGBRegressor(random_state=42, n_jobs=-1)
        
    elif model_type == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [8, 12, 16],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
    elif model_type == 'LightGBM':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [4, 6, 8],
            'num_leaves': [31, 50, 70]
        }
        base_model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
    
    print(f"🔍 Searching through {len(param_grid)} hyperparameters...")
    print(f"⏳ This may take several minutes...")
    
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=3, 
        scoring='r2', 
        n_jobs=-1, 
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    print(f"\n🎯 Best cross-validation R² score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


# ================================================================= #
#                    6. MAIN TRAINING PIPELINE
# ================================================================= #

def main_training_pipeline():
    """Complete training pipeline with EDA, model comparison, and tuning."""
    print("\n" + "="*70)
    print("🚀 COMPREHENSIVE AQI MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Load and preprocess
    df = load_and_preprocess_data(DATA_FILE)
    
    # Step 2: EDA
    perform_eda(df)
    
    # Step 3: Feature engineering
    df, lag_cols = create_features(df)
    all_features = [f for f in POLLUTANT_FEATURES if f in df.columns] + TIME_FEATURES + lag_cols
    df.dropna(inplace=True)
    
    # Prepare data
    X = df[all_features]
    Y = df[TARGETS]
    
    print(f"\n📊 Final dataset shape: {X.shape}")
    print(f"✅ Features: {X.shape[1]} | Samples: {X.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 4: Model Training & Comparison
    print("\n" + "="*70)
    print("🎯 STEP 4: MODEL TRAINING & COMPARATIVE ANALYSIS")
    print("="*70)
    
    all_results = {}
    best_models = {}
    
    for target in TARGETS:
        print(f"\n{'='*70}")
        print(f"Target: {target.upper()}")
        print('='*70)
        
        # Transform target if needed
        y = np.log1p(Y[target]) if target == 'co' else Y[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Compare all models
        results_df = train_and_compare_models(X_train, X_test, y_train, y_test, target)
        all_results[target] = results_df
        
        # Save results
        results_df.to_csv(f"{RESULTS_DIR}/comparison_{target}.csv", index=False)
        print(f"\n✅ Results saved to {RESULTS_DIR}/comparison_{target}.csv")
        
        # Get best model name
        best_model_name = results_df.iloc[0]['Model']
        best_models[target] = best_model_name
        print(f"\n🏆 Best model for {target.upper()}: {best_model_name} (R² = {results_df.iloc[0]['R² Score']:.4f})")
    
    # Step 5: Hyperparameter Tuning for top models
    print("\n" + "="*70)
    print("🔧 PERFORMING HYPERPARAMETER TUNING FOR TOP MODELS")
    print("="*70)
    
    tuned_models = {}
    for target in TARGETS[:2]:  # Tune for first 2 targets as example
        y = np.log1p(Y[target]) if target == 'co' else Y[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        best_model, best_params = hyperparameter_tuning(X_train, y_train, 'XGBoost')
        tuned_models[target] = best_model
    
    # Save tuned models
    joblib.dump(tuned_models, f"{RESULTS_DIR}/tuned_models.joblib")
    
    # Step 6: Save best models for each pollutant (for prediction system)
    print("\n" + "="*70)
    print("💾 SAVING BEST MODELS FOR PREDICTION SYSTEM")
    print("="*70)
    
    final_best_models = {}
    for target in TARGETS:
        print(f"\n🔄 Re-training best model for {target.upper()}...")
        
        y = np.log1p(Y[target]) if target == 'co' else Y[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Get the best model type from comparison
        best_model_name = best_models[target]
        models = get_all_models()
        
        if best_model_name in models:
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)
            final_best_models[target] = best_model
            
            # Save individual model
            model_file = f"{RESULTS_DIR}/best_model_{target}.joblib"
            joblib.dump(best_model, model_file)
            print(f"✅ Saved {best_model_name} for {target.upper()} to {model_file}")
        else:
            print(f"⚠️  Could not find model {best_model_name}")
    
    # Save metadata
    joblib.dump({
        'scaler': scaler,
        'features': all_features,
        'best_models': best_models
    }, f"{RESULTS_DIR}/model_metadata.joblib")
    
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📁 All results saved in: {RESULTS_DIR}/")
    print(f"\n📊 CSV Files (Model Comparisons):")
    for target in TARGETS:
        print(f"   - comparison_{target}.csv")
    print(f"\n🖼️  PNG Files (EDA Visualizations):")
    print(f"   - eda_distributions.png")
    print(f"   - eda_correlation.png")
    print(f"   - eda_timeseries.png")
    print(f"\n🔧 Joblib Files (Trained Models):")
    print(f"   - tuned_models.joblib (hyperparameter-tuned models)")
    print(f"   - model_metadata.joblib (scaler + features + best model info)")
    for target in TARGETS:
        print(f"   - best_model_{target}.joblib")
    
    # Summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY - BEST MODELS PER POLLUTANT")
    print("="*70)
    for target, model_name in best_models.items():
        r2_score_val = all_results[target].iloc[0]['R² Score']
        print(f"🎯 {target.upper():<10} | {model_name:<25} | R² = {r2_score_val:.4f}")


# ================================================================= #
#                    EXECUTION
# ================================================================= #

if __name__ == "__main__":
    main_training_pipeline()