import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
import xgboost as xgb
import warnings
from typing import Dict, List, Union, Tuple

class CropWaterPredictor:
    def __init__(self, model_type: str = 'rf'):
        """
        Initialize the predictor with choice of model
        
        Args:
            model_type (str): 'rf' for Random Forest or 'xgb' for XGBoost
        """
        self.model = None
        self.scaler = StandardScaler()
        self.model_type = model_type
        self.feature_importances = None
        self.feature_names = None
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features from raw data with additional environmental factors
        """
        # Extract temporal features
        data['month'] = pd.to_datetime(data['date']).dt.month
        data['season'] = pd.to_datetime(data['date']).dt.quarter
        data['days_since_planting'] = (pd.to_datetime(data['date']) - 
                                     pd.to_datetime(data['planting_date'])).dt.days
        
        # Calculate derived features
        data['vapor_pressure_deficit'] = self._calculate_vpd(
            data['temperature'], 
            data['humidity']
        )
        
        # Define feature columns including new ones
        feature_columns = [
            'temperature', 'humidity', 'rainfall', 'soil_moisture',
            'month', 'season', 'days_since_planting', 'vapor_pressure_deficit'
        ]
        
        # Add soil type if available
        if 'soil_type' in data.columns:
            soil_dummies = pd.get_dummies(data['soil_type'], prefix='soil')
            feature_columns.extend(soil_dummies.columns)
        
        # Create crop type dummy variables
        crop_dummies = pd.get_dummies(data['crop_type'], prefix='crop')
        features = pd.concat([data[feature_columns], crop_dummies], axis=1)
        
        self.feature_names = features.columns
        return features
    
    def _calculate_vpd(self, temperature: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate Vapor Pressure Deficit"""
        # Convert temperature to Celsius if needed
        saturation_pressure = 0.611 * np.exp((17.27 * temperature) / (temperature + 237.3))
        actual_pressure = saturation_pressure * (humidity / 100)
        return saturation_pressure - actual_pressure

    def train(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        Train the model with enhanced validation and metrics
        
        Returns:
            Dict containing various performance metrics
        """
        # Validate input data
        self._validate_input_data(historical_data)
        
        # Prepare features and target
        features = self.prepare_features(historical_data)
        target = historical_data['water_requirement']
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            scaled_features, target, test_size=0.2, random_state=42
        )
        
        # Initialize model based on type
        if self.model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'xgb':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=7,
                learning_rate=0.1,
                random_state=42
            )
            
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        metrics = self._calculate_performance_metrics(
            X_train, X_val, y_train, y_val
        )
        
        # Store feature importances
        self.feature_importances = self._get_feature_importances()
        
        return metrics
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data structure and values"""
        required_columns = {
            'date', 'temperature', 'humidity', 'rainfall',
            'soil_moisture', 'crop_type', 'planting_date', 'water_requirement'
        }
        
        if missing_cols := required_columns - set(data.columns):
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Validate data types and ranges
        if not pd.to_datetime(data['date'], errors='coerce').notna().all():
            raise ValueError("Invalid date format in 'date' column")
            
        if (data['humidity'] < 0).any() or (data['humidity'] > 100).any():
            raise ValueError("Humidity values must be between 0 and 100")
            
        if (data['temperature'] < -50).any() or (data['temperature'] > 60).any():
            raise ValueError("Temperature values seem out of realistic range")

    def predict_water_needs(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict water requirements with uncertainty estimates
        """
        if self.model is None:
            raise ValueError("Model needs to be trained first!")
            
        # Validate and prepare features
        self._validate_input_data(current_data)
        features = self.prepare_features(current_data)
        scaled_features = self.scaler.transform(features)
        
        # Make predictions
        predictions = self.model.predict(scaled_features)
        
        # Add predictions to the input data
        result = current_data.copy()
        result['predicted_water_requirement'] = predictions
        
        # Add confidence intervals for Random Forest
        if self.model_type == 'rf':
            predictions_all_trees = np.array([
                tree.predict(scaled_features)
                for tree in self.model.estimators_
            ])
            result['prediction_std'] = predictions_all_trees.std(axis=0)
            result['prediction_lower'] = predictions - 1.96 * result['prediction_std']
            result['prediction_upper'] = predictions + 1.96 * result['prediction_std']
        
        return result

    def _calculate_performance_metrics(
        self, X_train: np.ndarray, X_val: np.ndarray, 
        y_train: np.ndarray, y_val: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        metrics = {
            'train_r2': r2_score(y_train, train_pred),
            'val_r2': r2_score(y_val, val_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'cv_score': np.mean(cross_val_score(self.model, X_train, y_train, cv=5))
        }
        
        return metrics

    def _get_feature_importances(self) -> pd.Series:
        """Get feature importance scores"""
        if self.model_type == 'rf':
            importances = self.model.feature_importances_
        else:  # XGBoost
            importances = self.model.feature_importances_
            
        return pd.Series(importances, index=self.feature_names)

    def plot_feature_importances(self) -> None:
        """Plot feature importance chart"""
        plt.figure(figsize=(12, 6))
        self.feature_importances.sort_values().plot(kind='barh')
        plt.title('Feature Importances')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()

    def plot_learning_curves(self, X: np.ndarray, y: np.ndarray) -> None:
        """Plot learning curves to diagnose bias/variance"""
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'b-', label='Training score')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 'r-', label='Cross-validation score')
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    def plot_predictions_vs_actual(self, actual: np.ndarray, predicted: np.ndarray) -> None:
        """Plot predicted vs actual values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(actual, predicted, alpha=0.5)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        plt.xlabel('Actual Water Requirement')
        plt.ylabel('Predicted Water Requirement')
        plt.title('Predicted vs Actual Values')
        plt.tight_layout()
        plt.show()

def load_and_process_realtime_data(
    sensor_data: pd.DataFrame, 
    crop_info: pd.DataFrame
) -> pd.DataFrame:
    """Process real-time sensor data with enhanced error checking"""
    try:
        # Combine sensor data with crop information
        data = pd.merge(sensor_data, crop_info, on='field_id', how='inner')
        
        # Check for missing values
        if data.isnull().any().any():
            warnings.warn("Missing values detected in the input data")
            
        # Check for duplicate readings
        if data.duplicated(['field_id', 'date']).any():
            warnings.warn("Duplicate readings detected for some fields")
            
        return data
        
    except Exception as e:
        raise ValueError(f"Error processing real-time data: {str(e)}")

# Example usage with enhanced features
if __name__ == "__main__":
    # Generate more realistic sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=1000)
    
    # Simulate seasonal temperature variations
    seasonal_temp = 25 + 10 * np.sin(2 * np.pi * np.arange(1000) / 365)
    
    historical_data = pd.DataFrame({
        'date': dates,
        'temperature': seasonal_temp + np.random.normal(0, 2, 1000),
        'humidity': np.random.normal(60, 10, 1000).clip(0, 100),
        'rainfall': np.random.exponential(5, 1000),
        'soil_moisture': np.random.normal(30, 5, 1000).clip(0, 100),
        'soil_type': np.random.choice(['sandy', 'loamy', 'clay'], 1000),
        'crop_type': np.random.choice(['corn', 'wheat', 'soybeans'], 1000),
        'planting_date': dates - pd.Timedelta(days=30),
        'water_requirement': np.random.normal(50, 10, 1000)
    })
    
    # Initialize and train the model
    predictor = CropWaterPredictor(model_type='rf')
    metrics = predictor.train(historical_data)
    
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    
    # Plot feature importances
    predictor.plot_feature_importances()
    
    # Make predictions with uncertainty estimates
    current_data = pd.DataFrame({
        'field_id': [1, 2, 3],
        'date': pd.date_range(start='2024-01-01', periods=3),
        'temperature': [24.5, 26.2, 23.8],
        'humidity': [65.2, 58.7, 62.1],
        'rainfall': [2.5, 0.0, 1.2],
        'soil_moisture': [28.5, 25.2, 30.1],
        'soil_type': ['sandy', 'loamy', 'clay'],
        'crop_type': ['corn', 'wheat', 'soybeans'],
        'planting_date': ['2024-01-01', '2024-01-01', '2024-01-01']
    })
    
    predictions = predictor.predict_water_needs(current_data)
    print("\nPredictions with Uncertainty Estimates:")
    print(predictions[[
        'field_id', 'crop_type', 'predicted_water_requirement',
        'prediction_lower', 'prediction_upper'
    ]])