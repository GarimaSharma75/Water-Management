import pandas as pd
import numpy as np
from crop_water_predictor import CropWaterPredictor

# Create sample data
dates = pd.date_range(start='2023-01-01', periods=1000)
historical_data = pd.DataFrame({
    'date': dates,
    'temperature': np.random.normal(25, 5, 1000),
    'humidity': np.random.normal(60, 10, 1000).clip(0, 100),
    'rainfall': np.random.exponential(5, 1000),
    'soil_moisture': np.random.normal(30, 5, 1000).clip(0, 100),
    'soil_type': np.random.choice(['sandy', 'loamy', 'clay'], 1000),
    'crop_type': np.random.choice(['corn', 'wheat', 'soybeans'], 1000),
    'planting_date': dates - pd.Timedelta(days=30),
    'water_requirement': np.random.normal(50, 10, 1000)
})

# Initialize and train
predictor = CropWaterPredictor(model_type='rf')
metrics = predictor.train(historical_data)

# Print metrics
print("\nTraining Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.3f}")

# Plot feature importances
predictor.plot_feature_importances()

# Make predictions
current_data = pd.DataFrame({
    'field_id': [1, 2],
    'date': ['2024-01-01', '2024-01-01'],
    'temperature': [24.5, 26.2],
    'humidity': [65.2, 58.7],
    'rainfall': [2.5, 0.0],
    'soil_moisture': [28.5, 25.2],
    'soil_type': ['sandy', 'loamy'],
    'crop_type': ['corn', 'wheat'],
    'planting_date': ['2023-12-01', '2023-12-01']
})

predictions = predictor.predict_water_needs(current_data)
print("\nPredictions:")
print(predictions[[
    'field_id', 'crop_type', 'predicted_water_requirement',
    'prediction_lower', 'prediction_upper'
]])