"""
Enhanced model utilities with performance metrics display
"""
import os
import joblib
import pandas as pd
import json

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"
METRICS_FILE = "model_metrics.json"


def save_model_metrics(metrics):
    """Save model performance metrics for display in UI"""
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f)


def load_model_metrics():
    """Load saved model performance metrics"""
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


def get_model_info():
    """Get comprehensive model information for UI display"""
    import sklearn
    from model_utils import load_model
    
    model, pipeline = load_model()
    
    if model is None:
        return {
            "status": "not_trained",
            "message": "Model not available"
        }
    
    # Get model name
    model_name = model.__class__.__name__
    
    # Get feature names from pipeline
    feature_names = []
    if pipeline is not None:
        # Numerical features
        num_features = ['longitude', 'latitude', 'housing_median_age', 
                       'total_rooms', 'total_bedrooms', 'population',
                       'households', 'median_income']
        # Categorical features
        cat_features = ['ocean_proximity']
        feature_names = num_features + cat_features
    
    # Load metrics if available
    metrics = load_model_metrics()
    
    info = {
        "status": "ready",
        "model_name": model_name,
        "framework": "scikit-learn " + sklearn.__version__,
        "features_count": len(feature_names),
        "features": feature_names,
        "metrics": metrics
    }
    
    # Add model-specific info
    if hasattr(model, 'n_estimators'):
        info['n_estimators'] = model.n_estimators
    if hasattr(model, 'max_depth') and model.max_depth is not None:
        info['max_depth'] = model.max_depth
    
    return info
