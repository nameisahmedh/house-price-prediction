"""
Model utility functions for California House Prediction
Handles model loading, predictions, and data preprocessing
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"


def build_pipeline(num_features, cat_features):
    """
    Build preprocessing pipeline for data transformation
    
    Args:
        num_features: List of numerical feature column names
        cat_features: List of categorical feature column names
    
    Returns:
        ColumnTransformer pipeline
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ])

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features),
    ])
    
    return full_pipeline


def load_model():
    """
    Load trained model and preprocessing pipeline
    
    Returns:
        tuple: (model, pipeline) or (None, None) if files don't exist
    """
    if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
        return None, None
    
    try:
        model = joblib.load(MODEL_FILE)
        pipeline = joblib.load(PIPELINE_FILE)
        return model, pipeline
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def train_model():
    """
    Train a new model on the housing dataset
    
    Returns:
        dict: Training results with status and message
    """
    try:
        # Load dataset
        housing = pd.read_csv("housing.csv")
        housing["income_cat"] = pd.cut(
            housing["median_income"], 
            bins=[0., 1.5, 3.0, 4.5, 6., np.inf], 
            labels=[1, 2, 3, 4, 5]
        )

        # Create stratified split
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        for train_index, test_index in split.split(housing, housing["income_cat"]):
            housing.loc[test_index].drop("income_cat", axis=1).to_csv("input.csv", index=False)
            housing = housing.loc[train_index].drop("income_cat", axis=1)

        # Separate features and labels
        housing_labels = housing["median_house_value"].copy()
        housing_features = housing.drop("median_house_value", axis=1)

        # Define feature columns
        num_features = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
        cat_features = ["ocean_proximity"]

        # Build and fit pipeline
        pipeline = build_pipeline(num_features, cat_features)
        housing_prepared = pipeline.fit_transform(housing_features)

        # Train model
        model = RandomForestRegressor(random_state=42, n_estimators=100)
        model.fit(housing_prepared, housing_labels)

        # Save model and pipeline
        joblib.dump(model, MODEL_FILE)
        joblib.dump(pipeline, PIPELINE_FILE)

        return {
            "status": "success",
            "message": "Model trained successfully!",
            "samples": len(housing_features)
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Training failed: {str(e)}"
        }


def predict_single(features):
    """
    Make prediction for a single input
    
    Args:
        features: Dictionary with feature values
    
    Returns:
        dict: Prediction result
    """
    model, pipeline = load_model()
    
    if model is None or pipeline is None:
        return {
            "status": "error",
            "message": "Model not found. Please train the model first."
        }
    
    try:
        # Create DataFrame from input features
        df = pd.DataFrame([features])
        
        # Validate required columns
        required_columns = [
            'longitude', 'latitude', 'housing_median_age', 
            'total_rooms', 'total_bedrooms', 'population',
            'households', 'median_income', 'ocean_proximity'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return {
                "status": "error",
                "message": f"Missing required columns: {', '.join(missing_cols)}"
            }
        
        # Transform and predict
        transformed = pipeline.transform(df)
        prediction = model.predict(transformed)[0]
        
        return {
            "status": "success",
            "prediction": float(prediction),
            "formatted_prediction": f"${prediction:,.2f}"
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }


def predict_batch(dataframe):
    """
    Make predictions for multiple rows
    
    Args:
        dataframe: pandas DataFrame with features
    
    Returns:
        dict: Batch prediction results
    """
    model, pipeline = load_model()
    
    if model is None or pipeline is None:
        return {
            "status": "error",
            "message": "Model not found. Please train the model first."
        }
    
    try:
        # Validate columns
        required_columns = [
            'longitude', 'latitude', 'housing_median_age', 
            'total_rooms', 'total_bedrooms', 'population',
            'households', 'median_income', 'ocean_proximity'
        ]
        
        missing_cols = [col for col in required_columns if col not in dataframe.columns]
        if missing_cols:
            return {
                "status": "error",
                "message": f"Missing required columns: {', '.join(missing_cols)}"
            }
        
        # Transform and predict
        transformed = pipeline.transform(dataframe)
        predictions = model.predict(transformed)
        
        # Add predictions to dataframe
        result_df = dataframe.copy()
        result_df['predicted_house_value'] = predictions
        
        return {
            "status": "success",
            "predictions": predictions.tolist(),
            "dataframe": result_df,
            "count": len(predictions),
            "statistics": {
                "mean": float(predictions.mean()),
                "median": float(np.median(predictions)),
                "min": float(predictions.min()),
                "max": float(predictions.max()),
                "std": float(predictions.std())
            }
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Batch prediction failed: {str(e)}"
        }


def get_feature_info():
    """
    Get information about expected input features
    
    Returns:
        dict: Feature information
    """
    return {
        "features": [
            {
                "name": "longitude",
                "description": "Longitude coordinate",
                "type": "float",
                "example": -122.23
            },
            {
                "name": "latitude",
                "description": "Latitude coordinate",
                "type": "float",
                "example": 37.88
            },
            {
                "name": "housing_median_age",
                "description": "Median age of houses in the block",
                "type": "float",
                "example": 41.0
            },
            {
                "name": "total_rooms",
                "description": "Total number of rooms",
                "type": "float",
                "example": 880.0
            },
            {
                "name": "total_bedrooms",
                "description": "Total number of bedrooms",
                "type": "float",
                "example": 129.0
            },
            {
                "name": "population",
                "description": "Population in the block",
                "type": "float",
                "example": 322.0
            },
            {
                "name": "households",
                "description": "Number of households",
                "type": "float",
                "example": 126.0
            },
            {
                "name": "median_income",
                "description": "Median income (in tens of thousands)",
                "type": "float",
                "example": 8.3252
            },
            {
                "name": "ocean_proximity",
                "description": "Proximity to ocean",
                "type": "categorical",
                "options": ["NEAR BAY", "NEAR OCEAN", "<1H OCEAN", "INLAND", "ISLAND"],
                "example": "NEAR BAY"
            }
        ]
    }
