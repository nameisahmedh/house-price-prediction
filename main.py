import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score


MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"


def build_pipeline(num_features,cat_features):
    # Create pipelines for numerical and categorical features
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ])

    # Construct the full pipeline
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features),
    ])
    return full_pipeline


if not os.path.exists(MODEL_FILE):
    # TRAINING PHASE

    housing = pd.read_csv("housing.csv")
    housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

    for train_index,test_index in split.split(housing, housing["income_cat"]):
        housing.loc[test_index].drop("income_cat", axis=1).to_csv("input.csv",index=False)
        housing = housing.loc[train_index].drop("income_cat",axis=1)

    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    num_features = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_features = ["ocean_proximity"]

    pipeline = build_pipeline(num_features, cat_features)
    housing_prepared = pipeline.fit_transform(housing_features)
    print(housing_prepared)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)


    # Save model and pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model trained. Congrats!")

else:
    # INFERENCE PHASE

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    print("Model loaded")

    # Load new data
    input_data = pd.read_csv("input.csv")

    # Prepare data
    transformed_input = pipeline.transform(input_data)

    # Make predictions
    predictions = model.predict(transformed_input)
    input_data["median_house_value"] = predictions

    input_data.to_csv("output.csv",index=False)
    print("Inference complete. Results saved to output.csv")