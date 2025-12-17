"""
Model Comparison Script for California Housing Dataset
Compares multiple regression models and selects the best one
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time

# Models to compare
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR

# Try to import XGBoost (optional)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")


def build_pipeline(num_features, cat_features):
    """Build preprocessing pipeline"""
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


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluate a model and return metrics"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print('='*60)
    
    start_time = time.time()
    
    # Train model
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Cross-validation RMSE
    cv_scores = -cross_val_score(model, X_train, y_train, 
                                  scoring="neg_root_mean_squared_error", 
                                  cv=5, n_jobs=-1)
    cv_rmse = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"Training Time: {train_time:.2f}s")
    print(f"\nTraining Set:")
    print(f"  RMSE: ${train_rmse:,.2f}")
    print(f"  R² Score: {train_r2:.4f}")
    print(f"  MAE: ${train_mae:,.2f}")
    print(f"\nTest Set:")
    print(f"  RMSE: ${test_rmse:,.2f}")
    print(f"  R² Score: {test_r2:.4f}")
    print(f"  MAE: ${test_mae:,.2f}")
    print(f"\nCross-Validation (5-fold):")
    print(f"  CV RMSE: ${cv_rmse:,.2f} (±${cv_std:,.2f})")
    
    return {
        'model_name': model_name,
        'model': model,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'cv_rmse': cv_rmse,
        'cv_std': cv_std,
        'train_time': train_time
    }


def main():
    """Main comparison function"""
    print("\n" + "="*60)
    print("CALIFORNIA HOUSING PRICE - MODEL COMPARISON")
    print("="*60)
    
    # Load data
    print("\nLoading dataset...")
    housing = pd.read_csv("housing.csv")
    
    # Create income categories for stratified split
    housing["income_cat"] = pd.cut(
        housing["median_income"], 
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf], 
        labels=[1, 2, 3, 4, 5]
    )
    
    # Stratified split
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
        strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)
    
    # Separate features and labels
    X_train = strat_train_set.drop("median_house_value", axis=1)
    y_train = strat_train_set["median_house_value"].copy()
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    
    # Build pipeline
    num_features = X_train.drop("ocean_proximity", axis=1).columns.tolist()
    cat_features = ["ocean_proximity"]
    pipeline = build_pipeline(num_features, cat_features)
    
    # Transform data
    print("Preprocessing data...")
    X_train_prepared = pipeline.fit_transform(X_train)
    X_test_prepared = pipeline.transform(X_test)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Define models to compare
    models = [
        (LinearRegression(), "Linear Regression"),
        (Ridge(alpha=1.0), "Ridge Regression"),
        (Lasso(alpha=1.0), "Lasso Regression"),
        (DecisionTreeRegressor(random_state=42), "Decision Tree"),
        (RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), 
         "Random Forest (100 trees)"),
        (RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1), 
         "Random Forest (200 trees)"),
        (ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1), 
         "Extra Trees"),
        (GradientBoostingRegressor(n_estimators=100, random_state=42), 
         "Gradient Boosting"),
    ]
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models.append((
            XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "XGBoost"
        ))
    
    # Evaluate all models
    results = []
    for model, name in models:
        result = evaluate_model(model, X_train_prepared, y_train, 
                               X_test_prepared, y_test, name)
        results.append(result)
    
    # Sort by test RMSE
    results_sorted = sorted(results, key=lambda x: x['test_rmse'])
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY - RANKED BY TEST RMSE")
    print("="*60)
    print(f"{'Rank':<5} {'Model':<30} {'Test RMSE':<15} {'Test R²':<10}")
    print("-"*60)
    
    for i, result in enumerate(results_sorted, 1):
        print(f"{i:<5} {result['model_name']:<30} "
              f"${result['test_rmse']:>12,.0f}  {result['test_r2']:>8.4f}")
    
    # Best model
    best_result = results_sorted[0]
    print("\n" + "="*60)
    print(f"🏆 BEST MODEL: {best_result['model_name']}")
    print("="*60)
    print(f"Test RMSE: ${best_result['test_rmse']:,.2f}")
    print(f"Test R² Score: {best_result['test_r2']:.4f}")
    print(f"Test MAE: ${best_result['test_mae']:,.2f}")
    print(f"CV RMSE: ${best_result['cv_rmse']:,.2f} (±${best_result['cv_std']:,.2f})")
    print(f"Training Time: {best_result['train_time']:.2f}s")
    
    # Save best model
    print(f"\nSaving best model ({best_result['model_name']})...")
    joblib.dump(best_result['model'], "model.pkl")
    joblib.dump(pipeline, "pipeline.pkl")
    
    # Save results summary
    results_df = pd.DataFrame([{
        'Model': r['model_name'],
        'Test RMSE': r['test_rmse'],
        'Test R²': r['test_r2'],
        'Test MAE': r['test_mae'],
        'CV RMSE': r['cv_rmse'],
        'CV Std': r['cv_std'],
        'Train Time (s)': r['train_time']
    } for r in results_sorted])
    
    results_df.to_csv("model_comparison_results.csv", index=False)
    print("Results saved to: model_comparison_results.csv")
    
    print("\n✅ Model comparison complete!")
    return best_result


if __name__ == "__main__":
    main()
