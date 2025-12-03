import numpy as np
import pandas as pd # Although pandas is not strictly used for data manipulation in this script, it's a requested standard library.
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def run_regression_pipeline_experiment():
    """
    Generates a synthetic regression dataset, creates a pipeline with StandardScaler and Ridge,
    performs GridSearchCV to find the best Ridge alpha, and reports the results.
    """

    # 1. Generate a synthetic regression dataset
    print("1. Generating synthetic regression dataset...")
    X, y = make_regression(n_samples=500, n_features=5, noise=0.5, random_state=42)
    print(f"   Dataset generated: X shape {X.shape}, y shape {y.shape}")
    print(f"   First 5 samples of X:\n{X[:5]}")
    print(f"   First 5 samples of y:\n{y[:5]}")

    # 2. Create an sklearn.pipeline.Pipeline
    print("\n2. Creating machine learning pipeline (StandardScaler -> Ridge)...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])
    print(f"   Pipeline steps: {pipeline.steps}")

    # 3. Define a hyperparameter grid for the Ridge regressor
    print("\n3. Defining hyperparameter grid for Ridge 'alpha'...")
    param_grid = {
        'ridge__alpha': [0.1, 1.0, 10.0, 100.0]
    }
    print(f"   Parameter grid: {param_grid}")

    # 4. Use sklearn.model_selection.GridSearchCV
    print("\n4. Executing GridSearchCV with 3-fold cross-validation and neg_mean_squared_error scoring...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        n_jobs=-1, # Use all available cores
        verbose=1
    )

    grid_search.fit(X, y)
    print("   GridSearchCV fitting complete.")

    # 5. Report the best hyperparameters and the corresponding best score
    print("\n5. Reporting best hyperparameters and best score...")
    best_params = grid_search.best_params_
    best_neg_mse = grid_search.best_score_
    
    # Convert negative MSE to positive MSE
    best_mse = -best_neg_mse

    print(f"   Best Hyperparameters found: {best_params}")
    print(f"   Corresponding Best Mean Squared Error (MSE): {best_mse:.4f}")

if __name__ == "__main__":
    run_regression_pipeline_experiment()