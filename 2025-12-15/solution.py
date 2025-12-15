import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def run_regression_pipeline():
    """
    Generates a synthetic regression dataset, constructs an ML pipeline with
    feature selection, performs hyperparameter tuning using GridSearchCV,
    and reports the best results.
    """

    # 1. Generate synthetic regression dataset
    # Generate 5 informative features and target variable
    X_informative, y = make_regression(
        n_samples=1000,
        n_features=5,
        n_informative=5,
        n_targets=1,
        noise=10,
        random_state=42
    )

    # Generate 5 completely random, uninformative features
    X_uninformative = np.random.rand(1000, 5)

    # Concatenate to create a feature matrix X with 10 features
    X = np.hstack((X_informative, X_uninformative))

    print("--- Dataset Generation Complete ---")
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}\n")

    # 2. Create an sklearn.pipeline.Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selectkbest', SelectKBest(score_func=f_regression)),
        ('regressor', LinearRegression())
    ])

    print("--- ML Pipeline Created ---")
    print(pipeline)
    print("\n")

    # 3. Define a hyperparameter grid for GridSearchCV
    param_grid = {
        'selectkbest__k': [3, 5, 7, 10] # k can be up to 10, the total number of features
    }

    print("--- Hyperparameter Grid Defined ---")
    print(param_grid)
    print("\n")

    # 4. Use GridSearchCV with the pipeline and the defined parameter grid
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,  # Use all available CPU cores
        verbose=1
    )

    print("--- Starting GridSearchCV ---")
    grid_search.fit(X, y)
    print("--- GridSearchCV Complete ---\n")

    # 5. Report the best k value found, corresponding score, and selected feature indices
    best_k = grid_search.best_params_['selectkbest__k']
    # Convert neg_mean_squared_error to positive MSE
    best_mse = -grid_search.best_score_

    # Get the best estimator from the grid search
    best_pipeline = grid_search.best_estimator_

    # Extract the SelectKBest step from the best pipeline
    best_selectkbest_model = best_pipeline.named_steps['selectkbest']

    # Get the indices of the features selected by the best SelectKBest model
    selected_feature_indices = best_selectkbest_model.get_support(indices=True)

    print("--- Results ---")
    print(f"Best k value (number of features selected): {best_k}")
    print(f"Best cross-validation MSE: {best_mse:.4f}")
    print(f"Indices of features selected by the best model: {selected_feature_indices}")
    print("\nNote: Features 0-4 are informative, features 5-9 are uninformative.")

if __name__ == '__main__':
    run_regression_pipeline()