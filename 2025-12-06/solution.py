import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# 1. Generate a synthetic regression dataset
n_samples = 500
n_features = 10 # make_regression can create more features than informative ones
n_informative = 3
noise = 20.0
random_state = 42

X, y = make_regression(n_samples=n_samples, n_features=n_features,
                       n_informative=n_informative, noise=noise,
                       random_state=random_state)

# 2. Create two distinct sklearn.pipeline.Pipeline objects
# pipeline_simple: StandardScaler followed by LinearRegression
pipeline_simple = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# pipeline_poly: PolynomialFeatures (degree=2), then StandardScaler, then LinearRegression
pipeline_poly = Pipeline([
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)), # include_bias=False to avoid redundant intercept
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# 3. Evaluate both pipelines using cross_val_score
cv_folds = 5
scoring_metric = 'neg_mean_squared_error'

# Evaluate pipeline_simple
scores_simple = cross_val_score(pipeline_simple, X, y, cv=cv_folds, scoring=scoring_metric)

# Evaluate pipeline_poly
scores_poly = cross_val_score(pipeline_poly, X, y, cv=cv_folds, scoring=scoring_metric)

# 4. Print the mean and standard deviation of the Mean Squared Error (MSE)
# Convert neg_mean_squared_error to positive MSE values
mse_simple = -scores_simple
mse_poly = -scores_poly

mean_mse_simple = np.mean(mse_simple)
std_mse_simple = np.std(mse_simple)

mean_mse_poly = np.mean(mse_poly)
std_mse_poly = np.std(mse_poly)

print("--- Pipeline Evaluation Results ---")
print(f"Pipeline: Simple (StandardScaler -> LinearRegression)")
print(f"  Mean MSE: {mean_mse_simple:.4f}")
print(f"  Std Dev MSE: {std_mse_simple:.4f}")
print("\n")
print(f"Pipeline: Poly (PolynomialFeatures(degree=2) -> StandardScaler -> LinearRegression)")
print(f"  Mean MSE: {mean_mse_poly:.4f}")
print(f"  Std Dev MSE: {std_mse_poly:.4f}")