import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score

# 1. Generate Synthetic Regression Dataset
n_samples = 500
n_features = 7
noise_level = 10
X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise_level, random_state=42)

# Convert X to a Pandas DataFrame for easier column selection by name
feature_names = [f'feature_{i}' for i in range(n_features)]
X_df = pd.DataFrame(X, columns=feature_names)

# 2. Implement CustomPolynomialFeatures Transformer
class CustomPolynomialFeatures(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer that applies PolynomialFeatures only to a
    specified subset of features. It relies on ColumnTransformer to pass the
    correct subset of X to its transform method.
    """
    def __init__(self, features_to_transform, degree=2, include_bias=False):
        self.features_to_transform = features_to_transform # Stores feature names as per requirement
        self.degree = degree
        self.include_bias = include_bias
        self.poly_transformer = PolynomialFeatures(degree=self.degree, include_bias=self.include_bias)

    def fit(self, X, y=None):
        # When used in ColumnTransformer, X here will already be the subset
        # of features specified by features_to_transform
        self.poly_transformer.fit(X)
        return self

    def transform(self, X):
        # X here will be the subset of features passed by ColumnTransformer
        return self.poly_transformer.transform(X)

# 3. Define Feature Groups for Transformation
# Features for polynomial transformation
poly_features_names = ['feature_0', 'feature_1', 'feature_2'] # 3 features

# Features for standard scaling (the remaining numerical features)
scaler_features_names = [f for f in feature_names if f not in poly_features_names] # 4 features

# 4. Create sklearn.compose.ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        # Apply CustomPolynomialFeatures to specific features
        ('custom_poly', CustomPolynomialFeatures(
            features_to_transform=poly_features_names, 
            degree=2, 
            include_bias=False
         ), poly_features_names),
        
        # Apply StandardScaler to other numerical features
        ('scaler', StandardScaler(), scaler_features_names)
    ],
    remainder='drop' # Explicitly drop any features not specified (all 7 features are handled)
)

# 5. Build the Full sklearn.pipeline.Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(random_state=42)) # Using Ridge regressor
])

# 6. Evaluate the pipeline's performance using cross_val_score
cv_scores = cross_val_score(
    pipeline,
    X_df, # Pass the DataFrame to leverage column names
    y,
    cv=5, # 5-fold cross-validation
    scoring='neg_mean_squared_error',
    n_jobs=-1 # Use all available CPU cores
)

# 7. Print the mean and standard deviation of the Mean Squared Error (MSE)
# Convert negative MSE scores to positive MSE values
mse_scores = -cv_scores
mean_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)

print("Pipeline Cross-Validation Results:")
print(f"  Mean Squared Error (MSE): {mean_mse:.4f}")
print(f"  Standard Deviation of MSE: {std_mse:.4f}")