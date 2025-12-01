import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. Generate a synthetic classification dataset and introduce missing values
# Base numerical dataset
n_samples = 1200
n_numerical_features = 5
X_numerical, y = make_classification(
    n_samples=n_samples,
    n_features=n_numerical_features,
    n_informative=4,
    n_redundant=1,
    n_classes=2,
    random_state=42
)

# Convert to DataFrame
feature_names = [f'num_feature_{i}' for i in range(n_numerical_features)]
X_df = pd.DataFrame(X_numerical, columns=feature_names)

# Add two categorical features
X_df['cat_feature_A'] = np.random.choice(['Type_X', 'Type_Y', 'Type_Z'], size=n_samples)
X_df['cat_feature_B'] = np.random.choice(['Small', 'Medium', 'Large', 'XL', 'XXL'], size=n_samples)

# Introduce missing values into two numerical features
# For 'num_feature_0'
missing_indices_0 = np.random.choice(X_df.index, size=int(0.05 * n_samples), replace=False)
X_df.loc[missing_indices_0, 'num_feature_0'] = np.nan
# For 'num_feature_2'
missing_indices_2 = np.random.choice(X_df.index, size=int(0.03 * n_samples), replace=False)
X_df.loc[missing_indices_2, 'num_feature_2'] = np.nan

print("--- Dataset Information ---")
print(f"Dataset shape: {X_df.shape}")
print(f"Target shape: {y.shape}")
print("Missing values introduced:")
print(X_df[['num_feature_0', 'num_feature_2']].isnull().sum())
print("-" * 30)

# Define feature types
numerical_features = [f'num_feature_{i}' for i in range(n_numerical_features)]
categorical_features = ['cat_feature_A', 'cat_feature_B']

# 2. Create an sklearn.compose.ColumnTransformer for preprocessing
# Numerical pipeline: Impute with mean, then StandardScale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: OneHotEncode
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 3. Construct an sklearn.pipeline.Pipeline
# The pipeline first applies the ColumnTransformer and then trains a RandomForestClassifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])

print("\n--- Pipeline Structure ---")
print(pipeline)
print("-" * 30)

# 4. Evaluate the complete pipeline's performance using 5-fold cross-validation
print("\n--- Cross-validation Performance ---")
cv_scores = cross_val_score(pipeline, X_df, y, cv=5, scoring='accuracy', n_jobs=-1)

print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean accuracy: {cv_scores.mean():.4f}")
print(f"Standard deviation of accuracy: {cv_scores.std():.4f}")
print("-" * 30)