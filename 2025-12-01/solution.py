import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Set random seed for reproducibility
np.random.seed(42)

# 1. Generate the Synthetic Dataset and Introduce Complexities
# Base numerical classification dataset
X_numerical, y = make_classification(
    n_samples=1500,        # At least 1000 samples
    n_features=5,          # 5 numerical features
    n_informative=3,
    n_redundant=1,
    n_repeated=0,
    n_classes=2,
    random_state=42
)

# Convert to DataFrame for easier manipulation and to add categorical features
X_df = pd.DataFrame(X_numerical, columns=[f'num_feature_{i}' for i in range(5)])

# Manually add two new categorical columns
# Categorical feature 1 (3 unique values)
cat_feature_1_values = ['Category_A', 'Category_B', 'Category_C']
X_df['cat_feature_1'] = np.random.choice(cat_feature_1_values, size=len(X_df))

# Categorical feature 2 (5 unique values)
cat_feature_2_values = ['Level_X', 'Level_Y', 'Level_Z', 'Level_W', 'Level_V']
X_df['cat_feature_2'] = np.random.choice(cat_feature_2_values, size=len(X_df))

# Introduce missing values (np.nan) into two of the original numerical features
# Introduce NaNs in 'num_feature_0'
missing_idx_0 = np.random.choice(X_df.index, size=int(0.05 * len(X_df)), replace=False) # 5% missing
X_df.loc[missing_idx_0, 'num_feature_0'] = np.nan

# Introduce NaNs in 'num_feature_2'
missing_idx_2 = np.random.choice(X_df.index, size=int(0.07 * len(X_df)), replace=False) # 7% missing
X_df.loc[missing_idx_2, 'num_feature_2'] = np.nan

print("--- Dataset Information ---")
print("Dataset Head with Missing Values and Categorical Features:")
print(X_df.head())
print("\nMissing values per numerical feature:")
print(X_df[['num_feature_0', 'num_feature_1', 'num_feature_2', 'num_feature_3', 'num_feature_4']].isnull().sum())
print(f"\nTotal samples: {len(X_df)}, Total features: {X_df.shape[1]}")
print(f"Target variable shape: {y.shape}")

# Define feature types for ColumnTransformer
numerical_features = [f'num_feature_{i}' for i in range(5)]
categorical_features = ['cat_feature_1', 'cat_feature_2']

# 2. Define Preprocessing Steps for Numerical and Categorical Features
# Preprocessing pipeline for numerical features: impute with mean, then scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features: OneHotEncoder
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# 3. Construct the ColumnTransformer
# This transformer applies different preprocessing steps to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Assemble the Full Machine Learning Pipeline
# The full pipeline first preprocesses data and then trains a RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf_classifier)
])

print("\n--- ML Pipeline Details ---")
print("Full Pipeline Steps:")
print(full_pipeline)

# 5. Evaluate the Pipeline using Cross-Validation
print("\n--- Pipeline Evaluation ---")
print("Performing 5-fold cross-validation (scoring='accuracy')...")
# Use n_jobs=-1 to utilize all available CPU cores for faster computation
cv_scores = cross_val_score(full_pipeline, X_df, y, cv=5, scoring='accuracy', n_jobs=-1)

# 6. Report Performance Metrics
mean_accuracy = np.mean(cv_scores)
std_accuracy = np.std(cv_scores)

print(f"\nCross-validation accuracy scores: {cv_scores}")
print(f"Mean accuracy: {mean_accuracy:.4f}")
print(f"Standard deviation of accuracy: {std_accuracy:.4f}")
print("\n--- Script Finished ---")