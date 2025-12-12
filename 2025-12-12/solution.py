import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

# --- 1. Generate Synthetic Dataset and Initial Preparation ---
# Generate synthetic binary classification dataset
n_samples = 1500  # At least 1000 samples
n_features = 4    # Exactly 4 numerical features
n_classes = 2     # Binary classification
random_state = 42

X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_features, # All features are informative
    n_redundant=0,
    n_repeated=0,
    n_classes=n_classes,
    random_state=random_state
)

# Convert to pandas DataFrame
feature_names = [f'feature_{i}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("--- Step 1: Dataset Generation and Initial Preparation ---")
print(f"Generated dataset shape: {df.shape}")
print("First 5 rows of the DataFrame:")
print(df.head())
print("\n")

# --- 2. Feature Discretization ---
# Select two original numerical features for binning
features_to_bin = ['feature_0', 'feature_1']
num_bins = 3 # Discretize into 3 bins

for feat in features_to_bin:
    new_bin_col_name = f'binned_{feat}'
    # Apply pd.cut to discretize features into 3 bins
    df[new_bin_col_name] = pd.cut(
        df[feat],
        bins=num_bins,
        labels=[f'{feat}_low', f'{feat}_medium', f'{feat}_high']
    )

print("--- Step 2: Feature Discretization ---")
print(f"Original features binned: {features_to_bin}")
print(f"New binned categorical features created: {[f'binned_{f}' for f in features_to_bin]}")
print("First 5 rows with original and new binned features:")
print(df[features_to_bin + [f'binned_{f}' for f in features_to_bin]].head())
print("\n")

# Identify features for the ColumnTransformer
# Remaining original numerical features (not binned)
numerical_features_for_scaling = [f for f in feature_names if f not in features_to_bin]
# Newly created binned categorical features
categorical_features_for_ohe = [f'binned_{f}' for f in features_to_bin]

print("--- Feature Identification for ColumnTransformer ---")
print(f"Numerical features for StandardScaler: {numerical_features_for_scaling}")
print(f"Categorical features for OneHotEncoder: {categorical_features_for_ohe}")
print("\n")

# --- 3. Define Preprocessing with ColumnTransformer ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features_for_scaling),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_for_ohe)
    ],
    remainder='drop' # Drop any columns not explicitly handled (e.g., original feature_0, feature_1)
)

print("--- Step 3: ColumnTransformer Defined ---")
print(f"ColumnTransformer setup: {preprocessor.transformers}")
print("\n")

# --- 4. Construct the Machine Learning Pipeline ---
model = GradientBoostingClassifier(random_state=random_state)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

print("--- Step 4: ML Pipeline Constructed ---")
print(pipeline)
print("\n")

# --- 5. Evaluate Pipeline Performance using Cross-Validation ---
# Prepare data for the pipeline:
# X_data should only include the features that will be processed by the ColumnTransformer
X_data = df[numerical_features_for_scaling + categorical_features_for_ohe]
y_data = df['target']

print("--- Step 5: Pipeline Performance Evaluation (Cross-Validation) ---")
print(f"Features used for training X_data: {list(X_data.columns)}")
print(f"Shape of X_data for pipeline: {X_data.shape}")
print(f"Shape of y_data for pipeline: {y_data.shape}")

# Perform 5-fold cross-validation
cv_scores = cross_val_score(pipeline, X_data, y_data, cv=5, scoring='accuracy')

print(f"Cross-validation scores (accuracy): {cv_scores}")
print(f"Mean accuracy: {np.mean(cv_scores):.4f}")
print(f"Standard deviation of accuracy: {np.std(cv_scores):.4f}")