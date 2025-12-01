import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Set a random seed for reproducibility
random_state = 42
np.random.seed(random_state) # For consistent NaN introduction

# 1. Generate a synthetic classification dataset and introduce missing values
# Generate a base numerical dataset using make_classification
n_samples = 1000
n_numerical_features = 5
X_base, y = make_classification(
    n_samples=n_samples,
    n_features=n_numerical_features,
    n_informative=n_numerical_features, # All features are informative
    n_redundant=0,
    n_repeated=0,
    n_classes=2, # Binary classification
    random_state=random_state
)

# Convert the base numerical data to a Pandas DataFrame
numerical_feature_names = [f'num_feature_{i+1}' for i in range(n_numerical_features)]
X = pd.DataFrame(X_base, columns=numerical_feature_names)

# Manually add two new categorical features
# Categorical feature 1 with 3 unique values
cat1_values = ['CategoryA', 'CategoryB', 'CategoryC']
X['cat_feature_1'] = np.random.choice(cat1_values, n_samples, replace=True)

# Categorical feature 2 with 5 unique values
cat2_values = ['TypeX', 'TypeY', 'TypeZ', 'TypeW', 'TypeV']
X['cat_feature_2'] = np.random.choice(cat2_values, n_samples, replace=True)

# Introduce missing values (np.nan) into two of the numerical features
missing_percentage = 0.07 # Approximately 7% missing values
features_to_introduce_nan = ['num_feature_1', 'num_feature_3']

for col in features_to_introduce_nan:
    n_missing = int(n_samples * missing_percentage)
    # Randomly select indices to set as NaN
    missing_indices = np.random.choice(X.index, n_missing, replace=False)
    X.loc[missing_indices, col] = np.nan

print("--- Dataset Snapshot (first 5 rows) ---")
print(X.head())
print("\nMissing values per numerical feature:")
print(X[numerical_feature_names].isnull().sum())
print("-" * 40)

# Define column names for the ColumnTransformer
numerical_features = numerical_feature_names
categorical_features = ['cat_feature_1', 'cat_feature_2']

# 2. Create an sklearn.compose.ColumnTransformer to preprocess the data
# Preprocessing for numerical features: Impute NaNs with mean, then StandardScale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features: One-Hot Encode
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Drop any columns not explicitly specified
)

# 3. Construct an sklearn.pipeline.Pipeline
# The pipeline first applies the ColumnTransformer and then trains a RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

print("\n--- Pipeline Structure ---")
print(pipeline)
print("-" * 40)

# 4. Evaluate the complete pipeline's performance using 5-fold cross-validation
print("\n--- Evaluating Pipeline with 5-fold Cross-Validation (Accuracy) ---")
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy', n_jobs=-1) # n_jobs=-1 uses all available cores

# Report the mean accuracy and its standard deviation
mean_accuracy = np.mean(cv_scores)
std_accuracy = np.std(cv_scores)

print(f"\nCross-validation Accuracy Scores: {cv_scores}")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")
print("\n--- Pipeline Evaluation Complete ---")