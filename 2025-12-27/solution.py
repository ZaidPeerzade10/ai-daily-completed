import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNeighborsImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Set a global random seed for reproducibility across all random operations
np.random.seed(42)

# 1. Generate a synthetic binary classification dataset
n_samples = 1000
n_features = 6
random_state_data = 42

X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_features, # All features are informative
    n_redundant=0,
    n_classes=2,
    random_state=random_state_data
)

feature_names = [f'feature_{i}' for i in range(n_features)]
df_X = pd.DataFrame(X, columns=feature_names)
df_y = pd.Series(y, name='target')

print("--- Initial Dataset Summary ---")
print(f"Features (X) shape: {df_X.shape}")
print(f"Target (y) shape: {df_y.shape}\n")

# 2. Introduce missing values into the DataFrame
missing_rates = {
    'feature_0': 0.15, # Approximately 15% missing
    'feature_1': 0.10, # Approximately 10% missing
    'feature_2': 0.05, # Approximately 5% missing
}

for feature, rate in missing_rates.items():
    num_missing = int(len(df_X) * rate)
    # Randomly select indices to set to NaN
    # np.random.choice is already seeded by np.random.seed(42)
    missing_indices = np.random.choice(df_X.index, num_missing, replace=False)
    df_X.loc[missing_indices, feature] = np.nan

print("--- Missing Values Introduced ---")
print("Number of missing values per feature:")
print(df_X[['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']].isnull().sum())
print(f"Total missing values in the dataset: {df_X.isnull().sum().sum()}\n")

# 3. Create an sklearn.pipeline.Pipeline with ColumnTransformer for preprocessing
# Define preprocessing steps for different feature subsets

# Pipeline for 'feature_0': Mean imputation followed by StandardScaler
feature_0_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Pipeline for 'feature_1': KNeighborsImputer (n_neighbors=5) followed by StandardScaler
feature_1_pipeline = Pipeline(steps=[
    ('imputer', KNeighborsImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

# Pipeline for 'feature_2': Median imputation followed by StandardScaler
feature_2_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline for the remaining numerical features ('feature_3' to 'feature_5'): StandardScaler directly
remaining_numerical_features = ['feature_3', 'feature_4', 'feature_5']
remaining_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Create the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('f0_preproc', feature_0_pipeline, ['feature_0']),
        ('f1_preproc', feature_1_pipeline, ['feature_1']),
        ('f2_preproc', feature_2_pipeline, ['feature_2']),
        ('remaining_preproc', remaining_pipeline, remaining_numerical_features)
    ],
    remainder='passthrough' # Keep other columns (if any) as they are, although not applicable here
)

# 4. Construct the complete Machine Learning Pipeline
random_state_rf = 42
classifier = RandomForestClassifier(random_state=random_state_rf)

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

print("--- Complete Pipeline Structure ---")
print(full_pipeline)
print("\n")

# 5. Evaluate the complete pipeline's performance using 5-fold cross-validation
print("--- Starting 5-fold Cross-Validation (scoring='accuracy') ---")
cv_scores = cross_val_score(full_pipeline, df_X, df_y, cv=5, scoring='accuracy')

# 6. Report the mean accuracy and its standard deviation
print("\n--- Cross-Validation Results ---")
print(f"Individual fold accuracies: {cv_scores}")
print(f"Mean accuracy: {cv_scores.mean():.4f}")
print(f"Standard deviation of accuracy: {cv_scores.std():.4f}")