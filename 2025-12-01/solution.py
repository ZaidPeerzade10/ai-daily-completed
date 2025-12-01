import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# 1. Generate and Prepare the Synthetic Dataset
# Generate numerical features and target
n_samples = 1200
n_numerical_features = 5
X_numerical_base, y = make_classification(
    n_samples=n_samples,
    n_features=n_numerical_features,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    random_state=42
)

# Manually generate two additional arrays for categorical features
# One with 3 unique values
cat_feature_A = np.random.randint(0, 3, size=n_samples)
# One with 5 unique values
cat_feature_B = np.random.randint(0, 5, size=n_samples)

# Combine numerical features and categorical features into a single Pandas DataFrame
numerical_feature_names = [f'num_feature_{i+1}' for i in range(n_numerical_features)]
categorical_feature_names = ['cat_feature_A', 'cat_feature_B']

X = pd.DataFrame(X_numerical_base, columns=numerical_feature_names)
X['cat_feature_A'] = cat_feature_A
X['cat_feature_B'] = cat_feature_B

# Introduce missing values (np.nan) into two of the numerical features
# Introduce NaNs into 'num_feature_1' (e.g., 5% of samples)
nan_indices_1 = np.random.choice(X.index, size=int(0.05 * n_samples), replace=False)
X.loc[nan_indices_1, 'num_feature_1'] = np.nan

# Introduce NaNs into 'num_feature_3' (e.g., 4% of samples)
nan_indices_3 = np.random.choice(X.index, size=int(0.04 * n_samples), replace=False)
X.loc[nan_indices_3, 'num_feature_3'] = np.nan

# Define lists of feature names for the ColumnTransformer
numerical_features = numerical_feature_names
categorical_features = categorical_feature_names

# 2. Define Preprocessing Steps for Numerical Features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# 3. Define Preprocessing Steps for Categorical Features
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# 4. Construct the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Drop any columns not specified
)

# 5. Instantiate the Machine Learning Model
classifier = RandomForestClassifier(random_state=42)

# 6. Build the Complete Scikit-learn Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', classifier)
])

# 7. Evaluate the Pipeline using Cross-Validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

# Report the mean accuracy and its standard deviation
mean_accuracy = np.mean(cv_scores)
std_accuracy = np.std(cv_scores)

print(f"Mean cross-validation accuracy: {mean_accuracy:.4f}")
print(f"Standard deviation of accuracy: {std_accuracy:.4f}")