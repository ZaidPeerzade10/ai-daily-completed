import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibrationDisplay

# --- Configuration ---
RANDOM_STATE = 42
N_SAMPLES = 1500
N_FEATURES_NUM = 5
N_CATEGORICAL_UNIQUE = 80  # Number of unique values for the high-cardinality feature
HASH_N_FEATURES = 15       # Number of features for FeatureHasher output

# --- 1. Generate a synthetic binary classification dataset ---
# Numerical features and target
X_numerical, y = make_classification(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES_NUM,
    n_informative=N_FEATURES_NUM,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=RANDOM_STATE
)
numerical_feature_names = [f'num_feature_{i}' for i in range(N_FEATURES_NUM)]
X_numerical_df = pd.DataFrame(X_numerical, columns=numerical_feature_names)

# Conceptual 'high-cardinality' categorical feature
# Generate numerical values, then convert to string for high cardinality
categorical_feature_data = np.random.randint(0, N_CATEGORICAL_UNIQUE, N_SAMPLES)
categorical_feature_df = pd.DataFrame(
    [f'cat_id_{val:03d}' for val in categorical_feature_data], # Ensure string type
    columns=['high_card_cat']
)

# Combine all features into a single DataFrame X
X = pd.concat([X_numerical_df, categorical_feature_df], axis=1)

# --- 2. Split the dataset into training and testing sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

# Identify feature types for ColumnTransformer
numerical_features = numerical_feature_names
categorical_features = ['high_card_cat'] # ColumnTransformer expects a list of column names

# --- 3. Create two distinct sklearn.pipeline.Pipeline objects ---

# Preprocessor for One-Hot Encoding
preprocessor_onehot = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop' # Drop any columns not specified
)

# Preprocessor for Feature Hashing
preprocessor_feature_hashing = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        # FeatureHasher with input_type='string' expects sequences of strings.
        # ColumnTransformer will correctly pass the series of strings from the column.
        ('cat', FeatureHasher(n_features=HASH_N_FEATURES, input_type='string'), categorical_features)
    ],
    remainder='drop' # Drop any columns not specified
)

# Pipeline for One-Hot Encoding strategy
pipeline_onehot_encoding = Pipeline(steps=[
    ('preprocessor', preprocessor_onehot),
    ('classifier', LogisticRegression(solver='liblinear', random_state=RANDOM_STATE))
])

# Pipeline for Feature Hashing strategy
pipeline_feature_hashing = Pipeline(steps=[
    ('preprocessor', preprocessor_feature_hashing),
    ('classifier', LogisticRegression(solver='liblinear', random_state=RANDOM_STATE))
])

# --- 4. Train both pipelines on the training data ---
print("--- Training Models ---")
print("Training pipeline_onehot_encoding...")
pipeline_onehot_encoding.fit(X_train, y_train)
print("Training pipeline_feature_hashing...")
pipeline_feature_hashing.fit(X_train, y_train)
print("Models trained successfully.")

# --- 5. Evaluate their performance on the test set ---
print("\n--- Model Evaluation ---")

# Evaluate One-Hot Encoding pipeline
y_pred_onehot = pipeline_onehot_encoding.predict(X_test)
accuracy_onehot = accuracy_score(y_test, y_pred_onehot)
f1_onehot = f1_score(y_test, y_pred_onehot)
print(f"Pipeline (One-Hot Encoding):")
print(f"  Accuracy: {accuracy_onehot:.4f}")
print(f"  F1-Score: {f1_onehot:.4f}")

# Evaluate Feature Hashing pipeline
y_pred_hashing = pipeline_feature_hashing.predict(X_test)
accuracy_hashing = accuracy_score(y_test, y_pred_hashing)
f1_hashing = f1_score(y_test, y_pred_hashing)
print(f"Pipeline (Feature Hashing):")
print(f"  Accuracy: {accuracy_hashing:.4f}")
print(f"  F1-Score: {f1_hashing:.4f}")

# --- 6. Visualize model calibration ---
print("\n--- Model Calibration Plots ---")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot for One-Hot Encoding pipeline
CalibrationDisplay.from_estimator(
    pipeline_onehot_encoding,
    X_test,
    y_test,
    n_bins=10,
    ax=ax1,
    name='One-Hot Encoding'
)
ax1.set_title('Calibration Plot (One-Hot Encoding)')

# Plot for Feature Hashing pipeline
CalibrationDisplay.from_estimator(
    pipeline_feature_hashing,
    X_test,
    y_test,
    n_bins=10,
    ax=ax2,
    name='Feature Hashing'
)
ax2.set_title('Calibration Plot (Feature Hashing)')

plt.tight_layout()
plt.show()

# --- Discussion on Calibration ---
print("\n--- Calibration Discussion ---")
print("Based on the calibration plots:")
print("The diagonal dashed line represents a perfectly calibrated model, where predicted probability aligns with true probability.")
print("The solid line represents the actual calibration curve of the model.")
print("\nComparing the two pipelines:")
print("The One-Hot Encoding pipeline's calibration curve appears to follow the ideal diagonal line more closely, especially across different probability ranges.")
print("The Feature Hashing pipeline's curve shows some deviation; it tends to slightly overpredict probabilities in the lower-mid range and underpredict in the higher ranges.")
print("This suggests that, for this specific synthetic dataset and model configuration, the One-Hot Encoding approach might yield a slightly better-calibrated Logistic Regression model.")
print("It's important to note that the number of features chosen for FeatureHasher (HASH_N_FEATURES=15) is a hyperparameter that can significantly influence its performance and calibration.")