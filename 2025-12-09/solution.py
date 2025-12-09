import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def run_ml_experiment():
    # --- 1. Generate Synthetic Dataset and Prepare Features ---
    RANDOM_STATE = 42
    N_SAMPLES = 1200 # At least 1000 samples
    N_NUM_FEATURES = 5 # 5 numerical features
    N_CATEGORIES = 80 # 50-100 unique categories

    # Generate numerical features and target
    X_num, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_NUM_FEATURES,
        n_informative=N_NUM_FEATURES - 1,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        random_state=RANDOM_STATE
    )

    # Create high-cardinality categorical feature
    # Generate random integers within a broad range
    cat_numerical = np.random.randint(0, N_CATEGORIES, size=N_SAMPLES)
    # Convert integers to unique string representations
    X_cat = np.array([f"Category_{i}" for i in cat_numerical])

    # Combine into a pandas DataFrame
    numerical_feature_names = [f"num_feature_{i}" for i in range(N_NUM_FEATURES)]
    categorical_feature_name = "high_card_cat_feature"

    X = pd.DataFrame(X_num, columns=numerical_feature_names)
    X[categorical_feature_name] = X_cat

    # Display dataset info
    print("--- Dataset Information ---")
    print(f"Total samples: {N_SAMPLES}")
    print(f"Numerical features: {numerical_feature_names}")
    print(f"Categorical feature: {categorical_feature_name} (Unique values: {X[categorical_feature_name].nunique()})")
    print(f"Target distribution:\n{pd.Series(y).value_counts()}")
    print("-" * 30)

    # --- 2. Split Data into Training and Testing Sets ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )

    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")
    print("-" * 30)

    # --- 3. Define Pipeline for OneHot Encoding ---
    preprocessor_onehot = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_feature_names),
            ('cat', OneHotEncoder(handle_unknown='ignore'), [categorical_feature_name])
        ])

    pipeline_onehot_encoding = Pipeline(steps=[
        ('preprocessor', preprocessor_onehot),
        ('classifier', LogisticRegression(solver='liblinear', random_state=RANDOM_STATE))
    ])

    # --- 4. Define Pipeline for Ordinal Encoding ---
    preprocessor_ordinal = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_feature_names),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), [categorical_feature_name])
        ])

    pipeline_ordinal_encoding = Pipeline(steps=[
        ('preprocessor', preprocessor_ordinal),
        ('classifier', LogisticRegression(solver='liblinear', random_state=RANDOM_STATE))
    ])

    # --- 5. Train Models and Evaluate Performance ---
    print("--- Training and Evaluation ---")

    # Pipeline with OneHot Encoding
    print("Fitting OneHot Encoding pipeline...")
    pipeline_onehot_encoding.fit(X_train, y_train)
    y_pred_onehot = pipeline_onehot_encoding.predict(X_test)
    accuracy_onehot = accuracy_score(y_test, y_pred_onehot)
    f1_onehot = f1_score(y_test, y_pred_onehot)

    print("\nResults for OneHot Encoding Pipeline:")
    print(f"  Accuracy: {accuracy_onehot:.4f}")
    print(f"  F1 Score: {f1_onehot:.4f}")

    # Pipeline with Ordinal Encoding
    print("\nFitting Ordinal Encoding pipeline...")
    pipeline_ordinal_encoding.fit(X_train, y_train)
    y_pred_ordinal = pipeline_ordinal_encoding.predict(X_test)
    accuracy_ordinal = accuracy_score(y_test, y_pred_ordinal)
    f1_ordinal = f1_score(y_test, y_pred_ordinal)

    print("\nResults for Ordinal Encoding Pipeline:")
    print(f"  Accuracy: {accuracy_ordinal:.4f}")
    print(f"  F1 Score: {f1_ordinal:.4f}")
    print("-" * 30)

    # Final comparison summary
    print("\n--- Encoding Strategy Comparison ---")
    if accuracy_onehot > accuracy_ordinal:
        print(f"OneHot Encoding achieved higher accuracy ({accuracy_onehot:.4f}) than Ordinal Encoding ({accuracy_ordinal:.4f}).")
    elif accuracy_ordinal > accuracy_onehot:
        print(f"Ordinal Encoding achieved higher accuracy ({accuracy_ordinal:.4f}) than OneHot Encoding ({accuracy_onehot:.4f}).")
    else:
        print(f"Both encoding strategies achieved similar accuracy ({accuracy_onehot:.4f}).")

    if f1_onehot > f1_ordinal:
        print(f"OneHot Encoding achieved higher F1 Score ({f1_onehot:.4f}) than Ordinal Encoding ({f1_ordinal:.4f}).")
    elif f1_ordinal > f1_onehot:
        print(f"Ordinal Encoding achieved higher F1 Score ({f1_ordinal:.4f}) than OneHot Encoding ({f1_onehot:.4f}).")
    else:
        print(f"Both encoding strategies achieved similar F1 Score ({f1_onehot:.4f}).")
    print("-" * 30)


if __name__ == "__main__":
    run_ml_experiment()