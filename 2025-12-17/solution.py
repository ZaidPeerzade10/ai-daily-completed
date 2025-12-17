import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def run_ml_experiment():
    """
    Generates a synthetic regression dataset, engineers cyclical time features,
    builds and evaluates two ML pipelines, and prints their performance.
    """

    print("Starting ML Experiment: Cyclical Feature Engineering vs. Raw Numerical Feature\n")

    # 1. Generate synthetic regression dataset
    n_samples = 1000
    n_original_features = 4
    X_original, y = make_regression(
        n_samples=n_samples,
        n_features=n_original_features,
        n_informative=n_original_features,
        noise=15,
        random_state=42
    )

    # Convert original features to DataFrame for easier feature naming
    original_feature_names = [f'original_feature_{i}' for i in range(n_original_features)]
    X_original_df = pd.DataFrame(X_original, columns=original_feature_names)

    # Create new numerical feature 'time_of_day'
    time_of_day = np.random.randint(0, 24, size=n_samples)

    # Create cyclical features from 'time_of_day'
    time_of_day_sin = np.sin(2 * np.pi * time_of_day / 24)
    time_of_day_cos = np.cos(2 * np.pi * time_of_day / 24)

    # Combine all features into a single DataFrame
    X = pd.DataFrame({
        'time_of_day': time_of_day,
        'time_of_day_sin': time_of_day_sin,
        'time_of_day_cos': time_of_day_cos
    })
    X = pd.concat([X_original_df, X], axis=1)

    print(f"Dataset created with {n_samples} samples and {X.shape[1]} features (including engineered).\n")
    print("Features available for pipelines:")
    for col in X.columns:
        print(f"  - {col}")
    print("-" * 50)

    # Define feature sets for ColumnTransformer
    features_for_scaling_raw_tod = original_feature_names + ['time_of_day']
    features_for_scaling_cyclical_tod = original_feature_names + ['time_of_day_sin', 'time_of_day_cos']

    # 2. Create pipeline_raw_tod
    preprocessor_raw_tod = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), features_for_scaling_raw_tod)
        ],
        remainder='passthrough' # Ensure other columns are not dropped if they were present
    )

    pipeline_raw_tod = Pipeline(steps=[
        ('preprocessor', preprocessor_raw_tod),
        ('regressor', Ridge(random_state=42))
    ])
    print("\n'pipeline_raw_tod' created, using Standard Scaler on original features and raw 'time_of_day'.")

    # 3. Create pipeline_cyclical_tod
    preprocessor_cyclical_tod = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), features_for_scaling_cyclical_tod)
        ],
        remainder='passthrough' # Ensure other columns are not dropped if they were present
    )

    pipeline_cyclical_tod = Pipeline(steps=[
        ('preprocessor', preprocessor_cyclical_tod),
        ('regressor', Ridge(random_state=42))
    ])
    print("'pipeline_cyclical_tod' created, using Standard Scaler on original features and cyclical 'time_of_day_sin'/'time_of_day_cos'.")
    print("-" * 50)

    # 4. Evaluate both pipelines
    print("\nEvaluating pipelines using 5-fold cross-validation with R-squared metric...")
    cv_folds = 5

    scores_raw_tod = cross_val_score(
        pipeline_raw_tod, X[features_for_scaling_raw_tod], y, cv=cv_folds, scoring='r2'
    )

    # For the cyclical pipeline, ensure only the relevant features are passed to the initial transformer
    scores_cyclical_tod = cross_val_score(
        pipeline_cyclical_tod, X[features_for_scaling_cyclical_tod], y, cv=cv_folds, scoring='r2'
    )

    # 5. Print results
    print("\n--- Pipeline Performance Summary ---")
    print(f"\nPipeline: 'pipeline_raw_tod' (using raw 'time_of_day')")
    print(f"  R-squared scores (5-fold CV): {scores_raw_tod}")
    print(f"  Mean R-squared: {np.mean(scores_raw_tod):.4f}")
    print(f"  Standard Deviation of R-squared: {np.std(scores_raw_tod):.4f}")

    print(f"\nPipeline: 'pipeline_cyclical_tod' (using sin/cos 'time_of_day')")
    print(f"  R-squared scores (5-fold CV): {scores_cyclical_tod}")
    print(f"  Mean R-squared: {np.mean(scores_cyclical_tod):.4f}")
    print(f"  Standard Deviation of R-squared: {np.std(scores_cyclical_tod):.4f}")

    # Interpretation
    print("\n--- Interpretation ---")
    if np.mean(scores_cyclical_tod) > np.mean(scores_raw_tod):
        print(f"The 'pipeline_cyclical_tod' achieved a higher mean R-squared score ({np.mean(scores_cyclical_tod):.4f}) ")
        print(f"compared to 'pipeline_raw_tod' ({np.mean(scores_raw_tod):.4f}).")
        print("This suggests that encoding 'time_of_day' as cyclical sine and cosine features ")
        print("helped the model better capture the underlying periodic relationships in the data, ")
        print("leading to improved predictive performance.")
    elif np.mean(scores_cyclical_tod) < np.mean(scores_raw_tod):
        print(f"The 'pipeline_raw_tod' achieved a higher mean R-squared score ({np.mean(scores_raw_tod):.4f}) ")
        print(f"compared to 'pipeline_cyclical_tod' ({np.mean(scores_cyclical_tod):.4f}).")
        print("In this specific synthetic dataset, the raw 'time_of_day' feature performed better, ")
        print("which might imply the relationship with the target was more linear or monotonic ")
        print("than cyclical, or the noise level obscured the cyclical pattern's benefit.")
    else:
        print("Both pipelines performed similarly, indicating that the cyclical encoding did not ")
        print("significantly improve or worsen performance in this specific scenario.")

    print("\nML Experiment Finished.")

if __name__ == "__main__":
    run_ml_experiment()