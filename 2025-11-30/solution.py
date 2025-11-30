import pandas as pd
import numpy as np

def run_feature_engineering_pipeline():
    """
    Generates a synthetic dataset, engineers interaction features,
    detects skewed features, applies log1p transformation,
    and displays the results.
    """
    print("--- Starting Feature Engineering Pipeline ---")

    # 1. Synthesize the Dataset
    np.random.seed(42) # for reproducibility
    n_rows = 1000
    data = {
        'feature_A': np.random.uniform(1, 10, size=n_rows),
        'feature_B': np.random.normal(5, 2, size=n_rows),
        'feature_C': np.random.exponential(2, size=n_rows), # Likely to be skewed
        'feature_D': np.random.poisson(3, size=n_rows) # Likely to be skewed
    }
    df = pd.DataFrame(data)

    print("\n--- Initial DataFrame Head ---")
    print(df.head())

    # 2. Engineer Interaction Features
    df['interaction_AB'] = df['feature_A'] * df['feature_B']
    df['interaction_CD'] = df['feature_C'] * df['feature_D']

    print("\n--- DataFrame Head after Interaction Features ---")
    print(df.head())

    # 3. Calculate Initial Skewness
    print("\n--- Initial Skewness of all numerical features ---")
    initial_skewness = df.skew()
    print(initial_skewness)

    # 4. Identify Skewed Features
    skewness_threshold = 0.75
    skewed_features = initial_skewness[initial_skewness > skewness_threshold].index.tolist()
    
    print(f"\n--- Features identified as skewed (skewness > {skewness_threshold}): ---")
    if skewed_features:
        for feature in skewed_features:
            print(f"- {feature}: {initial_skewness[feature]:.4f}")
    else:
        print("No features identified as highly skewed above the threshold.")

    # 5. Apply Log1p Transformation
    if skewed_features:
        print("\n--- Applying np.log1p transformation to identified skewed features... ---")
        for feature in skewed_features:
            # Ensure feature values are non-negative for log1p. 
            # Poisson and Exponential distributions guarantee this.
            # For uniform/normal, clipping might be necessary for general case, 
            # but for this problem, it's not expected to be an issue.
            df[feature] = np.log1p(df[feature])
    else:
        print("\nSkipping log1p transformation as no features were identified as highly skewed.")

    # 6. Display Transformed Data and Final Skewness
    print("\n--- DataFrame Head after Log1p Transformation ---")
    print(df.head())

    print("\n--- Final Skewness of all numerical features after transformation ---")
    final_skewness = df.skew()
    print(final_skewness)

    print("\n--- Pipeline Finished ---")

if __name__ == "__main__":
    run_feature_engineering_pipeline()