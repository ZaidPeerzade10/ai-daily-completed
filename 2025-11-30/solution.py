import pandas as pd
import numpy as np

def run_feature_engineering_task():
    """
    Generates a synthetic DataFrame, creates interaction features,
    identifies skewed numerical features, applies log1p transformation,
    and displays the results.
    """
    print("--- Starting Feature Engineering Task ---")

    # 1. Generate Synthetic DataFrame
    np.random.seed(42) # for reproducibility
    n_rows = 1000

    print(f"\n1. Generating a synthetic DataFrame with {n_rows} rows...")
    df = pd.DataFrame({
        'feature_A': np.random.uniform(0, 10, n_rows),
        'feature_B': np.random.normal(5, 2, n_rows),
        'feature_C': np.random.exponential(2, n_rows), # Likely skewed
        'feature_D': np.random.poisson(3, n_rows)       # Likely skewed
    })
    print("Initial DataFrame head:")
    print(df.head())

    # 2. Create two new interaction features
    print("\n2. Creating interaction features: 'interaction_AB' and 'interaction_CD'...")
    df['interaction_AB'] = df['feature_A'] * df['feature_B']
    df['interaction_CD'] = df['feature_C'] * df['feature_D']
    print("DataFrame head with new interaction features:")
    print(df.head())

    # Calculate skewness for all numerical features
    print("\n3. Calculating initial skewness of all numerical features:")
    initial_skewness = df.skew(numeric_only=True)
    print(initial_skewness)

    # 4. Identify all numerical features that have a skewness value greater than 0.75
    skewed_features_to_transform = initial_skewness[initial_skewness > 0.75].index.tolist()
    print(f"\n4. Features identified for np.log1p transformation (skewness > 0.75):")
    if skewed_features_to_transform:
        print(skewed_features_to_transform)
    else:
        print("No features found with skewness > 0.75.")

    # 5. For each identified skewed feature, apply a np.log1p transformation
    print("\n5. Applying np.log1p transformation to identified skewed features...")
    for col in skewed_features_to_transform:
        df[col] = np.log1p(df[col])
        print(f"  - Transformed '{col}' using np.log1p.")

    # 6. Display the head of the modified DataFrame
    print("\n6. Displaying the head of the DataFrame after transformations:")
    print(df.head())

    # Display the skewness of all features after transformation
    print("\n7. Displaying the skewness of all numerical features AFTER transformations:")
    final_skewness = df.skew(numeric_only=True)
    print(final_skewness)

    print("\n--- Feature Engineering Task Completed ---")

if __name__ == "__main__":
    run_feature_engineering_task()