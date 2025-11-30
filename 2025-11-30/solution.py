import pandas as pd
import numpy as np

def run_feature_engineering_task():
    # 1. Generate Synthetic DataFrame
    np.random.seed(42) # For reproducibility
    n_rows = 1000
    df = pd.DataFrame({
        'feature_A': np.random.uniform(0, 10, n_rows),
        'feature_B': np.random.normal(0, 1, n_rows),
        'feature_C': np.random.exponential(scale=2.0, size=n_rows), # Likely skewed
        'feature_D': np.random.poisson(lam=3, size=n_rows) # Likely skewed
    })

    # 2. Create two new interaction features
    df['interaction_AB'] = df['feature_A'] * df['feature_B']
    df['interaction_CD'] = df['feature_C'] * df['feature_D']

    # 3. Identify all numerical features (original and new) that have a skewness value greater than 0.75
    initial_skewness = df.skew(numeric_only=True)
    skewed_features_to_transform = initial_skewness[initial_skewness > 0.75].index.tolist()

    print("Initial Skewness (features to be potentially transformed marked with *):")
    for feature, skew_val in initial_skewness.items():
        if feature in skewed_features_to_transform:
            print(f"  {feature}: {skew_val:.4f} *")
        else:
            print(f"  {feature}: {skew_val:.4f}")
    print(f"\nFeatures identified for np.log1p transformation: {skewed_features_to_transform}")


    # 4. For each identified skewed feature, apply a np.log1p transformation
    for feature in skewed_features_to_transform:
        df[feature] = np.log1p(df[feature])

    # 5. Display the head of the modified DataFrame
    print("\nModified DataFrame Head after transformations:")
    print(df.head())

    # 6. Display the skewness of all features after transformation
    post_transformation_skewness = df.skew(numeric_only=True)
    print("\nSkewness of all features after transformation:")
    print(post_transformation_skewness.to_string())

if __name__ == "__main__":
    run_feature_engineering_task()