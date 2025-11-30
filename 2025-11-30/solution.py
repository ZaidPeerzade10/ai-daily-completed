import pandas as pd
import numpy as np

def main():
    # 1. Generate Synthetic DataFrame
    np.random.seed(42) # for reproducibility
    n_rows = 1000

    data = {
        'feature_A': np.random.uniform(0, 10, n_rows),
        'feature_B': np.random.normal(5, 2, n_rows),
        'feature_C': np.random.exponential(2, n_rows), # Likely skewed
        'feature_D': np.random.poisson(3, n_rows)       # Likely skewed
    }
    df = pd.DataFrame(data)

    print("--- Original DataFrame Head ---")
    print(df.head())
    print("\n--- Original Features Skewness ---")
    print(df.skew())

    # 2. Create two new interaction features
    df['interaction_AB'] = df['feature_A'] * df['feature_B']
    df['interaction_CD'] = df['feature_C'] * df['feature_D']

    print("\n--- DataFrame Head after adding interaction features ---")
    print(df.head())

    # 3. Identify all numerical features (original and new)
    numerical_features = df.select_dtypes(include=np.number).columns

    # 4. Identify numerical features that have a skewness value greater than 0.75
    skewness_before_transform = df[numerical_features].skew()
    highly_skewed_features = skewness_before_transform[skewness_before_transform > 0.75].index.tolist()

    print(f"\n--- Features identified as highly skewed (> 0.75): ---")
    for feature in highly_skewed_features:
        print(f"- {feature}: {skewness_before_transform[feature]:.4f}")

    # 5. For each identified skewed feature, apply a np.log1p transformation
    for feature in highly_skewed_features:
        df[feature] = np.log1p(df[feature])
        print(f"Applied np.log1p transformation to '{feature}'.")

    # 6. Display the head of the modified DataFrame and the skewness of all features after transformation
    print("\n--- Head of DataFrame after transformations ---")
    print(df.head())

    print("\n--- Skewness of all numerical features after transformation ---")
    print(df[numerical_features].skew())

if __name__ == "__main__":
    main()