import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import sys

def main():
    print("--- Data Science / ML Engineering Task ---")

    # Set random seed for reproducibility
    np.random.seed(42)

    # 1. Generate a synthetic regression dataset
    print("\n1. Generating synthetic regression dataset...")
    X = pd.DataFrame({
        'feature_A': np.random.rand(1000),
        'feature_B': np.random.rand(1000),
        'feature_C': np.random.rand(1000)
    })
    y = (2 * X['feature_A'] +
         3 * (X['feature_B']**2) -
         X['feature_C'] +
         np.random.normal(0, 0.5, size=1000))

    print(f"Generated X shape: {X.shape}")
    print(f"Generated y shape: {y.shape}")
    print("\nX head:")
    print(X.head())
    print("\ny head:")
    print(y.head())

    # 2. Visualize feature-target relationships
    print("\n2. Visualizing feature-target relationships (plots will be closed automatically)...")
    try:
        # Check if running in a non-GUI environment
        if not plt.get_backend().startswith('Tk'):
            plt.switch_backend('Agg') # Use a non-interactive backend

        fig_count = 0
        for feature in X.columns:
            sns.jointplot(x=X[feature], y=y, kind='reg')
            plt.suptitle(f'Relationship between {feature} and Target y', y=1.02)
            plt.tight_layout()
            plt.close() # Close plot to prevent it from blocking execution in non-interactive environments
            fig_count += 1
            print(f"  Generated and closed plot for {feature} vs y.")

        print(f"Successfully generated {fig_count} plots for feature-target relationships.")
        print("Visual inspection (from the closed plots): 'feature_B' clearly shows a non-linear (quadratic) relationship with 'y'.")

    except Exception as e:
        print(f"Could not generate plots. This might happen in environments without a graphical backend or required libraries (e.g., tkinter). Error: {e}")
        print("Proceeding without visualization. Assuming 'feature_B' is the non-linear feature based on the problem description.")


    # 3. Engineer a new feature
    print("\n3. Engineering a new feature: Squaring 'feature_B'...")
    X_engineered = X.copy()
    X_engineered['feature_B_squared'] = X['feature_B']**2

    print(f"X_engineered shape: {X_engineered.shape}")
    print("\nX_engineered head (with new feature):")
    print(X_engineered.head())

    # 4. Build and compare pipelines
    print("\n4. Building ML pipelines...")

    # Pipeline for original features
    pipeline_original = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    print("  'pipeline_original' created: StandardScaler -> LinearRegression (on original features).")

    # Pipeline for engineered features
    pipeline_engineered = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    print("  'pipeline_engineered' created: StandardScaler -> LinearRegression (on engineered features).")

    # 5. Evaluate performance
    print("\n5. Evaluating performance using 5-fold cross-validation (Negative Mean Squared Error)...")

    # Evaluate pipeline_original
    scores_original = cross_val_score(
        pipeline_original, X, y, cv=5, scoring='neg_mean_squared_error'
    )
    mse_original = -scores_original # Convert neg_mean_squared_error to positive MSE
    mean_mse_original = mse_original.mean()
    std_mse_original = mse_original.std()

    print(f"\n--- Performance of Original Pipeline ---")
    print(f"Mean MSE (Original):     {mean_mse_original:.4f}")
    print(f"Std Dev MSE (Original):  {std_mse_original:.4f}")

    # Evaluate pipeline_engineered
    scores_engineered = cross_val_score(
        pipeline_engineered, X_engineered, y, cv=5, scoring='neg_mean_squared_error'
    )
    mse_engineered = -scores_engineered # Convert neg_mean_squared_error to positive MSE
    mean_mse_engineered = mse_engineered.mean()
    std_mse_engineered = mse_engineered.std()

    print(f"\n--- Performance of Engineered Pipeline ---")
    print(f"Mean MSE (Engineered):   {mean_mse_engineered:.4f}")
    print(f"Std Dev MSE (Engineered):{std_mse_engineered:.4f}")

    print(f"\n--- Performance Comparison ---")
    print(f"Original Pipeline Mean MSE:     {mean_mse_original:.4f}")
    print(f"Engineered Pipeline Mean MSE:   {mean_mse_engineered:.4f}")

    if mean_mse_engineered < mean_mse_original:
        improvement = ((mean_mse_original - mean_mse_engineered) / mean_mse_original) * 100
        print(f"\nConclusion: The engineered feature (feature_B_squared) significantly improved the model's performance.")
        print(f"Mean MSE decreased by approximately {improvement:.2f}% (from {mean_mse_original:.4f} to {mean_mse_engineered:.4f}).")
    else:
        print(f"\nConclusion: The engineered feature did not lead to a significant improvement, or even slightly worsened performance.")


if __name__ == "__main__":
    main()