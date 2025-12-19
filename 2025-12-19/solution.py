import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

def run_regression_analysis():
    """
    Generates a synthetic regression dataset, performs feature engineering,
    builds and evaluates ML pipelines, and visualizes feature importance.
    """

    # 1. Generate a synthetic regression dataset
    print("1. Generating synthetic regression dataset...")
    X_array, y = make_regression(
        n_samples=1000,
        n_features=6,
        n_informative=6,
        n_targets=1,
        noise=10.0,
        random_state=42
    )

    feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]
    X_original = pd.DataFrame(X_array, columns=feature_names)
    print(f"  X_original shape: {X_original.shape}")
    print(f"  y shape: {y.shape}\n")

    # 2. Feature Engineering: Create a new interaction feature
    print("2. Creating interaction feature 'feature_0_x_feature_1'...")
    X_with_interaction = X_original.copy(deep=True)
    X_with_interaction['feature_0_x_feature_1'] = X_with_interaction['feature_0'] * X_with_interaction['feature_1']
    print(f"  X_with_interaction shape: {X_with_interaction.shape}\n")

    # 3. Create two sklearn.pipeline.Pipeline objects
    print("3. Creating and evaluating ML pipelines...")

    # Pipeline without interaction feature
    pipeline_no_interaction = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    # Pipeline with interaction feature
    pipeline_with_interaction = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    # Evaluate pipeline_no_interaction
    scores_no_interaction = cross_val_score(
        pipeline_no_interaction,
        X_original,
        y,
        cv=5,
        scoring='neg_mean_squared_error'
    )
    mse_no_interaction = -scores_no_interaction # Convert neg_mean_squared_error to MSE
    print("  Pipeline WITHOUT interaction feature:")
    print(f"    Mean MSE: {np.mean(mse_no_interaction):.4f}")
    print(f"    Std Dev MSE: {np.std(mse_no_interaction):.4f}\n")

    # Evaluate pipeline_with_interaction
    scores_with_interaction = cross_val_score(
        pipeline_with_interaction,
        X_with_interaction,
        y,
        cv=5,
        scoring='neg_mean_squared_error'
    )
    mse_with_interaction = -scores_with_interaction # Convert neg_mean_squared_error to MSE
    print("  Pipeline WITH interaction feature:")
    print(f"    Mean MSE: {np.mean(mse_with_interaction):.4f}")
    print(f"    Std Dev MSE: {np.std(mse_with_interaction):.4f}\n")

    # 5. Feature Importance Visualization
    print("4. Training pipeline_with_interaction and visualizing feature importance...")

    # Train the pipeline_with_interaction on the entire dataset
    pipeline_with_interaction.fit(X_with_interaction, y)

    # Extract coefficients
    coefficients = pipeline_with_interaction.named_steps['regressor'].coef_
    feature_names_with_interaction = X_with_interaction.columns

    # Create a DataFrame for easy plotting
    coeff_df = pd.DataFrame({
        'Feature': feature_names_with_interaction,
        'Coefficient_Abs': np.abs(coefficients)
    })

    # Sort by absolute magnitude for better visualization
    coeff_df = coeff_df.sort_values(by='Coefficient_Abs', ascending=False)

    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(coeff_df['Feature'], coeff_df['Coefficient_Abs'])
    plt.xlabel('Feature Name')
    plt.ylabel('Absolute Coefficient Magnitude')
    plt.title('Linear Regression Coefficients (Absolute Magnitude)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    print("\nAnalysis complete. Plot displayed (may need to close plot window to exit script).")

if __name__ == "__main__":
    run_regression_analysis()