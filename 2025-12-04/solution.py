import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns

def generate_and_visualize_data():
    """
    Generates a synthetic dataset, adds a categorical feature, and creates
    various visualizations using seaborn and matplotlib.
    """

    # 1. Generate Synthetic Data and Create Initial DataFrame
    n_samples = 500
    n_features = 4
    n_clusters = 3
    random_state = 42 # for reproducibility

    X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                      centers=n_clusters, cluster_std=1.0, random_state=random_state)

    # Convert to pandas DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['cluster_id'] = y

    print("--- Synthetic Dataset Head ---")
    print(df.head())
    print("\n--- Dataset Info ---")
    df.info()
    print(f"\nUnique cluster_ids: {df['cluster_id'].unique()}")

    # 2. Add a Categorical Feature
    group_categories = ['Group A', 'Group B', 'Group C']
    df['group'] = np.random.choice(group_categories, size=n_samples, replace=True)

    print("\n--- Dataset Head with new 'group' feature ---")
    print(df.head())
    print(f"\nUnique group categories: {df['group'].unique()}")
    print(f"Distribution of 'group' categories:\n{df['group'].value_counts()}")


    # 3. Create a Pair Plot for numerical features, colored by cluster_id
    print("\n--- Generating Pair Plot... ---")
    plt.figure(figsize=(10, 8)) # This won't directly affect pairplot's internal figure size
    pair_plot = sns.pairplot(df, vars=feature_names, hue='cluster_id', palette='viridis')
    pair_plot.fig.suptitle('Pair Plot of Numerical Features by Cluster ID', y=1.02, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent title overlap
    plt.show()
    print("Pair Plot displayed.")

    # 4. Create Separated Histograms (or KDE plots) for feature_1 and feature_2 by 'group'
    print("\n--- Generating Histograms/KDEs by Group... ---")

    # For feature_1
    g1 = sns.FacetGrid(df, col='group', height=4, aspect=1.2, col_wrap=len(group_categories))
    g1.map(sns.histplot, 'feature_1', kde=True, stat='density', alpha=0.7, color='skyblue')
    g1.set_axis_labels('Feature 1 Value', 'Density')
    g1.set_titles(col_template="{col_name}")
    g1.fig.suptitle('Distribution of Feature 1 Across Different Groups', y=1.05, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    # For feature_2
    g2 = sns.FacetGrid(df, col='group', height=4, aspect=1.2, col_wrap=len(group_categories))
    g2.map(sns.histplot, 'feature_2', kde=True, stat='density', alpha=0.7, color='lightcoral')
    g2.set_axis_labels('Feature 2 Value', 'Density')
    g2.set_titles(col_template="{col_name}")
    g2.fig.suptitle('Distribution of Feature 2 Across Different Groups', y=1.05, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
    print("Separated Histograms/KDEs displayed.")


    # 5. Create a Box Plot (or violin plot) for feature_3 across cluster_ids
    print("\n--- Generating Box Plot for Feature 3 by Cluster ID... ---")
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='cluster_id', y='feature_3', palette='viridis')
    plt.title('Distribution of Feature 3 Across Different Clusters', fontsize=16)
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Feature 3 Value', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    print("Box Plot displayed.")

    print("\n--- Data visualization complete! ---")

if __name__ == "__main__":
    generate_and_visualize_data()