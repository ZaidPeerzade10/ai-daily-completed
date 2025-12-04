import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs

def generate_and_visualize_data():
    """
    Generates a synthetic dataset, adds a categorical feature, and creates
    several visualizations using seaborn and matplotlib.
    """

    # 1. Generate a synthetic dataset using make_blobs
    n_samples = 500
    n_features = 4
    n_clusters = 3
    random_state = 42

    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters,
                      cluster_std=1.0, random_state=random_state)

    # Convert to pandas DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)

    # Include the cluster labels as a feature
    df['cluster_id'] = y

    # 2. Add a new categorical feature (e.g., 'group')
    group_categories = ['Group A', 'Group B', 'Group C']
    df['group'] = np.random.choice(group_categories, size=len(df), p=[0.4, 0.3, 0.3]) # Uneven distribution for variety

    print("--- Synthetic Data Head ---")
    print(df.head())
    print("\n--- Synthetic Data Info ---")
    df.info()
    print(f"\nUnique clusters: {df['cluster_id'].nunique()}")
    print(f"Unique groups: {df['group'].nunique()}")
    print("\n")

    # 3. Create visualizations to explore the data

    # 3a. Pair plot for numerical features, colored by cluster_id
    print("Generating Pair Plot...")
    numeric_features = [f'feature_{i}' for i in range(n_features)]
    pair_plot = sns.pairplot(df, vars=numeric_features, hue='cluster_id', palette='viridis', diag_kind='kde')
    pair_plot.fig.suptitle('Pair Plot of Numerical Features by Cluster ID', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()
    print("Pair Plot displayed.\n")

    # 3b. Histograms/KDE plots for feature_1 and feature_2, separated by 'group'
    print("Generating Histograms/KDE plots by Group...")
    
    # Feature 1 distributions by group
    g1 = sns.FacetGrid(df, col='group', height=4, aspect=1.2, col_wrap=3)
    g1.map(sns.histplot, 'feature_1', kde=True, stat='density', alpha=0.7)
    g1.set_axis_labels("Feature 1 Value", "Density")
    g1.set_titles(col_template='{col_name} - Feature 1')
    g1.fig.suptitle('Distribution of Feature 1 by Group', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

    # Feature 2 distributions by group
    g2 = sns.FacetGrid(df, col='group', height=4, aspect=1.2, col_wrap=3)
    g2.map(sns.histplot, 'feature_2', kde=True, stat='density', alpha=0.7, color='skyblue')
    g2.set_axis_labels("Feature 2 Value", "Density")
    g2.set_titles(col_template='{col_name} - Feature 2')
    g2.fig.suptitle('Distribution of Feature 2 by Group', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()
    print("Histograms/KDE plots displayed.\n")

    # 3c. Box plot of feature_3 across different cluster_ids
    print("Generating Box Plot...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster_id', y='feature_3', data=df, palette='plasma')
    plt.title('Distribution of Feature 3 Across Different Cluster IDs', fontsize=16)
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Feature 3 Value', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    print("Box Plot displayed.\n")

    print("Data visualization task complete.")

if __name__ == "__main__":
    generate_and_visualize_data()