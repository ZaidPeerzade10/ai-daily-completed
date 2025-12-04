import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs

def run_visualization_task():
    # 1. Generate Synthetic Data and Initial DataFrame Creation
    n_samples = 1000
    n_features = 4
    n_clusters = 3
    random_state = 42

    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters,
                      cluster_std=1.0, random_state=random_state)

    # Convert to pandas DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['cluster_id'] = y

    print("--- Synthetic Dataset Head ---")
    print(df.head())
    print("\n--- Synthetic Dataset Info ---")
    df.info()
    print(f"\nUnique cluster_ids: {df['cluster_id'].unique()}")

    # 2. Add a New Categorical Feature
    group_categories = ['Group A', 'Group B', 'Group C']
    df['group'] = np.random.choice(group_categories, size=len(df))

    print("\n--- Dataset Head with new 'group' feature ---")
    print(df.head())
    print(f"\nUnique 'group' values: {df['group'].unique()}")

    # Set style for seaborn plots
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100 # Adjust for better display in many environments

    # 3. Create a Pair Plot for Numerical Features
    print("\n--- Generating Pair Plot ---")
    pair_plot = sns.pairplot(df, hue='cluster_id', vars=feature_names, palette='viridis', diag_kind='kde')
    pair_plot.fig.suptitle('Pair Plot of Numerical Features Colored by Cluster ID', y=1.02) # y adjusts title height
    plt.show()
    print("Pair Plot displayed.")

    # 4. Visualize Feature Distributions Separated by Categorical Group
    print("\n--- Generating Histograms/KDEs for Feature 1 and Feature 2 separated by 'group' ---")
    # For feature_1
    g1 = sns.FacetGrid(df, col='group', col_wrap=3, height=4, aspect=1.2)
    g1.map_dataframe(sns.histplot, x='feature_1', kde=True, bins=30)
    g1.set_axis_labels("Feature 1 Value", "Count / Density")
    g1.set_titles(col_name="Group: {col_name}")
    g1.fig.suptitle('Distribution of Feature 1 by Group', y=1.02, fontsize=16)
    g1.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent title overlap
    plt.show()
    print("Distribution of Feature 1 by Group displayed.")

    # For feature_2
    g2 = sns.FacetGrid(df, col='group', col_wrap=3, height=4, aspect=1.2)
    g2.map_dataframe(sns.histplot, x='feature_2', kde=True, bins=30, color='orange')
    g2.set_axis_labels("Feature 2 Value", "Count / Density")
    g2.set_titles(col_name="Group: {col_name}")
    g2.fig.suptitle('Distribution of Feature 2 by Group', y=1.02, fontsize=16)
    g2.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
    print("Distribution of Feature 2 by Group displayed.")

    # 5. Create a Box Plot for Feature Distribution Across Clusters
    print("\n--- Generating Box Plot for Feature 3 across Cluster IDs ---")
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='cluster_id', y='feature_3', palette='pastel')
    plt.title('Distribution of Feature 3 Across Different Cluster IDs')
    plt.xlabel('Cluster ID')
    plt.ylabel('Feature 3 Value')
    plt.show()
    print("Box Plot for Feature 3 across Cluster IDs displayed.")

if __name__ == "__main__":
    run_visualization_task()