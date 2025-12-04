import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs

# Set a consistent style for the plots for better aesthetics
sns.set_style("whitegrid")

# Set a random seed for reproducibility across the entire script
random_seed = 42
np.random.seed(random_seed) # For numpy random operations
# For sklearn and other libraries, their internal random_state will be used where available

# 1. Generate Synthetic Dataset and Create DataFrame
# Set parameters for make_blobs
n_samples = 550  # At least 500 samples as per requirement
n_features = 4   # 4 numerical features
n_clusters = 3   # 3 distinct clusters

X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters,
                  cluster_std=1.0, # Standard deviation of the clusters
                  random_state=random_seed) # For reproducible blob generation

# Convert the generated features and cluster labels into a pandas DataFrame
feature_names = [f'feature_{i}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['cluster_id'] = y
df['cluster_id'] = df['cluster_id'].astype('category') # Cast cluster_id as a categorical type

print("--- Generated Synthetic Dataset ---")
print("DataFrame Head:")
print(df.head())
print("\nDataFrame Info:")
df.info()

# 2. Add a new categorical feature (e.g., 'group')
group_categories = ['A', 'B', 'C'] # 3 distinct values
# Use np.random.choice to assign random categories to each sample
# Using a dedicated RNG for reproducibility for the new column
rng = np.random.default_rng(random_seed)
df['group'] = rng.choice(group_categories, size=len(df), replace=True)
df['group'] = df['group'].astype('category') # Cast 'group' as a categorical type

print("\n--- DataFrame with New Categorical Feature ---")
print("DataFrame Head with 'group' feature:")
print(df.head())
print("\nValue counts for 'group' feature:")
print(df['group'].value_counts())

# 3. Create a pair plot for numerical features, coloring points by 'cluster_id'
print("\n--- Generating Pair Plot ---")
# pairplot creates its own figure
pair_plot_fig = sns.pairplot(df, vars=feature_names, hue='cluster_id',
                             diag_kind='kde', # Show Kernel Density Estimate on the diagonal
                             palette='viridis') # A color palette for clusters
pair_plot_fig.fig.suptitle('Pair Plot of Numerical Features by Cluster ID', y=1.02) # Adjust y to prevent title overlap
plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for the suptitle
print("Pair Plot generated and displayed.")


# 4. Create histograms (or KDE plots) for feature_1 and feature_2, separated by 'group'
print("\n--- Generating Feature Distribution Plots by Group ---")

# FacetGrid for Feature 1 distribution by 'group'
g1 = sns.FacetGrid(df, col='group', height=4, aspect=1.2, col_wrap=3, palette='viridis')
g1.map(sns.histplot, 'feature_1', kde=True, stat='density', alpha=0.7, linewidth=0) # stat='density' for comparison
g1.set_axis_labels("Feature 1 Value", "Density")
g1.set_titles(col_name="Group {col_name}")
g1.fig.suptitle('Distribution of Feature 1 by Group', y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98])
print("Feature 1 distribution plot by Group generated.")

# FacetGrid for Feature 2 distribution by 'group'
g2 = sns.FacetGrid(df, col='group', height=4, aspect=1.2, col_wrap=3, palette='viridis')
g2.map(sns.histplot, 'feature_2', kde=True, stat='density', alpha=0.7, linewidth=0)
g2.set_axis_labels("Feature 2 Value", "Density")
g2.set_titles(col_name="Group {col_name}")
g2.fig.suptitle('Distribution of Feature 2 by Group', y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98])
print("Feature 2 distribution plot by Group generated.")


# 5. Create a box plot (or violin plot) showing the distribution of 'feature_3' across different 'cluster_id's
print("\n--- Generating Box Plot for Feature 3 by Cluster ID ---")
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='cluster_id', y='feature_3', palette='viridis')
plt.title('Distribution of Feature 3 Across Different Clusters')
plt.xlabel('Cluster ID')
plt.ylabel('Feature 3 Value')
plt.tight_layout()
print("Box Plot for Feature 3 by Cluster ID generated.")

# 6. Display all generated plots
print("\n--- Displaying all generated plots ---")
plt.show()
print("Script finished. All plots should now be visible.")