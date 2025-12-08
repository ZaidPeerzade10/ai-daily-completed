import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def run_clustering_analysis():
    """
    Generates synthetic data, applies K-Means clustering, evaluates it,
    reduces dimensionality, and visualizes the results.
    """

    # 1. Generate Synthetic Data
    n_samples = 750 # At least 700 samples
    n_features = 6
    n_clusters_true = 4
    random_state = 42

    X, y_true = make_blobs(n_samples=n_samples,
                           n_features=n_features,
                           centers=n_clusters_true,
                           cluster_std=1.0, # Standard deviation of the clusters
                           random_state=random_state)

    # Convert features to a pandas DataFrame
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    print("--- Synthetic Dataset Generated ---")
    print(f"Shape: {X_df.shape}")
    print(f"First 5 rows of features:\n{X_df.head()}\n")

    # 2. Apply K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters_true,
                    random_state=random_state,
                    n_init='auto') # 'auto' for modern sklearn versions
    kmeans.fit(X_df)
    cluster_labels = kmeans.labels_

    print("--- K-Means Clustering Applied ---")
    print(f"Discovered {len(np.unique(cluster_labels))} clusters.")
    print(f"Cluster label counts:\n{pd.Series(cluster_labels).value_counts().sort_index()}\n")

    # 3. Evaluate Clustering Quality
    silhouette_avg = silhouette_score(X_df, cluster_labels)
    print("--- Clustering Evaluation ---")
    print(f"Silhouette Score: {silhouette_avg:.3f}\n")

    # 4. Reduce Dimensionality for Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_df)
    X_pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])

    print("--- Dimensionality Reduction with PCA ---")
    print(f"Original features reduced to {X_pca_df.shape[1]} principal components.")
    print(f"Explained variance ratio by components: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}\n")


    # 5. Visualize the Clusters
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca_df['Principal Component 1'],
                          X_pca_df['Principal Component 2'],
                          c=cluster_labels,
                          cmap='viridis', # Choose a colormap
                          s=50,          # Marker size
                          alpha=0.8,     # Transparency
                          edgecolor='w'  # White edge around markers
                         )
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'K-Means Clustering on PCA-Reduced Data (Silhouette Score: {silhouette_avg:.3f})')
    plt.colorbar(scatter, label='Cluster Label')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_clustering_analysis()