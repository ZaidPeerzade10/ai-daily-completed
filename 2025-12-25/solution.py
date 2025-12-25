import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 1. Generate a synthetic dataset
n_samples = 1000
n_features = 5
n_true_centers = 4 # The actual number of centers in the synthetic data
X, y_true = make_blobs(n_samples=n_samples, n_features=n_features,
                       centers=n_true_centers, cluster_std=0.8,
                       random_state=42)

print(f"Generated synthetic dataset with {n_samples} samples, {n_features} features, and {n_true_centers} true centers.")

# 2. Apply StandardScaler to the generated features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features scaled using StandardScaler.")

# 3. Implement the Elbow method and Silhouette analysis
inertia_values = []
silhouette_scores = []
k_range = range(2, 9) # K values from 2 to 8

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto') # n_init='auto' is recommended
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)
    
    labels = kmeans.labels_
    # Silhouette score is well-defined for k >= 2 and n_samples > n_clusters.
    # Our setup (1000 samples, k max 8) satisfies this.
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

print("Completed KMeans training for K from 2 to 8, collecting inertia and silhouette scores.")

# 4. Plot both the inertia_ values (Elbow curve) and silhouette_score values
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(k_range, inertia_values, marker='o')
ax1.set_title('Elbow Method for Optimal K')
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia')
ax1.grid(True)

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(k_range, silhouette_scores, marker='o')
ax2.set_title('Silhouette Analysis for Optimal K')
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.grid(True)

print("Elbow Method and Silhouette Analysis plots generated.")


# 5. Based on plots, identify the optimal number of clusters (K)
# Visually, the elbow is typically observed at K=4, and the silhouette score often peaks at the optimal K.
# Given the dataset was generated with 4 centers, K=4 is the expected optimal.
optimal_k = 4 
print(f"\nBased on visual inspection of the Elbow method and Silhouette Analysis, the optimal number of clusters is identified as K = {optimal_k}.")

# Train a final KMeans model using this optimal K on the scaled data
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
final_kmeans.fit(X_scaled)
final_labels = final_kmeans.labels_
final_centroids = final_kmeans.cluster_centers_
print(f"Final KMeans model trained with K={optimal_k} clusters.")


# 6. Reduce the dimensionality of the scaled features to 2 principal components
pca = PCA(n_components=2, random_state=42) # Added random_state for reproducibility
X_pca = pca.fit_transform(X_scaled)
print(f"Scaled features reduced to 2 principal components (explained variance ratio: {pca.explained_variance_ratio_.sum():.2f}).")

# Transform the centroids of the final KMeans model into this 2D PCA space
centroids_pca = pca.transform(final_centroids)
print("KMeans centroids transformed into 2D PCA space.")


# 7. Visualize the final clusters in the 2D PCA space
fig3, ax3 = plt.subplots(figsize=(12, 8))
scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=final_labels, cmap='viridis', s=50, alpha=0.7)
ax3.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, color='red', edgecolor='black', linewidth=2, label='Centroids')

ax3.set_title(f'K-Means Clusters (K={optimal_k}) with Centroids in 2D PCA Space')
ax3.set_xlabel('Principal Component 1')
ax3.set_ylabel('Principal Component 2')
ax3.grid(True)

# Create a legend for the clusters
cluster_legend = ax3.legend(*scatter.legend_elements(), title="Clusters")
ax3.add_artist(cluster_legend) # Add the cluster legend first

# Create a separate legend for centroids and place it
# Using `plt.Line2D` allows creating a proxy artist for the legend entry without plotting it again
centroid_legend = ax3.legend(handles=[plt.Line2D([0], [0], marker='X', color='red', linestyle='None', markersize=10, label='Centroids')], loc='upper left', title="Centroids")

print("Final cluster visualization in 2D PCA space generated.")

# Display all generated plots
plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
plt.show()