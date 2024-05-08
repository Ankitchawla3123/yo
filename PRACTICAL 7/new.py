import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate synthetic dataset
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Function to perform K-means clustering and compute MSE for given parameters
def kmeans_and_mse(X, n_clusters, max_iter):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=10)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    mse = np.mean(np.square(X - centroids[labels]))
    return mse

# Parameters to compare
n_clusters = [2, 3, 4, 5]
max_iter = 100

# Plotting MSE for each number of clusters
plt.figure(figsize=(10, 6))
line_styles = ['-', '--', '-.', ':']
for i, cluster in enumerate(n_clusters):
    mse_values = []
    for _ in range(5):  # Repeat 5 times to account for randomness in initialization
        mse = kmeans_and_mse(X, cluster, max_iter)
        mse_values.append(mse)
    plt.plot(range(1, len(mse_values) + 1), mse_values, label=f'Clusters={cluster}', linestyle=line_styles[i % len(line_styles)])

plt.title('Mean Squared Error vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()
