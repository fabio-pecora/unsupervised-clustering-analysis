from pandas import factorize, read_csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

##################################################
# Question a
##################################################

filename = 'Frogs_MFCCs.csv'

data = read_csv(filename)

data.dropna(axis = 0, how ='any', inplace=True)
columns = data.columns

numericColumns = data.select_dtypes(include=['float64', 'int64']).columns

dataset = data[numericColumns].values
y = data['Family'].values

print(y)
target_names = data['Family'].unique()

y, target_names = factorize(data['Family'])

print(data[numericColumns])

import seaborn as sns

numeric_data = data.select_dtypes(include=['number'])

# Calculate the correlation matrix for the numerical data
correlation_matrix = numeric_data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap with Seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# Feature scaling
scaler = StandardScaler()
scaled_dataset = scaler.fit_transform(dataset)

pca = PCA()
dataset = pca.fit_transform(scaled_dataset)

cumulative_variance = pca.explained_variance_ratio_.cumsum()

plt.figure(figsize=(8,6))
plt.plot(cumulative_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

threshold = 0.95
num_components = next(i for i, cumulative_variance in enumerate(cumulative_variance) if cumulative_variance >= threshold) + 1

print(f'Number of components that explain {threshold*100}% of the variance: {num_components}')

pca = PCA(n_components=13)
dataset = pca.fit_transform(scaled_dataset)
print(dataset)
pca_df = pd.DataFrame(data=dataset, columns=[f'PC{i+1}' for i in range(dataset.shape[1])])
print(pca_df)

silhouette_scores = []
range_n_clusters = range(2, 9)

for k in range_n_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(dataset)
    silhouette_avg = silhouette_score(dataset, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {k}, the silhouette score is: {silhouette_avg}")
    
# Silhouette scores for k-means with clusters between 2 and 8
print("Silhouette_scores:", silhouette_scores)

# Plotting the silhouette scores
plt.figure(figsize=(6, 4))
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.title('Silhouette Scores for KMeans Clustering', fontsize=14)
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.grid(True)
plt.show()

# Determine the 2 best values for the number of clusters
best_clusters = np.argsort(silhouette_scores)[-2:] + 2  # Adding 2 because of 0-indexing
print("The best two clusters are", best_clusters)

##################################################
# Question b
##################################################

# Function for plotting the centroids for the clusters
def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)
    
# Function for plotting the Voronoi diagram for the clusters
def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))

    Z = np.zeros((resolution * resolution, X.shape[1]))
    means = X.mean(axis=0)
    Z[:, 0] = xx.ravel()
    Z[:, 1] = yy.ravel()
    for i in range(2, X.shape[1]):
        Z[:, i] = means[i]
    Z = clusterer.predict(Z)

    # Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

# Function for plotting the best two clusters
for i in best_clusters:
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit_predict(dataset)
    kmeans.cluster_centers_
    plt.figure(figsize=(6, 4))
    plot_decision_boundaries(kmeans, dataset)
    title = str(i) + ' Clusters'
    plt.title(title, fontsize=10)
    plt.show()


from sklearn.cluster import MiniBatchKMeans
minibatch_kmeans = MiniBatchKMeans(n_clusters=6, random_state=42)
minibatch_kmeans.fit(dataset)

minibatch_kmeans.inertia_

import urllib.request
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.int64)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    mnist["data"], mnist["target"], random_state=42)
    
    filename = "my_mnistttt.data"
X_mm = np.memmap(filename, dtype='float32', mode='write', shape=X_train.shape)
X_mm[:] = X_train

def load_next_batch(batch_size):
    return dataset[np.random.choice(len(dataset), batch_size, replace=False)]

p.random.seed(42)
    
from sklearn.cluster import KMeans, MiniBatchKMeans
from timeit import timeit
import matplotlib.pyplot as plt
import numpy as np

times = np.empty((100, 2))
inertias = np.empty((100, 2))

for k in range(1, 101): 
    kmeans_ = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    minibatch_kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        n_init=5,
        batch_size=1024,
        init_size=2048,
        max_iter=100,
    )
    print("\r{}/{}".format(k, 100), end="")  
    
    times[k - 1, 0] = timeit(lambda: kmeans_.fit(dataset), number=1)
    times[k - 1, 1] = timeit(lambda: minibatch_kmeans.fit(dataset), number=1)
    
    inertias[k - 1, 0] = kmeans_.inertia_
    inertias[k - 1, 1] = minibatch_kmeans.inertia_

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(range(1, 101), inertias[:, 0], "r--", label="K-Means")
plt.plot(range(1, 101), inertias[:, 1], "b.-", label="Mini-batch K-Means")
plt.xlabel("$k$", fontsize=16)
plt.ylabel("Inertia", fontsize=14)
plt.title("Inertia vs. Number of Clusters", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

plt.subplot(122)
plt.plot(range(1, 101), times[:, 0], "r--", label="K-Means")
plt.plot(range(1, 101), times[:, 1], "b.-", label="Mini-batch K-Means")
plt.xlabel("$k$", fontsize=16)
plt.ylabel("Training Time (seconds)", fontsize=14)
plt.title("Training Time vs. Number of Clusters", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.show()


from sklearn.cluster import MiniBatchKMeans


range_n_clusters_MB = best_clusters
print(range_n_clusters_MB)
silhouette_scores_MB_best = []
kmeans_per_k_MB = []

for n_clusters in (range_n_clusters_MB):

    kmeans_MB = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=1000)
    
    cluster_labels_MB = kmeans_MB.fit_predict(dataset)
    
    silhouette_avg_MB = silhouette_score(dataset, cluster_labels_MB)
    silhouette_scores_MB_best.append(silhouette_avg_MB)
    kmeans_per_k_MB.append(kmeans_MB)
    

    print(f"For n_clusters = {n_clusters}, the silhouette score is: {silhouette_avg_MB}")
    
plt.figure(figsize=(8, 5))
plt.plot(range_n_clusters_MB, silhouette_scores_MB_best, marker='o')
plt.title("Mini Batch Silhouette Scores for the Best Numbers of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.xticks(range_n_clusters_MB)
plt.grid()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter

kmeans_per_k_MB = [MiniBatchKMeans(n_clusters=k, random_state=42).fit(dataset) for k in range(2, 11)]

silhouette_scores_MB = [
    silhouette_samples(dataset, model.labels_).mean() for model in kmeans_per_k_MB
]

best_clusters = [ 5, 6]  
plt.figure(figsize=(12, len(best_clusters) * 3)) 
for idx, k in enumerate(best_clusters): 
    plt.subplot(len(best_clusters), 1, idx + 1)  
    kmeans_model = kmeans_per_k_MB[k - 2] 
    y = kmeans_model.labels_

    silhouette_coefficients_MB = silhouette_samples(dataset, y)

    padding = len(dataset) 
    pos = padding
    ticks = []
    for i in range(k):
        coeffs = silhouette_coefficients_MB[y == i]
        coeffs.sort()
        color = plt.cm.Spectral(i / k)
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding

    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    plt.ylabel("Cluster")
    plt.xlabel("Silhouette Coefficient")
    plt.axvline(x=silhouette_scores_MB[k - 2], color="red", linestyle="--")  
    plt.title(f"Silhouette Diagram for $k={k}$", fontsize=16)

plt.tight_layout()
plt.show()


def plot_clusters(dataset, y=None): 
    if y is None:
        y = np.zeros(dataset.shape[0])
    plt.scatter(dataset[:, 3], dataset[:, 4], c=y, s=10)
    plt.xlabel("$x_3$", fontsize=14)
    plt.ylabel("$x_4$", fontsize=14, rotation=0)

y = kmeans_MB.predict(dataset)
plt.figure(figsize=(8, 4))
plot_clusters(dataset, y)
plt.show()

def plot_data(dataset):
    plt.plot(dataset[:, 3], dataset[:, 4], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 3], centroids[:, 4],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 3], centroids[:, 4],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, dataset, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = dataset.min(axis=0) - 0.1
    maxs = dataset.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[3], maxs[4], resolution),
                         np.linspace(mins[4], maxs[4], resolution))

    Z = np.zeros((resolution * resolution, dataset.shape[1]))
    means = dataset.mean(axis=0)
    Z[:, 3] = xx.ravel()
    Z[:, 4] = yy.ravel()

    for i in range(2, dataset.shape[1]):
        Z[:, i] = means[i]

    Z = clusterer.predict(Z)
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[3], maxs[3], mins[4], maxs[4]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[3], maxs[3], mins[4], maxs[4]),
                linewidths=1, colors='k')
    plot_data(dataset)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_3$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_4$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans_MB, dataset)
plt.show()

best_n_clusters_MB = np.argsort(silhouette_scores)[-2:] + 2
best_scores_MB = [silhouette_scores[i] for i in best_n_clusters_MB]

print("Best number of clusters:", best_n_clusters_MB)
print("Corresponding silhouette scores:", best_scores_MB)


k1_MB = best_n_clusters_MB[0]
k2_MB = best_n_clusters_MB[1]

def fit_kmeans_and_voronoi(dataset, k):
    kmeans_MB = KMeans(n_clusters=k, random_state=42)
    kmeans_MB.fit(dataset)
    centroids_MB = kmeans_MB.cluster_centers_
    if len(centroids_MB) >= 8:
        vor = Voronoi(centroids_MB)
    else:
        vor = None
    return kmeans_MB, centroids_MB, vor

kmeans1_MB, centroids1, vor1 = fit_kmeans_and_voronoi(dataset, k1_MB)
kmeans2_MB, centroids2, vor2 = fit_kmeans_and_voronoi(dataset, k2_MB)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].scatter(dataset[:, 3], dataset[:, 4], c=kmeans1_MB.labels_, s=10)
if vor1 is not None:
    voronoi_plot_2d(vor1, ax=axs[0], show_vertices=False, line_colors='k', line_width=2, alpha=0.5)
axs[0].scatter(centroids1[:, 3], centroids1[:, 4], c='red', s=100, marker='X', label='Centroids')
axs[0].set_title(f'K-Means with k={k1_MB}')
axs[0].set_xlabel("$x_3$", fontsize=14)
axs[0].set_ylabel("$x_4$", fontsize=14, rotation=0)
axs[0].legend()

axs[1].scatter(dataset[:, 3], dataset[:, 4], c=kmeans2_MB.labels_, s=10)
if vor2 is not None:
    voronoi_plot_2d(vor2, ax=axs[1], show_vertices=False, line_colors='k', line_width=2, alpha=0.5)
axs[1].scatter(centroids2[:, 3], centroids2[:, 4], c='red', s=100, marker='X', label='Centroids')
axs[1].set_title(f'K-Means with k={k2_MB}')
axs[1].set_xlabel("$x_3$", fontsize=14)
axs[1].set_ylabel("$x_4$", fontsize=14, rotation=0)
axs[1].legend()

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans, MiniBatchKMeans

cluster_range = range(2, 11) 
silhouette_scores = []
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(dataset)
    silhouette_score = np.mean(silhouette_samples(dataset, kmeans.labels_)) 
    silhouette_scores.append(silhouette_score)

silhouette_scores_MB = []
for k in cluster_range:
    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
    minibatch_kmeans.fit(dataset)
    silhouette_score = np.mean(silhouette_samples(dataset, minibatch_kmeans.labels_)) 
    silhouette_scores_MB.append(silhouette_score)

print("Length of silhouette_scores:", len(silhouette_scores))
print("Length of silhouette_scores_MB:", len(silhouette_scores_MB))

plt.figure(figsize=(8, 5))
plt.plot(cluster_range, silhouette_scores, color='blue', label='KMeans') 
plt.plot(cluster_range, silhouette_scores_MB, color='orange', label='MiniBatchKMeans') 

plt.title('Silhouette Scores Comparison')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette score')
plt.legend() 
plt.show()

kmeans_labels = np.array(cluster_labels)
minibatch_kmeans_labels = np.array(cluster_labels_MB)

labels_df = pd.DataFrame({'KMeans': kmeans_labels, 'MiniBatchKMeans': minibatch_kmeans_labels})
correlation_matrix = labels_df.corr()

print("Correlation Matrix:")
print(correlation_matrix)

print("\n")

calinski_harabasz_kmeans = calinski_harabasz_score(dataset, kmeans_labels)
calinski_harabasz_mini_batch = calinski_harabasz_score(dataset, minibatch_kmeans_labels)

print("Calinski-Harabasz Index:")
print(f"K-Means: {calinski_harabasz_kmeans}")
print(f"Mini-Batch K-Means: {calinski_harabasz_mini_batch}")

print("\n")

davies_bouldin_kmeans = davies_bouldin_score(dataset, kmeans_labels)
davies_bouldin_mini_batch = davies_bouldin_score(dataset, minibatch_kmeans_labels)

print("Davies-Bouldin Index:")
print(f"K-Means: {davies_bouldin_kmeans}")
print(f"Mini-Batch K-Means: {davies_bouldin_mini_batch}")

