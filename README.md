# Unsupervised Clustering Analysis: Frogs & Fish Datasets

This project applies multiple unsupervised learning techniques to two real-world datasets (Frogs MFCCs and Fish features), exploring cluster structures using PCA, K-Means, DBSCAN, and Spectral Clustering.

## 📁 Datasets
- `Frogs_MFCCs.csv` – Audio features of frog calls
- `fish_data.csv` – Morphometric data of various fish species

## 🧠 Techniques Used
- PCA for dimensionality reduction
- K-Means and MiniBatchKMeans (with silhouette analysis & Voronoi plots)
- DBSCAN with custom KNN classification
- Spectral Clustering with RBF kernel
- Cluster evaluation metrics:
  - Silhouette Score
  - Calinski-Harabasz Index
  - Davies-Bouldin Index

## 📊 Visualizations
- Correlation heatmaps
- PCA cumulative explained variance plots
- Cluster decision boundaries and silhouette diagrams
- Voronoi diagrams of K-Means cluster centroids

## 🔍 Key Findings
- Optimal cluster counts were selected based on silhouette scores
- DBSCAN required careful tuning of `eps` and `min_samples`
- Spectral Clustering performed well under nearest-neighbor affinity

## 📦 Requirements
- Python 3.7+
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

## 📌 How to Run
1. Place both datasets (`Frogs_MFCCs.csv` and `fish_data.csv`) in the project directory
2. Run the notebook or script in a Python environment with required packages

