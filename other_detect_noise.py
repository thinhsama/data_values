import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy.stats import entropy
from tqdm import tqdm

# ------------------ PHƯƠNG PHÁP PHÁT HIỆN NHIỄU ------------------ #

def knn_consistency(X, y, k=5):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    neighbors = knn.kneighbors(X, return_distance=False)
    
    noise_scores = np.array([np.sum(y[neighbors[i]] != y[i]) / k for i in range(len(y))])
    return noise_scores

def kmeans_label_inconsistency(X, y, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    cluster_labels = [np.argmax(np.bincount(y[clusters == c])) for c in range(n_clusters)]
    noise_scores = np.array([y[i] != cluster_labels[clusters[i]] for i in range(len(y))])
    return noise_scores

def entropy_label_consistency(X, y, k=5):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    neighbors = knn.kneighbors(X, return_distance=False)
    
    noise_scores = np.array([entropy(np.bincount(y[neighbors[i]], minlength=2)) for i in range(len(y))])
    return noise_scores

def pca_outlier_detection(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    distances = np.linalg.norm(X_pca - np.mean(X_pca, axis=0), axis=1)
    return distances

def isolation_forest_outlier(X):
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(X)
    return -iso.decision_function(X)

def local_outlier_factor(X):
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    return -lof.fit_predict(X)


