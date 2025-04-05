import numpy as np
from enum import Enum
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from hdbscan import HDBSCAN

class ClusterModelC(Enum):
    """Enumeration for clustering algorithms."""
    KMEANS = 'K-Means'
    HDBSCAN = 'HDBSCAN'
    AGGLOMERATIVE = 'Agglomerative'
    GMM = 'GMM'

def is_city_cluster_divisible_valid(distance_matrix: np.ndarray, 
                            labels: np.ndarray) -> bool:
    """
    Check if city division into clusters is valid.
    
    Args:
        distance_matrix: Distance matrix between cities
        labels: Cluster labels for each city
        
    Returns:
        bool: True if division is valid
    """
    count_cties = distance_matrix.shape[0]
    count_clusters: int = len(np.unique(labels))
    is_valid: bool = count_cties % count_clusters == 0 and count_clusters != 1
    return is_valid


def max_distance_in_cluster(distance_matrix: np.ndarray, labels: np.ndarray, cluster_label: int) -> float:
    """
    Calculates the maximum distance within a cluster.

    Parameters:
    - distance_matrix (np.ndarray): A square distance matrix between objects.
    - labels (np.ndarray): An array of cluster labels for each object.
    - cluster_label (int): The label of the cluster for which the calculation is performed.

    Returns:
    - float: The maximum distance between any two objects within the specified cluster.
             Returns 0.0 if the cluster contains fewer than two objects.
    """
    indices = np.where(labels == cluster_label)[0]
    
    if len(indices) < 2:
        return 1.0

    submatrix = distance_matrix[np.ix_(indices, indices)]
    
    max_distance = np.max(submatrix)
    return max_distance

def cluster_radius(distance_matrix: np.ndarray, labels: np.ndarray, cluster_label: int) -> float:
    """
    Calculate the average distance between all points within a cluster (cluster radius).
    
    Args:
        distance_matrix: Square matrix of pairwise distances between all points (n x n)
        labels: Array of cluster labels for each point (length n)
        cluster_label: The label of the cluster to analyze
        
    Returns:
        float: Mean distance between all point pairs within the specified cluster.
               Returns 1.0 if cluster contains fewer than 2 points.
               
    Note:
        Uses only upper triangular part of distance matrix to avoid duplicate pairs
    """
    indices = np.where(labels == cluster_label)[0]
    
    if len(indices) < 2:
        return 1.0

    submatrix = distance_matrix[np.ix_(indices, indices)]
    
    upper_triangle = submatrix[np.triu_indices_from(submatrix, k=1)]
    
    mean_distance = np.mean(upper_triangle)
    
    return mean_distance

def distances_between_clusters(distance_matrix: np.array, labels: np.array) -> float:
    """
    Compute matrix of minimum distances between all cluster pairs.
    
    Args:
        distance_matrix: Square matrix of pairwise distances (n x n)
        labels: Array of cluster labels for each point (length n)
        
    Returns:
        np.ndarray: Symmetric matrix (k x k) where k is number of unique clusters,
                   with entries representing minimum distance between clusters.
                   Diagonal elements are 0.
    """
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    cluster_distance_matrix = np.full((k, k), np.inf)

    for i in range(k):
        for j in range(i + 1, k):
            label1, label2 = unique_labels[i], unique_labels[j]
            
            idx1 = np.where(labels == label1)[0]
            idx2 = np.where(labels == label2)[0]
            
            min_dist = np.min(distance_matrix[np.ix_(idx1, idx2)])

            cluster_distance_matrix[i, j] = min_dist
            cluster_distance_matrix[j, i] = min_dist
    
    np.fill_diagonal(cluster_distance_matrix, 0)
    return cluster_distance_matrix

def mean_cluster_distance(cluster_distance_matrix: np.array) -> float:
    """
    Calculates the average value among all distances between clusters,
    excluding the main diagonal.
    Copy

    Parameters:
    - cluster_distance_matrix: A (k x k) matrix of minimum distances between clusters.

    Returns:
    - The average distance between clusters.
    """

    mask = np.eye(cluster_distance_matrix.shape[0], dtype=bool)
    masked_values = cluster_distance_matrix[~mask]
    mean_distance = np.mean(masked_values) if masked_values.size > 0 else np.nan
    return mean_distance

def k_poly_approximation(x: float, coeffs: np.array = np.array([50.23391594, -10.79009267, 2.87041305])) -> float:
    """
    Compute polynomial approximation for cluster distance threshold.
    
    Uses a polynomial function to determine the threshold coefficient for phase transition
    based on the ratio of clusters to cities.
    
    Args:
        x: Input value (ratio of clusters to cities)
        coeffs: Polynomial coefficients (default empirically determined values)
        
    Returns:
        float: Threshold coefficient for phase transition detection
    """
    poly_func = np.poly1d(coeffs)
    y = poly_func(x)
    return y

def is_mean_cluster_distance_valid(distance_matrix: np.ndarray, labels: np.ndarray, k_approximator = k_poly_approximation) -> bool:
    """
    Validate if mean inter-cluster distance satisfies phase transition conditions.
    
    Args:
        distance_matrix: Pairwise distance matrix between cities
        labels: Cluster assignments for each city
        k_approximator: Function to compute threshold coefficient (default: k_poly_approximation)
        
    Returns:
        bool: True if mean cluster distance is valid for phase transition
    """
    count_clusters: int = len(np.unique(labels))
    count_cities: int = len(labels)
    ratio: float = count_clusters / count_cities
    k: float = k_approximator(ratio)
    cluster_radiuses: np.ndarray = np.array([max_distance_in_cluster(distance_matrix, labels, cluster_label)/2 for cluster_label in np.unique(labels)])
    cluster_distances: np.ndarray = distances_between_clusters(distance_matrix, labels)
    mean_clusters_distance: float = mean_cluster_distance(cluster_distances)
    mean_radius: float = np.mean(cluster_radiuses)
    is_valid = mean_radius * k <= mean_clusters_distance
    return is_valid

def is_cluster_city_distribution_valid(labels: np.array, min_ratio_coef: float = 3.0) -> bool:
    """
    Check if cluster size distribution indicates phase transition.
    
    Args:
        labels: Cluster assignments for each city
        min_ratio_coef: Minimum required ratio between largest and second-largest clusters
        
    Returns:
        bool: True if cluster size distribution suggests phase transition
    """
    unique_labels, clusters_cities = np.unique(labels, return_counts=True)
    count_clusters: int = len(unique_labels)

    if count_clusters == 2 and min(clusters_cities) == 1:
        return False
    
    unique_clusters_cities: np.array = np.unique(clusters_cities)
    sorted_unique_clusters_cities: np.array = np.sort(unique_clusters_cities)

    if len(sorted_unique_clusters_cities) == 1:
        max_cluster_ratio = 1
    else:
        max_cluster_ratio: float = sorted_unique_clusters_cities[-1] / sorted_unique_clusters_cities[-2]

    is_valid: bool = max_cluster_ratio >= min_ratio_coef

    return is_valid

def detect_clusters(distance_matrix: np.ndarray, method: ClusterModelC = ClusterModelC.AGGLOMERATIVE) -> np.array:
    """
    Perform cluster detection using specified algorithm.
    
    Args:
        distance_matrix: Pairwise distance matrix between cities
        method: Clustering algorithm to use (ClusterModelC enum)
        
    Returns:
        np.array: Cluster labels for each city
        
    Raises:
        ValueError: If unsupported clustering method is specified
    """
    best_count_clusters = 1
    best_silhouette = -1
    if method == ClusterModelC.HDBSCAN:
        model = HDBSCAN(min_cluster_size=2)
        labels = model.fit_predict(distance_matrix)
        return labels
        
    for count_clusters in range(1, distance_matrix.shape[0]):
        if method == ClusterModelC.KMEANS:
            model = KMeans(n_clusters=count_clusters, random_state=42)
        elif method == ClusterModelC.AGGLOMERATIVE:
            model = AgglomerativeClustering(n_clusters=count_clusters, metric='precomputed', linkage='complete')
        elif method == ClusterModelC.GMM:
            model = GaussianMixture(n_components=count_clusters, random_state=42)

        labels = model.fit_predict(distance_matrix)

        if len(np.unique(labels)) > 1:
            sil = silhouette_score(distance_matrix, labels)
            if sil > best_silhouette:
                best_silhouette = sil
                best_count_clusters = count_clusters

    if method == ClusterModelC.KMEANS:
        final_model = KMeans(n_clusters=best_count_clusters, random_state=42)
    elif method == ClusterModelC.AGGLOMERATIVE:
        final_model = AgglomerativeClustering(n_clusters=best_count_clusters, metric='precomputed', linkage='complete')
    elif method == ClusterModelC.GMM:
        final_model = GaussianMixture(n_components=best_count_clusters, random_state=42)

    final_labels = final_model.fit_predict(distance_matrix)

    return final_labels

def is_cluster_phase_transition(distance_matrix: np.ndarray, labels: np.ndarray) -> bool:
    """
    Determine if cluster configuration represents phase transition.
    
    Args:
        distance_matrix: Pairwise distance matrix between cities
        labels: Cluster assignments for each city
        
    Returns:
        bool: True if cluster configuration meets phase transition criteria
    """
    if is_city_cluster_divisible_valid(distance_matrix, labels) and (is_mean_cluster_distance_valid(distance_matrix, labels) or is_cluster_city_distribution_valid(labels)):
        return True
    return False

def detect_cluster_pt(distance_matrix, cluster_method: ClusterModelC = ClusterModelC.AGGLOMERATIVE) -> bool:
    """
    Detect cluster phase transition in TSP problem.
    
    Args:
        distance_matrix: Pairwise distance matrix between cities
        cluster_method: Clustering algorithm to use (default: Agglomerative)
        
    Returns:
        bool: True if phase transition is detected
    """
    cluster_labels = detect_clusters(method=cluster_method, distance_matrix=distance_matrix)
    is_clusters_pt = is_cluster_phase_transition(distance_matrix=distance_matrix, labels=cluster_labels)
    return is_clusters_pt
