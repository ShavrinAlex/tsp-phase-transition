import numpy as np 
from enum import Enum
from sklearn.cluster import KMeans, OPTICS, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class ClusterModelR(Enum):
    """Enumeration for clustering algorithms."""
    KMEANS='K-Means',
    OPTICS='OPTICS',
    AGGLOMERATIVE='Agglomerative'

def cities_clusters(labels: list) -> dict:
    """
    Groups city indices by their cluster labels and returns a dictionary
    describing which cities belong to which cluster.

    Parameters
    ----------
    labels : list
        A list where the i-th element represents the cluster index
        assigned to the i-th city.

    Returns
    -------
    dict
        A dictionary where each key is a cluster number (starting from 1),
        and the value is another dictionary with key 'cities' containing
        a list of city indices belonging to that cluster.

        Example:
        {
            1: {'cities': [0, 2, 5]},
            2: {'cities': [1, 3]},
            3: {'cities': [4]}
        }

    Notes
    -----
    - Cluster numbers in the input list may be arbitrary integers and do not
      need to be consecutive. In the output, they are reindexed starting from 1
      based on the order of their first appearance.
    """
    map = dict()
    for city, cluster in enumerate(labels):
        map.setdefault(cluster, []).append(city)

    cities_clusters = {cluster_number+1: {'cities': cities} for cluster_number, cities in enumerate(map.values())}
    return cities_clusters


def is_sequential_cluster(cluster: dict) -> bool:
    """
    Checks whether the cities in a cluster form a continuous sequence.

    Args:
        cluster (dict): Dictionary with key 'cities' that contains a list of city indices.

    Returns:
        bool: True if cities are sequential, False otherwise.
    """
    return set(range(min(cluster['cities']), max(cluster['cities']))).issubset(set(cluster['cities']))


def is_all_sequential_clusters(clusters: dict) -> bool:
    """
    Checks whether all clusters consist of sequential city indices.

    Args:
        clusters (dict): Dictionary of clusters.

    Returns:
        bool: True if all clusters are sequential, False otherwise.
    """
    return all(map(is_sequential_cluster, clusters.values()))


def calculate_cluster_params(cluster: np.array, cost_matrix: np.array) -> dict:
    """
    Calculates statistical parameters (mean and std) for a cluster
    based on the distances in the cost matrix.

    Args:
        cluster (np.array): List of city indices in the cluster.
        cost_matrix (np.array): Full distance matrix between cities.

    Returns:
        dict: Dictionary with 'mean' and 'std' of intra-cluster distances.
    """
    first_city = min(cluster)
    last_city = max(cluster)

    mask_upper_triangle = np.triu(np.ones_like(cost_matrix, dtype=bool), k=1)
    mask_lower_triangle = np.tril(np.ones_like(cost_matrix, dtype=bool), k=-1)

    upper_triangle_elements = cost_matrix[mask_upper_triangle]
    lower_triangle_elements = cost_matrix[mask_lower_triangle]

    valid_upper_elements = upper_triangle_elements[(mask_upper_triangle.nonzero()[1] >= first_city) & (mask_upper_triangle.nonzero()[1] <= last_city)]
    valid_lower_elements = lower_triangle_elements[(mask_lower_triangle.nonzero()[0] >= first_city) & (mask_lower_triangle.nonzero()[0] <= last_city)]

    combined = np.concatenate([valid_upper_elements, valid_lower_elements])
    mean_value = np.mean(combined)
    std = np.std(combined)
    
    return dict(mean=mean_value, std=std)

def calculate_clusters_k_parameter(clusters: dict) -> None:
    """
    Calculates the k-parameter for each cluster based on the difference
    of mean distances relative to the previous cluster.

    Args:
        clusters (dict): Dictionary of clusters with 'params' keys already filled with 'mean'.
    """
    previous_cluster = clusters.get(1)
    previous_cluster_mean = previous_cluster['params']['mean']
    for key in sorted(clusters.keys())[1:]:
      current_cluster = clusters[key]
      current_cluster_mean = current_cluster['params']['mean']
      mean_diff = abs(current_cluster_mean - previous_cluster_mean)
      current_cluster['params']['k'] = mean_diff  ** (1 / (key-1))
      previous_cluster = current_cluster
      previous_cluster_mean = current_cluster_mean

  
def is_valid_k(clusters: np.array) -> bool:
    """
    Checks if the average k-value meets the threshold condition for a phase transition.

    Args:
        clusters (np.array): Dictionary of clusters with 'k' values inside 'params'.

    Returns:
        bool: True if the average k is valid, False otherwise.
    """
    k_values = [cluster['params'].get('k') for cluster in clusters.values() if 'k' in cluster['params']]
    mean_k = np.mean(np.array(k_values))

    if mean_k <= -0.5*(len(clusters.keys())+1) + 6:
        return False
    return True

 
def is_radian_clusters_phase_transition(cost_matrix: np.array, labels: np.array) -> bool:
    """
    Determines whether a radial cluster-based phase transition has occurred.

    Args:
        cost_matrix (np.array): Distance matrix between cities.
        labels (np.array): Cluster labels assigned to each city.

    Returns:
        bool: True if a phase transition is detected, False otherwise.
    """
    clusters = cities_clusters(labels)

    if len(np.unique(labels)) == 1 or not is_all_sequential_clusters(clusters):
        return False

    for cluster in clusters.values():
        cluster['params'] = calculate_cluster_params(cluster['cities'], cost_matrix)
    calculate_clusters_k_parameter(clusters)


    return is_valid_k(clusters)

def normalize(data: np.array, scaler = StandardScaler) -> np.array:
    """
    Applies normalization to the data using a specified scaler.

    Args:
        data (np.array): Data to normalize.
        scaler (type): Scikit-learn scaler class (default: StandardScaler).

    Returns:
        np.array: Normalized data.
    """
    scaler = scaler()
    return scaler.fit_transform(data)

def get_preprocessed_data(cost_matrix: np.array) -> np.array:
    """
    Preprocesses the cost matrix into a 2D feature representation using t-SNE.

    Args:
        cost_matrix (np.array): Distance matrix between cities.

    Returns:
        np.array: 2D representation of each city for clustering.
    """
    count_rows = cost_matrix.shape[0]
    mask = ~np.eye(count_rows, dtype=bool)
    data1 = cost_matrix[mask].reshape(count_rows, -1)
    data2 = cost_matrix.T[mask].reshape(count_rows, -1)

    data = np.hstack((data1, data2))

    tsne = TSNE(n_components=2, random_state=42, perplexity=int(cost_matrix.shape[0]*0.3))
    reduced_features = normalize(tsne.fit_transform(data), StandardScaler)

    return reduced_features

def detect_clusters(cost_matrix: np.array, model: ClusterModelR = ClusterModelR.KMEANS) -> np.array:
    """
    Applies the specified clustering model to the cost matrix.

    Args:
        cost_matrix (np.array): Distance matrix.
        model (ClusterModelR): Clustering model to use.

    Returns:
        np.array: List of cluster labels.
    """
    match(model):
        case ClusterModelR.KMEANS:
            return cluster_KMEANS(cost_matrix)
        case ClusterModelR.OPTICS:
            return cluster_OPTICS(cost_matrix)
        case ClusterModelR.AGGLOMERATIVE:
            return cluster_AGGLO(cost_matrix)
        
def cluster_AGGLO(cost_matrix):
    data = get_preprocessed_data(cost_matrix)
    k_values = range(2, cost_matrix.shape[0])
    best_count_clusters = 1
    best_silhouette = -1

    for k in k_values:
      model = AgglomerativeClustering(n_clusters=k, linkage='complete')
      labels = model.fit_predict(data)

      if len(np.unique(labels)) > 1:
            score = silhouette_score(data, labels)
            if score > best_silhouette:
                    best_silhouette = score
                    best_count_clusters = k

    optimal_k = best_count_clusters

    model = AgglomerativeClustering(n_clusters=optimal_k, linkage='complete')
    optimal_labels = model.fit_predict(data)

    return optimal_labels

def cluster_KMEANS(cost_matrix: np.array) -> np.array:
    """
    Performs KMeans clustering with silhouette score to determine optimal number of clusters.

    Args:
        cost_matrix (np.array): Distance matrix.

    Returns:
        list: Optimal cluster labels for each city.
    """
    data = get_preprocessed_data(cost_matrix)

    k_values = range(2, cost_matrix.shape[0])
    best_count_clusters = 1
    best_silhouette = -1
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        labels = kmeans.labels_
        if len(np.unique(labels)) > 1:
            score = silhouette_score(data, labels)
            if score > best_silhouette:
                    best_silhouette = score
                    best_count_clusters = k

    optimal_k = best_count_clusters

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    optimal_labels = kmeans.fit_predict(data)
    return np.array(optimal_labels)

def cluster_OPTICS(cost_matrix: np.array) -> np.array:
    """
    Performs OPTICS clustering on the normalized distance matrix.

    Args:
        cost_matrix (np.array): Distance matrix.

    Returns:
        np.array: Cluster labels.
    """
    data = normalize(cost_matrix, StandardScaler)

    optics = OPTICS(min_samples=2)
    labels = optics.fit_predict(data)

    return np.array(labels)

def detect_radian_phase_transition(distance_matrix: np.array, cluster_method: ClusterModelR = ClusterModelR.KMEANS) -> bool:
    """
    Detects a radial phase transition based on distance matrix and clustering method.

    Args:
        distance_matrix (np.array): Distance matrix of the problem instance.
        cluster_method (ClusterModelR): Clustering model to use for cluster detection.

    Returns:
        bool: True if phase transition is detected, False otherwise.
    """
    cluster_labels = detect_clusters(model=cluster_method, cost_matrix=distance_matrix)

    is_radian_pt = is_radian_clusters_phase_transition(cost_matrix=distance_matrix, labels=cluster_labels)
    return is_radian_pt
