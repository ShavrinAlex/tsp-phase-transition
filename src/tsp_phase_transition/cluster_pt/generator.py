import numpy as np
from typing import Tuple, List
from scipy.spatial.distance import pdist, squareform

def generate_uniform_cluster(n: int, radius: float, center: np.ndarray) -> np.ndarray:
    """
    Generate cities uniformly distributed within a circle.
    
    Args:
        n: Number of cities
        radius: Cluster radius
        center: (x, y) coordinates of cluster center
        
    Returns:
        numpy.ndarray: Array of city coordinates (n x 2)
    """
    angles = np.random.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(np.random.uniform(0, radius**2, n))
    
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    
    return np.column_stack((x, y))

def generate_clusters(
    n: int,
    k: int,
    radius_partition: List[float],
    cities_partition: List[int],
    min_distance: float = 0,
    plane_size: float = 100,
    max_attempts: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate clustered cities with specified parameters for phase transition detection.
    
    Generates k clusters with:
    - Each cluster having specified radius and number of cities
    - Cluster centers spaced according to minimum distance requirements
    - Cities uniformly distributed within each cluster
    
    Args:
        n: Total number of cities to generate
        k: Number of clusters to generate
        radius_partition: List of radius values for each cluster (length k)
        cities_partition: List of city counts for each cluster (length k, sum must equal n)
        min_distance: Minimum required distance between cluster edges (default: 0)
        plane_size: Size of the square plane where clusters are placed (default: 100)
        max_attempts: Maximum attempts to place each cluster before giving up (default: 1000)
        
    Returns:
        Tuple containing:
        - points: Array of city coordinates (n x 2)
        - labels: Array of cluster labels for each city (n,)
        - centers: Array of cluster center coordinates (k x 2)
        
    Raises:
        ValueError: If input validation fails or cluster placement is impossible
    """

    if len(radius_partition) != k or len(cities_partition) != k:
        raise ValueError("Length of radius_partition and cities_partition must equal k")
    if sum(cities_partition) != n:
        raise ValueError("Sum of cities in cities_partition must equal n")
    if any(r <= 0 for r in radius_partition):
        raise ValueError("Cluster radii must be positive")
    if any(c <= 0 for c in cities_partition):
        raise ValueError("Each cluster must contain at least one city")

    centers = []
    for i in range(k):
        current_radius = radius_partition[i]
        placed = False
        
        for attempt in range(max_attempts):
            new_center = np.random.uniform(current_radius, 
                                         plane_size - current_radius, 
                                         2)
            
            valid = True
            for j, existing_center in enumerate(centers):
                existing_radius = radius_partition[j]
                center_distance = np.linalg.norm(new_center - existing_center)
                required_distance = current_radius + existing_radius + min_distance
                
                if center_distance < required_distance:
                    valid = False
                    break
            
            if valid:
                centers.append(new_center)
                placed = True
                break
        
        if not placed:
            raise ValueError(
                f"Failed to place cluster {i} after {max_attempts} attempts. "
                f"Consider increasing plane_size or decreasing min_distance."
            )

    centers = np.array(centers)

    all_points = []
    all_labels = []
    
    for cluster_idx in range(k):
        cluster_points = generate_uniform_cluster(
            n=cities_partition[cluster_idx],
            radius=radius_partition[cluster_idx],
            center=centers[cluster_idx]
        )
        all_points.append(cluster_points)
        all_labels.extend([cluster_idx] * cities_partition[cluster_idx])

    points = np.vstack(all_points)
    labels = np.array(all_labels)
    
    shuffle_idx = np.random.permutation(len(points))
    return points[shuffle_idx], labels[shuffle_idx], centers, squareform(pdist(points))