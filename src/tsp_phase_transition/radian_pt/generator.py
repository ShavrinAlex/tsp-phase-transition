import numpy as np
import random
import math

def create_labels_by_radius(radius: np.array) -> np.array:
    """
    Creates a label array based on the given radius counts.

    For each value in the `radius` array, appends that many copies of its index
    to the result. For example, if radius = [2, 3], the result will be [0, 0, 1, 1, 1].

    Parameters
    ----------
    radius : np.array
        A 1D array where each element represents the number of items at that radius.

    Returns
    -------
    np.array
        An array of labels where each label corresponds to a radius index.
    """
    labels = []
    for i in range(radius.shape[0]):
        labels.extend([i]*radius[i])
    return np.array(labels)


def get_s(count: int, k: int, s1_mean, s1_std):
    """
    Generates a list of dictionaries containing mean and standard deviation values.

    Starts with the initial mean (`s1_mean`) and standard deviation (`s1_std`),
    then recursively updates the mean by raising it to the power of `k` for each step.

    Parameters
    ----------
    count : int
        Number of elements (layers) to generate.
    k : int
        Power by which the s1_mean is updated at each step.
    s1_mean : float
        The starting mean value.
    s1_std : float
        The standard deviation to assign to each element.

    Returns
    -------
    list of dict
        A list of dictionaries, each containing a "mean" and "std" key.
    """
    std = s1_std
    cur_mean = s1_mean
    s = [dict(mean=cur_mean, std=std)]
    for i in range(1, count):
        s.append(dict(mean=cur_mean + k**i, std=std))
    return s

def generate_pt_matrix(count_cities: int, radius: np.array, k: int, generate_params: dict):
    """
    Generates a weighted distance (or similarity) matrix with clustered structure based on radius and shape parameters.

    This function divides cities into radial clusters and assigns weights to each pair of cities using
    a custom-shaped region defined by Gaussian-like layers.

    Parameters
    ----------
    count_cities : int
        Total number of cities (nodes) in the matrix.
    radius : np.array
        1D array where each value represents the number of cities in each radial cluster.
    k : int
        Power parameter used to modify cluster mean values.
    generate_params : dict
        Dictionary containing generation parameters:
            - 's1_mean': float, initial mean value for weight generation
            - 's1_std': float, standard deviation
            - 'center': dict with keys 'x' and 'y', defining the center of the cluster region
            - 'rotation': float, rotation angle in degrees
            - 'is_symmetric': bool, whether the resulting matrix should be symmetric

    Returns
    -------
    np.ndarray
        A square matrix (count_cities x count_cities) filled with generated weights, symmetric if specified.

    Notes
    -----
    - Self-loops (diagonal values) are set to zero.
    """
    count_radius = len(radius)

    s = get_s(count=count_radius, k=k, s1_mean=generate_params['s1_mean'],  s1_std=generate_params['s1_std'])

    weights = np.zeros((count_cities, count_cities))
    for index, cluster in enumerate(s[::-1]):
      mean = cluster['mean']
      std = cluster['std']
      weights = generate_custom_shape_cluster(weights, generate_params['center'], dict(start=mean-std, stop=mean+std), np.sum(radius[index:]), rotation=generate_params['rotation'], roundness=0)
    if generate_params['is_symmetric']:
      weights = (weights + weights.T) / 2

    np.fill_diagonal(weights, 0)
    return weights


def generate_custom_shape_cluster(matrix, center, generate_params, size=5, rotation=0, roundness=0, stretch=1):
    """
    Modifies a matrix by filling in a rotated and optionally rounded square or elliptical region
    with logarithmic random values from a specified range.

    Parameters
    ----------
    matrix : np.ndarray
        The square matrix to modify.
    center : dict
        Dictionary with keys 'x' and 'y' defining the center of the shape.
    generate_params : dict
        Dictionary with 'start' and 'stop' defining the range for random integer values (used in log scale).
    size : int, optional
        Size (radius) of the region to generate (default is 5).
    rotation : float, optional
        Rotation angle of the region in degrees (default is 0).
    roundness : float, optional
        Degree of roundness from 0 (square) to 100 (circle), default is 0.
    stretch : float, optional
        Vertical stretch factor for the ellipse shape (default is 1).

    Returns
    -------
    np.ndarray
        The modified matrix with generated values in the shape region.

    Notes
    -----
    - Points within the defined region are assigned log10-scaled random values.
    - The shape is an interpolation between a square and a stretched ellipse.
    """
    n = matrix.shape[0]
    center_x, center_y = center['x'], center['y']

    angle_rad = math.radians(rotation)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    diagonal_radius = size * math.sqrt(2)

    roundness = max(0, min(roundness, 100)) / 100.0

    for i in range(n):
        for j in range(n):
            x = i - center_x + 1
            y = j - center_y + 1

            x_rot = cos_a * x - sin_a * y
            y_rot = sin_a * x + cos_a * y

            abs_x = abs(x_rot) / size
            abs_y = abs(y_rot) / (size * stretch)

            square_condition = max(abs_x, abs_y)

            ellipse_condition = (x_rot**2 + (y_rot / stretch)**2)**0.5 / diagonal_radius

            shape_value = (1 - roundness) * square_condition + roundness * ellipse_condition

            if shape_value <= 1:
                matrix[i, j] = random.randint(generate_params['start'], generate_params['stop'])

    return matrix