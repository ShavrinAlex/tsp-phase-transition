from enum import Enum
import numpy as np
from typing import Generator, Tuple

class SortDirection(Enum):
    """Enumeration for partition sorting directions."""
    ASC = 1  # Ascending order
    DESC = 2  # Descending order
    RAND = 3  # Random order

def partitions_generator(n: int, k: int, min_part: int = 1) -> Generator[Tuple[int, ...], None, None]:
    """
    Generate all partitions of integer n into k parts with each part >= l.
    
    Args:
        n: The integer to partition
        k: The length of partitions
        min_part: The minimum partition element size (default: 1)
        
    Yields:
        Tuples representing partitions of n into k parts
    """
    if k < 1:
        return
    if k == 1:
        if n >= min_part:
            yield (n,)
        return
    for i in range(min_part, n//k + 1):
        for result in partitions_generator(n - i, k - 1, i):
            yield (i,) + result

def get_partitions(n: int, k: int, min_part: int = 1) -> np.ndarray:
    """
    Get all partitions of n into k parts as a numpy array.
    
    Args:
        n: The integer to partition
        k: The number of parts
        min_part: Minimum value for each part (default: 1)
        
    Returns:
        numpy.ndarray: Array of partitions (each row is a partition)
    """
    partitions = list(partitions_generator(n, k, min_part))
    return np.array(partitions)

def get_most_balanced_partition_from_partitions(partitions: np.ndarray) -> np.ndarray:
    """
    Find the most balanced partition (with minimal standard deviation).
    
    Args:
        partitions: Array of partitions (from get_partitions)
        
    Returns:
        numpy.ndarray: The most balanced partition
    """
    std_devs = np.std(partitions, axis=1)
    min_index = np.argmin(std_devs)
    return partitions[min_index]

def get_most_balanced_partition(n: int, k: int, min_part: int = 1) -> np.ndarray:
    """
    Find the most balanced partition (with minimal standard deviation).
    
    Args:
        n: The integer to partition
        k: The length of partitions
        min_part: The minimum partition element size (default: 1)
        
    Returns:
        numpy.ndarray: The most balanced partition
    """
    partitions = get_partitions(n, k, min_part)
    return get_most_balanced_partition_from_partitions(partitions)

def get_random_partition_from_partitions(partitions: np.ndarray, 
                        sort_flag: SortDirection = SortDirection.ASC) -> np.ndarray:
    """
    Get a random partition with optional sorting.
    
    Args:
        partitions: Array of partitions (from get_partitions)
        sort_flag: How to sort the partition (ASC, DESC, or RAND)
        
    Returns:
        numpy.ndarray: Randomly selected (and optionally sorted) partition
    """
    random_index = np.random.choice(partitions.shape[0])
    random_partition = partitions[random_index]
    
    if sort_flag == SortDirection.ASC:
        return np.sort(random_partition)
    elif sort_flag == SortDirection.DESC:
        return np.sort(random_partition)[::-1]
    else:  # SortDirection.RAND
        return np.random.permutation(random_partition)
    
def get_random_partition(n: int, k: int, min_part: int = 1, sort_flag: SortDirection = SortDirection.ASC):
    """
    Get a random partition with optional sorting.
    
    Args:
        n: The integer to partition
        k: The length of partitions
        min_part: The minimum partition element size (default: 1)
        sort_flag: How to sort the partition (ASC, DESC, or RAND)
        
    Returns:
        numpy.ndarray: Randomly selected (and optionally sorted) partition
    """
    partitions = get_partitions(n, k, min_part)
    return get_random_partition_from_partitions(partitions, sort_flag)
    