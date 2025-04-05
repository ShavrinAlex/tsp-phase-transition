import numpy as np
import matplotlib.pyplot as plt


def show_heatmap(cost_matrix: np.ndarray):
    """
    Display a heatmap visualization of the cost matrix using matplotlib.
    
    Args:
        cost_matrix: Square cost/distance matrix (n x n numpy array) where
                   n is the number of cities/points.
    
    Note:
        - Uses 'viridis' colormap by default
        - Automatically adjusts figure layout
    """
    plt.figure(figsize=(8, 6))
    
    # Создаем тепловую карту
    plt.imshow(cost_matrix, cmap='viridis', interpolation='nearest')
    
    # Добавляем цветовую шкалу
    cbar = plt.colorbar()
    cbar.set_label('Значения')
    
    # Настраиваем оси
    plt.xlabel('Города')
    plt.ylabel('Города')
    
    
    plt.tight_layout()
    plt.show()

def show_clusters(coords: np.array, 
                 centers: np.array = None, 
                 radiuses: np.array = None, 
                 labels: np.array = None):
    """
    Visualize clustered points with optional centers, radii, and cluster labels.
    
    Args:
        coords: Array of point coordinates (N x 2)
        centers: Optional array of cluster centers (M x 2)
        radiuses: Optional array of cluster radii (M,)
        labels: Optional array of cluster labels (N,)
    
    Raises:
        ValueError: If length of radiuses doesn't match number of centers
    
    Note:
        - Uses 'tab20' colormap for cluster labels
        - Automatically handles equal aspect ratio
        - Adds legend only when centers are shown
    """
    plt.figure(figsize=(10, 8))
    
    # Настройка цветов в зависимости от наличия меток
    if labels is not None:
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab20', alpha=0.6)
    else:
        scatter = plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6)
    
    # Отрисовка центров, если они переданы
    if centers is not None:
        centers_scatter = plt.scatter(centers[:, 0], centers[:, 1], 
                                    c='black', s=100, marker='x', 
                                    label='Centers')
        
        # Отрисовка радиусов, если переданы и centers есть
        if radiuses is not None:
            if len(radiuses) != len(centers):
                raise ValueError("Количество радиусов должно совпадать с количеством центров")
            
            for i, center in enumerate(centers):
                circle = plt.Circle(center, radiuses[i], 
                                  color='gray', fill=True, 
                                  linestyle='-', alpha=0.4)
                plt.gca().add_patch(circle)
    
    plt.axis('equal')
    plt.xlabel("X Координата")
    plt.ylabel("Y Координата")
    
    # Добавление легенды только если есть центры
    if centers is not None:
        plt.legend()
    
    # Добавление цветовой шкалы если есть метки
    if labels is not None:
        plt.colorbar(scatter, label='Cluster labels')
    
    plt.show()