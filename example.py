from tsp_phase_transition.utils import vizualizer
from tsp_phase_transition.partition.generator import get_partitions, get_random_partition, SortDirection, get_most_balanced_partition
from tsp_phase_transition.cluster_pt import generator as cluster_pt_generator, detector as cluster_pt_detector
from tsp_phase_transition.radian_pt import generator as radian_pt_generator, detector as radian_pt_detector

N = 12 # Количество городов
k = 3 # Количество кластеров

""" Пример генерации разбиения (для формирования соотношений городов в кластерах и др)"""
partitions = get_partitions(N, k, min_part=1)
print("partitions: ", partitions)

partition = get_random_partition(N, k, 1, sort_flag=SortDirection.RAND) # Генерация случайного разбиения с сортировкой значений внутри
print("random partition: ", partition)

""" Пример генерации кластерного фазового перехода"""
plane_size = 300 # Указывем размеры генерации
cities_partition = get_random_partition(N, k, 1, sort_flag=SortDirection.RAND) # Генерируем случайное распределение городов по кластерам
radius_partition = get_random_partition(plane_size//5, k, 1, sort_flag=SortDirection.RAND) # Генерируем радиусы для кластеров
print("radius partition: ", radius_partition)
points, labels, centers, dist_matrix = cluster_pt_generator.generate_clusters(N, k, radius_partition, cities_partition, plane_size=plane_size)
vizualizer.show_clusters(coords=points, centers=centers, radiuses=radius_partition, labels=labels) # Отображаем сгенерированные кластеры
vizualizer.show_heatmap(dist_matrix) # Отображение матрицы в виде тепловой карты

""" Пример определения кластерного фазового перехода"""
is_cluster_pt = cluster_pt_detector.detect_cluster_pt(dist_matrix, cluster_method=cluster_pt_detector.ClusterModelC.AGGLOMERATIVE)
print("is cluster phase transition: ", is_cluster_pt)

""" Пример генерации радиального фазового перехода"""
generate_params = dict(
    s1_mean = 5, # среднее значение в первом кластере
    s1_std = 4, # максимальное отклонение в кластерах
    is_symmetric = False,
    center=dict(x=0, y=0), # точка расхождения радиальных кластеров
    rotation=0, # угол поворота кластера
)
coef = 4 # коэффициент отличия между кластерами
radius = get_most_balanced_partition(n=N, k=k, min_part=2)
print("radius partition: ", radius)
cost_matrix = radian_pt_generator.generate_pt_matrix(N, radius, coef, generate_params)
vizualizer.show_heatmap(cost_matrix)

""" Пример определения радиального фазового перехода"""
is_radian_pt = radian_pt_detector.detect_radian_phase_transition(distance_matrix=cost_matrix, cluster_method=radian_pt_detector.ClusterModelR.KMEANS)
print("is radian phase transition: ", is_radian_pt)
