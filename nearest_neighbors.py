import pandas as pd
import numpy as np


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def k_nearest_neighbors(data, n, k):
    query_point = data[n]

    distances = [(i, euclidean_distance(query_point, data[i])) for i in range(len(data)) if i != n]
    sorted_distances = sorted(distances, key=lambda x: x[1])[:k]

    nearest_neighbors_indices = [index for index, _ in sorted_distances]
    nearest_neighbors = [data[index] for index in nearest_neighbors_indices]

    return nearest_neighbors


data = pd.read_csv("penguins.csv")
data = data.dropna()

new_data = data[["bill_length_mm", "bill_depth_mm"]].values

neighbors = k_nearest_neighbors(new_data, int(input()), int(input()))

for neighbor in neighbors:
    print(neighbor)
