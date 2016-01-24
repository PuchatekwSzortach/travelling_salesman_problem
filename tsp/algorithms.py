import numpy as np

def get_random_distances_grid(cities_number, max_distance):
    """
    :param cities_number:
    :param max_distance:
    :return: matrix of distances from. E.g. element (i,j) defines distance from city i to city j
    """

    asymetric_distances_grid = np.random.random_integers(1, max_distance, size=(cities_number, cities_number))
    distances_grid = (asymetric_distances_grid + asymetric_distances_grid.transpose()) / 2

    # Distance from a city to itself should be 0
    distances_grid[np.diag_indices(cities_number)] = 0

    return distances_grid
