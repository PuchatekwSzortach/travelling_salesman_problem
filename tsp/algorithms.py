import numpy as np
import itertools

def get_random_distances_matrix(cities_number, max_distance):
    """
    :param cities_number:
    :param max_distance:
    :return: matrix of distances from. E.g. element (i,j) defines distance from city i to city j
    """

    asymmetric_distances_matrix = np.random.random_integers(1, max_distance, size=(cities_number, cities_number))
    distances_matrix = (asymmetric_distances_matrix + asymmetric_distances_matrix.transpose()) / 2

    # Distance from a city to itself should be 0
    distances_matrix[np.diag_indices(cities_number)] = 0

    return distances_matrix


def get_trip_distance(distances_matrix, path):
    """
    :param distances_matrix: Matrix of distances between cities
    :param path: List of city indices
    :return: Trip distance
    """

    distance = 0

    for index in range(len(path))[1:]:

        distance += distances_matrix[path[index - 1], path[index]]

    return distance


class BruteForceTSPSolver:
    """
    Travelling salesman problem solver that uses brute force to compute optimal solution.
    """

    def __init__(self, distances_matrix):

        self.distances_matrix = distances_matrix

    def solve(self):
        """
        Solve travelling salesman problem solver problem given distances grid.
        Returns an array of cities indices defining optimal trip.
        :param distances_matrix:
        :return:
        """

        # Generate all possible paths
        cities_number = self.distances_matrix.shape[0]
        paths = itertools.permutations(range(cities_number))

        best_path = next(paths)
        best_path_distance = get_trip_distance(self.distances_matrix, best_path)

        for path in paths:

            path_distance = get_trip_distance(self.distances_matrix, path)

            if path_distance < best_path_distance:

                best_path = path
                best_path_distance = path_distance

        return best_path
