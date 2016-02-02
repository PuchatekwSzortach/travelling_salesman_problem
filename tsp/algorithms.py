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


class BoltzmannMachineTSPSolver:
    """
    Travelling salesman problem solver that uses Boltzmann machine to compute a solution
    """

    def __init__(self, distances_matrix):

        self.distances_matrix = distances_matrix

        nodes_number = distances_matrix.shape[0]

        # Grid of nodes represents possible path combinations.
        # Each row represents a city and each column a position in the tour.
        # So node(1, 2) represents city no 3 visited at step no 4
        self.nodes = np.zeros([nodes_number, nodes_number])

        max_distance = np.max(self.distances_matrix)

        # Compute bias and penalty according to inequalities penalty > bias > 2 * max_distance
        bias = 2.5 * max_distance
        penalty = 3 * max_distance

        # Each node from the 2D grid is connected to all other nodes (including itself), hence
        # weights matrix is 4D.
        # For each node at grid position (i, j):
        # Node (i, j) has a self-connection of weight b representing desirability of visiting city i at stage j
        # Node (i, j) is connected to all other units in row i with a penalty weight -p.
        # This represents the constraints that the same city is not visited more than once.
        # Node (i, j) is connected to all other units in the column j with a penalty weight -p.
        # This represents the constrained that only one city can be visited at a single trip stage
        # Node (i, j) is connected to nodes(k, j+1) for 0 <= k < nodes_number, k != i with weight -d(i,k)
        # where d is a matrix of distance.
        # This represents cost of transitioning from city i at stage j to city k at stage j+1
        # Node (i, j) is connected to nodes(k, j-1) for 0 <= k < nodes_number, k != i with weight -d(i,k)
        # where d is a matrix of distance.
        # This represents cost of transitioning from city k at stage j - 1to city i at stage j
        self.weights = self.get_initialized_weights_matrix(nodes_number, bias, penalty)

    def get_initialized_weights_matrix(self, nodes_number, bias, penalty):

        weights = np.zeros([nodes_number, nodes_number, nodes_number, nodes_number])
        return weights