import numpy as np
import itertools
import pprint

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

        print("Weights")
        for city_index in range(nodes_number):
            for tour_step_index in range(nodes_number):
                print("City index {}".format(city_index))
                print("Tour step index {}".format(tour_step_index))
                pprint.pprint(self.weights[city_index, tour_step_index])

    def get_initialized_weights_matrix(self, nodes_number, bias, penalty):

        weights = np.zeros([nodes_number, nodes_number, nodes_number, nodes_number])

        for city_index in range(nodes_number):

            for tour_step_index in range(nodes_number):

                # Select a 2D matrix of weights for this node
                node_weights = weights[city_index, tour_step_index]

                # distances = self.distances_matrix[city_index, :]
                # # Set distances to other cities on adjacent trips
                #
                # # For trip at previous stage. For first step wire it back to last step, so that starting position
                # # doesn't matter
                # previous_tour_step_index = tour_step_index - 1 if tour_step_index > 0 else nodes_number - 1
                # node_weights[previous_tour_step_index, :] = -distances
                #
                # # For trip at next stage. For last step wire it back to first step, so that starting position
                # # doesn't matter
                # next_tour_step_index = tour_step_index + 1 if tour_step_index < nodes_number - 1 else 0
                # node_weights[next_tour_step_index, :] = -distances

                # Penalty for visiting other cities at that tour step
                node_weights[:, tour_step_index] = -penalty

                # Penalty for visiting same city at other tour steps
                node_weights[city_index, :] = -penalty

                node_weights[city_index, tour_step_index] = bias

        return weights

    def get_weights_to_cities_at_previous_tour_step(self, city_index, tour_step, nodes_number):

        previous_city_index = city_index - 1

        # Wrap around if to last city in the matrix if needed
        if previous_city_index == -1:
            previous_city_index = nodes_number - 1

        weights = self.distances_matrix[city_index, :]


