import numpy as np
import itertools


def get_random_distances_matrix(cities_number, max_distance):
    """
    :param cities_number:
    :param max_distance:
    :return: matrix of distances between cities. E.g. element (i,j) defines distance from city i to city j
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


# Given a square matrix of binary values, check if it represents a legal configuration for a
# travelling salesman problem. Configuration is deemed to be legal if
# in each row and each column exactly one node is set
def is_nodes_configuration_legal(nodes):

    for row_index in range(nodes.shape[0]):

        if np.sum(nodes[row_index, :]) != 1:
            return False

    for column_index in range(nodes.shape[1]):

        if np.sum(nodes[:, column_index]) != 1:
            return False

    # Seems we passed all the checks, so configuration is legal
    return True


# Given a square matrix of binary values representing a legal tsp path, return the path as a list of city indices
# visited at each path stage.
def get_path_from_nodes_configuration(nodes):

    path = []

    for row_index in range(nodes.shape[0]):

        path.append(np.argmax(nodes[row_index, :]))

    return path


class BruteForceTSPSolver:
    """
    Travelling salesman problem solver that uses brute force to compute optimal solution.
    """

    def __init__(self, distances_matrix):

        self.distances_matrix = distances_matrix

    def solve(self):
        """
        Solve travelling salesman.
        Returns a list of cities indices defining optimal trip.
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


class BruteForceTSPWorstPathSolver:
    """
    Travelling salesman problem solver that uses brute force to compute worst possible solution.
    """

    def __init__(self, distances_matrix):

        self.distances_matrix = distances_matrix

    def solve(self):
        """
        Solve travelling salesman problem.
        Returns a list of cities indices defining worst possible trip.
        """

        # Generate all possible paths
        cities_number = self.distances_matrix.shape[0]
        paths = itertools.permutations(range(cities_number))

        worst_path = next(paths)
        worst_path_distance = get_trip_distance(self.distances_matrix, worst_path)

        for path in paths:

            path_distance = get_trip_distance(self.distances_matrix, path)

            if path_distance > worst_path_distance:

                worst_path = path
                worst_path_distance = path_distance

        return worst_path


class BoltzmannMachineTSPSolver:
    """
    Travelling salesman problem solver that uses Boltzmann machine to compute a solution
    """

    def __init__(self, distances_matrix):

        self.distances_matrix = distances_matrix
        self.cities_number = distances_matrix.shape[0]
        self.nodes_number = self.cities_number * self.cities_number

        # Grid of nodes represents possible path combinations.
        # Each row represents a city and each column a position in the tour.
        # So node(1, 2) represents city no 2 visited at step no 3.
        # And since we need to have an initial legal path, set initial nodes
        # configuration to an eye matrix
        self.nodes = np.eye(self.cities_number)

        max_distance = np.max(self.distances_matrix)

        # Compute bias and penalty according to inequalities penalty > bias > 2 * max_distance
        bias = 2.5 * max_distance
        penalty = 2 * bias

        self.weights = self._get_weights_matrix(bias, penalty)
        self.temperature = self._get_initial_temperature(bias, penalty)

    def _get_weights_matrix(self, bias, penalty):
        """
        Computes weights matrix.
        Each node from the 2D grid is connected to all other nodes (including itself), hence
        weights matrix is 4D.

        For each node at grid position (i, j):

            Node has a self-connection of weight b representing desirability of visiting city i at stage j

            Node is connected to all other units in row i with a penalty weight -p.
            This represents the constraint that the same city is not visited more than once.

            Node is connected to all other units in the column j with a penalty weight -p.
            This represents the constraint that only one city can be visited at a single trip stage

            Node is connected to nodes(k, j+1) for 0 <= k < nodes_number, k != i with weight -d(i,k)
            where d is a distance matrix.
            This represents cost of transitioning from city i at stage j to city k at stage j+1

            Node is connected to nodes(k, j-1) for 0 <= k < nodes_number, k != i with weight -d(i,k)
            where d is a distance matrix.
            This represents cost of transitioning from city k at stage j - 1 to city i at stage j

        :param bias:
        :param penalty:
        :return: 4D weights matrix representing connections from each node to all network nodes
        """

        weights = np.zeros([self.cities_number, self.cities_number, self.cities_number, self.cities_number])

        for city_index in range(self.cities_number):

            for tour_step_index in range(self.cities_number):

                # Select a 2D matrix of weights for this node
                node_weights = weights[city_index, tour_step_index]

                # Distances between current city and other cities
                distances = self.distances_matrix[city_index, :]

                # Set costs for going from city at previous trip step to current city.
                # If this is the first trip step, use last step column, so that starting position
                # doesn't matter
                previous_tour_step_index = tour_step_index - 1 if tour_step_index > 0 else self.cities_number - 1
                node_weights[:, previous_tour_step_index] = -distances

                # Set costs for going from city at current trip step to city in next trip step.
                # If this is the last trip step, use first step column, so that finish position
                # doesn't matter
                next_tour_step_index = tour_step_index + 1 if tour_step_index < self.cities_number - 1 else 0
                node_weights[:, next_tour_step_index] = -distances

                # Penalty for visiting other cities at that tour step
                node_weights[:, tour_step_index] = -penalty

                # Penalty for visiting same city at other tour steps
                node_weights[city_index, :] = -penalty

                # Benefit gained from visiting the city
                node_weights[city_index, tour_step_index] = bias

        return weights

    def _get_initial_temperature(self, bias, penalty):
        """
        Compute initial temperature.
        We want initial temperature to be so high that any change in consensus, both positive and negative,
        will be equally likely to be accepted. This effectively means temperature should be significantly higher
        than highest change in consensus that can occur - say 100 times higher (at which point probability
        of accepting the change will be roughly within 1% from fifty-fifty for even highest consensus change).
        A high consensus change would result from moving into a highly illegal configuration, say
        revisiting the same city cities_number times and being in all cities at the same time.
        :param bias:
        :param penalty:
        :return: initial temperature
        """

        total_penalty = penalty * (self.cities_number - 1) * (self.cities_number - 1)
        return 100 * (total_penalty - bias)

    def solve(self):

        last_legal_configuration = self.nodes.copy()

        # At temperature 0.1 probability of configuration change is virtually 0
        while self.temperature > 0.1:

            for _ in range(self.nodes_number**2):

                # Get random coordinates for a node
                i, j = np.random.random_integers(0, self.cities_number - 1, 2)

                consensus_change = self._get_consensus_change(i, j)
                change_probability = self._get_activation_change_probability(consensus_change, self.temperature)

                # Change node activation with change_probability
                if np.random.binomial(1, change_probability) == 1:

                    # Change node activation
                    self.nodes[i, j] = 1 - self.nodes[i, j]

                    if is_nodes_configuration_legal(self.nodes):
                        last_legal_configuration = self.nodes.copy()

            # Decrease temperature
            self.temperature *= 0.98

        return get_path_from_nodes_configuration(last_legal_configuration)

    def _get_consensus_change(self, i, j):

        node_value = self.nodes[i, j]
        sign = 1 - node_value

        node_weights = self.weights[i, j]

        weights_effect = np.sum(node_weights * self.nodes)

        # To compute weights effect of other nodes in a single step, we used an element-wise matrix
        # multiplication that. That included in weights effect of the node we consider and we will have
        # to remove that effect now. We also need to include benefit from visiting the node,
        # so do both operations in a single step.
        weights_effect += (1 - node_value) * node_weights[i, j]

        return sign * weights_effect

    def _get_activation_change_probability(self, consensus_change, temperature):

        exponential_argument = -1 * consensus_change / temperature

        # To avoid overflow problems for values that would lead to a huge exponent just use a large value
        # that approximates correct result well enough.
        # Once exponent get high enough it doesn't really matter if we get 0.0001% or 0.00001% probability anyway
        exponential = np.exp(exponential_argument) if exponential_argument < 100 else 1e40

        return 1 / (1 + exponential)
