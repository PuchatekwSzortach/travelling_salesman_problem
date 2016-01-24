import numpy as np


def get_random_distances_grid(cities_number, max_distance):
    """
    :param cities_number:
    :param max_distance:
    :return: matrix of distances from. E.g. element (i,j) defines distance from city i to city j
    """

    asymmetric_distances_grid = np.random.random_integers(1, max_distance, size=(cities_number, cities_number))
    distances_grid = (asymmetric_distances_grid + asymmetric_distances_grid.transpose()) / 2

    # Distance from a city to itself should be 0
    distances_grid[np.diag_indices(cities_number)] = 0

    return distances_grid


def get_trip_distance(distances_grid, trip_itinerary):
    """
    :param distances_grid: Matrix of distances between cities
    :param trip_itinerary: List of city indices
    :return: Trip distance
    """
    0


class BruteForceTSPSolver:
    """
    Travelling salesman problem solver that uses brute force to compute optimal solution.
    """

    def __init__(self):
        print("TSP BRUTE")

    def solve(self, distances_grid):
        """
        Solve travelling salesman problem solver problem given distances grid.
        Returns an array of cities indices defining optimal trip.
        :param distances_grid:
        :return:
        """

        return [1, 2, 3]
