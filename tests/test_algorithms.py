import numpy as np
import pytest

import tsp.algorithms


@pytest.fixture()
def four_cities_grid():
    distances_matrix = np.zeros([4, 4])

    # Build distances manually
    distances_matrix[0, 1] = distances_matrix[1, 0] = 100
    distances_matrix[0, 2] = distances_matrix[2, 0] = 10
    distances_matrix[0, 3] = distances_matrix[3, 0] = 200

    distances_matrix[1, 2] = distances_matrix[2, 1] = 200
    distances_matrix[1, 3] = distances_matrix[3, 1] = 10

    distances_matrix[2, 3] = distances_matrix[3, 2] = 100

    return distances_matrix


def test_get_random_distances_matrix_one_city():

    distances_matrix = tsp.algorithms.get_random_distances_matrix(1, 100)

    assert 0 == distances_matrix


def test_get_random_distances_matrix_two_cities():

    cities_number = 2
    max_distance = 100

    distances_matrix = tsp.algorithms.get_random_distances_matrix(cities_number, max_distance)

    assert (cities_number, cities_number) == distances_matrix.shape

    # Check distance from city to itself is 0
    assert np.all(0 == distances_matrix.diagonal())

    # And no distance is larger than max distance
    assert np.all(distances_matrix <= max_distance)

    off_diagonal_mask = np.ones([cities_number, cities_number], dtype=bool)
    off_diagonal_mask[np.diag_indices(cities_number)] = False

    # Check distances are non zero
    assert np.all(distances_matrix[off_diagonal_mask] > 0)

    # Check that distance from city i to j is the same as from j to i
    assert np.all(distances_matrix == distances_matrix.transpose())


def test_get_random_distances_matrix_four_cities():

    cities_number = 4
    max_distance = 100

    distances_matrix = tsp.algorithms.get_random_distances_matrix(cities_number, max_distance)

    assert (cities_number, cities_number) == distances_matrix.shape

    # Check distance from city to itself is 0
    assert np.all(0 == distances_matrix.diagonal())

    # And no distance is larger than max distance
    assert np.all(distances_matrix <= max_distance)

    off_diagonal_mask = np.ones([cities_number, cities_number], dtype=bool)
    off_diagonal_mask[np.diag_indices(cities_number)] = False

    # Check distances are non zero
    assert np.all(distances_matrix[off_diagonal_mask] > 0)

    # Check that distance from city i to j is the same as from j to i
    assert np.all(distances_matrix == distances_matrix.transpose())


def test_get_trip_distance_two_cities():

    distances_matrix = np.zeros([2, 2])
    distances_matrix[0, 1] = distances_matrix[1, 0] = 100

    assert 100 == tsp.algorithms.get_trip_distance(distances_matrix, [0, 1])


def test_get_trip_distance_four_cities_optimal_path(four_cities_grid):

    trip = [0, 2, 3, 1]
    assert 120 == tsp.algorithms.get_trip_distance(four_cities_grid, trip)


def test_get_trip_distance_four_cities_bad_path(four_cities_grid):

    trip = [0, 1, 2, 3]
    assert 400 == tsp.algorithms.get_trip_distance(four_cities_grid, trip)


def test_brute_force_solution_four_cities(four_cities_grid):

    optimal_trip = tsp.algorithms.BruteForceTSPSolver(four_cities_grid).solve()
    assert 120 == tsp.algorithms.get_trip_distance(four_cities_grid, optimal_trip)

