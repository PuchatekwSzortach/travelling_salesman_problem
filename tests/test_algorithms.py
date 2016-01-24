import numpy as np

import tsp.algorithms


def test_get_random_distances_grid_one_city():

    distances_grid = tsp.algorithms.get_random_distances_grid(1, 100)

    assert 0 == distances_grid


def test_get_random_distances_grid_two_cities():

    cities_number = 2
    max_distance = 100

    distances_grid = tsp.algorithms.get_random_distances_grid(cities_number, max_distance)

    assert (cities_number, cities_number) == distances_grid.shape

    # Check distance from city to itself is 0
    assert np.all(0 == distances_grid.diagonal())

    # And no distance is larger than max distance
    assert np.all(distances_grid <= max_distance)

    off_diagonal_mask = np.ones([cities_number, cities_number], dtype=bool)
    off_diagonal_mask[np.diag_indices(cities_number)] = False

    # Check distances are non zero
    assert np.all(distances_grid[off_diagonal_mask] > 0)

    # Check that distance from city i to j is the same as from j to i
    assert np.all(distances_grid == distances_grid.transpose())


def test_get_random_distances_grid_four_cities():

    cities_number = 4
    max_distance = 100

    distances_grid = tsp.algorithms.get_random_distances_grid(cities_number, max_distance)

    assert (cities_number, cities_number) == distances_grid.shape

    # Check distance from city to itself is 0
    assert np.all(0 == distances_grid.diagonal())

    # And no distance is larger than max distance
    assert np.all(distances_grid <= max_distance)

    off_diagonal_mask = np.ones([cities_number, cities_number], dtype=bool)
    off_diagonal_mask[np.diag_indices(cities_number)] = False

    # Check distances are non zero
    assert np.all(distances_grid[off_diagonal_mask] > 0)

    # Check that distance from city i to j is the same as from j to i
    assert np.all(distances_grid == distances_grid.transpose())




# def test_brute_force_solution_four_cities():
#
#     distances_grid = np.zeros([4, 4])
#
#     # Build distances manually
#     distances_grid[0, 1] = distances_grid[1, 0] = 100
#     distances_grid[0, 2] = distances_grid[2, 0] = 10
#     distances_grid[0, 3] = distances_grid[3, 0] = 200
#
#     distances_grid[1, 2] = distances_grid[2, 1] = 200
#     distances_grid[1, 3] = distances_grid[3, 1] = 10
#
#     distances_grid[2, 3] = distances_grid[3, 2] = 100
#
#     optimal_trip = tsp.algorithms.BruteForceTSPSolver().solve(distances_grid)
#
#