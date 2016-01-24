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


    print()
    print(distances_grid)
    print(distances_grid.transpose())

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


    print()
    print(distances_grid)
    print(distances_grid.transpose())

    # Check distances are non zero
    assert np.all(distances_grid[off_diagonal_mask] > 0)

    # Check that distance from city i to j is the same as from j to i
    assert np.all(distances_grid == distances_grid.transpose())
