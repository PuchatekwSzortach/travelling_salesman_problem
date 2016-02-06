# travelling_salesman_problem

This repository contains three different travelling salesman problem solvers.
First two, `BruteForceTSPSolver` and `BruteForceTSPWorstPathSolver` simply perform a brute force search to find best and worst possible paths.
Third solver, `BoltzmannMachineTSPSolver`, uses a Boltzmann machine to find a solution. Please note that Boltzmann machine doesn't guarantee it will find an optimal solution. It is however faster than a brute force search for trips containing more than ~11 cities.

Utilities to create random distances matrix and compute paths lengths are also provided.

Sample use:

```python

import tsp.algorithms
import time

cities_number = 5
    max_distance = 100

    distances_matrix = tsp.algorithms.get_random_distances_matrix(cities_number, max_distance)

    start = time.time()
    optimal_path = tsp.algorithms.BruteForceTSPSolver(distances_matrix).solve()
    print("Optimal path is " + str(optimal_path))
    print("Distance is " + str(tsp.algorithms.get_trip_distance(distances_matrix, optimal_path)))
    print("Computational time is: {0:.2f} seconds".format(time.time() - start))

    start = time.time()
    worst_path = tsp.algorithms.BruteForceTSPWorstPathSolver(distances_matrix).solve()
    print("\nWorst path is " + str(worst_path))
    print("Distance is " + str(tsp.algorithms.get_trip_distance(distances_matrix, worst_path)))
    print("Computational time is: {0:.2f} seconds".format(time.time() - start))

    start = time.time()
    boltzmann_path = tsp.algorithms.BoltzmannMachineTSPSolver(distances_matrix).solve()
    print("\nBoltzmann path is " + str(boltzmann_path))
    print("Distance is " + str(tsp.algorithms.get_trip_distance(distances_matrix, boltzmann_path)))
    print("Computational time is: {0:.2f} seconds".format(time.time() - start))
```
