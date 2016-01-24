import tsp.algorithms

if __name__ == "__main__":

    cities_number = 4
    max_distance = 100

    distances_grid = tsp.algorithms.get_random_distances_grid(cities_number, max_distance)

    brute_solver = tsp.algorithms.BruteForceTSPSolver()

    path = brute_solver.solve(distances_grid)

    print("Path is " + str(path))
    print("Distance is " + str(tsp.algorithms.get_trip_distance(distances_grid, path)))

