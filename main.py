import tsp.algorithms

if __name__ == "__main__":

    cities_number = 4
    max_distance = 100

    distances_matrix = tsp.algorithms.get_random_distances_matrix(cities_number, max_distance)

    path = tsp.algorithms.BruteForceTSPSolver(distances_matrix).solve()

    print("Path is " + str(path))
    print("Distance is " + str(tsp.algorithms.get_trip_distance(distances_matrix, path)))

