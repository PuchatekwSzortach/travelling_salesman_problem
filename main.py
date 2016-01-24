import tsp.algorithms

if __name__ == "__main__":

    cities_number = 4
    max_distance = 100

    distances_grid = tsp.algorithms.get_random_distances_grid(cities_number, max_distance)

    brute_solver = tsp.algorithms.BruteForceTSPSolver()

