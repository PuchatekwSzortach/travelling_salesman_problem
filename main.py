import tsp.algorithms
import time


if __name__ == "__main__":

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
    print("\nBoltzmann_path is " + str(boltzmann_path))
    print("Distance is " + str(tsp.algorithms.get_trip_distance(distances_matrix, boltzmann_path)))
    print("Computational time is: {0:.2f} seconds".format(time.time() - start))

