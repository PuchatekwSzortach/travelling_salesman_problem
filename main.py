import pprint
import tsp.algorithms


if __name__ == "__main__":

    cities_number = 5
    max_distance = 100

    distances_matrix = tsp.algorithms.get_random_distances_matrix(cities_number, max_distance)

    optimal_path = tsp.algorithms.BruteForceTSPSolver(distances_matrix).solve()

    print("Optimal path is " + str(optimal_path))
    print("Distance is " + str(tsp.algorithms.get_trip_distance(distances_matrix, optimal_path)))

    boltzmann_path = tsp.algorithms.BoltzmannMachineTSPSolver(distances_matrix).solve()
    print("Boltzmann_path is " + str(boltzmann_path))
    print("Distance is " + str(tsp.algorithms.get_trip_distance(distances_matrix, boltzmann_path)))

