import numpy as np
import pickle
import os


def print_data(print_name: str, est_solver_time: int = None) -> None:
    """
    Print data for a given plot name.
    :param plot_name: The name of the print (with corresponding data names)
    """
    for file in os.listdir("../Data"):
        if file.startswith(print_name):
            with open(os.path.join("../Data", file), 'rb') as f:
                number_of_satellites = pickle.load(f)
                solver_times = pickle.load(f)
                number_of_satellites_log = np.log10(number_of_satellites)
                solver_times_log = np.log10(solver_times)

                print(f"Number of satellites: {number_of_satellites}")
                print(f"Solver times (s): {solver_times}")

                grad = (solver_times_log[-1] - solver_times_log[0]) / (number_of_satellites_log[-1] - number_of_satellites_log[0])

                if est_solver_time is not None:
                    for n_sat in est_solver_time:
                        est_time = solver_times_log[0] + grad * (np.log10(n_sat) - number_of_satellites_log[0])
                        print(f"Estimated time for {n_sat} satellites: {10**est_time}")


if __name__ == '__main__':
    print_data('basic_SLS_SLS', [12, 13, 14, 3000])
