import numpy as np
import pickle
import os
import scipy as sp
import scipy.interpolate
from scipy.stats import linregress

data = [12, 34, 29, 38, 34, 51, 29, 34, 47, 34, 55, 94, 68, 81]
x = np.arange(1, len(data) + 1)
y = np.array(data)
res = linregress(x, y)


def lin_regress(xx, yy):
    logx = np.log10(xx)
    logy = np.log10(yy)
    res = linregress(logx, logy)

    return res


def log_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp


def print_data(print_name: str, est_solver_time: int = None, interp_kind='linear') -> None:
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

                # grad = (solver_times_log[-1] - solver_times_log[0]) / (number_of_satellites_log[-1] - number_of_satellites_log[0])

                est_sol_times = []
                if est_solver_time is not None:
                    interp = log_interp1d(number_of_satellites, solver_times, kind=interp_kind)
                    regression = lin_regress(number_of_satellites, solver_times)
                    for n_sat in est_solver_time:
                        if n_sat > number_of_satellites[-1]:
                            # est_time = solver_times_log[0] + grad * (np.log10(n_sat) - number_of_satellites_log[0])
                            est_time = regression.intercept + regression.slope * np.log10(n_sat)
                            est_sol_times.append(10 ** est_time)
                            # print(f"Estimated time for {n_sat} satellites: {10**est_time}")
                        else:
                            # print(f"Estimated time for {n_sat} satellites: {interp(n_sat)}")
                            est_sol_times.append(interp(n_sat))

                    print(f"{print_name} & {'&'.join(f'{x:4.4f}' for x in est_sol_times)} ")


if __name__ == '__main__':
    # print_data('basic_SLS_Gurobi', [3, 10, 100, 1000])
    # print_data('transformed_SLS_Gurobi', [3, 10, 100, 1000])
    print_data('robust_SLS_Gurobi', [1, 10, 100])
    print()
    print_data('robust_SLS_GurobiExact', [1, 10, 100])
    print()
    print_data('robust_SLS_cuOSQP', [1, 10, 100])
    print()
    print_data('robust_SLS_Toolbox', [1, 10, 100], interp_kind='linear')
