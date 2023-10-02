import numpy as np
from Optimisation.Vector.OSQP import time_optimisation as OSPQ_timer
from Optimisation.Vector.Gurobi import time_optimisation as Gurobi_timer
from Optimisation.Vector.QuadProg import time_optimisation as QuadProg_timer
from Optimisation.Vector.CVXPY import time_optimisation as CVXPY_timer
# from Controllers.SimulatedAnnealing import time_optimisation as SimAn_timer
from Results.SolverComparison.Scripts.plotFromData import plot_data
import pickle

prediction_horizon = 10
main_naming_identifier = 'basic_MPC'

timers_to_run = [QuadProg_timer, OSPQ_timer, Gurobi_timer, CVXPY_timer]
timer_names = ['QuadProg', 'OSQP', 'Gurobi', 'CVXPY']
satellite_array = [np.logspace(np.log10(3), np.log10(5), num=3, dtype=int),
                   np.logspace(np.log10(3), np.log10(700), num=15, dtype=int),
                   np.logspace(np.log10(3), np.log10(300), num=15, dtype=int),
                   np.logspace(np.log10(3), np.log10(50), num=15, dtype=int)]

time_array = [np.zeros_like(satellite_arr, dtype=float) for satellite_arr in satellite_array]

for idx, timer in enumerate(timers_to_run):
    print(f'Starting on {timer_names[idx]}')

    for i, number_of_satellites in enumerate(satellite_array[idx]):
        time_array[idx][i] = timers_to_run[idx](int(number_of_satellites), prediction_horizon=prediction_horizon)

    # Save orbital sim with all data
    file_name = '../Data/' + main_naming_identifier + '_' + timer_names[idx]
    with open(file_name, 'wb') as file:
        pickle.dump(satellite_array[idx], file)
        pickle.dump(time_array[idx], file)
        pickle.dump(timer_names[idx], file)

plot_data(main_naming_identifier)
# plt.show()
