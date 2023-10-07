import numpy as np
from Optimisation.Matrix.Gurobi_matrix import time_optimisation as Gurobi_timer
from Optimisation.Matrix.CVXPY_matrix import time_optimisation as CVXPY_timer
from Optimisation.Matrix.SLS_benchmark import time_optimisation as SLS_timer
from Results.SolverComparison.Scripts.plotFromData import plot_data
import pickle

prediction_horizon = 10
main_naming_identifier = 'basic_SLS'
scaling_factor = 1.1
initial_number_of_satellites = 3
maximum_time_in_seconds = 2

timers_to_run = [Gurobi_timer, CVXPY_timer, CVXPY_timer, SLS_timer]
timer_names = ['Gurobi', 'CVXPY(OSQP)', 'CVXPY(Gurobi)', 'SLS']
# timers_to_run = [CVXPY_timer, SLS_timer]
# timer_names = ['CVXPY(OSQP)', 'SLS']

satellite_array = [[initial_number_of_satellites] for i in range(len(timer_names))]
time_array = [list() for i in range(len(timer_names))]
arguments = {timer_name: {'prediction_horizon': prediction_horizon} for timer_name in timer_names}
arguments['CVXPY(OSQP)']['solver'] = 'OSQP'

for idx, timer in enumerate(timers_to_run):
    print(f'Starting on {timer_names[idx]}')

    last_time = 0
    while True:
        last_time = timers_to_run[idx](satellite_array[idx][-1], **arguments[timer_names[idx]])
        time_array[idx].append(last_time)
        if last_time >= maximum_time_in_seconds:
            break

        satellite_array[idx].append(max(satellite_array[idx][-1] + 1, int(scaling_factor * satellite_array[idx][-1])))

    # Save orbital sim with all data
    file_name = '../Data/' + main_naming_identifier + '_' + timer_names[idx]
    with open(file_name, 'wb') as file:
        pickle.dump(np.array(satellite_array[idx]), file)
        pickle.dump(np.array(time_array[idx]), file)
        pickle.dump(timer_names[idx], file)


plot_data(main_naming_identifier)
# plt.show()
