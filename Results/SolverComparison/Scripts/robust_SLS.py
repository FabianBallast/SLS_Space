import numpy as np
from Optimisation.Robust.Gurobi_robust import time_optimisation as Gurobi_timer
from Optimisation.Robust.Gurobi_robust_exact import time_optimisation as Gurobi_timer_exact
# from Optimisation.Robust.OSQP_robust import time_optimisation as OSQP_timer
from Results.SolverComparison.Scripts.plotFromData import plot_data
import pickle

prediction_horizon = 6
main_naming_identifier = 'robust_SLS'
timers_to_run = [Gurobi_timer,
                 # Gurobi_timer_exact,
                 #OSQP_timer
                ]
timer_names = ['Gurobi',
               'GurobiExact',
               'OSQP']
scaling_factor = 1.1
initial_number_of_satellites = 1
maximum_time_in_seconds = 100

satellite_array = [[initial_number_of_satellites] for i in range(len(timer_names))]
time_array = [list() for i in range(len(timer_names))]
arguments = {timer_name: {'prediction_horizon': prediction_horizon} for timer_name in timer_names}

for idx, timer in enumerate(timers_to_run):
    print(f'Starting on {timer_names[idx]}')

    while True:
        last_time = timers_to_run[idx](satellite_array[idx][-1], **arguments[timer_names[idx]])
        time_array[idx].append(last_time)

        if satellite_array[idx][-1] >= 100:
            break

        satellite_array[idx].append(max(satellite_array[idx][-1] + 1, int(scaling_factor * satellite_array[idx][-1])))



    # Save orbital sim with all data
    file_name = '../Data/' + main_naming_identifier + '_' + timer_names[idx]
    with open(file_name, 'wb') as file:
        pickle.dump(satellite_array[idx], file)
        pickle.dump(time_array[idx], file)
        pickle.dump(timer_names[idx], file)

plot_data(main_naming_identifier)
# plt.show()
