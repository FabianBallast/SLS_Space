import numpy as np
from Optimisation.OSQP_Solver import time_optimisation as OSPQ_timer
from Optimisation.Vector.Gurobi import time_optimisation as Gurobi_timer
from Optimisation.Vector.QuadProg import time_optimisation as QuadProg_timer
from Controllers.SimulatedAnnealing import time_optimisation as SimAn_timer
from matplotlib import pyplot as plt

prediction_horizon = 2

# Simulated Annealing
print(f'Starting on Simulated Annealing with {prediction_horizon=}')
satellite_array_siman = np.logspace(np.log10(3), np.log10(15), num=10, dtype=int)
time_array_siman = np.zeros_like(satellite_array_siman, dtype=float)

for idx, number_of_satellites in enumerate(satellite_array_siman):
    print(f"{number_of_satellites=}")
    time_array_siman[idx] = SimAn_timer(int(number_of_satellites), prediction_horizon=40)

# QuadProg
print(f'Starting on Quadprog with {prediction_horizon=}')
satellite_array_quadprog = np.logspace(np.log10(3), np.log10(15), num=10, dtype=int)
time_array_quadprog = np.zeros_like(satellite_array_quadprog, dtype=float)

for idx, number_of_satellites in enumerate(satellite_array_quadprog):
    print(f"{number_of_satellites=}")
    time_array_quadprog[idx] = QuadProg_timer(int(number_of_satellites), prediction_horizon=10)

# OSQP
print(f'Starting on OSQP with {prediction_horizon=}')
satellite_array_osqp = np.logspace(np.log10(3), np.log10(700), num=15, dtype=int)
time_array_osqp = np.zeros_like(satellite_array_osqp, dtype=float)

for idx, number_of_satellites in enumerate(satellite_array_osqp):
    print(f"{number_of_satellites=}")
    time_array_osqp[idx] = OSPQ_timer(int(number_of_satellites), prediction_horizon=10)

# Gurobi
print(f'Starting on Gurobi with {prediction_horizon=}')
satellite_array_gurobi = np.logspace(np.log10(3), np.log10(300), num=15, dtype=int)
time_array_gurobi = np.zeros_like(satellite_array_gurobi, dtype=float)

for idx, number_of_satellites in enumerate(satellite_array_gurobi):
    print(f"{number_of_satellites=}")
    time_array_gurobi[idx] = Gurobi_timer(int(number_of_satellites), prediction_horizon=10)






plt.figure()
plt.loglog(satellite_array_siman, time_array_siman, label="Simulated Annealing")
plt.loglog(satellite_array_quadprog, time_array_quadprog, label="QuadProg")
plt.loglog(satellite_array_gurobi, time_array_gurobi, label="Gurobi")
plt.loglog(satellite_array_osqp, time_array_osqp, label='OSPQ')

plt.legend(fontsize=14)
plt.xlabel(r'$\mathrm{Number \;of \;satellites \;[-]}$', fontsize=14)
plt.ylabel(r'$\mathrm{Time \;[s]}$', fontsize=14)
plt.grid(True)
plt.show()
