import time
import random
from cvxpy import *
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from Dynamics.HCWDynamics import RelCylHCW as dyn
from Scenarios.MainScenarios import ScenarioEnum

# Create single model
dynamics = dyn(ScenarioEnum.simple_scenario_translation_HCW_scaled.value)
model_single = dynamics.create_model(20)

# Create large model
number_of_satellites = 10
Ad = sparse.kron(sparse.eye(number_of_satellites), sparse.csc_matrix(model_single.A))
Bd = sparse.kron(sparse.eye(number_of_satellites), sparse.csc_matrix(model_single.B))
[nx, nu] = Bd.shape

# Constraints
umin = -np.array(dynamics.get_input_constraint() * number_of_satellites)
umax =  np.array(dynamics.get_input_constraint() * number_of_satellites)
xmin = -np.array(dynamics.get_state_constraint() * number_of_satellites)
xmax =  np.array(dynamics.get_state_constraint() * number_of_satellites)

# Objective function
Q = sparse.kron(sparse.eye(number_of_satellites), sparse.csc_matrix(dynamics.get_state_cost_matrix_sqrt()).power(2))
QN = Q
R = sparse.kron(sparse.eye(number_of_satellites), sparse.csc_matrix(dynamics.get_input_cost_matrix_sqrt()[:3]).power(2))

# Initial and reference states
x0 = np.zeros(nx)
possible_angles = np.linspace(0, 2 * np.pi, number_of_satellites + 2, endpoint=False)

random.seed(129)
selected_indices = np.sort(random.sample(range(number_of_satellites + 2), number_of_satellites))
x0[1::6] = possible_angles[selected_indices]
xr = np.zeros(nx)
xr[1::6] = np.linspace(0, 2 * np.pi, number_of_satellites, endpoint=False)

# Prediction horizon
N = 10

# Define problem
u = Variable((nu, N))
x = Variable((nx, N + 1))
x_init = Parameter(nx)
objective = 0
constraints = [x[:, 0] == x_init]
for k in range(N):
    objective += quad_form(x[:, k] - xr, Q) + quad_form(u[:, k], R)
    constraints += [x[:, k + 1] == Ad @ x[:, k] + Bd @ u[:, k]]
    constraints += [xmin <= x[:, k], x[:, k] <= xmax]
    constraints += [umin <= u[:, k], u[:, k] <= umax]
objective += quad_form(x[:, N] - xr, QN)
prob = Problem(Minimize(objective), constraints)

t_0 = time.time()
runs = 1
nsim = 15
x = np.zeros((nx, nsim + 1))
x[:, 0] = x0

for run in range(runs):
    print(run)
    x0 = np.zeros(nx)
    x0[1::6] = possible_angles[selected_indices]

    # Simulate in closed loop
    for i in range(nsim):
        x_init.value = x0
        prob.solve(solver=GUROBI, warm_start=nsim > 0)

        x0 = Ad.dot(x0) + Bd.dot(u[:, 0].value)
        x[:, i + 1] = x0

t_end = time.time()

print(f"Average elapsed time: {(t_end - t_0) / runs / nsim}")

plt.figure()
plt.plot(x[1::6].T)
plt.show()

