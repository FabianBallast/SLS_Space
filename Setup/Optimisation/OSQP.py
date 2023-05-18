import random
import time
import osqp
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from Dynamics.HCWDynamics import RelCylHCW as dyn
from Scenarios.MainScenarios import ScenarioEnum

# Create single model
dynamics = dyn(ScenarioEnum.simple_scenario_translation_HCW_scaled.value)
model_single = dynamics.create_model(30)

# Create large model
number_of_satellites = 100
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

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
# - quadratic objective
P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                       sparse.kron(sparse.eye(N), R)], format='csc')
# - linear objective
q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
               np.zeros(N*nu)])
# - linear dynamics
Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-x0, np.zeros(N*nx)])
ueq = leq
# - input and state constraints
Aineq = sparse.eye((N+1)*nx + N*nu)
lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
# - OSQP constraints
A = sparse.vstack([Aeq, Aineq], format='csc')
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace
prob.setup(P, q, A, l, u, warm_start=True, verbose=False)

# Simulate in closed loop
t_0 = time.time()
runs = 1
nsim = 30
x = np.zeros((nx, nsim + 1))
x[:, 0] = x0
input = np.zeros((nu, nsim))

for run in range(runs):
    for i in range(nsim):
        # Solve
        res = prob.solve()

        # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        # Apply first control input to the plant
        ctrl = res.x[-N*nu:-(N-1)*nu]
        x0 = Ad.dot(x0) + Bd.dot(ctrl)
        x[:, i + 1] = x0
        input[:, i] = ctrl

        # Update initial state
        l[:nx] = -x0
        u[:nx] = -x0
        prob.update(l=l, u=u)

t_end = time.time()

print(f"Average elapsed time: {(t_end - t_0) / runs / nsim}")

plt.figure()
plt.plot(np.rad2deg(x[1::6].T))

plt.figure()
plt.plot(input[1::3].T)
plt.show()

