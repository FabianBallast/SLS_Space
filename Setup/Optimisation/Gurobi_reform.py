import gurobipy as gp
from gurobipy import GRB
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from Dynamics.HCWDynamics import RelCylHCW as dyn
from Scenarios.MainScenarios import ScenarioEnum
import random
import time


def time_optimisation(number_of_satellites: int, prediction_horizon: int):

    # Create a new model
    m = gp.Model("MPC")

    # Create single model
    scenario = ScenarioEnum.simple_scenario_translation_HCW_scaled.value
    scenario.number_of_satellites = number_of_satellites
    scenario.control.tFIR = prediction_horizon
    dynamics = dyn(scenario)
    model_single = dynamics.create_model(scenario.control.control_timestep)

    # horizon
    N = scenario.control.tFIR

    # Create large model
    number_of_satellites = scenario.number_of_satellites
    Ad = sparse.kron(sparse.eye(number_of_satellites), sparse.csr_matrix(model_single.A))
    Bd = sparse.kron(sparse.eye(number_of_satellites), sparse.csr_matrix(model_single.B))
    [nx, nu] = Bd.shape

    # Constraints
    umin = -np.array(dynamics.get_input_constraint() * number_of_satellites)
    umax =  np.array(dynamics.get_input_constraint() * number_of_satellites)
    xmin = -np.array(dynamics.get_state_constraint() * number_of_satellites)
    xmax =  np.array(dynamics.get_state_constraint() * number_of_satellites)

    # Objective function
    Q = sparse.kron(sparse.eye(number_of_satellites), sparse.csr_matrix(dynamics.get_state_cost_matrix_sqrt()).power(2))
    R = sparse.kron(sparse.eye(number_of_satellites), sparse.csr_matrix(dynamics.get_input_cost_matrix_sqrt()[:3]).power(2))

    # Initial and reference states
    x0 = np.zeros(nx)
    possible_angles = np.linspace(0, 2 * np.pi, number_of_satellites + 2, endpoint=False)

    random.seed(129)
    selected_indices = np.sort(random.sample(range(number_of_satellites + 2), number_of_satellites))
    x0[1::6] = possible_angles[selected_indices]
    xr = np.zeros(nx)
    ref_rel_angles = np.linspace(0, 2 * np.pi, scenario.number_of_satellites, endpoint=False)
    ref_rel_angles -= np.mean(ref_rel_angles - possible_angles[selected_indices])
    xr[1::6] = ref_rel_angles

    # MPC Formulation
    # xmin = np.tile(xmin - xr, (N + 1, 1))
    # xmax = np.tile(xmax - xr, (N + 1, 1))
    # umin = np.tile(umin, (N, 1))
    # umax = np.tile(umax, (N, 1))

    x = m.addMVar(shape=(N+1, nx * nx), name='x')
    u = m.addMVar(shape=(N, nu * nx), name='u')

    constraint_list = []
    Fx = sparse.kron(sparse.eye(nx), x0-xr)
    Fu = sparse.kron(sparse.eye(nu), x0-xr)
    FQF = Fx.T @ Q @ Fx
    FRF = Fu.T @ R @ Fu

    A_f = sparse.kron(Ad, sparse.eye(nx))
    B_f = sparse.kron(Bd, sparse.eye(nx))

    for n in range(N):
        constraint_list.append(m.addConstr(Fx @ x[n] <= xmax - xr))
        constraint_list.append(m.addConstr(Fx @ x[n] >= xmin - xr))
        constraint_list.append(m.addConstr(Fu @ u[n] <= umax))
        constraint_list.append(m.addConstr(Fu @ u[n] >= umin))

    constraint_list.append(m.addConstr(Fx @ x[n] <= xmax))
    constraint_list.append(m.addConstr(Fx @ x[n] >= xmin))

    m.addConstr(x[0] == np.eye(nx).flatten())

    for n in range(N):
        m.addConstr(x[n+1] == A_f @ x[n] + B_f @ u[n])

    obj1 = sum(x[k, :] @ FQF @ x[k, :] for k in range(N + 1))
    obj2 = sum(u[k, :] @ FRF @ u[k, :] for k in range(N))

    m.setObjective(obj1 + obj2, GRB.MINIMIZE)
    m.setParam("OutputFlag", 0)
    m.setParam("OptimalityTol", 1e-4)

    # Simulate in closed loop
    # t_0 = time.time()
    t_0 = 0
    runs = 1
    nsim = 10
    states = np.zeros((nx, nsim + 1))
    states[:, 0] = x0
    input = np.zeros((nu, nsim))

    for run in range(runs):
        for i in range(nsim):
            # Solve
            m.optimize()

            all_vars = m.getVars()
            values = m.getAttr("X", all_vars)
            names = m.getAttr("VarName", all_vars)
            # print(values, names)

            phi_x = np.zeros(nx * nx)
            phi_u = np.zeros(nx * nu)
            for j in range(nx * nx, 2 * nx * nx):
                phi_x[j - nx**2] = values[names.index(f'x[{j}]')]

            states[:, i+1] = phi_x.reshape((nx, nx)) @ (states[:, i] - xr) + xr

            for j in range(0, nu * nx):
                phi_u[j] = values[names.index(f'u[{j}]')]

            input[:, i] = phi_u.reshape((nu, nx)) @ (states[:, i] - xr)
            # print(input)
            for constraint in constraint_list:
                m.remove(constraint)

            x0 = states[:, i+1]
            constraint_list = []
            Fx = sparse.kron(sparse.eye(nx), x0 - xr)
            Fu = sparse.kron(sparse.eye(nu), x0 - xr)
            FQF = Fx.T @ Q @ Fx
            FRF = Fu.T @ R @ Fu

            for n in range(N):
                constraint_list.append(m.addConstr(Fx @ x[n] <= xmax - xr))
                constraint_list.append(m.addConstr(Fx @ x[n] >= xmin - xr))
                constraint_list.append(m.addConstr(Fu @ u[n] <= umax))
                constraint_list.append(m.addConstr(Fu @ u[n] >= umin))

            constraint_list.append(m.addConstr(Fx @ x[n] <= xmax))
            constraint_list.append(m.addConstr(Fx @ x[n] >= xmin))
            obj1 = sum(x[k, :] @ FQF @ x[k, :] for k in range(N + 1))
            obj2 = sum(u[k, :] @ FRF @ u[k, :] for k in range(N))

            m.setObjective(obj1 + obj2, GRB.MINIMIZE)
            m.update()

            if i == 0:
                t_0 = time.time()

    t_end = time.time()

    avg_time = (t_end - t_0) / runs / (nsim- 1)
    # print(f"Average elapsed time: {avg_time}")

    plt.figure()
    plt.plot(np.rad2deg(states[1::6].T - xr[1::6]))

    plt.figure()
    plt.plot(input[1::3].T)
    plt.show()
    return avg_time


if __name__ == '__main__':
    print(time_optimisation(3, 10))
