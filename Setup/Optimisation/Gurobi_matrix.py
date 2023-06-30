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
    xmin = np.tile(xmin - xr, (N + 1, 1))
    xmax = np.tile(xmax - xr, (N + 1, 1))
    umin = np.tile(umin, (N, 1))
    umax = np.tile(umax, (N, 1))

    Phi_x = m.addMVar(shape=(N + 1, nx, nx), name='Phi_x')
    Phi_u = m.addMVar(shape=(N, nu, nx), name='Phi_u')
    x = m.addMVar(shape=(N+1, nx), lb=xmin, ub=xmax, name='x')
    u = m.addMVar(shape=(N, nu), lb=umin, ub=umax, name='u')

    constraint_list = []
    for n in range(N):
        for row in range(nx):
            constraint_list.append(m.addConstr(Phi_x[n, row, :] @ (x0 - xr) == x[n, row]))

        for row in range(nu):
            constraint_list.append(m.addConstr(Phi_u[n, row, :] @ (x0 - xr) == u[n, row]))

    for row in range(nx):
        constraint_list.append(m.addConstr(Phi_x[N, row, :] @ (x0 - xr) == x[N, row]))

    for row in range(nx):
        m.addConstr(Phi_x[0, row, :] == np.eye(nx)[row, :])
    for k in range(N):
        for column in range(nx):
            m.addConstr(Phi_x[k+1, :, column] == Ad @ Phi_x[k, :, column] + Bd @ Phi_u[k, :, column])

    # m.addConstr()
    # for k in range(N):
    #     for row in range(nx):
    #         m.addConstr(Phi_x[k, row, :] @ (x0 - xr) <= (xmax - xr)[row])
    #         m.addConstr(Phi_x[k, row, :] @ (x0 - xr) >= (xmin - xr)[row])
    #
    #     for row in range(nu):
    #         m.addConstr(Phi_u[k, row, :] @ (x0 - xr) <= umax[row])
    #         m.addConstr(Phi_u[k, row, :] @ (x0 - xr) >= umin[row])
    #
    # for row in range(nx):
    #     m.addConstr(Phi_x[N, row, :] @ (x0 - xr) <= (xmax - xr)[row])
    #     m.addConstr(Phi_x[N, row, :] @ (x0 - xr) >= (xmin - xr)[row])

    obj1 = sum(x[k, :] @ Q @ x[k, :] for k in range(N + 1))
    obj2 = sum(u[k, :] @ R @ u[k, :] for k in range(N))

    m.setObjective(obj1 + obj2, GRB.MINIMIZE)
    m.setParam("OutputFlag", 0)
    m.setParam("OptimalityTol", 1e-4)

    # m.optimize()
    #
    # all_vars = m.getVars()
    # values = m.getAttr("X", all_vars)
    # names = m.getAttr("VarName", all_vars)
    # new_state = np.zeros_like(x0)
    # input = np.zeros((nu, ))
    # for i in range(nx, 2 * nx):
    #    new_state[i - nx] = values[names.index(f'x[{i}]')]
    #
    # for i in range(0, nu):
    #     input[i] = values[names.index(f'u[{i}]')]

    # print(new_state, input)
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
                phi_x[j - nx**2] = values[names.index(f'Phi_x[{j}]')]

            states[:, i+1] = phi_x.reshape((nx, nx)) @ (states[:, i] - xr) + xr

            for j in range(0, nu * nx):
                phi_u[j] = values[names.index(f'Phi_u[{j}]')]

            input[:, i] = phi_u.reshape((nu, nx)) @ (states[:, i] - xr)

            for constraint in constraint_list:
                m.remove(constraint)

            x0 = states[:, i+1]
            constraint_list = []
            for n in range(N):
                for row in range(nx):
                    constraint_list.append(m.addConstr(Phi_x[n, row, :] @ (x0 - xr) == x[n, row]))

                for row in range(nu):
                    constraint_list.append(m.addConstr(Phi_u[n, row, :] @ (x0 - xr) == u[n, row]))

            for row in range(nx):
                constraint_list.append(m.addConstr(Phi_x[N, row, :] @ (x0 - xr) == x[N, row]))

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
    print(time_optimisation(10, 10))
