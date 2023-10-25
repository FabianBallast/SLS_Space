import gurobipy as gp
from gurobipy import GRB
import time
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from Optimisation.createProblem import create_sparse_problem
from Scenarios.MainScenarios import ScenarioEnum, Scenario
from Optimisation.sparseHelperFunctions import *


def time_optimisation(number_of_satellites: int, prediction_horizon: int = None, plot_results: bool = False,
                      scenario: Scenario = ScenarioEnum.simple_scenario_translation_blend_scaled.value) -> float:
    """
    Measure the time it takes to optimise the controller.

    :param number_of_satellites: Number of satellites to optimise.
    :param prediction_horizon: Prediction horizon of the controller.
    :param plot_results: Whether to plot the results.
    :param scenario: Scenario to run. Defaults to HCW scenario.
    :return: Float with average time (after first iteration has passed).
    """
    problem = create_sparse_problem(number_of_satellites, prediction_horizon, scenario)
    # problem['x0'][1] = 1
    # Create a new model
    m = gp.Model("MPC")

    # mask_A, mask_B = get_masks(problem['A'], problem['B'])
    nx, nu = problem['B'].shape
    mask_A = np.ones((nx, nx), dtype=bool)
    mask_B = np.ones((nu, nx), dtype=bool)
    x_vars = np.sum(mask_A)
    u_vars = np.sum(mask_B)

    x_max = np.tile(problem['x_max'], (1, problem['N'])).flatten()
    u_max = np.tile(problem['u_max'], (1, problem['N'])).flatten()

    phi_x = m.addMVar(problem['N'] * x_vars, name='phi_x', lb=-np.inf, ub=np.inf)
    phi_u = m.addMVar(problem['N'] * u_vars, name='phi_u', lb=-np.inf, ub=np.inf)
    x = m.addMVar(problem['N'] * problem['nx'], name='x', lb=-x_max, ub=x_max)
    u = m.addMVar(problem['N'] * problem['nu'], name='u', lb=-u_max, ub=u_max)

    A_f, B_f = sparse_state_matrix_replacement(problem['A'], problem['B'], mask_A, mask_B)
    Fx, Fu = find_fx_and_fu(mask_A, mask_B, problem['x0'])

    Q = sparse.kron(sparse.eye(problem['N']), problem['Q'])
    R = sparse.kron(sparse.eye(problem['N']), problem['R'])

    Fx = sparse.kron(sparse.eye(problem['N']), Fx)
    Fu = sparse.kron(sparse.eye(problem['N']), Fu)

    m.addConstr(phi_x[:x_vars] == A_f @ np.eye(problem['nx'])[mask_A] + B_f @ phi_u[:u_vars])

    for n in range(1, problem['N']):
        m.addConstr(phi_x[n * x_vars: (n + 1) * x_vars] == A_f @ phi_x[(n - 1) * x_vars: n * x_vars] + B_f @ phi_u[
                                                                                                             n * u_vars: (
                                                                                                                                     n + 1) * u_vars])

    constraint_list = []
    constraint_list.append(m.addConstr(Fx @ phi_x == x))
    constraint_list.append(m.addConstr(Fu @ phi_u == u))

    m.setObjective(x @ Q @ x + u @ R @ u, GRB.MINIMIZE)
    m.setParam("OutputFlag", 0)
    # m.setParam("OptimalityTol", 1e-3)
    m.setParam("BarConvTol", 1e-3)

    # Simulate in closed loop
    t_0 = 0
    nsim = 11
    runtime = 0
    states = np.zeros((problem['nx'], nsim + 1))
    states[:, 0] = problem['x0']
    inputs = np.zeros((problem['nu'], nsim))

    for i in range(nsim):
        # Solve
        m.optimize()
        runtime += m.Runtime

        states[:, i + 1] = x.X[:problem['nx']]
        inputs[:, i] = u.X[:problem['nu']]

        # print(states[:9, i+1])
        # print(inputs[:9, i])
        # print()

        m.remove(constraint_list)

        Fx, Fu = find_fx_and_fu(mask_A, mask_B, states[:, i + 1])
        # constraint_list = [0, 1]
        Fx = sparse.kron(sparse.eye(problem['N']), Fx)
        Fu = sparse.kron(sparse.eye(problem['N']), Fu)

        # start = time.time()
        constraint_list[0] = m.addConstr(Fx @ phi_x == x)
        constraint_list[1] = m.addConstr(Fu @ phi_u == u)

        m.update()
        # print(time.time() - start)
        # print()

        if i == 0:
            t_0 = time.time()
            runtime = 0

    t_end = time.time()

    avg_time = (t_end - t_0) / (nsim - 1)
    print(f"Average elapsed time Gurobi reformed for {number_of_satellites} satellites: {avg_time}")
    print(f"Runtime avg.: {runtime / (nsim - 1)}")

    if plot_results:
        plt.figure()
        plt.plot(np.rad2deg(states[1::6].T))
        plt.show()
    return avg_time


if __name__ == '__main__':
    time_optimisation(10, plot_results=True)
