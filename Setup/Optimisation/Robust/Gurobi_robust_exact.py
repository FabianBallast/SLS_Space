import gurobipy as gp
from gurobipy import GRB, norm
import time
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from Optimisation.createProblem import create_sparse_problem
from Scenarios.MainScenarios import ScenarioEnum, Scenario
from Optimisation.sparseHelperFunctions import *


def time_optimisation(number_of_satellites: int, prediction_horizon: int = None, plot_results: bool = False,
                      scenario: Scenario = ScenarioEnum.robustness_comparison_advanced_robust_noise.value) -> float:
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

    x_max = np.tile(problem['x_max'], (prediction_horizon + 1, 1))
    x_max[0] = np.inf
    u_max = np.tile(problem['u_max'], (prediction_horizon, 1))

    mask_A, mask_B = get_masks(sparse.coo_matrix(problem['A']), sparse.coo_matrix(problem['B']))
    x_vars = np.sum(mask_A)
    u_vars = np.sum(mask_B)
    number_of_blocks = int((prediction_horizon + 1) / 2 * prediction_horizon + 0.001)

    block_ordering = sparse.csc_matrix(np.tril(np.ones((prediction_horizon, prediction_horizon))))
    block_ordering.data = np.arange(number_of_blocks)
    block_ordering = np.array(block_ordering.todense())

    # one_norm_matrix_x, one_norm_matrix_u = find_fx_and_fu(mask_A, mask_B, np.ones_like(reference_state))
    one_norm_kron_mat = np.zeros((prediction_horizon, number_of_blocks))
    for n in range(prediction_horizon):
        one_norm_kron_mat[n, block_ordering[n, 1:n + 1]] = 1

    # print(x_max.shape)
    x = m.addMVar(prediction_horizon * problem['nx'], name='x', lb=-np.inf)
    u = m.addMVar(prediction_horizon * problem['nu'], name='u', lb=-np.inf)
    phi_x = m.addMVar(number_of_blocks * x_vars, name='phi_x', lb=-np.inf)  # , ub=np.inf)
    phi_u = m.addMVar(number_of_blocks * u_vars, name='phi_u', lb=-np.inf)  # , ub=np.inf)
    sigma = m.addMVar(prediction_horizon * problem['nx'], name='sigma')

    x_inf = m.addMVar(prediction_horizon, name='x_inf')
    u_inf = m.addMVar(prediction_horizon, name='u_inf')
    phi_x_inf = m.addMVar(number_of_blocks, name='phi_x_inf')
    phi_u_inf = m.addMVar(number_of_blocks, name='phi_u_inf')

    phi_x_one = m.addMVar(number_of_blocks * problem['nx'], name='phi_x_one')
    phi_u_one = m.addMVar(number_of_blocks * problem['nu'], name='phi_u_one')

    A_f, B_f = sparse_state_matrix_replacement(sparse.coo_matrix(problem['A']),
                                               sparse.coo_matrix(problem['B']), mask_A, mask_B)

    Ax = -sparse.eye(prediction_horizon * x_vars) + sparse.kron(sparse.eye(prediction_horizon, k=-1),
                                                                          A_f)
    Bu = sparse.kron(sparse.eye(prediction_horizon), B_f)
    # rhs = np.zeros(prediction_horizon * x_vars)
    # rhs[:x_vars] = -A_f @ np.eye(problem['nx')[mask_A]

    sigma_matrix = get_sigma_matrix(mask_A)
    A_f_extended = sparse.vstack([A_f, np.zeros(((prediction_horizon - 1) * x_vars, A_f.shape[1]))])
    rhs_base = -A_f_extended @ sigma_matrix

    m.addConstr(Ax @ phi_x[:prediction_horizon * x_vars] + Bu @ phi_u[:prediction_horizon * u_vars] == rhs_base @ np.ones(problem['nx']))

    for n in range(1, prediction_horizon):
        Ax = -sparse.eye((prediction_horizon - n) * x_vars) + sparse.kron(
            sparse.eye((prediction_horizon - n), k=-1), A_f)
        Bu = sparse.kron(sparse.eye(prediction_horizon - n), B_f)

        try:
            m.addConstr(Ax @ phi_x[block_ordering[n, n] * x_vars:block_ordering[n + 1, n + 1] * x_vars] +
                                    Bu @ phi_u[block_ordering[n, n] * u_vars:block_ordering[n + 1, n + 1] * u_vars] == rhs_base[:(prediction_horizon - n) * x_vars] @
                                    sigma[(n - 1) * problem['nx']:n * problem['nx']])
        except IndexError:
            m.addConstr(Ax @ phi_x[block_ordering[n, n] * x_vars:] +
                                    Bu @ phi_u[block_ordering[n, n] * u_vars:] == rhs_base[:(prediction_horizon - n) * x_vars] @ sigma[ (n - 1) * problem['nx']:n * problem['nx']])


    for i in range(prediction_horizon):
        m.addConstr(x_inf[i] == gp.norm(x[i * problem['nx']: (i + 1) * problem['nx']], gp.GRB.INFINITY))
        m.addConstr(u_inf[i] == gp.norm(u[i * problem['nu']: (i + 1) * problem['nu']], gp.GRB.INFINITY))

    one_norm_matrix_x, one_norm_matrix_u = find_fx_and_fu(mask_A, mask_B, np.ones(problem['nx']))
    # print(one_norm_matrix_x)
    # print(one_norm_matrix_u)
    for i in range(number_of_blocks):
        # m.addConstr(phi_x_inf[i] == gp.norm(phi_x_one[i * problem['nx': (i + 1) * problem['nx'], gp.GRB.INFINITY))
        # m.addConstr(phi_u_inf[i] == gp.norm(phi_u_one[i * problem['nu: (i + 1) * problem['nu], gp.GRB.INFINITY))
        m.addConstr(
            phi_x_inf[i] == gp.max_(phi_x_one[i * problem['nx'] + j] for j in range(problem['nx'])))
        m.addConstr(
            phi_u_inf[i] == gp.max_(phi_u_one[i * problem['nu'] + j] for j in range(problem['nu'])))

        for j in range(problem['nx']):
            ind = np.linspace(i * x_vars, (i + 1) * x_vars, num=x_vars, endpoint=False, dtype=int)[
                np.array(one_norm_matrix_x.todense())[j, :] > 0]
            m.addConstr(
                phi_x_one[i * problem['nx'] + j] == gp.norm(phi_x[ind], 1.0))

        for j in range(problem['nu']):
            ind = np.linspace(i * u_vars, (i + 1) * u_vars, num=u_vars, endpoint=False, dtype=int)[
                np.array(one_norm_matrix_u.todense())[j, :] > 0]
            # print(ind)
            m.addConstr(
                phi_u_one[i * problem['nu'] + j] == gp.norm(phi_u[ind], 1.0))

    # Upper bound on lumped disturbance
    epsilon_matrix_A = np.array([problem['e_A0'], problem['e_A1'], problem['e_A2'],
                                 problem['e_A3'], problem['e_A4'], problem['e_A5']])

    epsilon_matrix_B = np.array([problem['e_B0'], problem['e_B1'], problem['e_B2'],
                                 problem['e_B3'], problem['e_B4'], problem['e_B5']])

    sigma_w = np.array([problem['sigma_w0'], problem['sigma_w1'], problem['sigma_w2'],
                        problem['sigma_w3'], problem['sigma_w4'], problem['sigma_w5']])

    epsilon_matrix_A = np.kron(np.ones(number_of_satellites), epsilon_matrix_A)
    epsilon_matrix_B = np.kron(np.ones(number_of_satellites), epsilon_matrix_B)
    sigma_w = np.kron(np.ones(number_of_satellites), sigma_w)

    # t = 1
    m.addConstr(epsilon_matrix_A * (x_inf[0] + sigma[:problem['nx']]) +
                            epsilon_matrix_B * (
                                    u_inf[1] + gp.quicksum(phi_u_inf[block_ordering[1, 1:2]])) +
                            sigma_w <= sigma[problem['nx']:2 * problem['nx']])
    for n in range(2, prediction_horizon):
        m.addConstr(
            epsilon_matrix_A * (
                    x_inf[n - 1] + gp.quicksum(phi_x_inf[block_ordering[n - 1, 1:n]]) + sigma[( n - 1) * problem['nx']: n * problem['nx']]) +
            epsilon_matrix_B * (u_inf[n] + gp.quicksum(phi_u_inf[block_ordering[n, 1:n + 1]])) +
            sigma_w <= sigma[n * problem['nx']: (n + 1) * problem['nx']])

    # Tightened constraints
    m.addConstr(x[:problem['nx']] + sigma[:problem['nx']] <= x_max[1])
    m.addConstr(x[:problem['nx']] - sigma[:problem['nx']] >= -x_max[1])
    m.addConstr(u[:problem['nu']] <= u_max[0])
    m.addConstr(u[:problem['nu']] >= -u_max[0])

    for n in range(1, prediction_horizon):
        for j in range(problem['nx']):
            # print(block_ordering[n, 1:n + 1] * problem['nx' + j)
            m.addConstr(
                x[n * problem['nx'] + j] + gp.quicksum(phi_x_one[block_ordering[n, 1:n + 1] * problem['nx'] + j]) +
                + sigma[n * problem['nx'] + j] <= x_max[n + 1, j])
            m.addConstr(x[n * problem['nx'] + j] - gp.quicksum(
                phi_x_one[block_ordering[n, 1:n + 1] * problem['nx'] + j]) - sigma[n * problem['nx'] + j] >= -
                                    x_max[n + 1, j])
        for j in range(problem['nu']):
            m.addConstr(
                u[n * problem['nu'] + j] + gp.quicksum(
                    phi_u_one[block_ordering[n, 1:n + 1] * problem['nu'] + j]) <=
                u_max[n, j])
            m.addConstr(u[n * problem['nu'] + j] - gp.quicksum(
                phi_u_one[block_ordering[n, 1:n + 1] * problem['nu'] + j]) >= -u_max[n, j])


    # Temporary constraints
    constraint_list = []
    Fx, Fu = find_fx_and_fu(mask_A, mask_B, problem['x0'])
    Fx = sparse.kron(sparse.eye(prediction_horizon), Fx)
    Fu = sparse.kron(sparse.eye(prediction_horizon), Fu)

    constraint_list.append(m.addConstr(
        epsilon_matrix_A * np.max(np.abs(problem['x0'])) + epsilon_matrix_B * u_inf[0] + sigma_w <= sigma[:problem['nx']]))

    constraint_list.append(m.addConstr(Fx @ phi_x[:prediction_horizon * x_vars] == x))
    constraint_list.append(m.addConstr(Fu @ phi_u[:prediction_horizon * u_vars] == u))

    # Objective

    obj1 = x @ sparse.kron(sparse.eye(prediction_horizon), problem['Q']) @ x
    obj1 += 4 * x[-problem['nx']:] @ problem['Q'] @ x[-problem['nx']:]
    obj2 = u @ sparse.kron(sparse.eye(prediction_horizon), problem['R']) @ u
    # obj_test = gp.quicksum(phi_x_one) + gp.quicksum(phi_u_one)
    m.setObjective(obj1 + obj2, gp.GRB.MINIMIZE)
    m.setParam("OutputFlag", 0)
    m.setParam("OptimalityTol", 1e-3)
    # m.setParam("BarConvTol", 1e-4)

    # Simulate in closed loop
    t_0 = 0
    nsim = 11
    runtime = 0
    states = np.zeros((problem['nx'], nsim + 1))
    states[:, 0] = problem['x0']
    inputs = np.zeros((problem['nu'], nsim))

    for i in range(nsim):
        print(i)
        # Solve
        m.optimize()
        runtime += m.Runtime
        print(m.Runtime)

        states[:, i+1] = x.X[:problem['nx']]
        inputs[:, i] = u.X[:problem['nu']]

        # print(phi_x.X[:x_vars])
        # print()
        # print(phi_x_inf.X[:18])

        m.remove(constraint_list)

        Fx, Fu = find_fx_and_fu(mask_A, mask_B, states[:, i+1])
        Fx = sparse.kron(sparse.eye(prediction_horizon), Fx)
        Fu = sparse.kron(sparse.eye(prediction_horizon), Fu)

        constraint_list.append(m.addConstr(
            epsilon_matrix_A * np.max(np.abs(states[:, i+1])) + epsilon_matrix_B * u_inf[0] + sigma_w <= sigma[:problem[
                'nx']]))

        constraint_list.append(m.addConstr(Fx @ phi_x[:prediction_horizon * x_vars] == x))
        constraint_list.append(m.addConstr(Fu @ phi_u[:prediction_horizon * u_vars] == u))

        m.update()

        if i == 0:
            t_0 = time.time()
            runtime = 0

    t_end = time.time()

    avg_time = (t_end - t_0) / (nsim - 1)
    print(f"Average elapsed time Gurobi robust for {number_of_satellites} satellites: {avg_time}")
    print(f"Runtime avg.: {runtime / (nsim - 1)}")

    if plot_results:
        plt.figure()
        plt.plot(np.rad2deg(states[1::6].T))
        plt.ylabel('Angle')
        plt.grid(True)

        plt.figure()
        plt.plot(states[0::6].T)
        plt.ylabel('Radius')
        plt.grid(True)

        plt.figure()
        plt.plot(inputs[0::3].T)
        plt.ylabel('u_r')
        plt.grid(True)

        plt.figure()
        plt.plot(inputs[1::3].T)
        plt.ylabel('u_t')
        plt.grid(True)
        plt.show()
    return avg_time



if __name__ == '__main__':
    time_optimisation(3, prediction_horizon=6, plot_results=True)
