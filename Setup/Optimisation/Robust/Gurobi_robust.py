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

    mask_A, mask_B = get_masks(problem['A'], problem['B'])
    x_vars = np.sum(mask_A)
    u_vars = np.sum(mask_B)
    number_of_blocks = int((problem['N'] + 1) / 2 * problem['N'] + 0.001)

    block_ordering = sparse.csc_matrix(np.tril(np.ones((problem['N'], problem['N']))))
    block_ordering.data = np.arange(number_of_blocks)
    block_ordering = np.array(block_ordering.todense())

    one_norm_matrix_x_single, one_norm_matrix_u_single = find_fx_and_fu(mask_A, mask_B, np.ones_like(problem['x0']))
    one_norm_kron_mat = np.zeros((problem['N'], number_of_blocks))
    for n in range(problem['N']):
        one_norm_kron_mat[n, block_ordering[n, 1:n + 1]] = 1

    x_lim = np.tile(problem['x_max'], (1, problem['N'])).flatten()
    u_lim = np.tile(problem['u_max'], (1, problem['N'])).flatten()

    x = m.addMVar(problem['N'] * problem['nx'], name='x', lb=-np.inf) #, ub=x_lim)
    u = m.addMVar(problem['N'] * problem['nu'], name='u', lb=-np.inf) #, ub=u_lim)
    phi_x = m.addMVar(number_of_blocks * x_vars, name='phi_x', lb=-np.inf) #, ub=np.inf)
    phi_u = m.addMVar(number_of_blocks * u_vars, name='phi_u', lb=-np.inf) #, ub=np.inf)
    sigma = m.addMVar(problem['N'] * problem['nx'], name='sigma')

    phi_x_pos = m.addMVar(number_of_blocks * x_vars, name='phi_x_pos') #, lb=0, ub=np.inf)
    phi_x_neg = m.addMVar(number_of_blocks * x_vars, name='phi_x_neg') #, lb=0, ub=np.inf)
    phi_x_abs = m.addMVar(number_of_blocks * x_vars, name='phi_x_abs') #, lb=0, ub=np.inf)

    phi_u_pos = m.addMVar(number_of_blocks * u_vars, name='phi_u_pos') #, lb=0, ub=np.inf)
    phi_u_neg = m.addMVar(number_of_blocks * u_vars, name='phi_u_neg') #, lb=0, ub=np.inf)
    phi_u_abs = m.addMVar(number_of_blocks * u_vars, name='phi_u_abs') #, lb=0, ub=np.inf)

    x_pos = m.addMVar(problem['N'] * problem['nx'], name='x_pos') #, lb=0, ub=x_max)
    x_neg = m.addMVar(problem['N'] * problem['nx'], name='x_neg') #, lb=0, ub=x_max)
    x_abs = m.addMVar(problem['N'] * problem['nx'], name='x_abs') #, lb=0, ub=x_max)

    u_pos = m.addMVar(problem['N'] * problem['nu'], name='u_pos') #, lb=0, ub=u_max)
    u_neg = m.addMVar(problem['N'] * problem['nu'], name='u_neg') #, lb=0, ub=u_max)
    u_abs = m.addMVar(problem['N'] * problem['nu'], name='u_abs') #, lb=0, ub=u_max)

    x_max = m.addMVar(problem['N'], name='x_max')
    phi_x_max = m.addMVar(number_of_blocks, name='phi_x_max')
    u_max = m.addMVar(problem['N'], name='u_max')
    phi_u_max = m.addMVar(number_of_blocks, name='phi_u_max')

    A_f, B_f = sparse_state_matrix_replacement(problem['A'], problem['B'], mask_A, mask_B)
    Fx, Fu = find_fx_and_fu(mask_A, mask_B, problem['x0'])

    Q = sparse.kron(np.diag(np.concatenate((np.ones(problem['N']-1), np.array([5])))), problem['Q'])
    R = sparse.kron(sparse.eye(problem['N']), problem['R'])

    Fx = sparse.kron(sparse.eye(problem['N']), Fx)
    Fu = sparse.kron(sparse.eye(problem['N']), Fu)

    one_norm_matrix_x = sparse.kron(one_norm_kron_mat, one_norm_matrix_x_single)
    one_norm_matrix_u = sparse.kron(one_norm_kron_mat, one_norm_matrix_u_single)

    Ax = -sparse.eye(problem['N'] * x_vars) + sparse.kron(sparse.eye(problem['N'], k=-1), A_f)
    Bu = sparse.kron(sparse.eye(problem['N']), B_f)
    sigma_matrix = get_sigma_matrix(mask_A)
    A_f_extended = sparse.vstack([A_f, np.zeros(((prediction_horizon - 1) * x_vars, A_f.shape[1]))])
    rhs_base = -A_f_extended @ sigma_matrix

    m.addConstr(
        Ax @ phi_x[:prediction_horizon * x_vars] + Bu @ phi_u[:prediction_horizon * u_vars] == rhs_base @ np.ones(problem['nx']))


    for n in range(1, problem['N']):
        Ax = -sparse.eye((problem['N']-n) * x_vars) + sparse.kron(sparse.eye((problem['N'] - n), k=-1), A_f)
        Bu = sparse.kron(sparse.eye(problem['N'] - n), B_f)

        try:
            m.addConstr(Ax @ phi_x[block_ordering[n, n] * x_vars:block_ordering[n + 1, n + 1] * x_vars] +
                        Bu @ phi_u[block_ordering[n, n] * u_vars:block_ordering[n + 1, n + 1] * u_vars] == rhs_base[:(prediction_horizon - n) * x_vars] @
                        sigma[(n - 1) * problem['nx']:n * problem['nx']])
        except IndexError:
            m.addConstr(Ax @ phi_x[block_ordering[n, n] * x_vars:] +
                        Bu @ phi_u[block_ordering[n, n] * u_vars:] == rhs_base[:(prediction_horizon - n) * x_vars] @ sigma[(n - 1) * problem['nx']:n *problem['nx']])

    # m.addConstr(phi_x[] == A_f @ np.eye(problem['nx'])[mask_A] + _f @ phi_u[:u_vars])
    #
    # for n in range(1, problem['N']):
    #     m.addConstr(phi_x[n * x_vars: (n+1) * x_vars] == A_f @ phi_x[(n-1) * x_vars: n * x_vars] + B_f @ phi_u[n * u_vars: (n+1) * u_vars])

    # robust constraints
    m.addConstr(phi_x_pos - phi_x_neg == phi_x)
    m.addConstr(phi_x_pos + phi_x_neg == phi_x_abs)

    m.addConstr(phi_u_pos - phi_u_neg == phi_u)
    m.addConstr(phi_u_pos + phi_u_neg == phi_u_abs)

    m.addConstr(x_pos - x_neg == x)
    m.addConstr(x_pos + x_neg == x_abs)

    m.addConstr(u_pos - u_neg == u)
    m.addConstr(u_pos + u_neg == u_abs)

    for n in range(problem['N']):
        m.addConstr(x_abs[n * problem['nx']:(n + 1) * problem['nx']] <= x_max[n])
        m.addConstr(u_abs[n * problem['nu']:(n + 1) * problem['nu']] <= u_max[n])

    for n in range(number_of_blocks):
        m.addConstr(one_norm_matrix_x_single @ phi_x_abs[n * x_vars:(n + 1) * x_vars] <= phi_x_max[n])
        m.addConstr(one_norm_matrix_u_single @ phi_u_abs[n * u_vars:(n + 1) * u_vars] <= phi_u_max[n])

        # m.addConstr(phi_x_abs[n * x_vars:(n + 1) * x_vars] <= phi_x_max[n])
        # m.addConstr(phi_u_abs[n * u_vars:(n + 1) * u_vars] <= phi_u_max[n])

    # Infinity norm constraints
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
    m.addConstr(epsilon_matrix_A * (x_max[0] + sigma[:problem['nx']]) +
                epsilon_matrix_B * (u_max[1] + gp.quicksum(phi_u_max[block_ordering[1, 1:2]])) +
                sigma_w <= sigma[problem['nx']:2 * problem['nx']])
    for n in range(2, problem['N']):
        m.addConstr(epsilon_matrix_A * (x_max[n-1] + gp.quicksum(phi_x_max[block_ordering[n-1, 1:n]]) + sigma[( n - 1) * problem['nx']: n * problem['nx']]) +
                    epsilon_matrix_B * (u_max[n] + gp.quicksum(phi_u_max[block_ordering[n, 1:n+1]])) +
                    sigma_w <= sigma[n * problem['nx']: (n + 1) * problem['nx']])

    # 1-norm constraints
    m.addConstr(x + one_norm_matrix_x @ phi_x_abs + sigma <= x_lim)
    m.addConstr(x - one_norm_matrix_x @ phi_x_abs - sigma >= -x_lim)
    m.addConstr(u + one_norm_matrix_u @ phi_u_abs <= u_lim)
    m.addConstr(u - one_norm_matrix_u @ phi_u_abs >= -u_lim)
    # m.addConstr(x <= x_lim)
    # m.addConstr(x >= -x_lim)
    # m.addConstr(u <= u_lim)
    # m.addConstr(u >= -u_lim)

    # m.addConstr(inf_norms_x[0] == norm(problem['x0'], GRB.INFINITY))
    # m.addConstr(inf_norms_x[1] == norm(u[:problem['nu']], GRB.INFINITY))
    #
    # m.addConstr(problem['e_A'] * inf_norms_x[0] + problem['e_B'] * inf_norms_x[1] + problem['sigma_w'] <= sigma[0])
    constraint_list = []
    constraint_list.append(m.addConstr(
        epsilon_matrix_A * np.max(np.abs(problem['x0'])) + epsilon_matrix_B * u_max[0] + sigma_w <= sigma[:problem['nx']]))

    constraint_list.append(m.addConstr(Fx @ phi_x[:problem['N'] * x_vars] == x))
    constraint_list.append(m.addConstr(Fu @ phi_u[:problem['N'] * u_vars] == u))

    # scaling = 1e-1
    # obj_robust = scaling * gp.quicksum(phi_x_abs) + scaling * gp.quicksum(phi_u_abs)
    # obj_robust += scaling * gp.quicksum(x_abs) + scaling * gp.quicksum(x_abs)
    # obj_robust += scaling * gp.quicksum(x_max) + scaling * gp.quicksum(u_max)
    # obj_robust += scaling * gp.quicksum(phi_x_max) + scaling * gp.quicksum(phi_u_max)
    # obj_robust = 0
    m.setObjective(x @ Q @ x + u @ R @ u, GRB.MINIMIZE)
    m.setParam("OutputFlag", 0)
    m.setParam("OptimalityTol", 1e-2)
    m.setParam("BarConvTol", 1e-5)

    # Simulate in closed loop
    t_0 = 0
    nsim = 31
    runtime = 0
    states = np.zeros((problem['nx'], nsim + 1))
    states[:, 0] = problem['x0']
    inputs = np.zeros((problem['nu'], nsim))

    for i in range(nsim):
        # Solve
        m.optimize()
        runtime += m.Runtime

        # print(x.X.T @ Q @ x.X + u.X.T @ R @ u.X)
        # print(obj_robust.getValue())

        states[:, i+1] = x.X[:problem['nx']]
        inputs[:, i] = u.X[:problem['nu']]

        # print(x_abs.X)
        # print(phi_x.X)
        # print(sigma.X)

        # print(phi_x.X[:x_vars])
        # print()
        # print(phi_x_abs.X[:x_vars])
        # print()
        # print(phi_x_max.X[:18])


        m.remove(constraint_list)

        Fx, Fu = find_fx_and_fu(mask_A, mask_B, states[:, i + 1])
        constraint_list = []
        Fx = sparse.kron(sparse.eye(problem['N']), Fx)
        Fu = sparse.kron(sparse.eye(problem['N']), Fu)

        constraint_list.append(m.addConstr(
            epsilon_matrix_A * np.max(np.abs(states[:, i+1])) + epsilon_matrix_B * u_max[0] + sigma_w <= sigma[:problem['nx']]))
        constraint_list.append(m.addConstr(Fx @ phi_x[:problem['N'] * x_vars] == x))
        constraint_list.append(m.addConstr(Fu @ phi_u[:problem['N'] * u_vars] == u))

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


# def time_optimisation_2(number_of_satellites: int, prediction_horizon: int = None, plot_results: bool = False,
#                       scenario: Scenario = ScenarioEnum.robustness_comparison_advanced_robust_noise.value) -> float:
#     """
#     Measure the time it takes to optimise the controller.
#
#     :param number_of_satellites: Number of satellites to optimise.
#     :param prediction_horizon: Prediction horizon of the controller.
#     :param plot_results: Whether to plot the results.
#     :param scenario: Scenario to run. Defaults to HCW scenario.
#     :return: Float with average time (after first iteration has passed).
#     """
#     problem = create_sparse_problem(number_of_satellites, prediction_horizon, scenario)
#     # problem['x0'][1] = 1
#     # Create a new model
#
#     m = gp.Model("MPC")
#
#     mask_A, mask_B = get_masks(problem['A'], problem['B'])
#     x_vars = np.sum(mask_A)
#     u_vars = np.sum(mask_B)
#     number_of_blocks = int((problem['N'] + 1) / 2 * problem['N'] + 0.001)
#
#     block_ordering = sparse.csc_matrix(np.tril(np.ones((problem['N'], problem['N']))))
#     block_ordering.data = np.arange(number_of_blocks)
#     block_ordering = np.array(block_ordering.todense())
#
#     one_norm_matrix_x_single, one_norm_matrix_u_single = find_fx_and_fu(mask_A, mask_B, np.ones_like(problem['x0']))
#     one_norm_kron_mat = np.zeros((problem['N'], number_of_blocks))
#     for n in range(problem['N']):
#         one_norm_kron_mat[n, block_ordering[n, 1:n + 1]] = 1
#
#     x_lim = np.tile(problem['x_max'], (1, problem['N'])).flatten()
#     u_lim = np.tile(problem['u_max'], (1, problem['N'])).flatten()
#
#     x = m.addMVar(problem['N'] * problem['nx'], name='x', lb=-np.inf) #, ub=x_lim)
#     u = m.addMVar(problem['N'] * problem['nu'], name='u', lb=-np.inf) #, ub=u_lim)
#     phi_x = m.addMVar(number_of_blocks * x_vars, name='phi_x', lb=-np.inf) #, ub=np.inf)
#     phi_u = m.addMVar(number_of_blocks * u_vars, name='phi_u', lb=-np.inf) #, ub=np.inf)
#     sigma = m.addMVar(problem['N'] * problem['nx'], name='sigma')
#
#     # phi_x_pos = m.addMVar(number_of_blocks * x_vars, name='phi_x_pos') #, lb=0, ub=np.inf)
#     # phi_x_neg = m.addMVar(number_of_blocks * x_vars, name='phi_x_neg') #, lb=0, ub=np.inf)
#     phi_x_abs = m.addMVar(number_of_blocks * x_vars, name='phi_x_abs') #, lb=0, ub=np.inf)
#
#     # phi_u_pos = m.addMVar(number_of_blocks * u_vars, name='phi_u_pos') #, lb=0, ub=np.inf)
#     # phi_u_neg = m.addMVar(number_of_blocks * u_vars, name='phi_u_neg') #, lb=0, ub=np.inf)
#     phi_u_abs = m.addMVar(number_of_blocks * u_vars, name='phi_u_abs') #, lb=0, ub=np.inf)
#
#     # x_pos = m.addMVar(problem['N'] * problem['nx'], name='x_pos') #, lb=0, ub=x_max)
#     # x_neg = m.addMVar(problem['N'] * problem['nx'], name='x_neg') #, lb=0, ub=x_max)
#     x_abs = m.addMVar(problem['N'] * problem['nx'], name='x_abs') #, lb=0, ub=x_max)
#
#     # u_pos = m.addMVar(problem['N'] * problem['nu'], name='u_pos') #, lb=0, ub=u_max)
#     # u_neg = m.addMVar(problem['N'] * problem['nu'], name='u_neg') #, lb=0, ub=u_max)
#     u_abs = m.addMVar(problem['N'] * problem['nu'], name='u_abs') #, lb=0, ub=u_max)
#
#     x_max = m.addMVar(problem['N'], name='x_max')
#     phi_x_max = m.addMVar(number_of_blocks, name='phi_x_max')
#     u_max = m.addMVar(problem['N'], name='u_max')
#     phi_u_max = m.addMVar(number_of_blocks, name='phi_u_max')
#
#     A_f, B_f = sparse_state_matrix_replacement(problem['A'], problem['B'], mask_A, mask_B)
#     Fx, Fu = find_fx_and_fu(mask_A, mask_B, problem['x0'])
#
#     Q = sparse.kron(np.diag(np.concatenate((np.ones(problem['N']-1), np.array([5])))), problem['Q'])
#     R = sparse.kron(sparse.eye(problem['N']), problem['R'])
#
#     Fx = sparse.kron(sparse.eye(problem['N']), Fx)
#     Fu = sparse.kron(sparse.eye(problem['N']), Fu)
#
#     one_norm_matrix_x = sparse.kron(one_norm_kron_mat, one_norm_matrix_x_single)
#     one_norm_matrix_u = sparse.kron(one_norm_kron_mat, one_norm_matrix_u_single)
#
#     Ax = -sparse.eye(problem['N'] * x_vars) + sparse.kron(sparse.eye(problem['N'], k=-1), A_f)
#     Bu = sparse.kron(sparse.eye(problem['N']), B_f)
#     sigma_matrix = get_sigma_matrix(mask_A)
#     A_f_extended = sparse.vstack([A_f, np.zeros(((prediction_horizon - 1) * x_vars, A_f.shape[1]))])
#     rhs_base = -A_f_extended @ sigma_matrix
#
#     m.addConstr(
#         Ax @ phi_x[:prediction_horizon * x_vars] + Bu @ phi_u[:prediction_horizon * u_vars] == rhs_base @ np.ones(problem['nx']))
#
#
#     for n in range(1, problem['N']):
#         Ax = -sparse.eye((problem['N']-n) * x_vars) + sparse.kron(sparse.eye((problem['N'] - n), k=-1), A_f)
#         Bu = sparse.kron(sparse.eye(problem['N'] - n), B_f)
#
#         try:
#             m.addConstr(Ax @ phi_x[block_ordering[n, n] * x_vars:block_ordering[n + 1, n + 1] * x_vars] +
#                         Bu @ phi_u[block_ordering[n, n] * u_vars:block_ordering[n + 1, n + 1] * u_vars] == rhs_base[:(prediction_horizon - n) * x_vars] @
#                         sigma[(n - 1) * problem['nx']:n * problem['nx']])
#         except IndexError:
#             m.addConstr(Ax @ phi_x[block_ordering[n, n] * x_vars:] +
#                         Bu @ phi_u[block_ordering[n, n] * u_vars:] == rhs_base[:(prediction_horizon - n) * x_vars] @ sigma[(n - 1) * problem['nx']:n *problem['nx']])
#
#     # m.addConstr(phi_x[] == A_f @ np.eye(problem['nx'])[mask_A] + B_f @ phi_u[:u_vars])
#     #
#     # for n in range(1, problem['N']):
#     #     m.addConstr(phi_x[n * x_vars: (n+1) * x_vars] == A_f @ phi_x[(n-1) * x_vars: n * x_vars] + B_f @ phi_u[n * u_vars: (n+1) * u_vars])
#
#     # robust constraints
#     m.addConstr(phi_x <= phi_x_abs)
#     m.addConstr(-phi_x <= phi_x_abs)
#
#     m.addConstr(phi_u <= phi_u_abs)
#     m.addConstr(-phi_u <= phi_u_abs)
#
#     m.addConstr(x <= x_abs)
#     m.addConstr(-x <= x_abs)
#
#     m.addConstr(u <= u_abs)
#     m.addConstr(-u <= u_abs)
#
#     for n in range(problem['N']):
#         m.addConstr(x_abs[n * problem['nx']:(n + 1) * problem['nx']] <= x_max[n])
#         m.addConstr(u_abs[n * problem['nu']:(n + 1) * problem['nu']] <= u_max[n])
#
#     for n in range(number_of_blocks):
#         m.addConstr(one_norm_matrix_x_single @ phi_x_abs[n * x_vars:(n + 1) * x_vars] <= phi_x_max[n])
#         m.addConstr(one_norm_matrix_u_single @ phi_u_abs[n * u_vars:(n + 1) * u_vars] <= phi_u_max[n])
#
#         # m.addConstr(phi_x_abs[n * x_vars:(n + 1) * x_vars] <= phi_x_max[n])
#         # m.addConstr(phi_u_abs[n * u_vars:(n + 1) * u_vars] <= phi_u_max[n])
#
#     # Infinity norm constraints
#     epsilon_matrix_A = np.array([problem['e_A0'], problem['e_A1'], problem['e_A2'],
#                                  problem['e_A3'], problem['e_A4'], problem['e_A5']])
#
#     epsilon_matrix_B = np.array([problem['e_B0'], problem['e_B1'], problem['e_B2'],
#                                  problem['e_B3'], problem['e_B4'], problem['e_B5']])
#
#     sigma_w = np.array([problem['sigma_w0'], problem['sigma_w1'], problem['sigma_w2'],
#                         problem['sigma_w3'], problem['sigma_w4'], problem['sigma_w5']])
#
#     epsilon_matrix_A = np.kron(np.ones(number_of_satellites), epsilon_matrix_A)
#     epsilon_matrix_B = np.kron(np.ones(number_of_satellites), epsilon_matrix_B)
#     sigma_w = np.kron(np.ones(number_of_satellites), sigma_w)
#
#     # t = 1
#     m.addConstr(epsilon_matrix_A * (x_max[0] + sigma[:problem['nx']]) +
#                 epsilon_matrix_B * (u_max[1] + gp.quicksum(phi_u_max[block_ordering[1, 1:2]])) +
#                 sigma_w <= sigma[problem['nx']:2 * problem['nx']])
#     for n in range(2, problem['N']):
#         m.addConstr(epsilon_matrix_A * (x_max[n-1] + gp.quicksum(phi_x_max[block_ordering[n-1, 1:n]]) + sigma[( n - 1) * problem['nx']: n * problem['nx']]) +
#                     epsilon_matrix_B * (u_max[n] + gp.quicksum(phi_u_max[block_ordering[n, 1:n+1]])) +
#                     sigma_w <= sigma[n * problem['nx']: (n + 1) * problem['nx']])
#
#     # 1-norm constraints
#     m.addConstr(x + one_norm_matrix_x @ phi_x_abs + sigma <= x_lim)
#     m.addConstr(x - one_norm_matrix_x @ phi_x_abs - sigma >= -x_lim)
#     m.addConstr(u + one_norm_matrix_u @ phi_u_abs <= u_lim)
#     m.addConstr(u - one_norm_matrix_u @ phi_u_abs >= -u_lim)
#     # m.addConstr(x <= x_lim)
#     # m.addConstr(x >= -x_lim)
#     # m.addConstr(u <= u_lim)
#     # m.addConstr(u >= -u_lim)
#
#     # m.addConstr(inf_norms_x[0] == norm(problem['x0'], GRB.INFINITY))
#     # m.addConstr(inf_norms_x[1] == norm(u[:problem['nu']], GRB.INFINITY))
#     #
#     # m.addConstr(problem['e_A'] * inf_norms_x[0] + problem['e_B'] * inf_norms_x[1] + problem['sigma_w'] <= sigma[0])
#     constraint_list = []
#     constraint_list.append(m.addConstr(
#         epsilon_matrix_A * np.max(np.abs(problem['x0'])) + epsilon_matrix_B * u_max[0] + sigma_w <= sigma[:problem['nx']]))
#
#     constraint_list.append(m.addConstr(Fx @ phi_x[:problem['N'] * x_vars] == x))
#     constraint_list.append(m.addConstr(Fu @ phi_u[:problem['N'] * u_vars] == u))
#
#     scaling = 1e-8
#     obj_robust = scaling * gp.quicksum(phi_x_abs) + scaling * gp.quicksum(phi_u_abs)
#     obj_robust += 0.01 * scaling * gp.quicksum(x_abs) + 0.01 * scaling * gp.quicksum(x_abs)
#     # obj_robust += scaling * gp.quicksum(x_max) + scaling * gp.quicksum(u_max)
#     # obj_robust += scaling * gp.quicksum(phi_x_max) + scaling * gp.quicksum(phi_u_max)
#     m.setObjective(x @ Q @ x + u @ R @ u + obj_robust, GRB.MINIMIZE)
#     m.setParam("OutputFlag", 0)
#     m.setParam("OptimalityTol", 1e-3)
#     m.setParam("BarConvTol", 1e-9)
#
#     # Simulate in closed loop
#     t_0 = 0
#     nsim = 11
#     runtime = 0
#     states = np.zeros((problem['nx'], nsim + 1))
#     states[:, 0] = problem['x0']
#     inputs = np.zeros((problem['nu'], nsim))
#
#     for i in range(nsim):
#         # Solve
#         m.optimize()
#         runtime += m.Runtime
#
#         # print(x.X @ Q @ x.X + u.X @ R @ u.X)
#         # print(obj_robust.value())
#
#         states[:, i+1] = x.X[:problem['nx']]
#         inputs[:, i] = u.X[:problem['nu']]
#
#         # print(phi_x.X[:x_vars])
#         # print()
#         # print(phi_x_abs.X[:x_vars])
#         # print()
#         # print(phi_x_max.X[:18])
#
#
#         m.remove(constraint_list)
#
#         Fx, Fu = find_fx_and_fu(mask_A, mask_B, states[:, i + 1])
#         constraint_list = []
#         Fx = sparse.kron(sparse.eye(problem['N']), Fx)
#         Fu = sparse.kron(sparse.eye(problem['N']), Fu)
#
#         constraint_list.append(m.addConstr(
#             epsilon_matrix_A * np.max(np.abs(states[:, i+1])) + epsilon_matrix_B * u_max[0] + sigma_w <= sigma[:problem['nx']]))
#         constraint_list.append(m.addConstr(Fx @ phi_x[:problem['N'] * x_vars] == x))
#         constraint_list.append(m.addConstr(Fu @ phi_u[:problem['N'] * u_vars] == u))
#
#         m.update()
#
#         if i == 0:
#             t_0 = time.time()
#             runtime = 0
#
#     t_end = time.time()
#
#     avg_time = (t_end - t_0) / (nsim - 1)
#     print(f"Average elapsed time Gurobi robust for {number_of_satellites} satellites: {avg_time}")
#     print(f"Runtime avg.: {runtime / (nsim - 1)}")
#
#     if plot_results:
#         plt.figure()
#         plt.plot(np.rad2deg(states[1::6].T))
#         plt.ylabel('Angle')
#         plt.grid(True)
#
#         plt.figure()
#         plt.plot(states[0::6].T)
#         plt.ylabel('Radius')
#         plt.grid(True)
#
#         plt.figure()
#         plt.plot(inputs[0::3].T)
#         plt.ylabel('u_r')
#         plt.grid(True)
#
#         plt.figure()
#         plt.plot(inputs[1::3].T)
#         plt.ylabel('u_t')
#         plt.grid(True)
#         plt.show()
#
#     return avg_time


if __name__ == '__main__':
    time_optimisation(3, prediction_horizon=6, plot_results=True)
