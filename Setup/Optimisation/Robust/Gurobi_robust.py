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
                      scenario: Scenario = ScenarioEnum.simple_scenario_translation_HCW_scaled.value) -> float:
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

    constraint_list = []
    m = gp.Model("MPC")

    mask_A, mask_B = get_masks(problem['A'], problem['B'])
    x_vars = np.sum(mask_A)
    u_vars = np.sum(mask_B)
    number_of_blocks = int((problem['N'] + 1) / 2 * problem['N'] + 0.001)

    block_ordering = sparse.csc_matrix(np.tril(np.ones((problem['N'], problem['N']))))
    block_ordering.data = np.arange(number_of_blocks)
    block_ordering = np.array(block_ordering.todense())

    one_norm_matrix_x, one_norm_matrix_u = find_fx_and_fu(mask_A, mask_B, np.ones_like(problem['x0']))
    one_norm_kron_mat = np.zeros((problem['N'], number_of_blocks))
    for n in range(problem['N']):
        one_norm_kron_mat[n, block_ordering[n, 1:n + 1]] = 1

    x_lim = np.tile(problem['x_max'], (1, problem['N'])).flatten()
    u_lim = np.tile(problem['u_max'], (1, problem['N'])).flatten()

    x = m.addMVar(problem['N'] * problem['nx'], name='x', lb=-np.inf) #, ub=x_lim)
    u = m.addMVar(problem['N'] * problem['nu'], name='u', lb=-np.inf) #, ub=u_lim)
    phi_x = m.addMVar(number_of_blocks * x_vars, name='phi_x', lb=-np.inf) #, ub=np.inf)
    phi_u = m.addMVar(number_of_blocks * u_vars, name='phi_u', lb=-np.inf) #, ub=np.inf)
    sigma = m.addMVar(problem['N'], name='sigma')

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

    Q = sparse.kron(sparse.eye(problem['N']), problem['Q'])
    R = sparse.kron(sparse.eye(problem['N']), problem['R'])

    Fx = sparse.kron(sparse.eye(problem['N']), Fx)
    Fu = sparse.kron(sparse.eye(problem['N']), Fu)

    one_norm_matrix_x = sparse.kron(one_norm_kron_mat, one_norm_matrix_x)
    one_norm_matrix_u = sparse.kron(one_norm_kron_mat, one_norm_matrix_u)

    Ax = -sparse.eye(problem['N'] * x_vars) + sparse.kron(sparse.eye(problem['N'], k=-1), A_f)
    Bu = sparse.kron(sparse.eye(problem['N']), B_f)
    rhs = np.zeros(problem['N'] * x_vars)
    rhs[:x_vars] = -A_f @ np.eye(problem['nx'])[mask_A]

    m.addConstr(Ax @ phi_x[:problem['N'] * x_vars] + Bu @ phi_u[:problem['N'] * u_vars] == rhs)

    for n in range(1, problem['N']):
        Ax = -sparse.eye((problem['N']-n) * x_vars) + sparse.kron(sparse.eye((problem['N'] - n), k=-1), A_f)
        Bu = sparse.kron(sparse.eye(problem['N'] - n), B_f)

        try:
            m.addConstr(Ax @ phi_x[block_ordering[n, n] * x_vars:block_ordering[n + 1, n + 1] * x_vars] +
                        Bu @ phi_u[block_ordering[n, n] * u_vars:block_ordering[n + 1, n + 1] * u_vars] == rhs[:(problem['N'] - n) * x_vars] * sigma[n-1])
        except IndexError:
            m.addConstr(Ax @ phi_x[block_ordering[n, n] * x_vars:] +
                        Bu @ phi_u[block_ordering[n, n] * u_vars:] == rhs[:(problem['N'] - n) * x_vars] * sigma[n-1])

    # m.addConstr(phi_x[] == A_f @ np.eye(problem['nx'])[mask_A] + B_f @ phi_u[:u_vars])
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
        m.addConstr(phi_x_abs[n * x_vars:(n + 1) * x_vars] <= phi_x_max[n])
        m.addConstr(phi_u_abs[n * u_vars:(n + 1) * u_vars] <= phi_u_max[n])

    # Infinity norm constraints
    constraint_list.append(m.addConstr(
        problem['e_A'] * np.max(np.abs(problem['x0'])) + problem['e_B'] * u_max[0] + problem['sigma_w'] <= sigma[0]))

    m.addConstr(problem['e_A'] * x_max[0] +
                problem['e_B'] * (u_max[1] + gp.quicksum(phi_u_max[block_ordering[1, 1:2]])) +
                problem['sigma_w'] <= sigma[1])
    for n in range(2, problem['N']):
        m.addConstr(problem['e_A'] * (x_max[n-1] + gp.quicksum(phi_x_max[block_ordering[n-1, 1:n]]) + sigma[n]) +
                    problem['e_B'] * (u_max[n] + gp.quicksum(phi_u_max[block_ordering[n, 1:n+1]])) +
                    problem['sigma_w'] <= sigma[n])

    # 1-norm constraints
    m.addConstr(x + one_norm_matrix_x @ phi_x_abs + sparse.kron(sparse.eye(problem['N']), np.ones((problem['nx'], 1))) @ sigma <= x_lim)
    m.addConstr(x - one_norm_matrix_x @ phi_x_abs - sparse.kron(sparse.eye(problem['N']), np.ones((problem['nx'], 1))) @ sigma >= -x_lim)
    m.addConstr(u + one_norm_matrix_u @ phi_u_abs <= u_lim)
    m.addConstr(u - one_norm_matrix_u @ phi_u_abs >= -u_lim)

    # m.addConstr(inf_norms_x[0] == norm(problem['x0'], GRB.INFINITY))
    # m.addConstr(inf_norms_x[1] == norm(u[:problem['nu']], GRB.INFINITY))
    #
    # m.addConstr(problem['e_A'] * inf_norms_x[0] + problem['e_B'] * inf_norms_x[1] + problem['sigma_w'] <= sigma[0])
    constraint_list.append(m.addConstr(Fx @ phi_x[:problem['N'] * x_vars] == x))
    constraint_list.append(m.addConstr(Fu @ phi_u[:problem['N'] * u_vars] == u))

    scaling = 1e-3
    obj_robust = scaling * gp.quicksum(phi_x_abs) + scaling * gp.quicksum(phi_u_abs)
    obj_robust += scaling * gp.quicksum(x_abs) + scaling * gp.quicksum(x_abs)
    obj_robust += scaling * gp.quicksum(x_max) + scaling * gp.quicksum(u_max)
    obj_robust += scaling * gp.quicksum(phi_x_max) + scaling * gp.quicksum(phi_u_max)
    m.setObjective(x @ Q @ x + u @ R @ u + obj_robust, GRB.MINIMIZE)
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
        # Solve
        m.optimize()
        runtime += m.Runtime

        states[:, i+1] = x.X[:problem['nx']]
        inputs[:, i] = u.X[:problem['nu']]

        m.remove(constraint_list)

        Fx, Fu = find_fx_and_fu(mask_A, mask_B, states[:, i + 1])
        constraint_list = []
        Fx = sparse.kron(sparse.eye(problem['N']), Fx)
        Fu = sparse.kron(sparse.eye(problem['N']), Fu)

        constraint_list.append(m.addConstr(
            problem['e_A'] * np.max(np.abs(states[:, i+1])) + problem['e_B'] * u_max[0] + problem['sigma_w'] <= sigma[
                0]))
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
        plt.show()
    return avg_time


def time_optimisation_2(number_of_satellites: int, prediction_horizon: int = None, plot_results: bool = False,
                        scenario: Scenario = ScenarioEnum.simple_scenario_translation_HCW_scaled.value) -> float:
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

    constraint_list = []
    m = gp.Model("MPC")

    mask_A, mask_B = get_masks(problem['A'], problem['B'])
    x_vars = np.sum(mask_A)
    u_vars = np.sum(mask_B)
    number_of_blocks = int((problem['N'] + 1) / 2 * problem['N'] + 0.001)

    block_ordering = sparse.csc_matrix(np.tril(np.ones((problem['N'], problem['N']))))
    block_ordering.data = np.arange(number_of_blocks)
    block_ordering = np.array(block_ordering.todense())

    one_norm_matrix_x, one_norm_matrix_u = find_fx_and_fu(mask_A, mask_B, np.ones_like(problem['x0']))
    one_norm_kron_mat = np.zeros((problem['N'], number_of_blocks))
    for n in range(problem['N']):
        one_norm_kron_mat[n, block_ordering[n, 1:n + 1]] = 1

    x_lim = np.tile(problem['x_max'], (1, problem['N'])).flatten()
    u_lim = np.tile(problem['u_max'], (1, problem['N'])).flatten()

    x = m.addMVar(problem['N'] * problem['nx'], name='x', lb=-np.inf)  # , ub=x_lim)
    u = m.addMVar(problem['N'] * problem['nu'], name='u', lb=-np.inf)  # , ub=u_lim)
    phi_x = m.addMVar(number_of_blocks * x_vars, name='phi_x', lb=-np.inf)  # , ub=np.inf)
    phi_u = m.addMVar(number_of_blocks * u_vars, name='phi_u', lb=-np.inf)  # , ub=np.inf)
    sigma = m.addMVar(problem['N'], name='sigma')

    inf_norm_x = m.addMVar(problem['N'], name='inf_norm_x')
    inf_norm_u = m.addMVar(problem['N'], name='inf_norm_u')
    inf_norm_phi_x = m.addMVar(number_of_blocks, name='inf_norm_phi_x')
    inf_norm_phi_u = m.addMVar(number_of_blocks, name='inf_norm_phi_u')

    # one_norm_x = m.addMVar(problem['N'], name='one_norm_x')
    # one_norm_u = m.addMVar(problem['N'], name='one_norm_u')
    one_norm_phi_x = m.addMVar(problem['N'] * problem['nx'], name='one_norm_phi_x')
    one_norm_phi_u = m.addMVar(problem['N'] * problem['nu'], name='one_norm_phi_u')

    A_f, B_f = sparse_state_matrix_replacement(problem['A'], problem['B'], mask_A, mask_B)
    Fx, Fu = find_fx_and_fu(mask_A, mask_B, problem['x0'])

    Q = sparse.kron(sparse.eye(problem['N']), problem['Q'])
    R = sparse.kron(sparse.eye(problem['N']), problem['R'])

    Fx = sparse.kron(sparse.eye(problem['N']), Fx)
    Fu = sparse.kron(sparse.eye(problem['N']), Fu)

    one_norm_matrix_x = np.array(sparse.kron(one_norm_kron_mat, one_norm_matrix_x).todense(), dtype=bool)
    one_norm_matrix_u = np.array(sparse.kron(one_norm_kron_mat, one_norm_matrix_u).todense(), dtype=bool)
    indices_array_x = np.arange(number_of_blocks * x_vars)
    indices_array_u = np.arange(number_of_blocks * u_vars)

    Ax = -sparse.eye(problem['N'] * x_vars) + sparse.kron(sparse.eye(problem['N'], k=-1), A_f)
    Bu = sparse.kron(sparse.eye(problem['N']), B_f)
    rhs = np.zeros(problem['N'] * x_vars)
    rhs[:x_vars] = -A_f @ np.eye(problem['nx'])[mask_A]

    m.addConstr(Ax @ phi_x[:problem['N'] * x_vars] + Bu @ phi_u[:problem['N'] * u_vars] == rhs)

    for n in range(1, problem['N'] - 1):
        Ax = -sparse.eye((problem['N'] - n) * x_vars) + sparse.kron(sparse.eye((problem['N'] - n), k=-1), A_f)
        Bu = sparse.kron(sparse.eye(problem['N'] - n), B_f)

        m.addConstr(Ax @ phi_x[block_ordering[n, n] * x_vars:block_ordering[n + 1, n + 1] * x_vars] +
                    Bu @ phi_u[block_ordering[n, n] * u_vars:block_ordering[n + 1, n + 1] * u_vars] == rhs[:(problem[
                                                                                                                 'N'] - n) * x_vars] *
                    sigma[n - 1])

    for n in range(problem['N']):
        m.addConstr(inf_norm_x[n] == gp.norm(x[n * problem['nx']:(n + 1) * problem['nx']], GRB.INFINITY))
        m.addConstr(inf_norm_u[n] == gp.norm(u[n * problem['nu']:(n + 1) * problem['nu']], GRB.INFINITY))
        # m.addConstr(one_norm_x[n] == gp.norm(x[n * problem['nx']:(n + 1) * problem['nx']], 1))
        # m.addConstr(one_norm_u[n] == gp.norm(u[n * problem['nu']:(n + 1) * problem['nu']], 1))

    for n in range(number_of_blocks):
        m.addConstr(inf_norm_phi_x[n] == gp.norm(phi_x[n * x_vars:(n + 1) * x_vars], GRB.INFINITY))
        m.addConstr(inf_norm_phi_u[n] == gp.norm(phi_u[n * u_vars:(n + 1) * u_vars], GRB.INFINITY))

    for n in range(problem['N'] * problem['nx']):
        m.addConstr(one_norm_phi_x[n] == gp.norm(phi_x[indices_array_x[one_norm_matrix_x[n, :]]], 1))
    for n in range(problem['N'] * problem['nu']):
        m.addConstr(one_norm_phi_u[n] == gp.norm(phi_u[indices_array_u[one_norm_matrix_u[n, :]]], 1))

    # Infinity norm constraints
    constraint_list.append(m.addConstr(
        problem['e_A'] * np.max(np.abs(problem['x0'])) + problem['e_B'] * inf_norm_u[0] + problem['sigma_w'] <= sigma[0]))

    m.addConstr(problem['e_A'] * inf_norm_x[0] +
                problem['e_B'] * (inf_norm_u[1] + gp.quicksum(inf_norm_phi_u[block_ordering[1, 1:2]])) +
                problem['sigma_w'] <= sigma[1])
    for n in range(2, problem['N']):
        m.addConstr(problem['e_A'] * (inf_norm_x[n - 1] + gp.quicksum(inf_norm_phi_x[block_ordering[n - 1, 1:n]]) + sigma[n]) +
                    problem['e_B'] * (inf_norm_u[n] + gp.quicksum(inf_norm_phi_u[block_ordering[n, 1:n + 1]])) +
                    problem['sigma_w'] <= sigma[n])

    # 1-norm constraints
    m.addConstr(x + one_norm_phi_x + sparse.kron(sparse.eye(problem['N']),
                                                                np.ones((problem['nx'], 1))) @ sigma <= x_lim)
    m.addConstr(x - one_norm_phi_x - sparse.kron(sparse.eye(problem['N']),
                                                                np.ones((problem['nx'], 1))) @ sigma >= -x_lim)
    m.addConstr(u + one_norm_phi_u <= u_lim)
    m.addConstr(u - one_norm_phi_u >= -u_lim)

    # m.addConstr(inf_norms_x[0] == norm(problem['x0'], GRB.INFINITY))
    # m.addConstr(inf_norms_x[1] == norm(u[:problem['nu']], GRB.INFINITY))
    #
    # m.addConstr(problem['e_A'] * inf_norms_x[0] + problem['e_B'] * inf_norms_x[1] + problem['sigma_w'] <= sigma[0])
    constraint_list.append(m.addConstr(Fx @ phi_x[:problem['N'] * x_vars] == x))
    constraint_list.append(m.addConstr(Fu @ phi_u[:problem['N'] * u_vars] == u))

    scaling = 1e-3
    # obj_robust = scaling * gp.quicksum(phi_x_abs) + scaling * gp.quicksum(phi_u_abs)
    # obj_robust += scaling * gp.quicksum(x_abs) + scaling * gp.quicksum(x_abs)
    # obj_robust += scaling * gp.quicksum(x_max) + scaling * gp.quicksum(u_max)
    # obj_robust += scaling * gp.quicksum(phi_x_max) + scaling * gp.quicksum(phi_u_max)
    m.setObjective(x @ Q @ x + u @ R @ u, GRB.MINIMIZE)
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
        # Solve
        m.optimize()
        runtime += m.Runtime
        print(m.Runtime)

        states[:, i + 1] = x.X[:problem['nx']]
        inputs[:, i] = u.X[:problem['nu']]

        m.remove(constraint_list)

        Fx, Fu = find_fx_and_fu(mask_A, mask_B, states[:, i + 1])
        constraint_list = []
        Fx = sparse.kron(sparse.eye(problem['N']), Fx)
        Fu = sparse.kron(sparse.eye(problem['N']), Fu)

        constraint_list.append(m.addConstr(
            problem['e_A'] * np.max(np.abs(states[:, i + 1])) + problem['e_B'] * inf_norm_u[0] + problem['sigma_w'] <=
            sigma[0]))
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
        plt.show()
    return avg_time


if __name__ == '__main__':
    time_optimisation(4, plot_results=True)