import osqp
import time
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from Optimisation.createProblem import create_sparse_problem
from Scenarios.MainScenarios import ScenarioEnum, Scenario
from Optimisation.sparseHelperFunctions import *


def time_optimisation(number_of_satellites: int, prediction_horizon: int = None, plot_results: bool = False,
                      scenario: Scenario = ScenarioEnum.robustness_comparison_advanced_robust_noise.value,
                      small_problem: bool = False) -> float:
    """
    Measure the time it takes to optimise the controller.

    :param number_of_satellites: Number of satellites to optimise.
    :param prediction_horizon: Prediction horizon of the controller.
    :param plot_results: Whether to plot the results.
    :param scenario: Scenario to run. Defaults to HCW scenario.
    :return: Float with average time (after first iteration has passed).
    """
    problem = create_sparse_problem(number_of_satellites, prediction_horizon, scenario)
    if small_problem:
        problem['x0'] /= 3

    Q = sparse.kron(sparse.block_diag([sparse.eye(problem['N'] - 1), 5 * sparse.eye(1)]), problem['Q'])
    R = sparse.kron(sparse.eye(problem['N']), problem['R'])

    mask_A, mask_B = get_masks(problem['A'], problem['B'])
    x_vars = np.sum(mask_A)
    u_vars = np.sum(mask_B)
    number_of_blocks = int((problem['N'] + 1) / 2 * problem['N'] + 0.001)
    Fx_full_base, Fu_full_base = find_fx_and_fu(mask_A, mask_B, np.ones_like(problem['x0_abs']))

    block_ordering = sparse.csc_matrix(np.tril(np.ones((problem['N'], problem['N']))))
    block_ordering.data = np.arange(number_of_blocks)
    block_ordering = np.array(block_ordering.todense())

    one_norm_matrix_x_single, one_norm_matrix_u_single = find_fx_and_fu(mask_A, mask_B, np.ones_like(problem['x0']))
    one_norm_kron_mat = np.zeros((problem['N'], number_of_blocks))
    for n in range(problem['N']):
        one_norm_kron_mat[n, block_ordering[n, 1:n + 1]] = 1

    one_norm_matrix_x = sparse.kron(one_norm_kron_mat, one_norm_matrix_x_single)
    one_norm_matrix_u = sparse.kron(one_norm_kron_mat, one_norm_matrix_u_single)

    Fx_full = sparse.kron(sparse.eye(problem['N']), Fx_full_base)
    Fu_full = sparse.kron(sparse.eye(problem['N']), Fu_full_base)

    x_min = np.tile(-problem['x_max'] + 0 * problem['x_ref'], (1, problem['N']))
    u_min = np.tile(-problem['u_max'], (1, problem['N']))
    x_max = np.tile(problem['x_max'] + 0 * problem['x_ref'], (1, problem['N']))
    u_max = np.tile(problem['u_max'], (1, problem['N']))

    indices_dict = find_indices(problem, number_of_blocks, x_vars, u_vars)
    total_vars = indices_dict['phi_u_max'][-1] + 1

    # Create a new model
    m = osqp.OSQP()
    length_x = len(indices_dict['x'])
    length_u = len(indices_dict['u'])
    length_x_and_u = length_x + length_u

    zero_length = total_vars - length_x_and_u
    P = sparse.block_diag([Q, R, sparse.csc_matrix((zero_length, zero_length))], format='csc')
    q = np.zeros(total_vars)
    # q[indices_dict['x']] = np.kron(np.concatenate((np.ones(problem['N']-1), np.array([5]))), -problem['Q'] @ problem['x_ref'])

    scaling = 1e-4

    q[indices_dict['phi_x_abs']] = scaling
    q[indices_dict['phi_u_abs']] = scaling
    q[indices_dict['x_abs']] = scaling
    q[indices_dict['u_abs']] = scaling
    q[indices_dict['phi_x_max']] = scaling
    q[indices_dict['phi_u_max']] = scaling
    q[indices_dict['x_max']] = scaling
    q[indices_dict['u_max']] = scaling

    A_f, B_f = sparse_state_matrix_replacement(problem['A'], problem['B'], mask_A, mask_B)
    # Ax = sparse.kron(sparse.eye(problem['N']), -sparse.eye(problem['nx'])) + \
    #      sparse.kron(sparse.eye(problem['N'], k=-1), problem['A'])
    # Bu = sparse.kron(sparse.eye(problem['N']), problem['B'])
    # Aeq_dynamics = sparse.hstack([Ax, Bu, sparse.csc_matrix((problem['N'] * problem['nx'],
    #                                                          total_vars - indices_dict['u'][-1] - 1))])
    Ax = sparse.kron(sparse.eye(problem['N']), -sparse.eye(problem['nx'])) + \
         sparse.kron(sparse.eye(problem['N'], k=-1), sparse.csc_matrix(problem['A']))
    Bu = sparse.kron(sparse.eye(problem['N']), sparse.csc_matrix(problem['B']), format='csr')
    Aeq_states = sparse.vstack([sparse.hstack([Ax, Bu]),
                                sparse.csc_matrix((sum(range(1, prediction_horizon)) * x_vars, length_x_and_u))])

    sigma_matrix = get_sigma_matrix(mask_A)
    A_f_extended = sparse.vstack([A_f, np.zeros(((prediction_horizon - 1) * x_vars, A_f.shape[1]))])
    rhs_base = A_f_extended @ sigma_matrix

    A_list = [sparse.csc_matrix((problem['N'] * problem['nx'], problem['N'] * x_vars))]
    B_list = [sparse.csc_matrix((problem['N'] * problem['nx'], problem['N'] * u_vars))]
    sigma_list = []

    for n in range(1, problem['N']):
        A_list.append(-sparse.eye((problem['N'] - n) * x_vars) + sparse.kron(sparse.eye((problem['N'] - n), k=-1), A_f))
        B_list.append(sparse.kron(sparse.eye(problem['N'] - n), B_f))

        if n > 0:
            sigma_list.append(rhs_base[:(prediction_horizon - n) * x_vars])

    sigma_list = [sparse.csc_matrix(sigma) for sigma in sigma_list]
    A_dyn = sparse.block_diag(A_list)
    B_dyn = sparse.block_diag(B_list)
    sigma = sparse.vstack([sparse.csc_matrix((problem['N'] * problem['nx'], (problem['N'] - 1) * problem['nx'])), sparse.block_diag(sigma_list)])
    Aeq_dynamics = sparse.hstack([Aeq_states,
                                  A_dyn, B_dyn, sigma,
                                  sparse.csc_matrix((A_dyn.shape[0],
                                                     total_vars - indices_dict['sigma'][-1] + problem['nx'] - 1))])

    rhs_dynamics = np.zeros(Aeq_dynamics.shape[0])
    rhs_dynamics[:problem['nx']] = -problem['A'] @ problem['x0']

    # Phi_x x_0 == x
    Aeq_constraint_full = sparse.hstack([-sparse.eye(length_x_and_u),
                                         sparse.vstack([Fx_full,
                                                        sparse.csc_matrix((length_u, problem['N'] * x_vars))]),
                                         sparse.csc_matrix(
                                             (length_x_and_u, (number_of_blocks - problem['N']) * x_vars)),
                                         sparse.vstack([sparse.csc_matrix((length_x, problem['N'] * u_vars)),
                                                        Fu_full]),
                                         sparse.csc_matrix((length_x_and_u,
                                                            total_vars - indices_dict['phi_u'][-1] - 1 + (
                                                                    number_of_blocks - problem['N']) * u_vars))])

    # Slack equations
    length_phi_x = len(indices_dict['phi_x'])
    length_phi_u = len(indices_dict['phi_u'])
    length_sigma = len(indices_dict['sigma'])
    phi_x_con_1 = sparse.hstack([sparse.csc_matrix((length_phi_x, length_x_and_u)),
                                 -sparse.eye(length_phi_x),
                                 sparse.csc_matrix((length_phi_x, length_phi_u + length_sigma)),
                                 sparse.eye(length_phi_x), -sparse.eye(length_phi_x),
                                 sparse.csc_matrix((length_phi_x,
                                                    total_vars - indices_dict['phi_x_neg'][-1] - 1))])

    phi_x_con_2 = sparse.hstack(
        [sparse.csc_matrix((length_phi_x, length_x_and_u + length_phi_x + length_phi_u + length_sigma)),
         sparse.eye(length_phi_x), sparse.eye(length_phi_x), -sparse.eye(length_phi_x),
         sparse.csc_matrix((length_phi_x,
                            total_vars - indices_dict['phi_x_abs'][-1] - 1))])

    phi_u_con_1 = sparse.hstack([sparse.csc_matrix((length_phi_u, length_x_and_u + length_phi_x)),
                                 -sparse.eye(length_phi_u),
                                 sparse.csc_matrix((length_phi_u, length_sigma + 3 * length_phi_x)),
                                 sparse.eye(length_phi_u), -sparse.eye(length_phi_u),
                                 sparse.csc_matrix((length_phi_u,
                                                    total_vars - indices_dict['phi_u_neg'][-1] - 1))])

    phi_u_con_2 = sparse.hstack(
        [sparse.csc_matrix((length_phi_u, length_x_and_u + length_phi_x +
                            length_phi_u + length_sigma + 3 * length_phi_x)),
         sparse.eye(length_phi_u), sparse.eye(length_phi_u), -sparse.eye(length_phi_u),
         sparse.csc_matrix((length_phi_u,
                            total_vars - indices_dict['phi_u_abs'][-1] - 1))])

    x_con_1 = sparse.hstack([-sparse.eye(length_x),
                             sparse.csc_matrix(
                                 (length_x, length_u + 4 * length_phi_x + 4 * length_phi_u + length_sigma)),
                             sparse.eye(length_x), -sparse.eye(length_x),
                             sparse.csc_matrix((length_x, total_vars - indices_dict['x_neg'][-1] - 1))])

    x_con_2 = sparse.hstack(
        [sparse.csc_matrix((length_x, length_x_and_u + 4 * length_phi_x + 4 * length_phi_u + length_sigma)),
         sparse.eye(length_x), sparse.eye(length_x), -sparse.eye(length_x),
         sparse.csc_matrix((length_x, total_vars - indices_dict['x_abs'][-1] - 1))])

    u_con_1 = sparse.hstack([sparse.csc_matrix((length_u, length_x)), -sparse.eye(length_u),
                             sparse.csc_matrix(
                                 (length_u, 4 * length_phi_x + 4 * length_phi_u + length_sigma + 3 * length_x)),
                             sparse.eye(length_u), -sparse.eye(length_u),
                             sparse.csc_matrix((length_u, total_vars - indices_dict['u_neg'][-1] - 1))])

    u_con_2 = sparse.hstack(
        [sparse.csc_matrix(
            (length_u, length_x_and_u + 4 * length_phi_x + 4 * length_phi_u + length_sigma + 3 * length_x)),
            sparse.eye(length_u), sparse.eye(length_u), -sparse.eye(length_u),
            sparse.csc_matrix((length_u, total_vars - indices_dict['u_abs'][-1] - 1))])

    Aeq_slack = sparse.vstack([phi_x_con_1, phi_x_con_2, phi_u_con_1, phi_u_con_2, x_con_1, x_con_2, u_con_1, u_con_2])

    # Inequality constraints
    x_max_con = sparse.hstack([sparse.csc_matrix((length_x, indices_dict['x_abs'][0])),
                               -sparse.eye(length_x), sparse.csc_matrix((length_x, 3 * length_u)),
                               sparse.kron(sparse.eye(problem['N']), np.ones((problem['nx'], 1))),
                               sparse.csc_matrix((length_x, total_vars - indices_dict['x_max'][-1] - 1))])

    u_max_con = sparse.hstack([sparse.csc_matrix((length_u, indices_dict['u_abs'][0])),
                               -sparse.eye(length_u), sparse.csc_matrix((length_u, problem['N'] + number_of_blocks)),
                               sparse.kron(sparse.eye(problem['N']), np.ones((problem['nu'], 1))),
                               sparse.csc_matrix((length_u, total_vars - indices_dict['u_max'][-1] - 1))])

    phi_x_max_con = sparse.hstack([sparse.csc_matrix((number_of_blocks * problem['nx'], indices_dict['phi_x_abs'][0])),
                                   -sparse.kron(np.eye(number_of_blocks), one_norm_matrix_x_single),
                                   sparse.csc_matrix((number_of_blocks * problem['nx'],   3 * length_phi_u + 3 * length_x + 3 * length_u + problem['N'])),
                                   sparse.kron(sparse.eye(number_of_blocks), np.ones((problem['nx'], 1))),
                                   sparse.csc_matrix((number_of_blocks * problem['nx'], total_vars - indices_dict['phi_x_max'][-1] - 1))])

    phi_u_max_con = sparse.hstack([sparse.csc_matrix((number_of_blocks * problem['nu'], indices_dict['phi_u_abs'][0])),
                                   -sparse.kron(np.eye(number_of_blocks), one_norm_matrix_u_single),
                                   sparse.csc_matrix((number_of_blocks * problem['nu'], 3 * length_x + 3 * length_u + 2 * problem['N'] + number_of_blocks)),
                                   sparse.kron(sparse.eye(number_of_blocks), np.ones((problem['nu'], 1)))])

    lb_max_constraint = np.zeros(length_x_and_u + number_of_blocks * (problem['nx'] + problem['nu']))
    ub_max_constraint = np.inf * np.ones_like(lb_max_constraint)

    epsilon_matrix_A = np.array([problem['e_A0'], problem['e_A1'], problem['e_A2'],
                                 problem['e_A3'], problem['e_A4'], problem['e_A5']])

    epsilon_matrix_B = np.array([problem['e_B0'], problem['e_B1'], problem['e_B2'],
                                 problem['e_B3'], problem['e_B4'], problem['e_B5']])

    sigma_w = np.array([problem['sigma_w0'], problem['sigma_w1'], problem['sigma_w2'],
                        problem['sigma_w3'], problem['sigma_w4'], problem['sigma_w5']])

    epsilon_matrix_A = np.kron(np.ones(number_of_satellites), epsilon_matrix_A)
    epsilon_matrix_B = np.kron(np.ones(number_of_satellites), epsilon_matrix_B)
    sigma_w = np.kron(np.ones(number_of_satellites), sigma_w)

    inf_norm_con = sparse.hstack([sparse.csc_matrix((length_sigma, length_x_and_u + length_phi_x + length_phi_u)),
                                  sparse.eye(length_sigma),
                                  sparse.csc_matrix(
                                      (length_sigma, 3 * length_phi_x + 3 * length_phi_u + 3 * length_x_and_u)),
                                  -sparse.kron(sparse.eye(prediction_horizon, k=-1), epsilon_matrix_A).T,
                                  -sparse.vstack([sparse.csc_matrix((2 * problem['nx'], number_of_blocks)),
                                                                   sparse.kron(one_norm_kron_mat[:-2], epsilon_matrix_A.reshape((-1, 1)))]),
                                  -sparse.kron(sparse.eye(prediction_horizon), epsilon_matrix_B).T,
                                  -sparse.vstack([sparse.csc_matrix((problem['nx'], number_of_blocks)),
                                                                   sparse.kron(one_norm_kron_mat[:-1], epsilon_matrix_B.reshape((-1, 1)))]),
                                  ])

    lb_inf_norm = np.kron(np.ones(prediction_horizon), sigma_w)
    lb_inf_norm[:problem['nx']] += epsilon_matrix_A * np.max(np.abs(problem['x0']))
    ub_inf_norm = np.ones_like(lb_inf_norm) * np.inf

    one_norm_con = sparse.hstack([sparse.vstack([sparse.hstack([sparse.eye(length_x), sparse.csc_matrix((length_x, length_u))]),
                                                 sparse.hstack([sparse.eye(length_x), sparse.csc_matrix((length_x, length_u))]),
                                                 sparse.hstack([sparse.csc_matrix((length_u, length_x)), sparse.eye(length_u)]),
                                                 sparse.hstack([sparse.csc_matrix((length_u, length_x)), sparse.eye(length_u)])]),
                                  sparse.csc_matrix((2 * length_x_and_u, length_phi_x + length_phi_u)),
                                  sparse.vstack([sparse.eye(problem['N'] * problem['nx']),
                                                 -sparse.eye(problem['N'] * problem['nx']),
                                                 sparse.csc_matrix((2 * length_u, length_sigma))]),
                                  sparse.csc_matrix((2 * length_x_and_u, 2 * length_phi_x)),
                                  sparse.vstack([one_norm_matrix_x, -one_norm_matrix_x,
                                                 sparse.csc_matrix((2 * length_u, length_phi_x))]),
                                  sparse.csc_matrix((2 * length_x_and_u, 2 * length_phi_u)),
                                  sparse.vstack([sparse.csc_matrix((2 * length_x, length_phi_u)),
                                                 one_norm_matrix_u, -one_norm_matrix_u]),
                                  sparse.csc_matrix(
                                      (2 * length_x_and_u, total_vars - indices_dict['phi_u_abs'][-1] - 1))])
    # one_norm_con = sparse.hstack(
    #     [sparse.vstack([sparse.hstack([sparse.eye(length_x), sparse.csc_matrix((length_x, length_u))]),
    #                     sparse.hstack([sparse.eye(length_x), sparse.csc_matrix((length_x, length_u))]),
    #                     sparse.hstack([sparse.csc_matrix((length_u, length_x)), sparse.eye(length_u)]),
    #                     sparse.hstack([sparse.csc_matrix((length_u, length_x)), sparse.eye(length_u)])]),
    #      sparse.csc_matrix((2 * length_x_and_u, total_vars - length_x_and_u))])

    lb_one_norm = np.ones(2 * length_x_and_u) * -np.inf
    ub_one_norm = np.ones_like(lb_one_norm) * np.inf

    ub_one_norm[:length_x] = x_max
    lb_one_norm[length_x:2 * length_x] = x_min
    ub_one_norm[2 * length_x: 2 * length_x + length_u] = u_max
    lb_one_norm[2 * length_x + length_u:] = u_min

    phi_x_pos_neg_sign = sparse.hstack([sparse.csc_matrix((2 * length_phi_x, length_x_and_u + length_phi_x + length_phi_u + length_sigma)),
                                        sparse.eye(2 * length_phi_x),
                                        sparse.csc_matrix((2 * length_phi_x, total_vars - indices_dict['phi_x_neg'][-1] - 1))])
    phi_u_pos_neg_sign = sparse.hstack(
        [sparse.csc_matrix((2 * length_phi_u, length_x_and_u + 4 * length_phi_x + length_phi_u + length_sigma)),
         sparse.eye(2 * length_phi_u),
         sparse.csc_matrix((2 * length_phi_u, total_vars - indices_dict['phi_u_neg'][-1] - 1))])

    x_pos_neg_sign = sparse.hstack([sparse.csc_matrix((2 * length_x, length_x_and_u + 4 * length_phi_x + 4 * length_phi_u + length_sigma)),
                                    sparse.eye(2 * length_x),
                                    sparse.csc_matrix((2 * length_x, total_vars - indices_dict['x_neg'][-1] - 1))])
    u_pos_neg_sign = sparse.hstack(
        [sparse.csc_matrix((2 * length_u, length_x_and_u + 4 * length_phi_x + 4 * length_phi_u + length_sigma + 3 * length_x)),
         sparse.eye(2 * length_u),
         sparse.csc_matrix((2 * length_u, total_vars - indices_dict['u_neg'][-1] - 1))])

    lb_signs = np.zeros(2 * (length_phi_x + length_phi_u + length_x_and_u))
    ub_signs = np.ones_like(lb_signs) * np.inf

    # - input and state constraints
    # Skip first state
    Aineq = sparse.vstack([x_max_con, u_max_con, phi_x_max_con, phi_u_max_con, inf_norm_con, one_norm_con,
                           phi_x_pos_neg_sign, phi_u_pos_neg_sign, x_pos_neg_sign, u_pos_neg_sign])
    lineq = np.hstack([lb_max_constraint, lb_inf_norm, lb_one_norm, lb_signs])
    uineq = np.hstack([ub_max_constraint, ub_inf_norm, ub_one_norm, ub_signs])

    # - OSQP constraints

    leq = np.zeros(Aeq_dynamics.shape[0] + Aeq_constraint_full.shape[0] + Aeq_slack.shape[0])
    leq[:Aeq_dynamics.shape[0]] = rhs_dynamics
    ueq = leq

    l = np.hstack([leq, lineq])
    u = np.hstack([ueq, uineq])

    # print(l.shape, A.shape, leq.shape, lineq.shape, Aeq.shape, Aineq.shape, P.shape)
    eps_abs = 1e-6
    eps_rel = 1e-6


    # Full A
    # Fx_full = sparse.kron(sparse.eye(problem['N']), Fx_full_base)
    # Fu_full = sparse.kron(sparse.eye(problem['N']), Fu_full_base)

    Aeq_constraint_change = sparse.hstack([-sparse.csc_matrix((length_x_and_u, length_x_and_u)),
                                           sparse.vstack([Fx_full,
                                                          sparse.csc_matrix((length_u, problem['N'] * x_vars))]),
                                           sparse.csc_matrix(
                                               (length_x_and_u, (number_of_blocks - problem['N']) * x_vars)),
                                           sparse.vstack([sparse.csc_matrix((length_x, problem['N'] * u_vars)),
                                                          Fu_full]),
                                           sparse.csc_matrix((length_x_and_u,
                                                              total_vars - indices_dict['phi_u'][-1] - 1 + (
                                                                      number_of_blocks - problem['N']) * u_vars))])

    A_full = sparse.vstack([Aeq_dynamics, Aeq_constraint_full, Aeq_slack, Aineq], format='csc')
    A_change = sparse.vstack([sparse.csc_matrix(Aeq_dynamics.shape), Aeq_constraint_change,
                              sparse.csc_matrix(Aeq_slack.shape), sparse.csc_matrix(Aineq.shape)], format='csc')
    A_full_mask = A_full != 0
    A_change_mask = A_change != 0

    A_full.data = np.arange(np.sum(A_full_mask))
    A_indices_change = np.array(A_full.T[A_change_mask.T]).flatten()

    Fx_values, Fu_values = update_fx_and_fu(Fx_full_base, Fu_full_base, problem['x0'])
    A_values_change = np.hstack([np.kron(np.ones(problem['N']), Fx_values), np.kron(np.ones(problem['N']), Fu_values)])

    # Setup workspace
    A_temp = sparse.vstack([Aeq_dynamics, Aeq_constraint_full, Aeq_slack, Aineq], format='csc')
    q= None
    m.setup(P, q, A_temp, l, u, warm_start=True, verbose=False, eps_abs=eps_abs, eps_rel=eps_rel, max_iter=500000)
    m.update(Ax=A_values_change, Ax_idx=A_indices_change)

    # Simulate in closed loop
    t_0 = 0
    runtime = 0
    solve_time = 0
    nsim = 3
    states = np.zeros((problem['nx'], nsim + 1))
    states[:, 0] = problem['x0']
    inputs = np.zeros((problem['nu'], nsim))

    for i in range(nsim):
        # Solve
        res = m.solve()

        runtime += res.info.run_time
        solve_time += res.info.solve_time

        # print(2 * res.info.obj_val + problem['x_ref'].T @ problem['Q'] @ problem['x_ref'] * (prediction_horizon + 4))
        print(i)

        # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        # print(res.x[indices_dict['x']][:10])
        # print(res.x[indices_dict['x_pos']][:10])
        # print(res.x[indices_dict['x_neg']][:10])
        # print(res.x[indices_dict['x_abs']][:10])

        states[:, i + 1] = res.x[:problem['nx']]
        inputs[:, i] = res.x[problem['N'] * problem['nx']: problem['N'] * problem['nx'] + problem['nu']]

        Fx_values, Fu_values = update_fx_and_fu(Fx_full_base, Fu_full_base, states[:, i + 1])
        A_values_change = np.hstack(
            [np.kron(np.ones(problem['N']), Fx_values), np.kron(np.ones(problem['N']), Fu_values)])

        # l[:problem['nx']] = -problem['A'] @ states[:, i + 1]
        # u[:problem['nx']] = l[:problem['nx']]
        lb_inf_norm = np.kron(np.ones(prediction_horizon), sigma_w)
        lb_inf_norm[:problem['nx']] += epsilon_matrix_A * np.max(np.abs(states[:, i + 1] - 0 * problem['x_ref']))
        lineq = np.hstack([lb_max_constraint, lb_inf_norm, lb_one_norm, lb_signs])

        rhs_dynamics[:problem['nx']] = -problem['A'] @ states[:, i + 1]
        leq[:Aeq_dynamics.shape[0]] = rhs_dynamics

        ueq = leq

        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])
        m.update(l=l, u=u, Ax=A_values_change, Ax_idx=A_indices_change)

        if i == 0:
            t_0 = time.time()
            runtime = 0
            solve_time = 0

    t_end = time.time()

    avg_time = (t_end - t_0) / (nsim - 1)
    print(f"Average elapsed time OSQP robust for {number_of_satellites} satellites: {avg_time}")
    print(f"Run time avg: {runtime / (nsim - 1)}, solve time avg: {solve_time / (nsim - 1)}")
    if plot_results:
        plt.figure()
        plt.plot(np.rad2deg(states[1::6].T - 0*problem['x_ref'][1::6].T))
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
    time_optimisation(3, prediction_horizon=6, plot_results=True, small_problem=True)
