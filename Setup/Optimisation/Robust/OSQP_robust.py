import osqp
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

    Q = sparse.kron(sparse.eye(problem['N']), problem['Q'])
    R = sparse.kron(sparse.eye(problem['N']), problem['R'])

    mask_A, mask_B = get_masks(problem['A'], problem['B'])
    x_vars = np.sum(mask_A)
    u_vars = np.sum(mask_B)
    Fx_full_base, Fu_full_base = find_fx_and_fu(mask_A, mask_B, np.ones_like(problem['x0_abs']))

    x_min = np.tile(-problem['x_max'] + problem['x_ref'], (1, problem['N']))
    u_min = np.tile(-problem['u_max'], (1, problem['N']))
    x_max = np.tile(problem['x_max'] + problem['x_ref'], (1, problem['N']))
    u_max = np.tile(problem['u_max'], (1, problem['N']))

    # Create a new model
    m = osqp.OSQP()
    total_size_phis = problem['N'] * (x_vars + u_vars)
    P = sparse.block_diag([Q, R, sparse.csc_matrix((total_size_phis, total_size_phis))], format='csc')
    q = np.hstack([np.kron(np.ones(problem['N']), -problem['Q'] @ problem['x_ref']),
                   np.zeros(problem['N'] * problem['nu'] + total_size_phis)])


    Ax = sparse.kron(sparse.eye(problem['N']), -sparse.eye(problem['nx'])) + \
         sparse.kron(sparse.eye(problem['N'], k=-1), problem['A'])
    Bu = sparse.kron(sparse.eye(problem['N']), problem['B'])
    Aeq_dynamics = sparse.hstack([Ax, Bu, sparse.csc_matrix((problem['N'] * problem['nx'], total_size_phis))])
    Aeq_constraint = sparse.hstack([-sparse.eye(problem['N'] * (problem['nx'] + problem['nu'])),
                                    sparse.block_diag([sparse.kron(sparse.eye(problem['N']), Fx_full_base),
                                                       sparse.kron(sparse.eye(problem['N']), Fu_full_base)], format='csc')])

    # - input and state constraints
    # Skip first state
    Aineq = sparse.hstack([sparse.eye(problem['N'] * (problem['nx'] + problem['nu'])),
                           sparse.csc_matrix((problem['N'] * (problem['nx'] + problem['nu']), total_size_phis))])

    # - OSQP constraints

    leq = np.zeros(Aeq_dynamics.shape[0] + Aeq_constraint.shape[0])
    leq[:problem['nx']] = -problem['A'] @ problem['x0_abs']
    ueq = leq

    lineq = np.hstack([x_min, u_min]).flatten()
    uineq = np.hstack([x_max, u_max]).flatten()

    l = np.hstack([leq, lineq])
    u = np.hstack([ueq, uineq])

    # print(l.shape, A.shape, leq.shape, lineq.shape, Aeq.shape, Aineq.shape, P.shape)
    eps_abs = 1e-3
    eps_rel = 1e-3

    # Full A
    Fx_full = sparse.kron(sparse.eye(problem['N']), Fx_full_base)
    Fu_full = sparse.kron(sparse.eye(problem['N']), Fu_full_base)
    Aeq_constraint_full = sparse.hstack([-sparse.eye(problem['N'] * (problem['nx'] + problem['nu'])),
                                         sparse.block_diag([Fx_full, Fu_full], format='csc')])
    Aeq_constraint_change = sparse.hstack([0 * sparse.eye(problem['N'] * (problem['nx'] + problem['nu'])),
                                         sparse.block_diag([Fx_full, Fu_full], format='csc')])

    A_full = sparse.vstack([Aeq_dynamics, Aeq_constraint_full, Aineq], format='csc')
    A_change = sparse.vstack([sparse.csc_matrix(Aeq_dynamics.shape), Aeq_constraint_change,
                              sparse.csc_matrix(Aineq.shape)], format='csc')
    A_full_mask = A_full != 0
    A_change_mask = A_change != 0

    A_full.data = np.arange(np.sum(A_full_mask))
    A_indices_change = np.array(A_full.T[A_change_mask.T]).flatten()

    Fx_values, Fu_values = update_fx_and_fu(Fx_full_base, Fu_full_base, problem['x0_abs'])
    A_values_change = np.hstack([np.kron(np.ones(problem['N']), Fx_values), np.kron(np.ones(problem['N']), Fu_values)])

    # Setup workspace
    A_temp = sparse.vstack([Aeq_dynamics, Aeq_constraint_full, Aineq], format='csc')
    m.setup(P, q, A_temp, l, u, warm_start=True, verbose=False, eps_abs=eps_abs, eps_rel=eps_rel, max_iter=500000)
    m.update(Ax=A_values_change, Ax_idx=A_indices_change)

    # Simulate in closed loop
    t_0 = 0
    runtime = 0
    solve_time = 0
    nsim = 11
    states = np.zeros((problem['nx'], nsim + 1))
    states[:, 0] = problem['x0_abs']
    inputs = np.zeros((problem['nu'], nsim))

    for i in range(nsim):
        # Solve
        res = m.solve()

        runtime += res.info.run_time
        solve_time += res.info.solve_time

        # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        states[:, i + 1] = res.x[:problem['nx']]
        inputs[:, i] = res.x[problem['N'] * problem['nx']: problem['N'] * problem['nx'] + problem['nu']]

        Fx_values, Fu_values = update_fx_and_fu(Fx_full_base, Fu_full_base, problem['x0_abs'])
        A_values_change = np.hstack(
            [np.kron(np.ones(problem['N']), Fx_values), np.kron(np.ones(problem['N']), Fu_values)])

        l[:problem['nx']] = -problem['A'] @ states[:, i + 1]
        u[:problem['nx']] = l[:problem['nx']]
        m.update(l=l, u=u, Ax=A_values_change, Ax_idx=A_indices_change)

        if i == 0:
            t_0 = time.time()
            runtime = 0
            solve_time = 0

    t_end = time.time()

    avg_time = (t_end - t_0) / (nsim - 1)
    print(f"Average elapsed time OSQP reformed for {number_of_satellites} satellites: {avg_time}")
    print(f"Run time avg: {runtime / (nsim - 1)}, solve time avg: {solve_time / (nsim - 1)}")
    if plot_results:
        plt.figure()
        plt.plot(np.rad2deg(states[1::6].T - problem['x_ref'][1::6].T))
        plt.show()

    return avg_time


if __name__ == '__main__':
    time_optimisation(100, plot_results=True)
