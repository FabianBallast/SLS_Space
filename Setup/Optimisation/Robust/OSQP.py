import osqp
import time
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from Optimisation.createProblem import create_sparse_problem
from Scenarios.MainScenarios import ScenarioEnum, Scenario


def time_optimisation_2(number_of_satellites: int, prediction_horizon: int = None, plot_results: bool = False,
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

    mask_A, mask_B = get_masks(problem['A'], problem['B'])
    x_vars = np.sum(mask_A)
    u_vars = np.sum(mask_B)
    number_of_blocks = int((problem['N'] + 1) / 2 * problem['N'] + 0.001)

    # Constraints
    leq = np.hstack([-problem['x0'], np.zeros(problem['N'] * problem['nx'])])
    ueq = leq

    lineq = np.hstack([np.kron(np.ones(problem['N']), -problem['x_max']),
                       np.kron(np.ones(problem['N']), -problem['u_max'])])
    uineq = np.hstack([np.kron(np.ones(problem['N']),  problem['x_max']),
                       np.kron(np.ones(problem['N']), problem['u_max'])])

    lb = np.hstack([leq, lineq])
    ub = np.hstack([ueq, uineq])

    Ax = sparse.kron(sparse.eye(problem['N'] + 1), -sparse.eye(problem['nx'])) + \
         sparse.kron(sparse.eye(problem['N'] + 1, k=-1), problem['A'])
    Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, problem['N'])), sparse.eye(problem['N'])]), problem['B'])
    Aeq = sparse.hstack([Ax, Bu])

    A_f, B_f = sparse_state_matrix_replacement(problem['A'], problem['B'], mask_A, mask_B)
    # Ax = sparse.kron(sparse.eye(problem['N']), -sparse.eye(problem['nx'])) + \
    #      sparse.kron(sparse.eye(problem['N'], k=-1), problem['A'])
    # Bu = sparse.kron(sparse.eye(problem['N']), problem['B'])
    # Aeq_dynamics = sparse.hstack([Ax, Bu, sparse.csc_matrix((problem['N'] * problem['nx'],
    #                                                          total_vars - indices_dict['u'][-1] - 1))])
    sigma_matrix = get_sigma_matrix(mask_A)
    A_f_extended = sparse.vstack([A_f, np.zeros(((prediction_horizon - 1) * x_vars, A_f.shape[1]))])
    rhs_base = A_f_extended @ sigma_matrix

    A_list = [sparse.csc_matrix((problem['N'] * x_vars))]
    B_list = []

    for n in range(1, problem['N']):
        A_list.append(-sparse.eye((problem['N'] - n) * x_vars) + sparse.kron(sparse.eye((problem['N'] - n), k=-1), A_f))
        B_list.append(sparse.kron(sparse.eye(problem['N'] - n), B_f))

    A_dyn = sparse.block_diag(A_list)
    B_dyn = sparse.block_diag(B_list)
    Aeq_dynamics = sparse.hstack([Aeq, A_dyn, B_dyn])

    # - input and state constraints
    # Skip first state
    Aineq = sparse.hstack([sparse.csc_matrix(((problem['N']) * problem['nx'] + problem['N'] * problem['nu'],
                                              problem['nx'])),
                           sparse.eye((problem['N']) * problem['nx'] + problem['N'] * problem['nu'])])

    # - OSQP constraints
    A = sparse.vstack([Aeq, Aineq], format='csc')

    # Define problem
    # - quadratic objective
    P = sparse.block_diag([sparse.kron(sparse.eye(problem['N']), problem['Q']), 5 * problem['QN'],
                           sparse.kron(sparse.eye(problem['N']), problem['R'])], format='csc')
    # - linear objective
    q = np.hstack([np.zeros((problem['N'] + 1) * problem['nx']),
                   np.zeros(problem['N'] * problem['nu'])])

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace
    prob.setup(P, q, A, lb, ub, warm_start=True, verbose=False, eps_abs=1e-5, eps_rel=1e-5)
               # , check_termination=50, eps_abs=1e-4, eps_rel=1e-4, max_iter=45000)

    t_0 = 0
    nsim = 31
    x = np.zeros((problem['nx'], nsim + 1))
    x[:, 0] = problem['x0']
    input = np.zeros((problem['nu'], nsim))

    for i in range(nsim):
        # Solve
        res = prob.solve()

        # Check solver status
        # if res.info.status != 'solved':
        #     raise ValueError(f'OSQP did not solve the problem in iteration {i}!\n{res.info.status}')

        # Apply first control input to the plant
        input[:, i] = res.x[-problem['N'] * problem['nu']:-(problem['N'] - 1) * problem['nu']]
        x[:, i + 1] = problem['A'].dot(x[:, i]) + problem['B'].dot(input[:, i])

        # Update initial state
        lb[:problem['nx']] = -x[:, i + 1]
        ub[:problem['nx']] = -x[:, i + 1]
        prob.update(l=lb, u=ub)

        if i == 0:
            t_0 = time.time()

    t_end = time.time()
    average_time = (t_end - t_0) / (nsim - 1)

    print(f"Average elapsed time OSQP for {number_of_satellites} satellites: {average_time:.3}s")

    if plot_results:
        plt.figure()
        plt.plot(np.rad2deg(x[1::6].T ))
        plt.ylabel('Angle')
        plt.grid(True)

        plt.figure()
        plt.plot(x[0::6].T)
        plt.ylabel('Radius')
        plt.grid(True)

        plt.figure()
        plt.plot(input[0::3].T)
        plt.ylabel('u_r')
        plt.grid(True)

        plt.figure()
        plt.plot(input[1::3].T)
        plt.ylabel('u_t')
        plt.grid(True)
        plt.show()

    return average_time


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

    # Constraints
    leq = np.hstack([-problem['A'] @ problem['x0_abs'], np.zeros((problem['N'] - 1) * problem['nx'])])
    ueq = leq

    lineq = np.hstack([np.kron(np.ones(problem['N']), -problem['x_max'] + problem['x_ref']),
                       np.kron(np.ones(problem['N']), -problem['u_max'])])
    uineq = np.hstack([np.kron(np.ones(problem['N']),  problem['x_max'] + problem['x_ref']),
                       np.kron(np.ones(problem['N']), problem['u_max'])])

    lb = np.hstack([leq, lineq])
    ub = np.hstack([ueq, uineq])

    Ax = sparse.kron(sparse.eye(problem['N']), -sparse.eye(problem['nx'])) + \
         sparse.kron(sparse.eye(problem['N'], k=-1), problem['A'])
    Bu = sparse.kron(sparse.eye(problem['N']), problem['B'])
    Aeq = sparse.hstack([Ax, Bu])

    # - input and state constraints
    # Skip first state
    Aineq = sparse.eye((problem['N']) * problem['nx'] + problem['N'] * problem['nu'])

    # - OSQP constraints
    A = sparse.vstack([Aeq, Aineq], format='csc')

    # Define problem
    # - quadratic objective
    P = sparse.block_diag([sparse.kron(sparse.eye(problem['N'] - 1), problem['Q']),5* problem['QN'],
                           sparse.kron(sparse.eye(problem['N']), problem['R'])], format='csc')
    # - linear objective
    q = np.hstack([np.kron(np.ones(problem['N']-1), -problem['Q']@problem['x_ref']), -5*problem['Q']@problem['x_ref'],
                  # -problem['QN']@problem['x_ref'],
                   np.zeros(problem['N'] * problem['nu'])])

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace
    prob.setup(P, q, A, lb, ub, warm_start=True, verbose=False, eps_abs=1e-7, eps_rel=1e-7)

    t_0 = 0
    nsim = 31
    x = np.zeros((problem['nx'], nsim + 1))
    x[:, 0] = problem['x0_abs']
    input = np.zeros((problem['nu'], nsim))

    for i in range(nsim):
        # Solve
        res = prob.solve()

        # Check solver status
        if res.info.status != 'solved':
            raise ValueError(f'OSQP did not solve the problem in iteration {i}!\n{res.info.status}')

        # Apply first control input to the plant
        input[:, i] = res.x[-problem['N'] * problem['nu']:-(problem['N'] - 1) * problem['nu']]
        x[:, i + 1] = res.x[0 * problem['nx']:1 * problem['nx']]

        # Update initial state
        lb[:problem['nx']] = -problem['A'] @ x[:, i + 1]
        ub[:problem['nx']] = -problem['A'] @ x[:, i + 1]
        prob.update(l=lb, u=ub)

        if i == 0:
            t_0 = time.time()

    t_end = time.time()
    average_time = (t_end - t_0) / (nsim - 1)

    print(f"Average elapsed time OSQP for {number_of_satellites} satellites: {average_time:.3}s")

    if plot_results:
        plt.figure()
        plt.plot(np.rad2deg(x[1::6].T - problem['x_ref'][1::6].T))
        plt.ylabel('Angle')
        plt.grid(True)

        plt.figure()
        plt.plot(x[0::6].T)
        plt.ylabel('Radius')
        plt.grid(True)

        plt.figure()
        plt.plot(input[0::3].T)
        plt.ylabel('u_r')
        plt.grid(True)

        plt.figure()
        plt.plot(input[1::3].T)
        plt.ylabel('u_t')
        plt.grid(True)
        plt.show()

    return average_time


if __name__ == '__main__':
    # time_optimisation(3,prediction_horizon=6, plot_results=True)
    time_optimisation_2(3, prediction_horizon=6, plot_results=True)
