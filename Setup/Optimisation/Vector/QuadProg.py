import quadprog
import time
import numpy as np
import scipy
from matplotlib import pyplot as plt
from Optimisation.createProblem import create_sparse_problem
from Scenarios.MainScenarios import ScenarioEnum, Scenario

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

    # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
    # - quadratic objective
    P = scipy.linalg.block_diag(np.kron(np.eye(problem['N']), problem['Q'].todense()), problem['QN'].todense(),
                                np.kron(np.eye(problem['N']), problem['R'].todense()))
    # - linear objective
    q = np.hstack([np.zeros((problem['N'] + 1) * problem['nx']), np.zeros(problem['N'] * problem['nu'])])

    # - linear dynamics
    Ax = np.kron(np.eye(problem['N'] + 1), -np.eye(problem['nx'])) + np.kron(np.eye(problem['N'] + 1, k=-1), problem['A'].todense())
    Bu = np.kron(np.vstack([np.zeros((1, problem['N'])), np.eye(problem['N'])]), problem['B'].todense())
    Aeq = np.hstack([Ax, Bu])

    # - input and state constraints
    Aineq = np.hstack([np.zeros(((problem['N']) * problem['nx'] + problem['N'] * problem['nu'], problem['nx'])),
                       np.eye((problem['N']) * problem['nx'] + problem['N'] * problem['nu'])])

    # - OSQP constraints
    A = np.vstack([Aeq, Aineq, -Aineq])

    leq = np.hstack([-problem['x0'], np.zeros(problem['N'] * problem['nx'])])

    lineq = np.hstack([np.kron(np.ones(problem['N']), -problem['x_max']),
                       np.kron(np.ones(problem['N']), -problem['u_max'])])
    uineq = np.hstack([np.kron(np.ones(problem['N']), problem['x_max']),
                       np.kron(np.ones(problem['N']), problem['u_max'])])

    constraint_b = np.hstack([leq, lineq, -uineq])
    constraint_b[:problem['nx']] = -problem['x0']

    t_0 = 0
    nsim = 11
    x = np.zeros((problem['nx'], nsim + 1))
    x[:, 0] = problem['x0']
    input = np.zeros((problem['nu'], nsim))

    for i in range(nsim):
        # Solve
        res = quadprog.solve_qp(P, -q.T, A.T, constraint_b, meq=len(leq))

        # Apply first control input to the plant
        input[:, i] = res[0][-problem['N'] * problem['nu']:-(problem['N'] - 1) * problem['nu']]
        x[:, i + 1] = res[0][problem['nx']:2 * problem['nx']]

        # Update initial state
        constraint_b[:problem['nx']] = -x[:, i + 1]

        if i == 0:
            t_0 = time.time()

    t_end = time.time()
    avg_time = (t_end - t_0) / (nsim - 1)
    print(f"Average elapsed time QuadProg for {number_of_satellites} satellites: {avg_time}")

    if plot_results:
        plt.figure()
        plt.plot(np.rad2deg(x[1::6].T))
        plt.show()

    return avg_time


if __name__ == '__main__':
    time_optimisation(3, plot_results=True)
