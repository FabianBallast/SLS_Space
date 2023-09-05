from cvxpy import *
import time
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from Optimisation.createProblem import create_sparse_problem
from Scenarios.MainScenarios import ScenarioEnum, Scenario


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

    # Create a new model
    Phi_x = Expression(((problem['N'] + 1) * problem['nx'], problem['nx']))
    Phi_u = Variable((problem['N'] * problem['nu'], problem['nx']))
    x_init = Parameter(problem['nx'])
    objective = 0
    constraints = [Phi_x[:problem['nx']] == sparse.eye(problem['nx'])]

    for k in range(problem['N']):
        objective += quad_form(Phi_x[k * problem['nx']:(k+1) * problem['nx']] @ x_init, problem['Q']) + \
                     quad_form(Phi_u[k * problem['nu']:(k+1) * problem['nu']] @ x_init, problem['R'])
        constraints += [Phi_x[(k+1) * problem['nx']:(k+2) * problem['nx']] == problem['A'] @ Phi_x[k * problem['nx']:(k+1) * problem['nx']] + problem['B'] @ Phi_u[k * problem['nu']:(k+1) * problem['nu']]]
        if k > 0:
            constraints += [-problem['x_max'] <= Phi_x[k * problem['nx']:(k+1) * problem['nx']] @ x_init, Phi_x[k * problem['nx']:(k+1) * problem['nx']] @ x_init <= problem['x_max']]
            constraints += [-problem['u_max'] <= Phi_u[k * problem['nu']:(k+1) * problem['nu']] @ x_init, Phi_u[k * problem['nu']:(k+1) * problem['nu']] @ x_init <= problem['u_max']]
    objective += quad_form(Phi_x[problem['N'] * problem['nx']:] @ x_init, problem['QN'])
    prob = Problem(Minimize(objective), constraints)

    t_0 = 0
    nsim = 11
    states = np.zeros((problem['nx'], nsim + 1))
    states[:, 0] = problem['x0']

    # Simulate in closed loop
    for i in range(nsim):
        x_init.value = states[:, i]
        prob.solve(solver=ECOS, warm_start=True) #, OptimalityTol=1e-3, OutputFlag=1)

        states[:, i + 1] = Phi_x[problem['nx']:2 * problem['nx']].value.dot(states[:, i])

        if i == 0:
            t_0 = time.time()

    t_end = time.time()
    average_time = (t_end - t_0) / (nsim - 1)

    print(f"Average elapsed time CVXPY for {number_of_satellites} satellites: {average_time:.3}s")
    if plot_results:
        plt.figure()
        plt.plot(np.rad2deg(states[1::6].T))
        plt.show()
    return average_time


if __name__ == '__main__':
    time_optimisation(4, plot_results=True)
