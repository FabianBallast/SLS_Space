import gurobipy as gp
from gurobipy import GRB
import time
import numpy as np
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

    # Create a new model
    m = gp.Model("MPC")

    x_max = np.tile(problem['x_max'], (problem['N'] + 1, 1))
    x_max[0] = np.inf
    u_max = np.tile(problem['u_max'], (problem['N'], 1))

    x = m.addMVar(shape=(problem['N'] + 1, problem['nx']), lb=-x_max, ub=x_max, name='x')
    u = m.addMVar(shape=(problem['N'], problem['nu']), lb=-u_max, ub=u_max, name='u')

    initial_state_constraint = m.addConstr(x[0, :] == problem['x0'])

    for k in range(problem['N']):
        m.addConstr(x[k + 1, :] == problem['A'] @ x[k, :] + problem['B'] @ u[k, :])

    obj1 = sum(x[k, :] @ problem['Q'] @ x[k, :] for k in range(problem['N'] + 1))
    obj2 = sum(u[k, :] @ problem['R'] @ u[k, :] for k in range(problem['N']))
    m.setObjective(obj1 + obj2, GRB.MINIMIZE)
    m.setParam("OutputFlag", 0)
    m.setParam("BarConvTol", 1e-3)

    t_0 = 0
    nsim = 11
    states = np.zeros((problem['nx'], nsim + 1))
    states[:, 0] = problem['x0']
    input = np.zeros((problem['nu'], nsim))

    for i in range(nsim):
        # Solve
        m.optimize()

        all_vars = m.getVars()
        values = m.getAttr("X", all_vars)
        names = m.getAttr("VarName", all_vars)
        # for j in range(problem['nx'], 2 * problem['nx']):
        #     states[j - problem['nx'], i+1] = values[names.index(f'x[{j}]')]
        #
        # for j in range(0, problem['nu']):
        #     input[j, i] = values[names.index(f'u[{j}]')]
        states[:, i + 1] = np.array([values[i + problem['nx']] for i in range(problem['nx'])])
        input[:, i] = np.array([values[i + problem['nx'] * (problem['N'] + 1)] for i in range(problem['nu'])])

        initial_state_constraint.rhs = states[:, i+1]

        if i == 0:
            t_0 = time.time()

    t_end = time.time()

    avg_time = (t_end - t_0) / (nsim - 1)
    print(f"Average elapsed time Gurobi for {number_of_satellites} satellites: {avg_time:.3}s")

    if plot_results:
        plt.figure()
        plt.plot(np.rad2deg(states[1::6].T))
        plt.show()
    return avg_time


if __name__ == '__main__':
    time_optimisation(3, prediction_horizon=6, plot_results=True)
