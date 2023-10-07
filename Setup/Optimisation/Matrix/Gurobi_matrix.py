import gurobipy as gp
from gurobipy import GRB
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
    m = gp.Model("MPC")

    # MPC Formulation
    x_max = np.tile(problem['x_max'], (1, problem['N'] + 1)).reshape((-1, ))
    x_max[:problem['nx']] = np.inf
    u_max = np.tile(problem['u_max'], (1, problem['N'])).reshape((-1, ))

    Phi_x = m.addMVar(shape=(problem['N'] * problem['nx'], problem['nx']), name='Phi_x')
    Phi_u = m.addMVar(shape=(problem['N'] * problem['nu'], problem['nx']), name='Phi_u')
    x = m.addMVar(shape=((problem['N'] + 1) * problem['nx']), lb=-x_max, ub=x_max, name='x')
    u = m.addMVar(shape=(problem['N'] * problem['nu']), lb=-u_max, ub=u_max, name='u')

    # Dynamics
    # m.addConstr(Phi_x[:problem['nx']] == np.eye(problem['nx']))

    m.addConstr(Phi_x[:problem['nx']] == problem['A'] @ sparse.eye(problem['nx']) + problem['B'] @ Phi_u[:problem['nu']])
    for k in range(1, problem['N']):
        m.addConstr(Phi_x[k * problem['nx']:(k + 1) * problem['nx']] == problem['A'] @ Phi_x[(k-1) * problem['nx']:k * problem['nx']] +
                                    problem['B'] @ Phi_u[k * problem['nu']:(k+1) * problem['nu']])

    # state_constraint_min = m.addConstr(Phi_x @ problem['x0'] >= -x_max[problem['nx']:])
    # state_constraint_max = m.addConstr(Phi_x @ problem['x0'] <= x_max[problem['nx']:])
    # input_constraint_min = m.addConstr(Phi_u @ problem['x0'] >= -u_max)
    # input_constraint_max = m.addConstr(Phi_u @ problem['x0'] <= u_max)

    initial_state_constraint = m.addConstr(x[:problem['nx']] == problem['x0'])
    state_constraint = m.addConstr(Phi_x @ problem['x0'] == x[problem['nx']:])
    input_constraint = m.addConstr(Phi_u @ problem['x0'] == u)

    # obj_test = (Phi_x @ problem['x0']) @ sparse.kron(sparse.eye(problem['N']), problem['Q']) @ (Phi_x @ problem['x0'])
    # obj_test += (Phi_u @ problem['x0']) @ sparse.kron(sparse.eye(problem['N']), problem['R']) @ (Phi_u @ problem['x0'])
    obj1 = x @ sparse.kron(sparse.eye(problem['N'] + 1), problem['Q']) @ x
    obj2 = u @ sparse.kron(sparse.eye(problem['N']), problem['R']) @ u

    m.setObjective(obj1 + obj2, GRB.MINIMIZE)
    # m.setObjective(obj_test_3 + obj_test_4, GRB.MINIMIZE)
    m.setParam("OutputFlag", 0)
    m.setParam("BarConvTol", 1e-3)

    t_0 = 0
    nsim = 5
    runtime = 0
    states = np.zeros((problem['nx'], nsim + 1))
    states[:, 0] = problem['x0']
    input = np.zeros((problem['nu'], nsim))

    for i in range(nsim):
        # Solve
        m.optimize()
        runtime += m.Runtime

        all_vars = m.getVars()
        values = m.getAttr("X", all_vars)

        # phi_x = np.array([values[i] for i in range(problem['nx']**2)]).reshape((problem['nx'], problem['nx']))
        # states[:, i + 1] = phi_x @ states[:, i]
        #
        # m.remove(m.getConstrs()[-state_constraint_min.size - state_constraint_max.size - input_constraint_min.size - input_constraint_max.size:])
        #
        # state_constraint_min = m.addConstr(Phi_x @ states[:, i + 1] >= -x_max[problem['nx']:])
        # state_constraint_max = m.addConstr(Phi_x @ states[:, i + 1] <= x_max[problem['nx']:])
        # input_constraint_min = m.addConstr(Phi_u @ states[:, i + 1] >= -u_max)
        # input_constraint_max = m.addConstr(Phi_u @ states[:, i + 1] <= u_max)

        # obj_test = (Phi_x @ states[:, i + 1]) @ sparse.kron(sparse.eye(problem['N']), problem['Q']) @ (
        #             Phi_x @ states[:, i + 1])
        # obj_test += (Phi_u @ states[:, i + 1]) @ sparse.kron(sparse.eye(problem['N']), problem['R']) @ (
        #             Phi_u @ states[:, i + 1])

        # m.setObjective(obj_test, GRB.MINIMIZE)

        Phi_x_length = problem['nx']**2 * problem['N']
        Phi_u_length = problem['nx'] * problem['nu'] * problem['N']
        states[:, i+1] = np.array([values[i + Phi_x_length + Phi_u_length + problem['nx']] for i in range(problem['nx'])])
        input[:, i] = np.array([values[i + Phi_x_length + Phi_u_length + problem['nx'] * (problem['N'] + 1)] for i in range(problem['nu'])])
        #
        m.remove(m.getConstrs()[-state_constraint.size - input_constraint.size:])

        initial_state_constraint.rhs = states[:, i+1]
        state_constraint = m.addConstr(Phi_x @ states[:, i+1] == x[problem['nx']:])
        input_constraint = m.addConstr(Phi_u @ states[:, i+1] == u)

        m.update()

        if i == 0:
            t_0 = time.time()
            runtime = 0

    t_end = time.time()

    if nsim > 1:
        avg_time = (t_end - t_0) / (nsim - 1)
    else:
        avg_time = t_end - t_0

    print(f"Average elapsed time Gurobi matrix for {number_of_satellites} satellites: {avg_time}")
    print(f"Runtime avg.: {runtime / (nsim - 1) }")

    if plot_results:
        plt.figure()
        plt.plot(np.rad2deg(states[1::6].T))
        plt.show()
    return avg_time


if __name__ == '__main__':
    time_optimisation(10, plot_results=True)
