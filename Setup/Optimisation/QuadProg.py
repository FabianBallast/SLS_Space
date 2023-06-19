import random
import time
import scipy
import quadprog
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from Dynamics.HCWDynamics import RelCylHCW as dyn
from Scenarios.MainScenarios import ScenarioEnum

def time_optimisation(number_of_satellites: int, prediction_horizon: int):

    # General values
    scenario = ScenarioEnum.simple_scenario_translation_HCW_scaled.value
    scenario.number_of_satellites = number_of_satellites
    scenario.control.tFIR = prediction_horizon
    dynamics = dyn(scenario)
    model_single = dynamics.create_model(scenario.control.control_timestep)

    # Create large model
    number_of_satellites = scenario.number_of_satellites
    Ad = np.kron(np.eye(number_of_satellites), model_single.A)
    Bd = np.kron(np.eye(number_of_satellites), model_single.B)
    [nx, nu] = Bd.shape

    # Initial and reference states
    x0 = np.zeros(nx)
    possible_angles = np.linspace(0, 2 * np.pi, number_of_satellites + 2, endpoint=False)

    random.seed(129)
    selected_indices = np.sort(random.sample(range(number_of_satellites + 2), number_of_satellites))
    x0[1::6] = possible_angles[selected_indices]
    xr = np.zeros(nx)
    ref_rel_angles = np.linspace(0, 2 * np.pi, scenario.number_of_satellites, endpoint=False)
    ref_rel_angles -= np.mean(ref_rel_angles - possible_angles[selected_indices])
    xr[1::6] = ref_rel_angles

    # Prediction horizon
    N = scenario.control.tFIR

    # Objective function
    Q = np.kron(np.eye(number_of_satellites), dynamics.get_state_cost_matrix_sqrt()**2)
    QN = Q
    R = np.kron(np.eye(number_of_satellites), dynamics.get_input_cost_matrix_sqrt()[3:]**2)

    # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
    # - quadratic objective
    P = scipy.linalg.block_diag(np.kron(np.eye(N), Q), QN, np.kron(np.eye(N), R))
    # - linear objective
    q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr), np.zeros(N * nu)])

    # - linear dynamics
    Ax = np.kron(np.eye(N + 1), -np.eye(nx)) + np.kron(np.eye(N + 1, k=-1), Ad)
    Bu = np.kron(np.vstack([np.zeros((1, N)), np.eye(N)]), Bd)
    Aeq = np.hstack([Ax, Bu])

    # - input and state constraints
    Aineq = np.eye((N + 1) * nx + N * nu)

    # - OSQP constraints
    A = np.vstack([Aeq, Aineq, -Aineq])

    # Constraints
    umin = -np.array(dynamics.get_input_constraint() * number_of_satellites)
    umax = np.array(dynamics.get_input_constraint() * number_of_satellites)
    xmin = -np.array(dynamics.get_state_constraint() * number_of_satellites)
    xmax = np.array(dynamics.get_state_constraint() * number_of_satellites)

    leq = np.hstack([-x0 * 0, np.zeros(N * nx)])
    ueq = leq

    lineq = np.hstack([np.kron(np.ones(N + 1), xmin), np.kron(np.ones(N), umin)])
    uineq = np.hstack([np.kron(np.ones(N + 1), xmax), np.kron(np.ones(N), umax)])

    constraint_b = np.hstack([leq, lineq, -uineq])
    constraint_b[:nx] = -x0

    t_0 = 0
    runs = 1
    nsim = 10
    x = np.zeros((nx, nsim + 1))
    x[:, 0] = x0
    input = np.zeros((nu, nsim))

    for run in range(runs):
        for i in range(nsim):
            # Solve
            res = quadprog.solve_qp(P, -q.T, A.T, constraint_b, meq=len(leq))

            # Apply first control input to the plant
            ctrl = res[0][-N * nu:-(N - 1) * nu]
            x0 = Ad.dot(x0) + Bd.dot(ctrl)
            x[:, i + 1] = x0
            input[:, i] = ctrl

            # Update initial state
            constraint_b[:nx] = -x0

            if i == 0:
                t_0 = time.time()

    t_end = time.time()
    avg_time = (t_end - t_0) / runs / (nsim - 1)
    # print(f"Average elapsed time: {avg_time}")

    # plt.figure()
    # plt.plot(np.rad2deg(x[1::6].T - xr[1::6]))
    #
    # plt.figure()
    # plt.plot(input[1::3].T)
    # plt.show()

    return avg_time


if __name__ == '__main__':
    print(time_optimisation(3))
    plt.show()
