import osqp
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
from Dynamics.HCWDynamics import RelCylHCW as dyn
from Scenarios.MainScenarios import ScenarioEnum
import random
import time


def time_optimisation(number_of_satellites: int, prediction_horizon: int):
    # Create a new model
    m = osqp.OSQP()

    # Create single model
    scenario = ScenarioEnum.simple_scenario_translation_HCW_scaled.value
    scenario.number_of_satellites = number_of_satellites
    scenario.control.tFIR = prediction_horizon
    dynamics = dyn(scenario)
    model_single = dynamics.create_model(scenario.control.control_timestep)

    # horizon
    N = scenario.control.tFIR

    # Create large model
    number_of_satellites = scenario.number_of_satellites
    Ad = sparse.kron(sparse.eye(number_of_satellites), sparse.csr_matrix(model_single.A))
    Bd = sparse.kron(sparse.eye(number_of_satellites), sparse.csr_matrix(model_single.B))
    [nx, nu] = Bd.shape

    # Constraints
    umin = -np.array(dynamics.get_input_constraint() * number_of_satellites)
    umax = np.array(dynamics.get_input_constraint() * number_of_satellites)
    xmin = -np.array(dynamics.get_state_constraint() * number_of_satellites)
    xmax = np.array(dynamics.get_state_constraint() * number_of_satellites)

    # Objective function
    Q = sparse.kron(sparse.eye(number_of_satellites), sparse.csr_matrix(dynamics.get_state_cost_matrix_sqrt()).power(2))
    R = sparse.kron(sparse.eye(number_of_satellites),
                    sparse.csr_matrix(dynamics.get_input_cost_matrix_sqrt()[:3]).power(2))

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

    # MPC Formulation
    # xmin = np.tile(xmin - xr, (N + 1, 1))
    # xmax = np.tile(xmax - xr, (N + 1, 1))
    # umin = np.tile(umin, (N, 1))
    # umax = np.tile(umax, (N, 1))

    Q = sparse.block_diag([sparse.kron(sparse.eye(N + 1), Q)])
    R = sparse.block_diag([sparse.kron(sparse.eye(N), R)])

    Fx = sparse.kron(sparse.eye(nx * (N + 1)), x0 - xr)
    Fu = sparse.kron(sparse.eye(nu * N), x0 - xr)
    Aineq = sparse.block_diag([Fx, Fu], format='csc')

    FQF = Fx.T @ Q @ Fx
    FRF = Fu.T @ R @ Fu
    P = sparse.block_diag([FQF, FRF], format='csc')

    A_f = sparse.kron(Ad, sparse.eye(nx))
    B_f = sparse.kron(Bd, sparse.eye(nx))

    Ax = sparse.kron(sparse.eye(N + 1), -sparse.eye(nx * nx)) + sparse.kron(sparse.eye(N + 1, k=-1), A_f)
    Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), B_f)
    Aeq = sparse.hstack([Ax, Bu])

    # - OSQP constraints
    A = sparse.vstack([Aeq, Aineq], format='csc')

    leq = np.hstack([-np.eye(nx).flatten(), np.zeros(N * nx * nx)])
    ueq = leq

    lineq = np.hstack([np.kron(np.ones(N + 1), xmin - xr), np.kron(np.ones(N), umin)])
    uineq = np.hstack([np.kron(np.ones(N + 1), xmax - xr), np.kron(np.ones(N), umax)])

    l = np.hstack([leq, lineq])
    u = np.hstack([ueq, uineq])

    # print(l.shape, A.shape, leq.shape, lineq.shape, Aeq.shape, Aineq.shape, P.shape)

    # Setup workspace
    m.setup(P, None, A, l, u, warm_start=True, verbose=False, eps_abs=1e-4, eps_rel=1e-4)
    # l[:nx] = -(x0 - xr)
    # u[:nx] = -(x0 - xr)
    # prob.update(l=l, u=u)

    # Simulate in closed loop
    # t_0 = time.time()
    t_0 = 0
    runs = 1
    nsim = 10
    states = np.zeros((nx, nsim + 1))
    states[:, 0] = x0
    input = np.zeros((nu, nsim))

    for run in range(runs):
        for i in range(nsim):
            # Solve
            res = m.solve()

            # Check solver status
            if res.info.status != 'solved':
                raise ValueError('OSQP did not solve the problem!')

            phi_x = res.x[nx * nx: 2 * nx * nx]
            phi_u = res.x[-N * nu * nx:-(N-1) * nu * nx]

            states[:, i + 1] = phi_x.reshape((nx, nx)) @ (states[:, i] - xr) + xr
            input[:, i] = phi_u.reshape((nu, nx)) @ (states[:, i] - xr)

            print(np.sum(np.abs(res.x[: (N+1) * nx * nx]) < 1e-6) / ((N+1) * nx * nx))

            x0 = states[:, i + 1]
            Fx = sparse.kron(sparse.eye(nx * (N + 1)), x0 - xr)
            Fu = sparse.kron(sparse.eye(nu * N), x0 - xr)
            Aineq = sparse.block_diag([Fx, Fu], format='csc')

            FQF = Fx.T @ Q @ Fx
            FRF = Fu.T @ R @ Fu
            P = sparse.block_diag([FQF, FRF], format='csc')
            # print(np.min(np.linalg.eig(P.todense())), np.max(np.linalg.eig(P.todense())))

            # - OSQP constraints
            A = sparse.vstack([Aeq, Aineq], format='csc')
            m = osqp.OSQP()
            m.setup(P, None, A, l, u, warm_start=True, verbose=False, eps_abs=1e-4, eps_rel=1e-4)

            if i == 0:
                t_0 = time.time()

    t_end = time.time()

    avg_time = (t_end - t_0) / runs / (nsim - 1)
    # print(f"Average elapsed time: {avg_time}")

    plt.figure()
    plt.plot(np.rad2deg(states[1::6].T - xr[1::6]))

    plt.figure()
    plt.plot(input[1::3].T)
    plt.show()
    return avg_time


if __name__ == '__main__':
    print(time_optimisation(3, 10))
