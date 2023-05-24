import numpy as np
import osqp
from scipy import sparse

from slspy import SynthesisAlgorithm, LTV_System, SLS_StateFeedback_FIR_Controller


class OSQP_Solver():
    """
    Class to use OSQP solver for SLS purposes.
    """
    def __init__(self, number_of_satellites: int, prediction_horizon: int, model: LTV_System,
                 reference_state: np.ndarray):
        self._problem = osqp.OSQP()
        self._P = None
        self._q = None
        self._A = None
        self._l = None
        self._u = None

        self.number_of_satellites = number_of_satellites
        self.prediction_horizon = prediction_horizon
        self.model = model
        _, self._nx, self._nu = self.model._B2.shape
        self.x_ref = reference_state.reshape((-1,))
        self.result = None

    def initialise_phi(self, sls) -> None:
        """
        No need to initialise Phi for the OSQP solver.
        Merely here for convenience to be compatible with default solver.

        :param sls: SLS manager.
        """
        pass

    def set_cost_matrices(self, Q_sqrt: np.ndarray, R_sqrt: np.ndarray) -> None:
        """
        Set the cost matrices for the optimisation problem.
        :param Q_sqrt: Square root of Q matrix for state cost.
        :param R_sqrt: Square root of R matrix for input cost.
        """
        Q = sparse.kron(sparse.eye(self.number_of_satellites),
                        sparse.csc_matrix(Q_sqrt).power(2))
        QN = Q
        R = sparse.kron(sparse.eye(self.number_of_satellites),
                        sparse.csc_matrix(R_sqrt).power(2))

        self._P = sparse.block_diag([sparse.kron(sparse.eye(self.prediction_horizon), Q), QN,
                                     sparse.kron(sparse.eye(self.prediction_horizon), R)], format='csc')
        # - linear objective
        self._q = np.hstack([np.kron(np.ones(self.prediction_horizon), -Q.dot(self.x_ref)), -QN.dot(self.x_ref),
                             np.zeros(self.prediction_horizon * self._nu)])

    def set_constraints(self, state_limit: list, input_limit: list):
        """
        Set the constraints including state and input limits.

        :param state_limit: Maximum value of each state.
        :param input_limit: Minimum value of each state.
        """
        # Limit Constraints
        xmin = -np.array(state_limit * self.number_of_satellites)
        xmax = np.array(state_limit * self.number_of_satellites)
        umin = -np.array(input_limit * self.number_of_satellites)
        umax = np.array(input_limit * self.number_of_satellites)

        # Linear dynamics
        # sparse_A = [sparse.csc_matrix(self.model._A[i]) for i in range(self.prediction_horizon)]
        Ax = sparse.kron(sparse.eye(self.prediction_horizon + 1), -sparse.eye(self._nx)) + \
             sparse.vstack([sparse.csc_matrix((self._nx, self._nx * (self.prediction_horizon + 1))),
                            sparse.hstack([sparse.block_diag(self.model._A, format='csc'),
                                           sparse.csc_matrix((self._nx * self.prediction_horizon, self._nx))])])

        sparse_B2 = [sparse.csc_matrix(self.model._B2[i]) for i in range(self.prediction_horizon)]
        Bu = sparse.vstack([sparse.csc_matrix((self._nx, self.prediction_horizon * self._nu)),
                            sparse.block_diag(sparse_B2, format='csc')])

        Aeq = sparse.hstack([Ax, Bu])
        leq = np.zeros((self.prediction_horizon + 1) * self._nx)  # Set x0 later (fist nx elements)
        ueq = leq

        # - input and state constraints
        Aineq = sparse.eye((self.prediction_horizon + 1) * self._nx + self.prediction_horizon * self._nu)
        lineq = np.hstack([np.kron(np.ones(self.prediction_horizon + 1), xmin),
                           np.kron(np.ones(self.prediction_horizon), umin)])
        uineq = np.hstack([np.kron(np.ones(self.prediction_horizon + 1), xmax),
                           np.kron(np.ones(self.prediction_horizon), umax)])

        # - OSQP constraints
        self._A = sparse.vstack([Aeq, Aineq], format='csc')
        self._l = np.hstack([leq, lineq])
        self._u = np.hstack([ueq, uineq])

    def update_x0(self) -> None:
        """
        Solve a specific problem by selecting a specific x0.

        :param x0: Array with the initial state.
        """
        x0 = self.model._x0.flatten()
        self._l[:self._nx] = -x0
        self._u[:self._nx] = -x0
        self._problem.update(l=self._l, u=self._u)

    def initialise_problem(self, **kwargs) -> None:
        """
        Initialise the problem.

        :param kwargs: Options for the solver, e.g. warm_start=True and verbose=False.
        """
        self._problem.setup(self._P, self._q, self._A, self._l, self._u, **kwargs)

    def solve(self) -> tuple[float, SLS_StateFeedback_FIR_Controller]:
        self.result = self._problem.solve()

        if self.result.info.status_val != 1:
            raise Exception(f"Could not find a solution! {self.result.info.status}")

        controller = SLS_StateFeedback_FIR_Controller(Nx=self._nx, Nu=self._nu, FIR_horizon=self.prediction_horizon)
        controller._Phi_x = [None] * (self.prediction_horizon + 1)
        controller._Phi_u = [None] * (self.prediction_horizon + 1)
        states = self.result.x[:(self.prediction_horizon + 1)*self._nx].reshape((-1, 1))
        inputs = self.result.x[(self.prediction_horizon + 1)*self._nx:].reshape((-1, 1))
        for t in range(self.prediction_horizon):
            controller._Phi_x[t + 1] = states[t * self._nx:(t + 1) * self._nx]
            controller._Phi_u[t + 1] = inputs[t * self._nu:(t + 1) * self._nu]

        return self.result.info.obj_val, controller


class OSQP_Synthesiser(SynthesisAlgorithm):

    def __init__(self, number_of_satellites: int, prediction_horizon: int, model: LTV_System,
                 reference_state: np.ndarray):
        super().__init__()
        self._solver = OSQP_Solver(number_of_satellites, prediction_horizon, model, reference_state)
        self.initialised = False

    def create_optimisation_problem(self, Q_sqrt: np.ndarray, R_sqrt: np.ndarray, state_limit: list, input_limit: list):
        self._solver.set_cost_matrices(Q_sqrt, R_sqrt)
        self._solver.set_constraints(state_limit, input_limit)

    def synthesizeControllerModel(self, x_ref: np.ndarray=None) -> SLS_StateFeedback_FIR_Controller:
        """
        Synthesise the controller and find the optimal inputs.

        :param x_ref: Not used here. Does not do anything.
        """
        if not self.initialised:
            self._solver.initialise_problem(warm_start=True, verbose=False)
            self.initialised = True

        self._solver.update_x0()
        objective_value, controller = self._solver.solve()

        return controller


if __name__ == '__main__':
    from Scenarios.MainScenarios import ScenarioEnum
    from Dynamics.HCWDynamics import RelCylHCW
    import random
    import time
    from matplotlib import pyplot as plt

    scenario = ScenarioEnum.simple_scenario_translation_HCW_scaled.value
    dynamics = RelCylHCW(scenario)

    system_state_size = dynamics.state_size
    total_state_size = scenario.number_of_satellites * system_state_size
    total_input_size = scenario.number_of_satellites * dynamics.input_size

    # Always create an LTV system. For LTI, this simply contains the same matrices over time.
    sys = LTV_System(Nx=total_state_size, Nu=total_input_size, Nw=3, tFIR=scenario.control.tFIR)
    sys._A = np.zeros((scenario.control.tFIR, total_state_size, total_state_size))
    sys._B2 = np.zeros((scenario.control.tFIR, total_state_size, total_input_size))
    for t in range(scenario.control.tFIR):
        model = dynamics.create_multi_satellite_model(scenario.control.control_timestep,
                                                      argument_of_latitude_list=[0]*scenario.number_of_satellites)
        sys._A[t], sys._B2[t] = model.A, model.B
    sys._C2 = np.eye(total_state_size)  # All states as measurements

    # Initial and reference states
    x0 = np.zeros(total_state_size)
    possible_angles = np.linspace(0, 2 * np.pi, scenario.number_of_satellites + 2, endpoint=False)

    random.seed(129)
    selected_indices = np.sort(random.sample(range(scenario.number_of_satellites + 2), scenario.number_of_satellites))
    x0[1::6] = possible_angles[selected_indices]
    xr = np.zeros_like(x0)
    xr[1::6] = np.linspace(0, 2 * np.pi, scenario.number_of_satellites, endpoint=False)

    Q_matrix_sqrt = dynamics.get_state_cost_matrix_sqrt()
    R_matrix_sqrt = dynamics.get_input_cost_matrix_sqrt()

    full_Q_matrix = np.kron(np.eye(scenario.number_of_satellites, dtype=int), Q_matrix_sqrt)
    full_R_matrix = np.kron(np.eye(scenario.number_of_satellites, dtype=int), R_matrix_sqrt)

    # Set them as matrices for the regulator
    sys._C1 = full_Q_matrix
    sys._D12 = full_R_matrix

    synthesizer = OSQP_Synthesiser(scenario.number_of_satellites, scenario.control.tFIR, sys, xr)
    synthesizer.create_optimisation_problem(dynamics.get_state_cost_matrix_sqrt(),
                                            dynamics.get_input_cost_matrix_sqrt()[3:],
                                            dynamics.get_state_constraint(),
                                            dynamics.get_input_constraint())

    nsim = 10
    x = np.zeros((total_state_size, nsim + 1))
    x[:, 0] = x0
    t_0 = time.time()
    t_last = t_0
    print('Start!')
    for t in range(nsim):
        sys.initialize(x[:, t])
        controller = synthesizer.synthesizeControllerModel(xr)
        x[:, t+1] = controller._Phi_x[2].flatten()

        time_now = time.time()
        print(f"Last elapsed time: {(time_now - t_last)}")
        t_last = time_now


    print("End!")

    plt.figure()
    plt.plot(np.rad2deg(x[1::6].T - xr[1::6]))
    plt.show()
