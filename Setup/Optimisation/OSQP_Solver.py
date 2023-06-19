import numpy as np
import osqp
import scipy
from scipy import sparse
from slspy import SynthesisAlgorithm, LTV_System, SLS_StateFeedback_FIR_Controller
from abc import ABC, abstractmethod
from Scenarios.MainScenarios import ScenarioEnum
from Dynamics.HCWDynamics import RelCylHCW
import random
import time

class OSQP_Solver(ABC):
    """
    Class to use OSQP solver for SLS purposes.
    """

    def __init__(self, number_of_satellites: int, prediction_horizon: int, model: LTV_System,
                 reference_state: np.ndarray, slack_variables_length: int, slack_variables_cost: list[int]):
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
        self._settings = None

        self._A_ineq = None  # Can be reused when A_eq is updated

        # Everything for slack variables
        self._slack_variables_length = slack_variables_length
        self._active_slack_variables = np.arange(0, 6)[np.array(slack_variables_cost) > 0]
        self._slack_variables_cost = np.array(slack_variables_cost)[self._active_slack_variables].tolist()
        self._number_of_slack_variables = len(self._active_slack_variables)
        self._total_number_of_slack_variables = 2 * self._number_of_slack_variables * self._slack_variables_length * self.number_of_satellites

    @abstractmethod
    def set_cost_matrices(self, Q_sqrt: np.ndarray, R_sqrt: np.ndarray) -> None:
        """
        Set the cost matrices for the optimisation problem.
        :param Q_sqrt: Square root of Q matrix for state cost.
        :param R_sqrt: Square root of R matrix for input cost.
        """
        pass

    @abstractmethod
    def set_constraints(self, state_limit: list, input_limit: list):
        """
        Set the constraints including state and input limits.

        :param state_limit: Maximum value of each state.
        :param input_limit: Minimum value of each state.
        """
        pass

    def update_x0(self) -> None:
        """
        Solve a specific problem by selecting a specific x0.

        :param x0: Array with the initial state.
        """
        x0 = self.model._x0.flatten()
        self._l[:self._nx] = -x0
        self._u[:self._nx] = -x0
        self._problem.update(l=self._l, u=self._u)

    @abstractmethod
    def update_model(self, model: LTV_System, initialised: bool) -> None:
        """
        Update the model used for synthesis.

        :param model: Model that is updated.
        :param initialised: Whether the model is already initialised.
        """
        pass

    def initialise_problem(self, **kwargs) -> None:
        """
        Initialise the problem.

        :param kwargs: Options for the solver, e.g. warm_start=True and verbose=False.
        """
        self._settings = kwargs
        self._problem.setup(self._P, self._q, self._A, self._l, self._u, **self._settings)

    def solve(self) -> tuple[float, SLS_StateFeedback_FIR_Controller]:
        self.result = self._problem.solve()

        if self.result.info.status_val != 1:
            raise Exception(f"Could not find a solution! {self.result.info.status}")

        controller = SLS_StateFeedback_FIR_Controller(Nx=self._nx, Nu=self._nu, FIR_horizon=self.prediction_horizon)
        controller._Phi_x = [None] * (self.prediction_horizon + 2)
        controller._Phi_u = [None] * (self.prediction_horizon + 1)
        states = self.result.x[:(self.prediction_horizon + 1) * self._nx].reshape((-1, 1))
        inputs = self.result.x[(self.prediction_horizon + 1) * self._nx:
                               (self.prediction_horizon + 1) * self._nx + self.prediction_horizon * self._nu].reshape(
            (-1, 1))

        for t in range(self.prediction_horizon):
            controller._Phi_x[t + 1] = states[t * self._nx:(t + 1) * self._nx]
            controller._Phi_u[t + 1] = inputs[t * self._nu:(t + 1) * self._nu]

        controller._Phi_x[self.prediction_horizon + 1] = states[self.prediction_horizon * self._nx:
                                                                (self.prediction_horizon + 1) * self._nx]

        return self.result.info.obj_val, controller

    @abstractmethod
    def find_A_eq(self):
        """
        Find the equality matrix for a given model.

        :return: Sparse matrix with A_eq
        """
        Ax = sparse.kron(sparse.eye(self.prediction_horizon + 1), -sparse.eye(self._nx)) + \
             sparse.vstack([sparse.csc_matrix((self._nx, self._nx * (self.prediction_horizon + 1))),
                            sparse.hstack([sparse.block_diag(self.model._A, format='csc'),
                                           sparse.csc_matrix((self._nx * self.prediction_horizon, self._nx))])])

        sparse_B2 = [sparse.csc_matrix(self.model._B2[i]) for i in range(self.prediction_horizon)]
        Bu = sparse.vstack([sparse.csc_matrix((self._nx, self.prediction_horizon * self._nu)),
                            sparse.block_diag(sparse_B2, format='csc')])

        sparse_variables = sparse.csc_matrix((Ax.shape[0], self._total_number_of_slack_variables))

        return sparse.hstack([Ax, Bu, sparse_variables])


class OSQP_Solver_Sparse(OSQP_Solver):

    def set_cost_matrices(self, Q_sqrt: np.ndarray, R_sqrt: np.ndarray) -> None:
        Q = sparse.kron(sparse.eye(self.number_of_satellites),
                        sparse.csc_matrix(Q_sqrt).power(2))
        QN = Q
        R = sparse.kron(sparse.eye(self.number_of_satellites),
                        sparse.csc_matrix(R_sqrt).power(2))

        self._P = sparse.block_diag([sparse.kron(sparse.eye(self.prediction_horizon), Q), QN,
                                     sparse.kron(sparse.eye(self.prediction_horizon), R),
                                     sparse.csc_matrix((self._total_number_of_slack_variables,
                                                        self._total_number_of_slack_variables))],
                                    format='csc')
        # - linear objective
        sparse_costs = self._slack_variables_cost * self.number_of_satellites * self._slack_variables_length * 2
        self._q = np.hstack([np.kron(np.ones(self.prediction_horizon), -Q.dot(self.x_ref)), -QN.dot(self.x_ref),
                             np.zeros(self.prediction_horizon * self._nu), np.array(sparse_costs)])

    def set_constraints(self, state_limit: list, input_limit: list):
        # Limit Constraints
        xmin = -np.array(state_limit * self.number_of_satellites)
        xmax = np.array(state_limit * self.number_of_satellites)
        umin = -np.array(input_limit * self.number_of_satellites)
        umax = np.array(input_limit * self.number_of_satellites)

        slack_min = np.zeros((self._total_number_of_slack_variables,))
        slack_max = np.ones_like(slack_min) * 1

        # Linear dynamics
        leq = np.zeros((self.prediction_horizon + 1) * self._nx)  # Set x0 later (fist nx elements)
        ueq = leq

        # - input and state constraints
        regular_ineq = sparse.eye((self.prediction_horizon + 1) * self._nx + self.prediction_horizon * self._nu)
        slack_variables_rhs = sparse.vstack([sparse.block_diag([-sparse.eye(self._total_number_of_slack_variables // 2),
                                                                sparse.eye(
                                                                    self._total_number_of_slack_variables // 2)]),
                                             sparse.csc_matrix(
                                                 (regular_ineq.shape[0] - self._total_number_of_slack_variables,
                                                  self._total_number_of_slack_variables))])

        slack_variables_bottom = sparse.hstack([sparse.csc_matrix((self._total_number_of_slack_variables,
                                                                   regular_ineq.shape[1])),
                                                sparse.eye(self._total_number_of_slack_variables)])

        self._A_ineq = sparse.vstack([sparse.hstack([regular_ineq, slack_variables_rhs]), slack_variables_bottom])

        lineq = np.hstack([np.kron(np.ones(self.prediction_horizon + 1), xmin),
                           np.kron(np.ones(self.prediction_horizon), umin),
                           slack_min])
        uineq = np.hstack([np.kron(np.ones(self.prediction_horizon + 1), xmax),
                           np.kron(np.ones(self.prediction_horizon), umax),
                           slack_max])

        # Ignore constraints for x0 (as it is already given, and can only make problem unfeasible if start is not allowed
        lineq[:self._nx] = -np.inf
        uineq[:self._nx] = np.inf

        # - OSQP constraints
        self._A = sparse.vstack([self.find_A_eq(), self._A_ineq], format='csc')
        self._l = np.hstack([leq, lineq])
        self._u = np.hstack([ueq, uineq])

    def update_model(self, model: LTV_System, initialised: bool) -> None:
        self.model = model
        self._A = sparse.vstack([self.find_A_eq(), self._A_ineq], format='csc')

        if initialised:
            # self._problem.update(Ax=self._A.data)
            self._problem = osqp.OSQP()
            self._problem.setup(self._P, self._q, self._A, self._l, self._u, **self._settings)

    def find_A_eq(self):
        Ax = sparse.kron(sparse.eye(self.prediction_horizon + 1), -sparse.eye(self._nx)) + \
             sparse.vstack([sparse.csc_matrix((self._nx, self._nx * (self.prediction_horizon + 1))),
                            sparse.hstack([sparse.block_diag(self.model._A, format='csc'),
                                           sparse.csc_matrix((self._nx * self.prediction_horizon, self._nx))])])

        sparse_B2 = [sparse.csc_matrix(self.model._B2[i]) for i in range(self.prediction_horizon)]
        Bu = sparse.vstack([sparse.csc_matrix((self._nx, self.prediction_horizon * self._nu)),
                            sparse.block_diag(sparse_B2, format='csc')])

        sparse_variables = sparse.csc_matrix((Ax.shape[0], self._total_number_of_slack_variables))

        return sparse.hstack([Ax, Bu, sparse_variables])


class OSQP_Solver_Dense(OSQP_Solver):

    def set_cost_matrices(self, Q_sqrt: np.ndarray, R_sqrt: np.ndarray) -> None:
        Q = np.kron(np.eye(self.number_of_satellites), Q_sqrt ** 2)
        QN = Q
        R = np.kron(np.eye(self.number_of_satellites), R_sqrt ** 2)

        self._P = scipy.linalg.block_diag(np.kron(np.eye(self.prediction_horizon), Q), QN,
                                          np.kron(np.eye(self.prediction_horizon), R),
                                          np.zeros((self._total_number_of_slack_variables,
                                                    self._total_number_of_slack_variables)))
        # - linear objective
        sparse_costs = self._slack_variables_cost * self.number_of_satellites * self._slack_variables_length * 2
        self._q = np.hstack([np.kron(np.ones(self.prediction_horizon), -Q.dot(self.x_ref)), -QN.dot(self.x_ref),
                             np.zeros(self.prediction_horizon * self._nu), np.array(sparse_costs)])

    def set_constraints(self, state_limit: list, input_limit: list):
        # Limit Constraints
        xmin = -np.array(state_limit * self.number_of_satellites)
        xmax = np.array(state_limit * self.number_of_satellites)
        umin = -np.array(input_limit * self.number_of_satellites)
        umax = np.array(input_limit * self.number_of_satellites)

        slack_min = np.zeros((self._total_number_of_slack_variables,))
        slack_max = np.ones_like(slack_min) * 1

        # Linear dynamics
        leq = np.zeros((self.prediction_horizon + 1) * self._nx)  # Set x0 later (fist nx elements)
        ueq = leq

        # - input and state constraints
        regular_ineq = np.eye((self.prediction_horizon + 1) * self._nx + self.prediction_horizon * self._nu)
        slack_variables_rhs = np.vstack([scipy.linalg.block_diag(-np.eye(self._total_number_of_slack_variables // 2),
                                                                 np.eye(
                                                                     self._total_number_of_slack_variables // 2)),
                                         np.zeros(
                                             (regular_ineq.shape[0] - self._total_number_of_slack_variables,
                                              self._total_number_of_slack_variables))])

        slack_variables_bottom = np.hstack([np.zeros((self._total_number_of_slack_variables,
                                                      regular_ineq.shape[1])),
                                            np.eye(self._total_number_of_slack_variables)])

        self._A_ineq = np.vstack([np.hstack([regular_ineq, slack_variables_rhs]), slack_variables_bottom])

        lineq = np.hstack([np.kron(np.ones(self.prediction_horizon + 1), xmin),
                           np.kron(np.ones(self.prediction_horizon), umin),
                           slack_min])
        uineq = np.hstack([np.kron(np.ones(self.prediction_horizon + 1), xmax),
                           np.kron(np.ones(self.prediction_horizon), umax),
                           slack_max])

        # Ignore constraints for x0 (as it is already given, and can only make problem unfeasible if start is not allowed
        lineq[:self._nx] = -np.inf
        uineq[:self._nx] = np.inf

        # - OSQP constraints
        self._A = np.vstack([self.find_A_eq(), self._A_ineq])
        self._l = np.hstack([leq, lineq])
        self._u = np.hstack([ueq, uineq])

    def update_model(self, model: LTV_System, initialised: bool) -> None:
        self.model = model
        self._A = np.vstack([self.find_A_eq(), self._A_ineq])

        if initialised:
            # self._problem.update(Ax=self._A.data)
            self._problem = osqp.OSQP()
            self._problem.setup(self._P, self._q, self._A, self._l, self._u, **self._settings)

    def find_A_eq(self):
        Ax = np.kron(np.eye(self.prediction_horizon + 1), -np.eye(self._nx)) + \
             np.vstack([np.zeros((self._nx, self._nx * (self.prediction_horizon + 1))),
                        np.hstack([scipy.linalg.block_diag(*self.model._A),
                                   np.zeros((self._nx * self.prediction_horizon, self._nx))])])

        sparse_B2 = [(self.model._B2[i]) for i in range(self.prediction_horizon)]
        Bu = np.vstack([np.zeros((self._nx, self.prediction_horizon * self._nu)),
                        scipy.linalg.block_diag(*sparse_B2)])

        sparse_variables = np.zeros((Ax.shape[0], self._total_number_of_slack_variables))

        return np.hstack([Ax, Bu, sparse_variables])


class OSQP_Synthesiser(SynthesisAlgorithm):

    def __init__(self, number_of_satellites: int, prediction_horizon: int, model: LTV_System,
                 reference_state: np.ndarray, sparse_variables_length: int, sparse_variables_costs: list[int],
                 sparse: bool = True):
        super().__init__()

        if sparse:
            self._solver = OSQP_Solver_Sparse(number_of_satellites, prediction_horizon, model, reference_state,
                                              sparse_variables_length, sparse_variables_costs)
        else:
            self._solver = OSQP_Solver_Dense(number_of_satellites, prediction_horizon, model, reference_state,
                                             sparse_variables_length, sparse_variables_costs)
        self.initialised = False

    def create_optimisation_problem(self, Q_sqrt: np.ndarray, R_sqrt: np.ndarray, state_limit: list, input_limit: list):
        self._solver.set_cost_matrices(Q_sqrt, R_sqrt)
        self._solver.set_constraints(state_limit, input_limit)

    def update_model(self, model: LTV_System) -> None:
        """
        Update the synthesiser with a new model.

        :param model: New model.
        """
        self._solver.update_model(model, self.initialised)

    def synthesizeControllerModel(self, x_ref: np.ndarray = None) -> SLS_StateFeedback_FIR_Controller:
        """
        Synthesise the controller and find the optimal inputs.

        :param x_ref: Not used here. Does not do anything.
        """
        if not self.initialised:
            # self._solver.initialise_problem(warm_start=True, verbose=False)
            self._solver.initialise_problem(warm_start=True, verbose=False, polish=True, check_termination=10,
                                            eps_abs=1e-4, eps_rel=1e-4, max_iter=100000)
            self.initialised = True

        self._solver.update_x0()
        objective_value, controller = self._solver.solve()

        return controller


def time_optimisation(number_of_satellites: int, prediction_horizon: int):
    scenario = ScenarioEnum.simple_scenario_translation_HCW_scaled.value
    scenario.number_of_satellites = number_of_satellites
    scenario.control.tFIR = prediction_horizon
    # print(scenario.number_of_satellites)
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
                                                      argument_of_latitude_list=[0] * scenario.number_of_satellites,
                                                      argument_of_periapsis_list=[0] * scenario.number_of_satellites)
        sys._A[t], sys._B2[t] = model.A, model.B
    sys._C2 = np.eye(total_state_size)  # All states as measurements

    # Initial and reference states
    x0 = np.zeros(total_state_size)
    possible_angles = np.linspace(0, 2 * np.pi, scenario.number_of_satellites + 2, endpoint=False)

    random.seed(129)
    selected_indices = np.sort(random.sample(range(scenario.number_of_satellites + 2), scenario.number_of_satellites))
    x0[1::6] = possible_angles[selected_indices]
    xr = np.zeros_like(x0)
    ref_rel_angles = np.linspace(0, 2 * np.pi, scenario.number_of_satellites, endpoint=False)
    ref_rel_angles -= np.mean(ref_rel_angles - possible_angles[selected_indices])
    xr[1::6] = ref_rel_angles

    Q_matrix_sqrt = dynamics.get_state_cost_matrix_sqrt()
    R_matrix_sqrt = dynamics.get_input_cost_matrix_sqrt()

    full_Q_matrix = np.kron(np.eye(scenario.number_of_satellites, dtype=int), Q_matrix_sqrt)
    full_R_matrix = np.kron(np.eye(scenario.number_of_satellites, dtype=int), R_matrix_sqrt)

    # Set them as matrices for the regulator
    sys._C1 = full_Q_matrix
    sys._D12 = full_R_matrix

    synthesizer = OSQP_Synthesiser(scenario.number_of_satellites, scenario.control.tFIR, sys, xr,
                                   dynamics.get_slack_variable_length(), dynamics.get_slack_costs())
    synthesizer.create_optimisation_problem(dynamics.get_state_cost_matrix_sqrt(),
                                            dynamics.get_input_cost_matrix_sqrt()[3:],
                                            dynamics.get_state_constraint(),
                                            dynamics.get_input_constraint())

    nsim = 10
    x = np.zeros((total_state_size, nsim + 1))
    x[:, 0] = x0
    t_0 = 0 # time.time()
    t_last = t_0
    # print('Start!')
    for t in range(nsim):
        sys.initialize(x[:, t])
        controller = synthesizer.synthesizeControllerModel(xr)
        x[:, t + 1] = controller._Phi_x[2].flatten()

        if t == 0:
            t_0 = time.time()
        # time_now = time.time()
        # print(f"Last elapsed time: {(time_now - t_last)}")
        # t_last = time_now

    avg_time = (time.time() - t_0) / (nsim - 1)
    # print("End!")
    # time_array[idx] = (time.time() - t_0) / nsim
    return avg_time


if __name__ == '__main__':
    from Scenarios.MainScenarios import ScenarioEnum
    from Dynamics.HCWDynamics import RelCylHCW
    import random
    import time
    from matplotlib import pyplot as plt

    scenario = ScenarioEnum.simple_scenario_translation_HCW_scaled.value
    scenario.number_of_satellites = 10
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
                                                      argument_of_latitude_list=[0] * scenario.number_of_satellites,
                                                      argument_of_periapsis_list=[0] * scenario.number_of_satellites)
        sys._A[t], sys._B2[t] = model.A, model.B
    sys._C2 = np.eye(total_state_size)  # All states as measurements

    # Initial and reference states
    x0 = np.zeros(total_state_size)
    possible_angles = np.linspace(0, 2 * np.pi, scenario.number_of_satellites + 2, endpoint=False)

    random.seed(129)
    selected_indices = np.sort(random.sample(range(scenario.number_of_satellites + 2), scenario.number_of_satellites))
    x0[1::6] = possible_angles[selected_indices]
    xr = np.zeros_like(x0)
    ref_rel_angles = np.linspace(0, 2 * np.pi, scenario.number_of_satellites, endpoint=False)
    ref_rel_angles -= np.mean(ref_rel_angles - possible_angles[selected_indices])
    xr[1::6] = ref_rel_angles

    Q_matrix_sqrt = dynamics.get_state_cost_matrix_sqrt()
    R_matrix_sqrt = dynamics.get_input_cost_matrix_sqrt()

    full_Q_matrix = np.kron(np.eye(scenario.number_of_satellites, dtype=int), Q_matrix_sqrt)
    full_R_matrix = np.kron(np.eye(scenario.number_of_satellites, dtype=int), R_matrix_sqrt)

    # Set them as matrices for the regulator
    sys._C1 = full_Q_matrix
    sys._D12 = full_R_matrix

    synthesizer = OSQP_Synthesiser(scenario.number_of_satellites, scenario.control.tFIR, sys, xr,
                                   dynamics.get_slack_variable_length(), dynamics.get_slack_costs(), sparse=True)
    synthesizer.create_optimisation_problem(dynamics.get_state_cost_matrix_sqrt(),
                                            dynamics.get_input_cost_matrix_sqrt()[3:],
                                            dynamics.get_state_constraint(),
                                            dynamics.get_input_constraint())

    nsim = 10
    x = np.zeros((total_state_size, nsim + 1))
    x[:, 0] = x0
    sys.initialize(x[:, 0])
    controller = synthesizer.synthesizeControllerModel(xr)
    x[:, 1] = controller._Phi_x[2].flatten()

    t_0 = time.time()
    t_last = t_0
    print('Start!')
    for t in range(1, nsim):
        sys.initialize(x[:, t])
        controller = synthesizer.synthesizeControllerModel(xr)
        x[:, t + 1] = controller._Phi_x[2].flatten()

        # time_now = time.time()
        # print(f"Last elapsed time: {(time_now - t_last)}")
        # t_last = time_now

    print("End!")
    print(f"Time average: {(time.time() - t_0) / (nsim - 1)}")
    plt.figure()
    plt.plot(np.rad2deg(x[1::6].T - xr[1::6]))
    plt.show()
