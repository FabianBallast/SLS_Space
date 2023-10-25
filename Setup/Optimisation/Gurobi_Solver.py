import numpy as np
import gurobipy as gp
from slspy import SynthesisAlgorithm, LTV_System, SLS_StateFeedback_FIR_Controller
from abc import ABC, abstractmethod
import time


class Gurobi_Solver(ABC):
    """
    Class to use OSQP solver for SLS purposes.
    """

    def __init__(self, number_of_satellites: int, prediction_horizon: int, model: LTV_System,
                 reference_state: np.ndarray, state_limit: np.ndarray, input_limit: np.ndarray):
        self.number_of_satellites = number_of_satellites
        self.prediction_horizon = prediction_horizon
        self.model = model
        _, self._nx, self._nu = self.model._B2.shape
        self.x_ref = reference_state.reshape((-1,))
        self.result = None
        self._settings = None

        x_max = np.tile(state_limit, (self.prediction_horizon + 1, self.number_of_satellites))
        x_max[0] = np.inf
        u_max = np.tile(input_limit, (self.prediction_horizon, self.number_of_satellites))

        self._problem = gp.Model('MPC')
        # print(x_max.shape)
        self.x = self._problem.addMVar(shape=(self.prediction_horizon + 1, self._nx), name='x', lb=-x_max, ub=x_max)
        self.u = self._problem.addMVar(shape=(self.prediction_horizon, self._nu), name='u', lb=-u_max, ub=u_max)

        self.initial_state_constraint = self._problem.addConstr(self.x[0, :] == np.zeros((self._nx)))
        self.dynamics_constraints = []
        for k in range(self.prediction_horizon):
            self.dynamics_constraints.append(self._problem.addConstr(
                self.x[k + 1, :] == self.model._A[k] @ self.x[k, :] + self.model._B2[k] @ self.u[k, :]))

        self._model_updated = False

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

    def update_x0(self, time_since_start: int = 0) -> None:
        """
        Solve a specific problem by selecting a specific x0.

        :param x0: Array with the initial state.
        :param time_since_start: time since the start of the simulation in s.
        """
        x0 = self.model._x0.flatten()
        self.initial_state_constraint.rhs = x0

        # print(self._model_updated)
        if self._model_updated:
            for constraint in self.dynamics_constraints:
                self._problem.remove(constraint)

            for k in range(self.prediction_horizon):
                self.dynamics_constraints[k] = self._problem.addConstr(self.x[k + 1, :] == self.model._A[k] @ self.x[k, :] + self.model._B2[k] @ self.u[k, :])

            self._problem.update()
            self._model_updated = False

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

        for name in kwargs:
            self._problem.setParam(name, kwargs[name])

    def solve(self) -> tuple[float, SLS_StateFeedback_FIR_Controller]:
        """
        Solve the problem and return the result + the controller.

        :return: Runtime and feedback controller.
        """
        self.result = self._problem.optimize()

        controller = SLS_StateFeedback_FIR_Controller(Nx=self._nx, Nu=self._nu, FIR_horizon=self.prediction_horizon)
        controller._Phi_x = [None] * (self.prediction_horizon + 2)
        controller._Phi_u = [None] * (self.prediction_horizon + 1)

        for t in range(self.prediction_horizon):
            controller._Phi_x[t + 1] = self.x.X[t:t+1].T
            controller._Phi_u[t + 1] = self.u.X[t:t+1].T

        # print(controller._Phi_x[1][0][0], controller._Phi_x[2][0][0])

        controller._Phi_x[self.prediction_horizon + 1] = self.x.X[self.prediction_horizon:].T

        return self._problem.Runtime, controller


class Gurobi_Solver_Sparse(Gurobi_Solver):
    """
    Sparse OSQP solver.
    """
    def set_cost_matrices(self, Q_sqrt: np.ndarray, R_sqrt: np.ndarray) -> None:
        """
        Set the cost matrices for the solver.

        :param Q_sqrt: Square root of state cost matrix.
        :param R_sqrt: Square root of input cost matrix.
        """
        obj1 = sum(self.x[k, :] @ np.kron(np.eye(self.number_of_satellites), Q_sqrt**2) @ self.x[k, :] for k in range(1, self.prediction_horizon + 1))
        # obj1 += 4 * self.x[-1, :] @ np.kron(np.eye(self.number_of_satellites), Q_sqrt**2) @ self.x[-1, :]
        # obj1 *= 10
        obj2 = sum(self.u[k, :] @ np.kron(np.eye(self.number_of_satellites), R_sqrt**2) @ self.u[k, :] for k in range(self.prediction_horizon))
        self._problem.setObjective(obj1 + obj2, gp.GRB.MINIMIZE)

    def set_constraints(self, state_limit: list, input_limit: list) -> None:
        """
        Set the state and input constraints.

        :param state_limit: Maximum state values per satellite in list-form.
        :param input_limit: Maximum input values per satellite in list-form.
        """
        pass

    def update_model(self, model: LTV_System, initialised: bool) -> None:
        """
        Update the model and check if it was already initialised.

        :param model: LTV model
        :param initialised: Bool whether the problem was already initialised.
        """
        self.model = model

        if initialised:
            self._model_updated = True


class Gurobi_Synthesiser(SynthesisAlgorithm):

    def __init__(self, number_of_satellites: int, prediction_horizon: int, model: LTV_System,
                 reference_state: np.ndarray, state_limit: np.ndarray, input_limit: np.ndarray,
                 sparse: bool = True):
        super().__init__()

        if sparse:
            self._solver = Gurobi_Solver_Sparse(number_of_satellites, prediction_horizon, model, reference_state,
                                                state_limit, input_limit)
        self.initialised = False

    def create_optimisation_problem(self, Q_sqrt: np.ndarray, R_sqrt: np.ndarray, state_limit: list, input_limit: list):
        """
        Create the matrices for the optimisation problem.

        :param Q_sqrt: Square root of state cost.
        :param R_sqrt: Square root of input cost.
        :param state_limit: Maximum state values.
        :param input_limit: Maximum input values.
        """
        self._solver.set_cost_matrices(Q_sqrt, R_sqrt)
        self._solver.set_constraints(state_limit, input_limit)

    def update_model(self, model: LTV_System) -> None:
        """
        Update the synthesiser with a new model.

        :param model: New model.
        """
        self._solver.update_model(model, self.initialised)

    def synthesizeControllerModel(self, x_ref: np.ndarray = None,
                                  time_since_start: int = 0) -> tuple[float, SLS_StateFeedback_FIR_Controller]:
        """
        Synthesise the controller and find the optimal inputs.

        :param x_ref: Not used here. Does not do anything.
        :param time_since_start: Time since start of the simulation in s.
        """
        time_start = time.time()
        if not self.initialised:
            # self._solver.initialise_problem(warm_start=True, verbose=False)
            self._solver.initialise_problem(OutputFlag=0)
            self.initialised = True

        self._solver.update_x0(time_since_start)
        solver_time, controller = self._solver.solve()

        return time.time() - time_start, controller


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

    synthesizer = Gurobi_Synthesiser(scenario.number_of_satellites, scenario.control.tFIR, sys, xr,
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

    synthesizer = Gurobi_Synthesiser(scenario.number_of_satellites, scenario.control.tFIR, sys, xr,
                                     dynamics.get_state_constraint(), dynamics.get_input_constraint(), sparse=True)
    synthesizer.create_optimisation_problem(dynamics.get_state_cost_matrix_sqrt(),
                                            dynamics.get_input_cost_matrix_sqrt()[3:],
                                            dynamics.get_state_constraint(),
                                            dynamics.get_input_constraint())

    nsim = 10
    x = np.zeros((total_state_size, nsim + 1))
    x[:, 0] = x0 - xr
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
        # print(x[:, t+1])
        # time_now = time.time()
        # print(f"Last elapsed time: {(time_now - t_last)}")
        # t_last = time_now

    print("End!")
    print(f"Time average: {(time.time() - t_0) / (nsim - 1)}")
    plt.figure()
    plt.plot(np.rad2deg(x[1::6].T))
    plt.show()
