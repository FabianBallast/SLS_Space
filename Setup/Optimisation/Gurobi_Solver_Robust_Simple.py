import numpy as np
import gurobipy as gp
from slspy import SynthesisAlgorithm, LTV_System, SLS_StateFeedback_FIR_Controller
from Optimisation.Gurobi_Solver import Gurobi_Solver
from Optimisation.sparseHelperFunctions import *
from Scenarios.RobustScenarios import Robustness
import time


class Gurobi_Solver_Robust_Simple(Gurobi_Solver):
    """
    Class to use OSQP solver for SLS purposes.
    """

    def __init__(self, number_of_satellites: int, prediction_horizon: int, model: LTV_System,
                 reference_state: np.ndarray, state_limit: np.ndarray, input_limit: np.ndarray,
                 robustness_data: Robustness):
        super().__init__(number_of_satellites, prediction_horizon, model, reference_state, state_limit, input_limit)

        self.x_max = np.tile(state_limit, (self.prediction_horizon + 1, self.number_of_satellites))
        self.x_max[0] = np.inf
        self.u_max = np.tile(input_limit, (self.prediction_horizon, self.number_of_satellites))

        self.mask_A, self.mask_B = get_masks(sparse.coo_matrix(model._A[0]), sparse.coo_matrix(model._B2[0]))
        self.x_vars = np.sum(self.mask_A)
        self.u_vars = np.sum(self.mask_B)
        number_of_blocks = int((self.prediction_horizon + 1) / 2 * self.prediction_horizon + 0.001)

        block_ordering = sparse.csc_matrix(np.tril(np.ones((self.prediction_horizon, self.prediction_horizon))))
        block_ordering.data = np.arange(number_of_blocks)
        block_ordering = np.array(block_ordering.todense())

        # one_norm_matrix_x, one_norm_matrix_u = find_fx_and_fu(self.mask_A, self.mask_B, np.ones_like(reference_state))
        one_norm_kron_mat = np.zeros((self.prediction_horizon, number_of_blocks))
        for n in range(self.prediction_horizon):
            one_norm_kron_mat[n, block_ordering[n, 1:n + 1]] = 1

        self._problem = gp.Model('MPC')
        # print(x_max.shape)
        self.x = self._problem.addMVar(self.prediction_horizon * self._nx, name='x', lb=-np.inf)
        self.u = self._problem.addMVar(self.prediction_horizon * self._nu, name='u', lb=-np.inf)
        self.phi_x = self._problem.addMVar(number_of_blocks * self.x_vars, name='phi_x', lb=-np.inf)  # , ub=np.inf)
        self.phi_u = self._problem.addMVar(number_of_blocks * self.u_vars, name='phi_u', lb=-np.inf)  # , ub=np.inf)
        self.sigma = self._problem.addMVar(self.prediction_horizon, name='sigma')

        self.x_inf = self._problem.addMVar(self.prediction_horizon, name='x_inf')
        self.u_inf = self._problem.addMVar(self.prediction_horizon, name='u_inf')
        self.phi_x_inf = self._problem.addMVar(number_of_blocks, name='phi_x_inf')
        self.phi_u_inf = self._problem.addMVar(number_of_blocks, name='phi_u_inf')

        self.phi_x_one = self._problem.addMVar(number_of_blocks * self._nx, name='phi_x_one')
        self.phi_u_one = self._problem.addMVar(number_of_blocks * self._nu, name='phi_u_one')

        A_f, B_f = sparse_state_matrix_replacement(sparse.coo_matrix(self.model._A[0]), sparse.coo_matrix(self.model._B2[0]), self.mask_A, self.mask_B)

        Ax = -sparse.eye(self.prediction_horizon * self.x_vars) + sparse.kron(sparse.eye(self.prediction_horizon, k=-1), A_f)
        Bu = sparse.kron(sparse.eye(self.prediction_horizon), B_f)
        rhs = np.zeros(self.prediction_horizon * self.x_vars)
        rhs[:self.x_vars] = -A_f @ np.eye(self._nx)[self.mask_A]

        self._problem.addConstr(Ax @ self.phi_x[:self.prediction_horizon * self.x_vars] + Bu @ self.phi_u[:self.prediction_horizon * self.u_vars] == rhs)

        for n in range(1, self.prediction_horizon):
            Ax = -sparse.eye((self.prediction_horizon - n) * self.x_vars) + sparse.kron(sparse.eye((self.prediction_horizon - n), k=-1), A_f)
            Bu = sparse.kron(sparse.eye(self.prediction_horizon - n), B_f)

            try:
                self._problem.addConstr(Ax @ self.phi_x[block_ordering[n, n] * self.x_vars:block_ordering[n + 1, n + 1] * self.x_vars] +
                            Bu @ self.phi_u[block_ordering[n, n] * self.u_vars:block_ordering[n + 1, n + 1] * self.u_vars] == rhs[:(self.prediction_horizon - n) * self.x_vars] *
                            self.sigma[n - 1])
            except IndexError:
                self._problem.addConstr(Ax @ self.phi_x[block_ordering[n, n] * self.x_vars:] +
                            Bu @ self.phi_u[block_ordering[n, n] * self.u_vars:] == rhs[:(self.prediction_horizon - n) * self.x_vars] * self.sigma[
                                n - 1])

        self._model_updated = False
        self.constraint_list = []

        self.robustness = robustness_data

        for i in range(self.prediction_horizon):
            self._problem.addConstr(self.x_inf[i] == gp.norm(self.x[i * self._nx: (i + 1) * self._nx], gp.GRB.INFINITY))
            self._problem.addConstr(self.u_inf[i] == gp.norm(self.u[i * self._nu: (i + 1) * self._nu], gp.GRB.INFINITY))

        one_norm_matrix_x, one_norm_matrix_u = find_fx_and_fu(self.mask_A, self.mask_B, np.ones(self._nx))
        for i in range(number_of_blocks):
            # self._problem.addConstr(self.phi_x_inf[i] == gp.norm(self.phi_x_one[i * self._nx: (i + 1) * self._nx], gp.GRB.INFINITY))
            # self._problem.addConstr(self.phi_u_inf[i] == gp.norm(self.phi_u_one[i * self._nu: (i + 1) * self._nu], gp.GRB.INFINITY))
            self._problem.addConstr(
                self.phi_x_inf[i] == gp.max_(self.phi_x_one[i * self._nx + j] for j in range(self._nx)))
            self._problem.addConstr(
                self.phi_u_inf[i] == gp.max_(self.phi_u_one[i * self._nu + j] for j in range(self._nu)))

            for j in range(self._nx):
                ind = np.linspace(i * self.x_vars, (i+1) * self.x_vars, num=self.x_vars, endpoint=False, dtype=int)[
                    np.array(one_norm_matrix_x.todense())[j, :] > 0]
                self._problem.addConstr(
                    self.phi_x_one[i * self._nx + j] == gp.norm(self.phi_x[ind], 1.0))

            for j in range(self._nu):
                ind = np.linspace(i * self.u_vars, (i + 1) * self.u_vars, num=self.u_vars, endpoint=False, dtype=int)[
                    np.array(one_norm_matrix_u.todense())[j, :] > 0]
                self._problem.addConstr(
                    self.phi_u_one[i * self._nu + j] == gp.norm(self.phi_u[ind], 1.0))

        # Upper bound on lumped disturbance
        self._problem.addConstr(self.robustness.e_A * (self.x_inf[0] + self.sigma[0]) +
                    self.robustness.e_B * (self.u_inf[1] + gp.quicksum(self.phi_u_inf[block_ordering[1, 1:2]])) +
                    self.robustness.sigma_w <= self.sigma[1])
        for n in range(2, self.prediction_horizon):
            self._problem.addConstr(
                self.robustness.e_A * (self.x_inf[n - 1] + gp.quicksum(self.phi_x_inf[block_ordering[n - 1, 1:n]]) + self.sigma[n-1]) +
                self.robustness.e_B * (self.u_inf[n] + gp.quicksum(self.phi_u_inf[block_ordering[n, 1:n + 1]])) +
                self.robustness.sigma_w <= self.sigma[n])

        # Tightened constraints
        self._problem.addConstr(self.x[:self._nx] + self.sigma[0] <= self.x_max[1])
        self._problem.addConstr(self.x[:self._nx] - self.sigma[0] >= -self.x_max[1])
        self._problem.addConstr(self.u[:self._nu] <= self.u_max[0])
        self._problem.addConstr(self.u[:self._nu] >= -self.u_max[0])

        for n in range(1, self.prediction_horizon):
            for j in range(self._nx):
                # print(block_ordering[n, 1:n + 1] * self._nx + j)
                self._problem.addConstr(self.x[n * self._nx + j] + gp.quicksum(self.phi_x_one[block_ordering[n, 1:n + 1] * self._nx + j]) + self.sigma[n] <= self.x_max[n+1, j])
                self._problem.addConstr(self.x[n * self._nx + j] - gp.quicksum(self.phi_x_one[block_ordering[n, 1:n + 1] * self._nx + j]) - self.sigma[n] >= -self.x_max[n+1, j])
            for j in range(self._nu):
                self._problem.addConstr(self.u[n * self._nu + j] + gp.quicksum(self.phi_u_one[block_ordering[n, 1:n + 1] * self._nu + j]) <= self.u_max[n, j])
                self._problem.addConstr(self.u[n * self._nu + j] - gp.quicksum(self.phi_u_one[block_ordering[n, 1:n + 1] * self._nu + j]) >= -self.u_max[n, j])

    def set_cost_matrices(self, Q_sqrt: np.ndarray, R_sqrt: np.ndarray) -> None:
        """
        Set the cost matrices for the solver.

        :param Q_sqrt: Square root of state cost matrix.
        :param R_sqrt: Square root of input cost matrix.
        """
        obj1 = self.x @ np.kron(np.eye(self.number_of_satellites * self.prediction_horizon), Q_sqrt ** 2) @ self.x
        obj1 += 4 * self.x[-self._nx:] @ Q_sqrt ** 2 @ self.x[-self._nx:]
        obj2 = self.u @ np.kron(np.eye(self.number_of_satellites * self.prediction_horizon), R_sqrt ** 2) @ self.u
        self._problem.setObjective(obj1 + obj2, gp.GRB.MINIMIZE)

    def set_constraints(self, state_limit: list, input_limit: list) -> None:
        """
        Set the state and input constraints.

        :param state_limit: Maximum state values per satellite in list-form.
        :param input_limit: Maximum input values per satellite in list-form.
        """
        pass

    def update_x0(self, time_since_start: int = 0) -> None:
        """
        Solve a specific problem by selecting a specific x0.

        :param x0: Array with the initial state.
        :param time_since_start: time since the start of the simulation in s.
        """
        x0 = self.model._x0.flatten()

        Fx, Fu = find_fx_and_fu(self.mask_A, self.mask_B, x0)
        Fx = sparse.kron(sparse.eye(self.prediction_horizon), Fx)
        Fu = sparse.kron(sparse.eye(self.prediction_horizon), Fu)

        self._problem.remove(self.constraint_list)


        # self.initial_state_constraint.rhs = x0

        self.constraint_list.append(self._problem.addConstr(
            self.robustness.e_A * np.max(np.abs(x0)) + self.robustness.e_B * self.u_inf[0] + self.robustness.sigma_w  <= self.sigma[
                0]))

        self.constraint_list.append(self._problem.addConstr(Fx @ self.phi_x[:self.prediction_horizon * self.x_vars] == self.x))
        self.constraint_list.append(self._problem.addConstr(Fu @ self.phi_u[:self.prediction_horizon * self.u_vars] == self.u))

        # print(self._model_updated)
        # if self._model_updated:
        #     for constraint in self.dynamics_constraints:
        #         self._problem.remove(constraint)
        #
        #     for k in range(self.prediction_horizon):
        #         self.dynamics_constraints[k] = self._problem.addConstr(self.x[k + 1, :] == self.model._A[k] @ self.x[k, :] + self.model._B2[k] @ self.u[k, :])
        #
        #     self._problem.update()
        #     self._model_updated = False

    def update_model(self, model: LTV_System, initialised: bool) -> None:
        """
        Update the model and check if it was already initialised.

        :param model: LTV model
        :param initialised: Bool whether the problem was already initialised.
        """
        self.model = model

        if initialised:
            self._model_updated = True

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
            controller._Phi_x[t + 2] = self.x.X[t * self._nx:(t+1) * self._nx].reshape((-1, 1))
            controller._Phi_u[t + 1] = self.u.X[t * self._nu:(t+1) * self._nu].reshape((-1, 1))

        controller._Phi_x[self.prediction_horizon + 1] = self.x.X[self.prediction_horizon * self._nx:].reshape((-1, 1))


        return self._problem.Runtime, controller


class Gurobi_Synthesiser_Robust_Simple(SynthesisAlgorithm):

    def __init__(self, number_of_satellites: int, prediction_horizon: int, model: LTV_System,
                 reference_state: np.ndarray, state_limit: np.ndarray, input_limit: np.ndarray, robustness: Robustness,
                 sparse: bool = True):
        super().__init__()

        if sparse:
            self._solver = Gurobi_Solver_Robust_Simple(number_of_satellites, prediction_horizon, model, reference_state,
                                                state_limit, input_limit, robustness)
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
            self._solver.initialise_problem(OutputFlag=0, MIPGap=1e-8)
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

    synthesizer = Gurobi_Synthesiser_Robust_Simple(scenario.number_of_satellites, scenario.control.tFIR, sys, xr,
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
    from Scenarios.RobustScenarios import RobustnessScenarios

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

    synthesizer = Gurobi_Synthesiser_Robust_Simple(scenario.number_of_satellites, scenario.control.tFIR, sys, xr,
                                     dynamics.get_state_constraint(), dynamics.get_input_constraint(), robustness=RobustnessScenarios.simple_robustness_test.value,
                                                   sparse=True)
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
