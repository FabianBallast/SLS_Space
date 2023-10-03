import numpy as np
import osqp
import scipy
from scipy import sparse, spatial
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
                 reference_state: np.ndarray, slack_variables_length: int, slack_variables_cost: list[int],
                 inter_planetary_constraints: bool = False, longitudes: list[float | int] = None,
                 reference_angles: list[float | int] = None, planar_state: list[bool] = None, inter_planar_state: list[bool] = None,
                 inter_planetary_limit: float | int = None, planetary_limit: float | int = None,
                 radial_limit: float | int = None, mean_motion: float = None, sampling_time: float | int = None):
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
        self.state_size = self.x_ref.shape[0] // self.number_of_satellites
        self._slack_variables_length = min(slack_variables_length, self.prediction_horizon)
        self._active_slack_variables = np.arange(0, self.state_size)[np.array(slack_variables_cost) > 0]
        self._slack_variables_cost = np.array(slack_variables_cost)[self._active_slack_variables].tolist()
        self._number_of_slack_variables = len(self._active_slack_variables)
        self._total_number_of_slack_variables = 2 * self._number_of_slack_variables * self._slack_variables_length * self.number_of_satellites

        # (Inter)planetary constraints
        self.add_inter_planetary_constraints = inter_planetary_constraints
        self.longitudes = np.array(longitudes)
        self.reference_angles = reference_angles
        self.planar_state = planar_state
        self.inter_planar_state = inter_planar_state
        self._A_plan = None
        self._A_inter_plan = None
        self._l_plan = None
        self._u_plan = None
        self._l_inter_plan = None
        self._u_inter_plan = None

        self.planetary_limit = planetary_limit
        self.inter_planetary_limit = inter_planetary_limit
        self.radial_limit = radial_limit
        self.mean_motion = mean_motion
        self.sampling_time = sampling_time
        self.state_limit = None
        self.input_limit = None

        self.P_base = None
        self.q_base = None
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
        self._l[:self._nx] = -x0
        self._u[:self._nx] = -x0

        if self.add_inter_planetary_constraints:
            self.find_planetary_constraints()
            self.find_inter_planetary_constraints(time_since_start)

            A_ineq, l_ineq, u_ineq, inter_plan_constraints = self.find_A_ineq()
            self._A = sparse.vstack([self.find_A_eq(), A_ineq], format='csc')

            leq = np.zeros((self.prediction_horizon + 1) * self._nx)
            leq[:self._nx] = -x0
            ueq = leq

            self._l = np.hstack([leq, l_ineq])
            self._u = np.hstack([ueq, u_ineq])

            self._P = sparse.block_diag([self.P_base, sparse.csc_matrix((inter_plan_constraints, inter_plan_constraints))], format='csc')
            self._q = np.hstack([self.q_base, np.ones((inter_plan_constraints, )) * self._slack_variables_cost[0]])

            self._problem = osqp.OSQP()
            self._problem.setup(self._P, self._q, self._A, self._l, self._u, **self._settings)
        elif self._model_updated:
            self._A = sparse.vstack([self.find_A_eq(), self._A_ineq], format='csc')
            self._problem = osqp.OSQP()
            self._problem.setup(self._P, self._q, self._A, self._l, self._u, **self._settings)
            self._model_updated = False
        else:
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
        """
        Solve the problem and return the result + the controller.

        :return: Objective value and feedback controller.
        """
        self.result = self._problem.solve()

        if self.result.info.status_val != 1:
            raise Exception(f"Could not find a solution! {self.result.info.status}")

        controller = SLS_StateFeedback_FIR_Controller(Nx=self._nx, Nu=self._nu, FIR_horizon=self.prediction_horizon)
        controller._Phi_x = [None] * (self.prediction_horizon + 2)
        controller._Phi_u = [None] * (self.prediction_horizon + 1)
        states = self.result.x[:(self.prediction_horizon + 1) * self._nx].reshape((-1, 1))
        inputs = self.result.x[(self.prediction_horizon + 1) * self._nx:
                               (self.prediction_horizon + 1) * self._nx +
                               self.prediction_horizon * self._nu].reshape((-1, 1))

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
        pass

    @abstractmethod
    def find_A_ineq(self):
        """
        Find the inequality constraints for a given model.

        :return: Sparse matrix with A_ineq
        """
        pass

    def find_planetary_constraints(self) -> None:
        """
        Find the constraints that should hold within the plane.
        """
        # Find satellites within the same plane
        # Absolute longitudes
        inter_planar_variables = self.model._x0.flatten()[self.inter_planar_state * self.number_of_satellites] + self.longitudes

        close_satellites = spatial.distance.cdist(inter_planar_variables.reshape((-1, 1)), inter_planar_variables.reshape((-1, 1)), metric='minkowski', p=1) < self.inter_planetary_limit
        indices = sparse.lil_matrix(close_satellites).rows

        satellites_within_same_plane = [indices[0]]
        for set_plane in indices[1:]:
            set_added = False
            for idx, planar_set in enumerate(satellites_within_same_plane):
                if len(np.intersect1d(set_plane, planar_set)) > 0:
                    satellites_within_same_plane[idx] = np.union1d(set_plane, planar_set)
                    set_added = True
                    break
            if not set_added:
                satellites_within_same_plane.append(set_plane)

        # For these satellites, find the constraint in terms of their relative angles with respect to the reference
        planetary_constraints = []
        for set in satellites_within_same_plane:
            if len(set) > 1:
                planar_variables = self.model._x0.flatten()[self.planar_state * self.number_of_satellites][set] + self.reference_angles[set, 0]
                set = np.array(set)[np.argsort(planar_variables)].tolist()
                min_angular_separation = self.planetary_limit
                reference_difference = self.reference_angles[set] - np.roll(self.reference_angles[set], 1)
                reference_difference[reference_difference < 0] += 2 * np.pi
                planetary_constraints.append(min_angular_separation - reference_difference)

        # Save constraints
        self._A_plan = sparse.csc_matrix((0, (self.prediction_horizon + 1) * self._nx))
        self._l_plan = np.zeros((0, 1))

        skipped = 0
        for idx, set in enumerate(satellites_within_same_plane):
            if len(set) > 1:
                set_full = np.array(set) * self.state_size + np.arange(0, self.state_size)[self.planar_state][0] +\
                       np.arange(0, self.prediction_horizon + 1).reshape((-1, 1)) * self._nx
                set_roll = np.roll(set_full, 1, axis=1)[1:].flatten()
                set_full = set_full[1:].flatten()
                rows = np.arange(0, len(set_full))
                A_part_pos = sparse.csc_array((np.ones_like(rows), (rows, set_full)), shape=(len(set_full), (self.prediction_horizon + 1) * self._nx))
                A_part_neg = sparse.csc_array((-np.ones_like(rows), (rows, set_roll)),
                                              shape=(len(set_roll), (self.prediction_horizon + 1) * self._nx))
                self._A_plan = sparse.vstack([self._A_plan, A_part_pos + A_part_neg])
                self._l_plan = np.vstack([self._l_plan, np.tile(planetary_constraints[idx - skipped], (self.prediction_horizon, 1))])
            else:
                skipped += 1

        self._l_plan = self._l_plan.flatten()
        self._u_plan = np.ones_like(self._l_plan) * np.inf

    def find_inter_planetary_constraints(self, time_since_start: float | int) -> None:
        """
        Find the constraints that should hold between satellites in different planes.

        :param time_since_start: Time since the start of the simulation in s.
        """
        # Find satellites within different planes that are close or will be close (near crossing)
        current_pos = self.mean_motion * time_since_start + np.array(self.reference_angles).flatten() + self.model._x0.flatten()[self.planar_state * self.number_of_satellites]
        time_at_0 = (2 * np.pi - current_pos % (2 * np.pi) ) / self.mean_motion
        time_at_180 = ((np.pi - current_pos) % (2 * np.pi) ) / self.mean_motion

        satellites_that_clash_0 = spatial.distance.cdist(time_at_0.reshape((-1, 1)), time_at_0.reshape((-1, 1)), metric='minkowski', p=1)
        satellites_that_clash_180 = spatial.distance.cdist(time_at_180.reshape((-1, 1)), time_at_180.reshape((-1, 1)), metric='minkowski', p=1)
        clash_time = self.planetary_limit / self.mean_motion
        satellites_that_clash_0 = satellites_that_clash_0 < clash_time
        satellites_that_clash_180 = satellites_that_clash_180 < clash_time

        # For these satellites, find the constraint in terms of their relative angles with respect to the reference
        indices_0 = sparse.lil_matrix(satellites_that_clash_0).rows
        indices_180 = sparse.lil_matrix(satellites_that_clash_180).rows

        satellite_clashes_0 = [indices_0[0]]
        for set_0 in indices_0[1:]:
            set_added = False
            for idx, planar_set in enumerate(satellite_clashes_0):
                if len(np.intersect1d(set_0, planar_set)) > 0:
                    satellite_clashes_0[idx] = np.union1d(set_0, planar_set)
                    set_added = True
                    break
            if not set_added:
                satellite_clashes_0.append(set_0)

        satellite_clashes_180 = [indices_180[0]]
        for set_180 in indices_180[1:]:
            set_added = False
            for idx, planar_set in enumerate(satellite_clashes_180):
                if len(np.intersect1d(set_180, planar_set)) > 0:
                    satellite_clashes_180[idx] = np.union1d(set_180, planar_set)
                    set_added = True
                    break
            if not set_added:
                satellite_clashes_180.append(set_180)

        # For these satellites, find the constraint in terms of their relative angles with respect to the reference
        inter_planetary_constraints_0 = []
        radius = np.array([True] + [False] * (self.state_size - 1)).tolist()
        for set in satellite_clashes_0:
            if len(set) > 1:
                # Find radius for each sat in set
                radius_variables = self.model._x0.flatten()[radius * self.number_of_satellites][set]

                # Do a quick sort for rough rearrangment
                set = np.array(set)[np.argsort(radius_variables)].tolist()
                radius_variables = self.model._x0.flatten()[radius * self.number_of_satellites][set]

                if self.radial_limit > 0:
                    # Find distance
                    radius_distance = spatial.distance.cdist(radius_variables.reshape((-1, 1)), radius_variables.reshape((-1, 1)), metric='minkowski', p=1)
                    indices = sparse.lil_matrix(radius_distance < self.radial_limit).rows

                    # Make sets with same radius
                    radial_sets = [indices[0]]
                    for test_set in indices[1:]:
                        set_added = False
                        for idx, planar_set in enumerate(radial_sets):
                            if len(np.intersect1d(test_set, planar_set)) > 0:
                                radial_sets[idx] = np.union1d(test_set, planar_set).tolist()
                                set_added = True
                                break
                        if not set_added:
                            radial_sets.append(test_set)

                    # Order sets themselves
                    ordered_set = []
                    for radial_set in radial_sets:
                        angles = self.model._x0.flatten()[self.planar_state * self.number_of_satellites][set][radial_set]
                        ordered_set.append(np.array(radial_set)[np.argsort(angles).tolist()])

                    set_indices = [x for l in ordered_set for x in l]
                    set = np.array(set)[set_indices].tolist()
                inter_planetary_constraints_0.append([self.radial_limit] * (len(set) - 1))

        inter_planetary_constraints_180 = []
        for set in satellite_clashes_180:
            if len(set) > 1:
                # Find radius for each sat in set
                radius_variables = self.model._x0.flatten()[radius * self.number_of_satellites][set]

                # Do a quick sort for rough rearrangment
                set = np.array(set)[np.argsort(radius_variables)].tolist()
                radius_variables = self.model._x0.flatten()[radius * self.number_of_satellites][set]

                if self.radial_limit > 0:
                    # Find distance
                    radius_distance = spatial.distance.cdist(radius_variables.reshape((-1, 1)),
                                                             radius_variables.reshape((-1, 1)), metric='minkowski', p=1)
                    indices = sparse.lil_matrix(radius_distance < self.radial_limit).rows

                    # Make sets with same radius
                    radial_sets = [indices[0]]
                    for test_set in indices[1:]:
                        set_added = False
                        for idx, planar_set in enumerate(radial_sets):
                            if len(np.intersect1d(test_set, planar_set)) > 0:
                                radial_sets[idx] = np.union1d(test_set, planar_set).tolist()
                                set_added = True
                                break
                        if not set_added:
                            radial_sets.append(test_set)

                    # Order sets themselves
                    ordered_set = []
                    for radial_set in radial_sets:
                        angles = self.model._x0.flatten()[self.planar_state * self.number_of_satellites][set][
                            radial_set]
                        ordered_set.append(np.array(radial_set)[np.argsort(angles).tolist()])

                    set_indices = [x for l in ordered_set for x in l]
                    set = np.array(set)[set_indices].tolist()
                inter_planetary_constraints_180.append([self.radial_limit] * (len(set) - 1))

        # Save constraints
        self._A_inter_plan = sparse.csc_matrix((0, (self.prediction_horizon + 1) * self._nx))
        self._l_inter_plan = np.zeros((0,))
        skipped = 0
        for idx, set in enumerate(satellite_clashes_0):
            if len(set) > 1:
                avg_time_at_0 = int(np.mean(time_at_0[set]) // self.sampling_time)
                time_length = 2
                if avg_time_at_0 == 0:  # Ignore values at t==0
                    set_full = np.array(set) * self.state_size + np.arange(avg_time_at_0, avg_time_at_0 + 2).reshape((-1, 1)) * self._nx
                    set_roll = np.roll(set_full, 1, axis=1)[1:, 1:].flatten()
                    set_full = set_full[1:, 1:].flatten()
                    time_length = 1
                elif avg_time_at_0 < self.prediction_horizon:  # Normal case
                    set_full = np.array(set) * self.state_size + np.arange(avg_time_at_0, avg_time_at_0 + 2).reshape(
                        (-1, 1)) * self._nx
                    set_roll = np.roll(set_full, 1, axis=1)[:, 1:].flatten()
                    set_full = set_full[:, 1:].flatten()
                elif avg_time_at_0 == self.prediction_horizon:  # Outside prediction horizon
                    set_full = np.array(set) * self.state_size + np.array(self.prediction_horizon).reshape((-1, 1)) * self._nx
                    set_roll = np.roll(set_full, 1, axis=1)[:, 1:].flatten()
                    set_full = set_full[:, 1:].flatten()
                    time_length = 1
                else:
                    continue

                rows = np.arange(0, len(set_full))
                A_part_pos = sparse.csc_array((np.ones_like(rows), (rows, set_full)),
                                              shape=(len(set_full), (self.prediction_horizon + 1) * self._nx))
                A_part_neg = sparse.csc_array((-np.ones_like(rows), (rows, set_roll)),
                                              shape=(len(set_roll), (self.prediction_horizon + 1) * self._nx))

                self._A_inter_plan = sparse.vstack([self._A_inter_plan, A_part_pos + A_part_neg])
                self._l_inter_plan = np.hstack(
                    [self._l_inter_plan, np.tile(inter_planetary_constraints_0[idx - skipped], (time_length))])
            else:
                skipped += 1

        skipped = 0
        for idx, set in enumerate(satellite_clashes_180):
            if len(set) > 1:
                avg_time_at_180 = int(np.mean(time_at_180[set]) // self.sampling_time)
                time_length = 2
                if avg_time_at_180 == 0:  # Ignore values at t==0
                    set_full = np.array(set) * self.state_size + np.arange(avg_time_at_180, avg_time_at_180 + 2).reshape(
                        (-1, 1)) * self._nx
                    set_roll = np.roll(set_full, 1, axis=1)[1:, 1:].flatten()
                    set_full = set_full[1:, 1:].flatten()
                    time_length = 1
                elif avg_time_at_180 < self.prediction_horizon:  # Normal case
                    set_full = np.array(set) * self.state_size + np.arange(avg_time_at_180, avg_time_at_180 + 2).reshape(
                        (-1, 1)) * self._nx
                    set_roll = np.roll(set_full, 1, axis=1)[:, 1:].flatten()
                    set_full = set_full[:, 1:].flatten()
                elif avg_time_at_180 == self.prediction_horizon:  # Outside prediction horizon
                    set_full = np.array(set) * self.state_size + np.array(self.prediction_horizon).reshape(
                        (-1, 1)) * self._nx
                    set_roll = np.roll(set_full, 1, axis=1)[:, 1:].flatten()
                    set_full = set_full[:, 1:].flatten()
                    time_length = 1
                else:
                    continue

                rows = np.arange(0, len(set_full))
                A_part_pos = sparse.csc_array((np.ones_like(rows), (rows, set_full)),
                                              shape=(len(set_full), (self.prediction_horizon + 1) * self._nx))
                A_part_neg = sparse.csc_array((-np.ones_like(rows), (rows, set_roll)),
                                              shape=(len(set_roll), (self.prediction_horizon + 1) * self._nx))
                self._A_inter_plan = sparse.vstack([self._A_inter_plan, A_part_pos + A_part_neg])
                self._l_inter_plan = np.hstack(
                    [self._l_inter_plan, np.tile(inter_planetary_constraints_180[idx - skipped], (time_length))])
            else:
                skipped += 1
        self._l_inter_plan = self._l_inter_plan.flatten()
        self._u_inter_plan = np.ones_like(self._l_inter_plan) * np.inf

class OSQP_Solver_Sparse(OSQP_Solver):
    """
    Sparse OSQP solver.
    """
    def set_cost_matrices(self, Q_sqrt: np.ndarray, R_sqrt: np.ndarray) -> None:
        """
        Set the cost matrices for the solver.

        :param Q_sqrt: Square root of state cost matrix.
        :param R_sqrt: Square root of input cost matrix.
        """
        Q = sparse.kron(sparse.eye(self.number_of_satellites),
                        sparse.csc_matrix(Q_sqrt).power(2))
        QN = Q
        R = sparse.kron(sparse.eye(self.number_of_satellites),
                        sparse.csc_matrix(R_sqrt).power(2))

        self._P = sparse.block_diag([sparse.kron(sparse.eye(self.prediction_horizon), Q), QN,
                                     sparse.kron(sparse.eye(self.prediction_horizon), R),
                                     sparse.csc_matrix((self._total_number_of_slack_variables,
                                                        self._total_number_of_slack_variables))], format='csc')
        # - linear objective
        sparse_costs = self._slack_variables_cost * self.number_of_satellites * self._slack_variables_length * 2
        self._q = np.hstack([np.kron(np.ones(self.prediction_horizon), -Q.dot(self.x_ref)), -QN.dot(self.x_ref),
                             np.zeros(self.prediction_horizon * self._nu), np.array(sparse_costs)])

        # self.P_base = self._P
        # self.q_base = self._q

    def set_constraints(self, state_limit: list, input_limit: list) -> None:
        """
        Set the state and input constraints.

        :param state_limit: Maximum state values per satellite in list-form.
        :param input_limit: Maximum input values per satellite in list-form.
        """
        self.state_limit = state_limit
        self.input_limit = input_limit

        # Linear dynamics
        leq = np.zeros((self.prediction_horizon + 1) * self._nx)  # Set x0 later (fist nx elements)
        ueq = leq

        self._A_ineq, l_ineq, u_ineq, _ = self.find_A_ineq()

        # - OSQP constraints
        self._A = sparse.vstack([self.find_A_eq(), self._A_ineq], format='csc')
        self._l = np.hstack([leq, l_ineq])
        self._u = np.hstack([ueq, u_ineq])

    def update_model(self, model: LTV_System, initialised: bool) -> None:
        """
        Update the model and check if it was already initialised.

        :param model: LTV model
        :param initialised: Bool whether the problem was already initialised.
        """
        self.model = model

        if initialised:
            self._model_updated = True
            # self._problem.update(Ax=self._A.data)
            # self._problem = osqp.OSQP()
            # self._problem.setup(self._P, self._q, self._A, self._l, self._u, **self._settings)

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

        if self._A_inter_plan is not None:
            sparse_variables = sparse.csc_matrix((Ax.shape[0], self._total_number_of_slack_variables + self._A_inter_plan.shape[0]))
        else:
            sparse_variables = sparse.csc_matrix((Ax.shape[0], self._total_number_of_slack_variables))

        return sparse.hstack([Ax, Bu, sparse_variables])

    def find_A_ineq(self):
        """
        Find the inequality matrix for a given model.

        :return: Sparse matrix with A_ineq, l_ineq and u_ineq
        """
        # Limit Constraints
        xmin = -np.array(self.state_limit * self.number_of_satellites)
        xmax = np.array(self.state_limit * self.number_of_satellites)
        umin = -np.array(self.input_limit * self.number_of_satellites)
        umax = np.array(self.input_limit * self.number_of_satellites)

        slack_min = np.zeros((self._total_number_of_slack_variables,))
        slack_max = np.ones_like(slack_min) * 1

        # input and state constraints
        slack_selection = sparse.kron(sparse.eye(self.number_of_satellites * self._slack_variables_length),
                                      sparse.csc_matrix(np.eye(self.state_size)[:, self._active_slack_variables]))

        regular_ineq = sparse.eye((self.prediction_horizon + 1) * self._nx + self.prediction_horizon * self._nu)

        slack_variables_rhs = sparse.vstack(
            [sparse.csc_matrix((self._nx, self._total_number_of_slack_variables)),  # x0 does not need slack variables
             sparse.hstack([-slack_selection, slack_selection]),
             sparse.csc_matrix((self.prediction_horizon * self._nu + (self.prediction_horizon - self._slack_variables_length) * self._nx,
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

        if self._A_plan is not None:
            self._A_ineq = sparse.vstack([self._A_ineq,
                                          sparse.hstack([self._A_plan,
                                                         sparse.csc_matrix((self._A_plan.shape[0],
                                                                            self.prediction_horizon * self._nu +
                                                                            self._total_number_of_slack_variables))])])
            lineq = np.hstack([lineq, self._l_plan])
            uineq = np.hstack([uineq, self._u_plan])

        inter_plan_constraints = 0
        if self._A_inter_plan is not None:
            self._A_ineq = sparse.vstack([self._A_ineq,
                                          sparse.hstack([self._A_inter_plan,
                                                         sparse.csc_matrix((self._A_inter_plan.shape[0],
                                                                            self.prediction_horizon * self._nu +
                                                                            self._total_number_of_slack_variables))])])
            lineq = np.hstack([lineq, self._l_inter_plan])
            uineq = np.hstack([uineq, self._u_inter_plan])

            # Add slack variables for rho
            inter_plan_constraints = self._l_inter_plan.shape[0]
            self._A_ineq = sparse.hstack([self._A_ineq,
                                          sparse.vstack([sparse.csc_matrix((self._A_ineq.shape[0] - inter_plan_constraints, inter_plan_constraints)),
                                                         -sparse.eye(inter_plan_constraints)])])
            self._A_ineq = sparse.vstack([self._A_ineq,
                                          sparse.hstack([sparse.csc_matrix((inter_plan_constraints, self._A_ineq.shape[1] - inter_plan_constraints)),
                                                        sparse.eye(inter_plan_constraints)])])
            # self._A_eq = sparse.hstack([self._A_eq, sparse.csc_matrix((self._A_eq.shape[0], number_of_constraints))])
            lineq = np.hstack([lineq, np.zeros(inter_plan_constraints)])
            uineq = np.hstack([uineq, np.ones(inter_plan_constraints)])

        # Ignore constraints for x0 (as it is already given, and can only make problem unfeasible if start is not allowed
        lineq[:self._nx] = -np.inf
        uineq[:self._nx] = np.inf

        return self._A_ineq, lineq, uineq, inter_plan_constraints

# class OSQP_Solver_Dense(OSQP_Solver):
#
#     def set_cost_matrices(self, Q_sqrt: np.ndarray, R_sqrt: np.ndarray) -> None:
#         Q = np.kron(np.eye(self.number_of_satellites), Q_sqrt ** 2)
#         QN = Q
#         R = np.kron(np.eye(self.number_of_satellites), R_sqrt ** 2)
#
#         self._P = scipy.linalg.block_diag(np.kron(np.eye(self.prediction_horizon), Q), QN,
#                                           np.kron(np.eye(self.prediction_horizon), R),
#                                           np.zeros((self._total_number_of_slack_variables,
#                                                     self._total_number_of_slack_variables)))
#         # - linear objective
#         sparse_costs = self._slack_variables_cost * self.number_of_satellites * self._slack_variables_length * 2
#         self._q = np.hstack([np.kron(np.ones(self.prediction_horizon), -Q.dot(self.x_ref)), -QN.dot(self.x_ref),
#                              np.zeros(self.prediction_horizon * self._nu), np.array(sparse_costs)])
#
#     def set_constraints(self, state_limit: list, input_limit: list):
#         # Limit Constraints
#         xmin = -np.array(state_limit * self.number_of_satellites)
#         xmax = np.array(state_limit * self.number_of_satellites)
#         umin = -np.array(input_limit * self.number_of_satellites)
#         umax = np.array(input_limit * self.number_of_satellites)
#
#         slack_min = np.zeros((self._total_number_of_slack_variables,))
#         slack_max = np.ones_like(slack_min) * 5
#
#         # Linear dynamics
#         leq = np.zeros((self.prediction_horizon + 1) * self._nx)  # Set x0 later (fist nx elements)
#         ueq = leq
#
#         # - input and state constraints
#         regular_ineq = np.eye((self.prediction_horizon + 1) * self._nx + self.prediction_horizon * self._nu)
#         slack_variables_rhs = np.vstack([scipy.linalg.block_diag(-np.eye(self._total_number_of_slack_variables // 2),
#                                                                  np.eye(
#                                                                      self._total_number_of_slack_variables // 2)),
#                                          np.zeros(
#                                              (regular_ineq.shape[0] - self._total_number_of_slack_variables,
#                                               self._total_number_of_slack_variables))])
#
#         slack_variables_bottom = np.hstack([np.zeros((self._total_number_of_slack_variables,
#                                                       regular_ineq.shape[1])),
#                                             np.eye(self._total_number_of_slack_variables)])
#
#         self._A_ineq = np.vstack([np.hstack([regular_ineq, slack_variables_rhs]), slack_variables_bottom])
#
#         lineq = np.hstack([np.kron(np.ones(self.prediction_horizon + 1), xmin),
#                            np.kron(np.ones(self.prediction_horizon), umin),
#                            slack_min])
#         uineq = np.hstack([np.kron(np.ones(self.prediction_horizon + 1), xmax),
#                            np.kron(np.ones(self.prediction_horizon), umax),
#                            slack_max])
#
#         # Ignore constraints for x0 (as it is already given, and can only make problem unfeasible if start is not allowed
#         lineq[:self._nx] = -np.inf
#         uineq[:self._nx] = np.inf
#
#         # - OSQP constraints
#         self._A = np.vstack([self.find_A_eq(), self._A_ineq])
#         self._l = np.hstack([leq, lineq])
#         self._u = np.hstack([ueq, uineq])
#
#     def update_model(self, model: LTV_System, initialised: bool) -> None:
#         self.model = model
#
#         self._A = np.vstack([self.find_A_eq(), self._A_ineq])
#
#         if initialised:
#             # self._problem.update(Ax=self._A.data)
#             self._problem = osqp.OSQP()
#             self._problem.setup(self._P, self._q, self._A, self._l, self._u, **self._settings)
#
#     def find_A_eq(self):
#         Ax = np.kron(np.eye(self.prediction_horizon + 1), -np.eye(self._nx)) + \
#              np.vstack([np.zeros((self._nx, self._nx * (self.prediction_horizon + 1))),
#                         np.hstack([scipy.linalg.block_diag(*self.model._A),
#                                    np.zeros((self._nx * self.prediction_horizon, self._nx))])])
#
#         sparse_B2 = [(self.model._B2[i]) for i in range(self.prediction_horizon)]
#         Bu = np.vstack([np.zeros((self._nx, self.prediction_horizon * self._nu)),
#                         scipy.linalg.block_diag(*sparse_B2)])
#
#         sparse_variables = np.zeros((Ax.shape[0], self._total_number_of_slack_variables))
#
#         return np.hstack([Ax, Bu, sparse_variables])


class OSQP_Synthesiser(SynthesisAlgorithm):

    def __init__(self, number_of_satellites: int, prediction_horizon: int, model: LTV_System,
                 reference_state: np.ndarray, sparse_variables_length: int, sparse_variables_costs: list[int],
                 sparse: bool = True, inter_planetary_constraints: bool = False, longitudes: list[float | int] = None,
                 reference_angles: list[float | int] = None, planar_state: list[bool] = None, inter_planar_state: list[bool] = None,
                 inter_planetary_limit: float | int = None, planetary_limit: float | int = None,
                 radial_limit: float | int = None, mean_motion: float = None, sampling_time: float | int = None):
        super().__init__()

        if sparse:
            self._solver = OSQP_Solver_Sparse(number_of_satellites, prediction_horizon, model, reference_state,
                                              sparse_variables_length, sparse_variables_costs,
                                              inter_planetary_constraints, longitudes, reference_angles, planar_state,
                                              inter_planar_state, inter_planetary_limit, planetary_limit, radial_limit,
                                              mean_motion, sampling_time)
        # else:
        #     self._solver = OSQP_Solver_Dense(number_of_satellites, prediction_horizon, model, reference_state,
        #                                      sparse_variables_length, sparse_variables_costs)
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
                                  time_since_start: int = 0) -> SLS_StateFeedback_FIR_Controller:
        """
        Synthesise the controller and find the optimal inputs.

        :param x_ref: Not used here. Does not do anything.
        :param time_since_start: Time since start of the simulation in s.
        """
        if not self.initialised:
            # self._solver.initialise_problem(warm_start=True, verbose=False)
            self._solver.initialise_problem(warm_start=True, verbose=False, polish=False, check_termination=10,
                                            eps_abs=1e-5, eps_rel=1e-5, max_iter=500000)
            self.initialised = True

        self._solver.update_x0(time_since_start)
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
