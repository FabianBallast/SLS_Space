import numpy as np
import gurobipy as gp

from Utils.CollisionAngles import find_collision_angle_vect
from slspy import SynthesisAlgorithm, LTV_System, SLS_StateFeedback_FIR_Controller
from abc import ABC, abstractmethod
import time
from Utils.Conversions import find_delta_Omega
from Utils.OmegaPrediction import predict_Omega
from Utils.CollisionMatrices import update_collision_vector
import scipy.sparse as sparse

class Gurobi_Solver(ABC):
    """
    Class to use OSQP solver for SLS purposes.
    """

    def __init__(self, number_of_satellites: int, prediction_horizon: int, model: LTV_System,
                 reference_state: np.ndarray, state_limit: np.ndarray, input_limit: np.ndarray,
                 in_plane_collision_setup=None, out_plane_collision_setup=None, omega_difference_start=None):
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

        if in_plane_collision_setup is not None:
            self.in_plane_collision_setup = in_plane_collision_setup
            in_plane_collision_A, in_plane_collision_b = in_plane_collision_setup
            for k in range(self.prediction_horizon):
                self._problem.addConstr(in_plane_collision_A @ self.x[k+1, :] >= in_plane_collision_b.flatten())

        if out_plane_collision_setup is not None:
            self.out_plane_collision_setup = True
            self.collision_matrix = out_plane_collision_setup[0]
            self.collision_matrix_large = np.kron(np.eye(self.prediction_horizon), self.collision_matrix)
            self.collision_vector = np.kron(np.ones(self.prediction_horizon), out_plane_collision_setup[1])
            self.safety_margin = out_plane_collision_setup[2]
            self.theta_reff_diff = np.kron(np.ones((self.prediction_horizon, )), out_plane_collision_setup[3])
            self.Omega_reff_diff = np.kron(np.ones(self.prediction_horizon), out_plane_collision_setup[4])

            self.omega_difference_start = omega_difference_start

            direction = np.zeros_like(self.omega_difference_start)
            entries2check = np.abs(self.omega_difference_start) > 0.00001
            direction[entries2check] = -self.omega_difference_start[entries2check] / np.abs(self.omega_difference_start[entries2check])

            self.last_collision_vector_Omega = self.collision_matrix_large @ np.kron(np.ones(self.prediction_horizon), self.omega_difference_start).reshape((-1, 1))

            self.delta_Omega_prediction = predict_Omega(self.omega_difference_start.reshape((1, -1)),
                                                        direction=direction.reshape((-1,)),
                                                        time_steps=self.prediction_horizon)

            Omega_differences_new = self.collision_matrix_large @ self.delta_Omega_prediction.reshape((-1, 1))

            self.collision_vector, self.last_collision_vector_Omega = update_collision_vector(
                self.collision_vector, Omega_differences_new, self.last_collision_vector_Omega, self.Omega_reff_diff)

            self.number_of_constraints = self.collision_matrix.shape[0]
            # self.theta_abs = self._problem.addMVar(self.number_of_constraints * self.prediction_horizon, name='theta_abs')
            # print(f"Number of constraints: {self.number_of_constraints}")
            # self.r_pos = self._problem.addMVar(self.number_of_constraints * self.prediction_horizon, name='r_pos')
            # self.r_neg = self._problem.addMVar(self.number_of_constraints * self.prediction_horizon, name='r_neg')
            # self.theta_pos = self._problem.addMVar(self.number_of_constraints * self.prediction_horizon, name='theta_pos')
            # self.theta_neg = self._problem.addMVar(self.number_of_constraints * self.prediction_horizon, name='theta_neg')

            self.lambda_selection_matrix = np.kron(np.eye(prediction_horizon), np.kron(self.collision_matrix,
                                                                                  np.array([0, 1, 0, 0, 0, 0])))
            self.Omega_selection_matrix = np.cos(np.pi / 4) * np.kron(np.eye(prediction_horizon), self.collision_matrix)
            self.r_selection_matrix = sparse.kron(sparse.eye(self.prediction_horizon),
                                             sparse.kron(self.collision_matrix, np.array([1, 0, 0, 0, 0, 0])))

            # self._problem.addConstr(r_selection_matrix @ self.x[1:].reshape((-1, )) == self.r_pos - self.r_neg)

            # self.Omega_constraint = self._problem.addConstr(self.lambda_selection_matrix @ self.x[1:].reshape((-1, )) -
            #                                            self.Omega_selection_matrix @ self.delta_Omega_prediction.reshape((-1, )) +
            #                                            self.theta_reff_diff - self.collision_vector == self.theta_pos - self.theta_neg)
            self.out_plane_constraint = None
            self.out_of_plane_dist_constraint_added = False
            self.active_constraints = []
            self.active_constraints_indices = None
            self.active_constraints_changed = False
            self.theta_comp = None
            self.radius_comp = None
        else:
            self.out_plane_collision_setup = None

        self._model_updated = False
        self.alpha = 0.2

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
        # print(self.in_plane_collision_setup[0] @ x0.reshape((-1, 1)) >= self.in_plane_collision_setup[1])
        # print(self.in_plane_collision_setup[0][11] @ x0.reshape((-1, 1)))
        # print(self.in_plane_collision_setup[1][11])
        # print(self._model_updated)
        if self.out_plane_collision_setup and not self.out_of_plane_dist_constraint_added:
            lambda_diff = np.kron(np.ones(self.prediction_horizon), np.kron(self.collision_matrix, np.array([0, 1, 0, 0, 0, 0])) @ x0)
            self.lambda_sign = lambda_diff / np.abs(lambda_diff)

            theta_diff = self.lambda_selection_matrix @ np.kron(np.ones(self.prediction_horizon), x0) - \
                                                       self.Omega_selection_matrix @ self.delta_Omega_prediction.reshape((-1, )) + \
                                                       self.theta_reff_diff - self.collision_vector

            theta_des = theta_diff % (2 * np.pi)
            theta_des -= (theta_des > np.pi) * 2 * np.pi
            theta_abs = np.abs(theta_des)
            self.theta_comp = theta_abs - theta_diff

            # radius_diff = self.r_selection_matrix @ np.kron(np.ones(self.prediction_horizon), x0)
            # self.radius_comp = -2 * radius_diff.copy()
            # self.radius_comp[self.radius_comp < 0] = 0

            self.out_plane_constraint = self._problem.addConstr(
                (self.r_selection_matrix @ self.x[1:].reshape((-1,))) * self.lambda_sign + self.alpha * (self.lambda_selection_matrix @ self.x[1:].reshape((-1, )) -
                                                       self.Omega_selection_matrix @ self.delta_Omega_prediction.reshape((-1, )) +
                                                       self.theta_reff_diff - self.collision_vector + self.theta_comp) >= np.kron(np.ones(self.prediction_horizon), self.safety_margin))
            self.out_of_plane_dist_constraint_added = True

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

        if self.out_plane_collision_setup is not None and self.number_of_constraints > 0:
            # lambda_selection_matrix = np.kron(np.eye(self.prediction_horizon), np.kron(self.collision_matrix,
            #                                                                       np.array([0, 1, 0, 0, 0, 0])))
            # delta_lambda = lambda_selection_matrix @ self.x.X[1:].reshape((-1,))
            # delta_Omega = self.Omega_selection_matrix @ self.delta_Omega_prediction.reshape((-1,))
            # delta_theta = delta_lambda - delta_Omega + self.theta_reff_diff
            #
            # print(np.rad2deg(delta_lambda).reshape((20, -1)))
            # print(np.rad2deg(delta_Omega).reshape((20, -1)))
            # print(np.rad2deg(delta_theta).reshape((20, -1)))
            # print(np.rad2deg(self.collision_vector).reshape((20, -1)))
            # print()

            r_selection_matrix = sparse.kron(sparse.eye(self.prediction_horizon),
                                             sparse.kron(self.collision_matrix, np.array([1, 0, 0, 0, 0, 0])))
            cons_value = (r_selection_matrix @ self.x.X[1:].reshape((-1,))) * self.lambda_sign + self.alpha * (
                        self.lambda_selection_matrix @ self.x.X[1:].reshape((-1,)) -
                        self.Omega_selection_matrix @ self.delta_Omega_prediction.reshape((-1,)) +
                        self.theta_reff_diff - self.collision_vector + self.theta_comp)

            angle_part = self.lambda_selection_matrix @ self.x.X[1:].reshape((-1,)) - \
                         self.Omega_selection_matrix @ self.delta_Omega_prediction.reshape((-1,)) + \
                         self.theta_reff_diff - self.collision_vector + self.theta_comp

            angle_signed = self.lambda_selection_matrix @ self.x.X[1:].reshape((-1,)) - \
                         self.Omega_selection_matrix @ self.delta_Omega_prediction.reshape((-1,)) + \
                         self.theta_reff_diff - self.collision_vector
            # print(np.min(cons_value))
            # print(np.min((sparse.kron(self.collision_matrix, np.array([1, 0, 0, 0, 0, 0])) @ self.x.X[1].reshape((-1,))) * self.lambda_sign[:self.number_of_constraints]))
            # print()
            # print((sparse.kron(self.collision_matrix, np.array([1, 0, 0, 0, 0, 0])) @ self.x.X[1].reshape((-1,))))
            #
            # true_abs = np.abs(self.lambda_selection_matrix @ self.x.X[1:].reshape((-1,)) - self.Omega_selection_matrix @ self.delta_Omega_prediction.reshape((-1,)) + self.theta_reff_diff - self.collision_vector)
            # comp_abs = self.lambda_selection_matrix @ self.x.X[1:].reshape((-1,)) - self.Omega_selection_matrix @ self.delta_Omega_prediction.reshape((-1,)) + self.theta_reff_diff - self.collision_vector + self.theta_comp
            # print(np.rad2deg(np.min(true_abs - comp_abs)), np.rad2deg(np.max(true_abs - comp_abs)))
            # print(np.rad2deg(np.min(true_abs)), np.rad2deg(np.min(comp_abs)))
            # print()
            # print(np.round(np.rad2deg(true_abs[:20]), 1))
            # print(np.round(np.rad2deg(comp_abs[:20]), 1))

            # radial_diff = self.collision_matrix @ self.x.X[0, 0::6].reshape((-1, 1))  # r2 - r1
            # Omega_vals = find_delta_Omega(self.x.X[0:1], self.omega_difference_start)  # [Omega_0 - Omega_0_ref, Omega_1 - Omega_1_ref, ...]
            # Omega_diff = self.collision_matrix @ Omega_vals.reshape((-1, 1))  # Omega_2 - Omega_1 - Omega_2_ref + Omega_1_ref
            # coll_angles = find_collision_angle_vect(np.pi / 4, Omega_diff + self.Omega_reff_diff[:self.number_of_constraints].reshape((-1, 1))) % (2 * np.pi)
            # coll_angles -= (coll_angles > np.pi) * 2 * np.pi
            # coll_angles = coll_angles.reshape((-1, 1))

            # print(np.rad2deg(Omega_vals).flatten())
            # print(np.rad2deg(Omega_diff).flatten())
            # print(np.rad2deg(coll_angles).flatten())
            # print((self.collision_matrix @ self.x.X[1, 1::6].reshape((-1, 1))).shape)
            # print(Omega_diff.shape)
            # print(self.theta_reff_diff[:self.number_of_constraints].shape)
            # print(coll_angles.shape)
            # print(Omega_vals)
            # print(np.abs(self.collision_matrix[19]) > 0.1)
            # print(np.arange(30)[np.abs(self.collision_matrix[19]) > 0.1])
            # print(f"Omega diff start {self.omega_difference_start.flatten()[np.abs(self.collision_matrix[19]) > 0.1]}")
            # print(f"Omega vals: {Omega_vals.flatten()[np.abs(self.collision_matrix[19]) > 0.1]}")
            # print(f"Omega diff: {Omega_diff[19]}")
            # print(f"Omega reff diff: {self.Omega_reff_diff[19].flatten()}")
            # print(f"Omega_abs: {(Omega_diff + self.Omega_reff_diff[:self.number_of_constraints].reshape((-1, 1)))[19]}")
            # print(f'sats: {np.arange(30)[np.abs(self.collision_matrix[19])> 0.1]}')
            # print(f"delta lambda: {(self.collision_matrix @ self.x.X[0, 1::6].reshape((-1, 1)))[19]}")
            # print(f"delta theta_ref: {self.theta_reff_diff[19]}")
            # print(f'delta Omega_diff: {np.cos(np.pi / 4) * Omega_diff[19]}')
            # print(f"coll angles: {coll_angles[19]}")
            # print(f"delta theta: {(self.collision_matrix @ self.x.X[0, 1::6].reshape((-1, 1)))[19] - np.cos(np.pi / 4) * Omega_diff[19] + self.theta_reff_diff[19]}")


            # theta_diff = (self.collision_matrix @ self.x.X[0, 1::6].reshape((-1, 1)) - np.cos(np.pi / 4) * Omega_diff + self.theta_reff_diff[:self.number_of_constraints].reshape((-1, 1)) - coll_angles) % (2 * np.pi)
            # theta_diff -= (theta_diff > np.pi) * 2 * np.pi
            #
            # # print(theta_diff.shape)
            # constraint_value = np.abs(radial_diff[23]).flatten() + self.alpha * np.abs(theta_diff[23]).flatten()
            #
            # if constraint_value < 0.015:
            #     r_vals_true = []
            #     r_vals_estimate = []
            #     theta_value_true = []
            #     theta_value_estimate = []
            #     theta_value_signed = []
            #     constraint_value_true = []
            #     constraint_value_estimate = []
            #     Omega_true = []
            #     Omega_est = []
            #
            #     coll_angle_true = []
            #     coll_angle_est = []
            #
            #     for t in range(1, 5):
            #         radial_diff = self.collision_matrix @ self.x.X[t, 0::6].reshape((-1, 1))
            #         Omega_vals = find_delta_Omega(self.x.X[t:t+1],
            #                                       self.omega_difference_start)  # [Omega_0 - Omega_0_ref, Omega_1 - Omega_1_ref, ...]
            #         Omega_diff = self.collision_matrix @ Omega_vals.reshape(
            #             (-1, 1))  # Omega_2 - Omega_1 - Omega_2_ref + Omega_1_ref
            #         coll_angles = find_collision_angle_vect(np.pi / 4, Omega_diff + self.Omega_reff_diff[
            #                                                                         :self.number_of_constraints].reshape(
            #             (-1, 1))) % (2 * np.pi)
            #         coll_angles -= (coll_angles > np.pi) * 2 * np.pi
            #         coll_angles = coll_angles.reshape((-1, 1))
            #
            #         theta_diff = (self.collision_matrix @ self.x.X[t, 1::6].reshape((-1, 1)) - np.cos(
            #             np.pi / 4) * Omega_diff + self.theta_reff_diff[:self.number_of_constraints].reshape(
            #             (-1, 1)) - coll_angles) % (2 * np.pi)
            #         theta_diff -= (theta_diff > np.pi) * 2 * np.pi
            #
            #         # print(theta_diff.shape)
            #         constraint_value = np.abs(radial_diff[23]).flatten() + self.alpha * np.abs(theta_diff[23]).flatten()
            #
            #         r_vals_true.append(np.abs(radial_diff[23]).flatten()[0])
            #         r_vals_estimate.append((radial_diff[23] * self.lambda_sign[23])[0])
            #
            #         theta_value_true.append(np.abs(theta_diff[23]).flatten()[0])
            #         theta_value_estimate.append(angle_part[23 + (t-1) * self.number_of_constraints])
            #         theta_value_signed.append(angle_signed[23 + (t-1) * self.number_of_constraints])
            #
            #         Omega_true.append(np.cos(np.pi / 4) * Omega_diff[23][0])
            #         Omega_est.append((self.Omega_selection_matrix @ self.delta_Omega_prediction.reshape((-1,)))[23 + (t-1) * self.number_of_constraints])
            #
            #         coll_angle_true.append(coll_angles[23][0])
            #         coll_angle_est.append(self.collision_vector[23 + (t-1) * self.number_of_constraints])
            #
            #         if angle_part[23 + (t-1) * self.number_of_constraints] > 2 * np.pi:
            #             # self.lambda_selection_matrix @ self.x.X[1:].reshape((-1,)) - \
            #             # self.Omega_selection_matrix @ self.delta_Omega_prediction.reshape((-1,)) + \
            #             # self.theta_reff_diff - self.collision_vector + self.theta_comp
            #             idx = 23 + (t-1) * self.number_of_constraints
            #             print((self.lambda_selection_matrix @ self.x.X[1:].reshape((-1,)))[idx])
            #             print((self.Omega_selection_matrix @ self.delta_Omega_prediction.reshape((-1,)))[idx])
            #             print(self.theta_reff_diff[idx])
            #             print(self.collision_vector[idx])
            #             print(self.theta_comp[idx])
            #             print()
            #
            #         constraint_value_true.append(constraint_value[0])
            #         constraint_value_estimate.append(cons_value[23 + (t-1) * self.number_of_constraints])
            #
            #     print(f"r_true: {r_vals_true}")
            #     print(f'r_est: {r_vals_estimate}')
            #     print(f"theta_true: {theta_value_true}")
            #     print(f"theta_est: {theta_value_estimate}")
            #     print(f"Theta signed: {theta_value_signed}")
            #     print(f"Omega true: {Omega_true}")
            #     print(f"Omega estimate: {Omega_est}")
            #     print(f"Coll angle true: {coll_angle_true}")
            #     print(f"Coll angle estimate: {coll_angle_est}")
            #     print(f"constraint true: {constraint_value_true}")
            #     print(f"constraint estimate: {constraint_value_estimate}")
            #
            #     print()


                    # theta_est = self.collision_matrix[23] @ self.x.X[0, 1::6].reshape((-1,)) - \
                    #     self.Omega_selection_matrix[23] @ self.delta_Omega_prediction.reshape((-1,)) + \
                    #     self.theta_reff_diff[23] - self.collision_vector[23] + self.theta_comp[23]
                    # theta_est = self.collision_matrix @ self.x.X[0, 1::6].reshape((-1,)) - \
                    # np.cos(np.pi / 4) * self.collision_matrix @ self.delta_Omega_prediction.reshape((-1,))[t * 30:(t+1) * 30] + \
                    #             (self.theta_reff_diff - self.collision_vector + self.theta_comp)[t * self.number_of_constraints:(t+1) * self.number_of_constraints]
                    # print(f"t={t}")
                    # print(f"|delta r| = {np.abs(radial_diff[23]).flatten()}, r_abs_est = {radial_diff[23] * self.lambda_sign[23]}")
                    # print(f"|theta diff| = {np.abs(theta_diff[23]).flatten()}, theta_abs_est={theta_est[23]}")
                    # print(f"Constraint value: {constraint_value}")
                    # print(f"Constraint value solver: {cons_value[23 + t * self.number_of_constraints ]}")
                    # # print(f"Argmin: {np.argmin(np.abs(radial_diff)+self.alpha * np.abs(theta_diff))}")
                    #
                    # radial_diff = self.collision_matrix @ self.x.X[t+1, 0::6].reshape((-1, 1))  # r2 - r1
                    # Omega_vals = find_delta_Omega(self.x.X[t+1:t+2], self.omega_difference_start)  # [Omega_0 - Omega_0_ref, Omega_1 - Omega_1_ref, ...]
                    # Omega_diff = self.collision_matrix @ Omega_vals.reshape((-1, 1))  # Omega_2 - Omega_1 - Omega_2_ref + Omega_1_ref
                    # coll_angles = find_collision_angle_vect(np.pi / 4, Omega_diff + self.Omega_reff_diff[:self.number_of_constraints].reshape((-1, 1))) % (2 * np.pi)
                    # coll_angles -= (coll_angles > np.pi) * 2 * np.pi
                    # coll_angles = coll_angles.reshape((-1, 1))
                    #
                    # theta_diff = (self.collision_matrix @ self.x.X[t+1, 1::6].reshape((-1, 1)) - np.cos(np.pi / 4) * Omega_diff + self.theta_reff_diff[:self.number_of_constraints].reshape((-1, 1)) - coll_angles) % (2 * np.pi)
                    # theta_diff -= (theta_diff > np.pi) * 2 * np.pi
                    # constraint_value = np.abs(radial_diff[23]).flatten() + self.alpha * np.abs(theta_diff[23]).flatten()
                    #
                    # theta_est = self.collision_matrix @ self.x.X[t+1, 1::6].reshape((-1,)) - \
                    #             np.cos(np.pi / 4) * self.collision_matrix @ self.delta_Omega_prediction.reshape((-1,))[
                    #                                                         t * 30:(t + 1) * 30] + \
                    #             (self.theta_reff_diff - self.collision_vector + self.theta_comp)[
                    #             t * self.number_of_constraints:(t + 1) * self.number_of_constraints]
                    # print()


            poly_deg = 15 if self.prediction_horizon > 6 else 3
            self.delta_Omega_prediction = predict_Omega(find_delta_Omega(self.x.X[2:], self.omega_difference_start), deg=poly_deg)
            Omega_differences_new = self.collision_matrix_large @ self.delta_Omega_prediction.reshape((-1, 1))
            self.collision_vector, self.last_collision_vector_Omega = update_collision_vector(self.collision_vector, Omega_differences_new, self.last_collision_vector_Omega, self.Omega_reff_diff)

            lambda_vals = self.x.X[2:, 1::6]
            lamda_pred = predict_Omega(lambda_vals, deg=1)

            theta_diff = (self.collision_matrix_large @ lamda_pred.reshape((-1, )) - \
                         self.Omega_selection_matrix @ self.delta_Omega_prediction.reshape((-1,)) + \
                         self.theta_reff_diff - self.collision_vector)# % (2 * np.pi)
            # theta_diff -= (theta_diff > np.pi) * 2 * np.pi
            theta_des = theta_diff % (2 * np.pi)
            theta_des -= (theta_des > np.pi) * 2 * np.pi
            theta_abs = np.abs(theta_des)
            self.theta_comp = theta_abs - theta_diff
            # print(theta_comp.shape)

            # radius_vals = self.x.X[2:, 0::6]
            # radius_pred = predict_Omega(radius_vals, deg=radius_vals.shape[0] - 2)
            # radius_diff = self.collision_matrix_large @ radius_pred.reshape((-1, ))
            # self.radius_comp = -2 * radius_diff.copy()
            # self.radius_comp[self.radius_comp < 0] = 0

            self.out_plane_constraint.rhs = np.kron(np.ones(self.prediction_horizon), self.safety_margin) + self.alpha * (self.Omega_selection_matrix @ self.delta_Omega_prediction.reshape((-1, )) - self.theta_reff_diff + self.collision_vector - self.theta_comp)

            # print(np.min(self.theta_pos.X[:self.number_of_constraints] + self.theta_neg.X[:self.number_of_constraints]),
            #       (self.r_pos.X[:self.number_of_constraints] + self.r_neg.X[:self.number_of_constraints])[np.argmin(self.theta_pos.X[:self.number_of_constraints] + self.theta_neg.X[:self.number_of_constraints])])
            # print(np.rad2deg(self.delta_Omega_prediction))
            # faulty_indices = np.arange(self.number_of_constraints * self.prediction_horizon)[(np.abs(self.r_pos.X) > 0.001) & (np.abs(self.r_neg.X) > 0.001)]
            # print("r")
            # print(faulty_indices)
            # print(self.r_pos.X[faulty_indices])
            # print(self.r_neg.X[faulty_indices])
            # print((np.kron(np.eye(self.prediction_horizon), np.kron(self.out_plane_collision_setup[0],
            #                                                                  np.array([1, 0, 0, 0, 0, 0]))) @ self.x.X[1:].reshape((-1, )))[faulty_indices])
            #
            # print("Lambda")
            # print(self.theta_pos.X[faulty_indices])
            # print(self.theta_neg.X[faulty_indices])
            # print((np.kron(np.eye(self.prediction_horizon), np.kron(self.out_plane_collision_setup[0],
            #                                                                           np.array([0, 1, 0, 0, 0, 0])))@ self.x.X[1:].reshape((-1, )))[faulty_indices])
            # print((self.Omega_selection_matrix @ self.delta_Omega_prediction.reshape((-1, )))[faulty_indices])
            # print(np.kron(np.ones((self.prediction_horizon, )), self.out_plane_collision_setup[3] )[faulty_indices])
            # print(np.kron(np.ones((self.prediction_horizon,)), self.out_plane_collision_setup[1])[faulty_indices])
            # print(np.kron(np.ones(self.prediction_horizon), self.out_plane_collision_setup[2])[faulty_indices])
            #
            # print()

            # print(np.sum())
            # print(np.sum(np.abs(self.r_pos.X + self.r_neg.X - np.abs(np.kron(np.eye(self.prediction_horizon), np.kron(self.out_plane_collision_setup[0],
            #                                                                  np.array([1, 0, 0, 0, 0, 0]))) @ self.x.X[1:].reshape((-1, )))) > 0.005))
            # print()
            # print(np.kron(np.eye(self.prediction_horizon), np.kron(self.out_plane_collision_setup[0],
            #                                                                  np.array([1, 0, 0, 0, 0, 0]))) @ self.x.X[1:].reshape((-1, )))

        # print(controller._Phi_x[1][0][0], controller._Phi_x[2][0][0])
        # for t in range(self.prediction_horizon):
        #     print(self.in_plane_collision_setup[0] @ self.x.X[t, :] >= self.in_plane_collision_setup[1].flatten())

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

        if self.out_plane_collision_setup and self.number_of_constraints > 0:
            scaling = 1e-1
            # obj3 = scaling * gp.quicksum(self.theta_pos + self.theta_neg)
            obj3 = 0
        else:
            obj3 = 0
        self._problem.setObjective(obj1 + obj2 + obj3, gp.GRB.MINIMIZE)

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
                 in_plane_collision_setup=None, out_plane_collision_setup=None,
                 omega_difference_start=None, sparse: bool = True):
        super().__init__()

        if sparse:
            self._solver = Gurobi_Solver_Sparse(number_of_satellites, prediction_horizon, model, reference_state,
                                                state_limit, input_limit, in_plane_collision_setup,
                                                out_plane_collision_setup, omega_difference_start)
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
