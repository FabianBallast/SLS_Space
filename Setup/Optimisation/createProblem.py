import numpy as np
from scipy import sparse
from Dynamics.HCWDynamics import RelCylHCW
from Dynamics.BlendDynamics import Blend
from Dynamics.ROEDynamics import QuasiROE
from Scenarios.MainScenarios import ScenarioEnum, Scenario
import random
from Scenarios.ControlScenarios import Model


def create_sparse_problem(number_of_satellites: int,
                          prediction_horizon: int = None,
                          scenario: Scenario = ScenarioEnum.simple_scenario_translation_blend_scaled.value) -> dict:
    """
    Create a sparse optimisation problem.

    :param number_of_satellites: Number of satellites as an integer.
    :param prediction_horizon: Prediction horizon for the controller.
    :param scenario: scenario to run.
    :return: dict with A, B, Q, R, QN, x0, x_max, u_max, N, nx and nu
    """
    problem = dict()

    # Setup scenario
    scenario.number_of_satellites = number_of_satellites
    if prediction_horizon is not None:
        scenario.control.tFIR = prediction_horizon

    # Create single model
    if scenario.model == Model.HCW:
        dynamics = RelCylHCW(scenario)
    elif scenario.model == Model.BLEND:
        dynamics = Blend(scenario)
    elif scenario.model == Model.ROE:
        dynamics = QuasiROE(scenario)
    else:
        raise ValueError("Dynamical model not supported.")

    model_single = dynamics.create_model(scenario.control.control_timestep)

    # Create large model
    problem['A'] = sparse.kron(sparse.eye(number_of_satellites), sparse.csc_matrix(model_single.A), format='csc')
    problem['B'] = sparse.kron(sparse.eye(number_of_satellites), sparse.csc_matrix(model_single.B), format='csc')
    [nx, nu] = problem['B'].shape
    problem['nx'] = nx
    problem['nu'] = nu

    # Constraints
    problem['u_max'] = np.array(dynamics.get_input_constraint() * number_of_satellites)
    problem['x_max'] = np.array(dynamics.get_state_constraint() * number_of_satellites)

    # Objective function
    problem['Q'] = sparse.kron(sparse.eye(number_of_satellites), sparse.csc_matrix(dynamics.get_state_cost_matrix_sqrt()).power(2))
    problem['QN'] = problem['Q']
    problem['R'] = sparse.kron(sparse.eye(number_of_satellites), sparse.csc_matrix(dynamics.get_input_cost_matrix_sqrt()[3:]).power(2))
    problem['R_SLSPY'] = sparse.kron(sparse.eye(number_of_satellites),
                                     sparse.csc_matrix(dynamics.get_input_cost_matrix_sqrt()))

    # Initial and reference states
    x0 = np.zeros(nx)
    dropouts = int(scenario.initial_state.dropouts * scenario.number_of_satellites) + 1
    possible_angles = np.linspace(0, 2 * np.pi, number_of_satellites + dropouts, endpoint=False)

    random.seed(129)
    selected_indices = np.sort(random.sample(range(number_of_satellites + dropouts), number_of_satellites))
    x0[1::6] = possible_angles[selected_indices]
    xr = np.zeros(nx)
    xr[1::6] = np.linspace(0, 2 * np.pi, number_of_satellites, endpoint=False)

    x0_rel = x0 - xr
    x0_rel[1::6] -= np.mean(x0_rel[1::6])

    if number_of_satellites == 1:
        x0_rel[1::6] += np.deg2rad(30)

    problem['x0'] = x0_rel
    problem['x0_abs'] = x0_rel + xr
    problem['x_ref'] = xr
    problem['N'] = scenario.control.tFIR

    # problem['e_A'] = 0.0079
    # problem['e_B'] = 0.00097
    problem['e_A'] = scenario.robustness.e_A
    problem['e_B'] = scenario.robustness.e_B
    problem['sigma_w'] = scenario.robustness.sigma_w

    problem['e_A0'] = scenario.robustness.e_A0
    problem['e_A1'] = scenario.robustness.e_A1
    problem['e_A2'] = scenario.robustness.e_A2
    problem['e_A3'] = scenario.robustness.e_A3
    problem['e_A4'] = scenario.robustness.e_A4
    problem['e_A5'] = scenario.robustness.e_A5

    problem['e_B0'] = scenario.robustness.e_B0
    problem['e_B1'] = scenario.robustness.e_B1
    problem['e_B2'] = scenario.robustness.e_B2
    problem['e_B3'] = scenario.robustness.e_B3
    problem['e_B4'] = scenario.robustness.e_B4
    problem['e_B5'] = scenario.robustness.e_B5

    problem['sigma_w0'] = scenario.robustness.sigma_w0
    problem['sigma_w1'] = scenario.robustness.sigma_w1
    problem['sigma_w2'] = scenario.robustness.sigma_w2
    problem['sigma_w3'] = scenario.robustness.sigma_w3
    problem['sigma_w4'] = scenario.robustness.sigma_w4
    problem['sigma_w5'] = scenario.robustness.sigma_w5


    return problem
