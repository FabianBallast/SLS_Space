from slspy import SLS, SLS_Obj_H2, SLS_Cons_Input, SLS_Cons_State, LTV_System
import time
import numpy as np
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

    # Create controller for matrices
    sys = LTV_System(Nx=problem['nx'], Nu=problem['nu'], Nw=3, tFIR=problem['N'])
    sys._A = np.zeros((problem['N'], problem['nx'], problem['nx']))
    sys._B2 = np.zeros((problem['N'], problem['nx'], problem['nu']))

    for t in range(problem['N']):
        sys._A[t], sys._B2[t] = problem['A'].todense(), problem['B'].todense()
        sys._C2 = np.eye(problem['nx'])

    # Set them as matrices for the regulator
    sys._C1 = np.sqrt(problem['Q'])
    sys._D12 = problem['R_SLSPY']

    # set SLS objective
    synthesizer = SLS(system_model=sys, FIR_horizon=scenario.control.tFIR, noise_free=True, fast_computation=False)
    synthesizer << SLS_Obj_H2()
    synthesizer << SLS_Cons_Input(state_feedback=True, maximum_input=problem['u_max'].reshape((-1, 1)))
    synthesizer += SLS_Cons_State(state_feedback=True, maximum_state=problem['x_max'].reshape((-1, 1)))

    # Optimise
    nsim = 11
    states = np.zeros((problem['nx'], nsim + 1))
    states[:, 0] = problem['x0']
    t_0 = 0

    for t in range(nsim):

        sys.initialize(states[:, t:t+1])

        # Synthesise controller
        controller = synthesizer.synthesizeControllerModel(0)

        # print(f"Predicted next state: {(self.controller._Phi_x[2] + self.x_ref).T}")
        # Update state and input
        states[:, t + 1] = controller._Phi_x[2] @ states[:, t]

        if t == 0:
            t_0 = time.time()

    t_end = time.time()
    average_time = (t_end - t_0) / (nsim - 1)

    print(f"Average elapsed time SLSPY for {number_of_satellites} satellites: {average_time:.3}s")

    if plot_results:
        plt.figure()
        plt.plot(np.rad2deg(states[1::6].T))
        plt.show()

    return average_time


if __name__ == '__main__':
    time_optimisation(15, plot_results=True)
# import numpy as np
# from slspy import SLS, SLS_Obj_H2, SLS_Cons_Input, SLS_Cons_State
# from Controllers.SLS_setup import SLSSetup
# from Scenarios.MainScenarios import ScenarioEnum
# from Dynamics.HCWDynamics import RelCylHCW as dyn
# from scipy import sparse
# import random
# import matplotlib.pyplot as plt
# import time
# from Optimisation.createProblem import create_sparse_problem
#
# prediction_horizon = None
# number_of_satellites = 4
# simulation_length = 4
# scenario = ScenarioEnum.simple_scenario_translation_HCW_scaled.value
#
# problem = create_sparse_problem(number_of_satellites, prediction_horizon, scenario)
#
# # General values
#
# # scenario.number_of_satellites = number_of_satellites
# # dynamics = dyn(scenario)
# # model_single = dynamics.create_model(scenario.control.control_timestep)
#
# # # Create large model
# # number_of_satellites = scenario.number_of_satellites
# # Ad = sparse.kron(sparse.eye(number_of_satellites), sparse.csc_matrix(model_single.A))
# # Bd = sparse.kron(sparse.eye(number_of_satellites), sparse.csc_matrix(model_single.B))
# # [nx, nu] = Bd.shape
#
# # Initial and reference states
# # x0 = np.zeros((nx, 1))
# # possible_angles = np.linspace(0, 2 * np.pi, number_of_satellites + 2, endpoint=False).reshape((scenario.number_of_satellites + 2, 1))
# #
# # random.seed(129)
# # selected_indices = np.sort(random.sample(range(number_of_satellites + 2), number_of_satellites))
# # x0[1::6] = possible_angles[selected_indices]
# # xr = np.zeros((nx, 1))
# # ref_rel_angles = np.linspace(0, 2 * np.pi, scenario.number_of_satellites, endpoint=False).reshape((scenario.number_of_satellites, 1))
# # ref_rel_angles -= np.mean(ref_rel_angles - possible_angles[selected_indices].reshape((scenario.number_of_satellites, 1)))
# # xr[1::6] = ref_rel_angles
#
# # Prediction horizon
# N = scenario.control.tFIR
#
#
# # Create controller for matrices
# # dynamics = dyn(scenario)
# # sls_setup = SLSSetup(scenario.control.control_timestep, system_dynamics=dynamics, prediction_horizon=scenario.control.tFIR)
# # sls_setup.create_system(scenario.number_of_satellites)
# # sls_setup.create_cost_matrices(dynamics.get_state_cost_matrix_sqrt(), dynamics.get_input_cost_matrix_sqrt())
#
#
# sys = LTV_System(Nx=problem['nx'], Nu=problem['nu'], Nw=3, tFIR=problem['N'])
# sys._A = np.zeros((problem['N'], problem['nx'], problem['nx']))
# sys._B2 = np.zeros((problem['N'], problem['nx'], problem['nu']))
#
# for t in range(problem['N']):
#     sys._A[t], sys._B2[t] = problem['A'].todense(), problem['B'].todense()
#     sys._C2 = np.eye(problem['nx'])
#
# # Set them as matrices for the regulator
# sys._C1 = np.sqrt(problem['Q'])
# sys._D12 = np.sqrt(problem['R_SLSPY'])
#
# # synthesizer = SLS(system_model=sls_setup.sys, FIR_horizon=scenario.control.tFIR, noise_free=True, fast_computation=False)
# #
# # # set SLS objective
# # synthesizer << SLS_Obj_H2()
# #
# # input_constraint = np.array(dynamics.get_input_constraint() * scenario.number_of_satellites).reshape((-1, 1))
# # synthesizer << SLS_Cons_Input(state_feedback=True, maximum_input=input_constraint)
# #
# # state_constraint = np.array(dynamics.get_state_constraint() * scenario.number_of_satellites).reshape((-1, 1))
# # synthesizer += SLS_Cons_State(state_feedback=True, maximum_state=state_constraint)
# # # synthesizer.initializePhi()
#
# synthesizer = SLS(system_model=sys, FIR_horizon=scenario.control.tFIR, noise_free=True, fast_computation=False)
# synthesizer << SLS_Obj_H2()
# synthesizer << SLS_Cons_Input(state_feedback=True, maximum_input=problem['u_max'].reshape((-1, 1)))
# synthesizer += SLS_Cons_State(state_feedback=True, maximum_state=problem['x_max'].reshape((-1, 1)))
#
# # Optimise
# x_states = np.zeros((problem['nx'], simulation_length * 1 + 1))
# # u_inputs = np.zeros((sls_setup.total_input_size, simulation_length))
#
# x_states[:, 0:1] = problem['x0'].reshape((-1, 1))
# t_0 = 0
#
# for t in range(simulation_length):
#
#     sls_setup.sys.initialize(x_states[:, t:t+1])
#
#     # Synthesise controller
#     controller = synthesizer.synthesizeControllerModel(0)
#
#     # print(f"Predicted next state: {(self.controller._Phi_x[2] + self.x_ref).T}")
#     # Update state and input
#     # u_inputs[:, t] = controller._Phi_u[1] @ x_states[:, t]
#     x_states[:, t + 1] = controller._Phi_x[2] @ x_states[:, t]
#
#     if t == 0:
#         t_0 = time.time()
#
# t_end = time.time()
# print(f"Average elapsed time: {(t_end - t_0) / (simulation_length - 1)}")
#
# plt.figure()
# plt.plot(np.rad2deg(x_states[1::6].T))
#
# plt.show()
