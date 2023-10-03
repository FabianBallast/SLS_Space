from matplotlib import pyplot as plt
from Dynamics.AttitudeDynamics import LinAttModel
from Dynamics.HCWDynamics import RelCylHCW
from Dynamics.ROEDynamics import QuasiROE, ROE
from Dynamics.DifferentialDragDynamics import DifferentialDragDynamics
from Dynamics.BlendDynamics import Blend, BlendSmall
from Dynamics.SystemDynamics import TranslationalDynamics
from Scenarios.MainScenarios import Scenario
from Space.OrbitalMechanics import OrbitalMechSimulator
from Controllers.SLS_setup import SLSSetup
from Controllers.SimulatedAnnealing import SimulatedAnnealing
from Scenarios.ControlScenarios import Model
from Utils.GenerateInitialPosition import generate_anomalies_and_longitudes, generate_reference
import numpy as np


class ScenarioHandler:
    """
    Create a class to deal with different scenarios and set up everything.
    """

    def __init__(self, scenario: Scenario) -> None:
        """
        Create the scenario handler object through a scenario.

        :param scenario: A dict with the scenario to run.
        """
        self.scenario = scenario
        self.controller = None
        self.orbital_mech = None

        self.control_inputs_thrust = np.zeros((self.scenario.number_of_satellites, 3, 1000))
        self.control_inputs_torque = np.zeros((self.scenario.number_of_satellites, 3, 1000))
        self.sls_states = None
        self.pos_states = None
        self.rot_states = None
        self.control_inputs = None

        self.dep_vars = None
        self.dep_vars_dict = None

        self.t_horizon_control = None
        self.control_simulation_ratio = None
        self.t_horizon_simulation = None
        self.t_full_simulation_steps = None

        self.orbital_element_offsets = None
        self.done_full_sim = False

        self.satellite_longitudes = self.scenario.orbital.longitude

        if isinstance(self.satellite_longitudes, list):
            self.number_of_planes = len(self.satellite_longitudes)
        else:
            self.number_of_planes = 1

        self.reference_satellites_added = False

    def create_sls_system(self) -> None:
        """
        Create the SLS optimiser.
        """
        if self.scenario.model == Model.ATTITUDE:
            control_model = LinAttModel(self.scenario)
        elif self.scenario.model == Model.ROE:
            control_model = QuasiROE(self.scenario)
        elif self.scenario.model == Model.HCW:
            control_model = RelCylHCW(self.scenario)
        elif self.scenario.model == Model.DIFFERENTIAL_DRAG:
            control_model = DifferentialDragDynamics(self.scenario)
        elif self.scenario.model == Model.ROE_V2:
            control_model = ROE(self.scenario)
        elif self.scenario.model == Model.BLEND:
            control_model = Blend(self.scenario)
        elif self.scenario.model == Model.BLEND_SMALL:
            control_model = BlendSmall(self.scenario)
        else:
            raise Exception(f"Model type {self.scenario.model} not recognized!")

        if self.scenario.model == Model.DIFFERENTIAL_DRAG:
            control_method = SimulatedAnnealing
        else:
            control_method = SLSSetup

        self.controller = control_method(sampling_time=self.scenario.control.control_timestep,
                                         system_dynamics=control_model,
                                         prediction_horizon=self.scenario.control.tFIR)
        self.controller.create_system(number_of_systems=self.scenario.number_of_satellites)

        # Add cost matrices
        if isinstance(self.controller, SLSSetup):
            self.controller.create_cost_matrices()

        # Create x0 and x_ref
        self.controller.create_x0(number_of_dropouts=int(self.scenario.initial_state.dropouts *
                                                         self.scenario.number_of_satellites) + 1)
        self.controller.create_reference()

    def create_storage_variables(self) -> None:
        """
        Create variables to store the results in.
        """
        self.t_horizon_control = int(np.ceil(self.scenario.simulation.simulation_duration /
                                             self.scenario.control.control_timestep)) + 1
        self.control_simulation_ratio = int(self.scenario.control.control_timestep /
                                            self.scenario.simulation.simulation_timestep)
        self.t_horizon_simulation = self.control_simulation_ratio * self.t_horizon_control
        self.t_full_simulation_steps = self.scenario.simulation.simulation_duration // \
                                       self.scenario.simulation.simulation_timestep

        # Also include reference for pos/rot states
        self.sls_states = np.zeros((self.t_horizon_simulation + 1, len(self.controller.x0)))
        self.pos_states = np.zeros((self.t_horizon_simulation + 1, (
                    self.scenario.number_of_satellites + self.number_of_planes * self.reference_satellites_added) * 6))
        self.rot_states = np.zeros((self.t_horizon_simulation + 1, (
                    self.scenario.number_of_satellites + self.number_of_planes * self.reference_satellites_added) * 7))

        self.control_inputs = np.zeros((self.t_horizon_simulation + 1, len(self.controller.x0) // 2))

    def __create_simulation(self) -> None:
        """
        Create an OrbitalMechSim and set most of the parameters, except initial positions.
        """
        self.orbital_mech = OrbitalMechSimulator(self.scenario, self.controller.x_ref[self.controller.angle_states])
        self.orbital_mech.set_mean_motion_and_orbital_diff(self.controller.dynamics.mean_motion,
                                                           self.controller.dynamics.get_orbital_differentiation())
        self.orbital_mech.create_bodies(number_of_satellites=self.scenario.number_of_satellites,
                                        satellite_mass=self.scenario.physics.mass,
                                        satellite_inertia=self.scenario.physics.inertia_tensor,
                                        add_reference_satellite=self.reference_satellites_added,
                                        use_parameters_from_scenario=self.scenario)

        # Create thrust models
        self.orbital_mech.create_engine_models_thrust(control_timestep=self.scenario.control.control_timestep,
                                                      thrust_inputs=self.control_inputs_thrust,
                                                      specific_impulse=self.scenario.physics.specific_impulse)

        # Create torque models/update them
        # if self.orbital_mech.input_torque_model is None:
        # self.orbital_mech.create_engine_models_torque(control_timestep=self.scenario.control.control_timestep,
        #                                               torque_inputs=self.control_inputs_torque,
        #                                               specific_impulse=self.scenario.physics.specific_impulse)

        # Add acceleration to the model
        if self.scenario.physics.J2_perturbation:
            self.orbital_mech.create_acceleration_model(order_of_zonal_harmonics=2)  # Check if this works!
        else:
            self.orbital_mech.create_acceleration_model(order_of_zonal_harmonics=0)

        # Add torques to the satellites. Currently only adds 2nd degree grav torque.
        # self.orbital_mech.create_torque_model()

        # Add dependent variables to track
        self.orbital_mech.set_dependent_variables_translation(add_keplerian_state=True,
                                                              add_rsw_rotation_matrix=False,
                                                              add_thrust_accel=True)
        # self.orbital_mech.set_dependent_variables_rotation(add_control_torque=True, add_torque_norm=False)

    def __initialise_simulation(self) -> None:
        """
        Initialise the simulation by setting the correct initial positions.
        """
        # longitude = self.scenario.orbital.longitude

        # Set the initial position of the satellites
        if isinstance(self.controller.dynamics, LinAttModel):
            true_anomalies = np.linspace(0, 360, self.scenario.number_of_satellites, endpoint=False).tolist()
        elif isinstance(self.controller.dynamics, DifferentialDragDynamics):
            true_anomalies = [0] + np.rad2deg(self.controller.x0[self.controller.angle_states].reshape((-1,))).tolist()
        else:
            number_of_dropouts = int(self.scenario.initial_state.dropouts * self.scenario.number_of_satellites) + 1
            true_anomalies, self.satellite_longitudes = generate_anomalies_and_longitudes(number_of_dropouts,
                                                                                          self.scenario.orbital.longitude,
                                                                                          self.scenario.number_of_satellites,
                                                                                          self.controller.dynamics)
            true_anomalies = np.rad2deg(true_anomalies).tolist()
            self.controller.x_ref[self.controller.angle_states] = generate_reference(self.scenario.number_of_satellites,
                                                                                     true_anomalies,
                                                                                     self.scenario.orbital.longitude)

        self.pos_states[0:1] = \
            self.orbital_mech.convert_orbital_elements_to_cartesian(true_anomalies=true_anomalies,
                                                                    orbit_height=self.scenario.physics.orbital_height,
                                                                    inclination=self.scenario.orbital.inclination,
                                                                    longitude=self.satellite_longitudes,
                                                                    eccentricity=self.scenario.orbital.eccentricity,
                                                                    argument_of_periapsis=self.scenario.orbital.argument_of_periapsis,
                                                                    orbital_element_offsets=self.orbital_element_offsets)

        # angular_vel = np.array([0, 0, -1 * self.controller.dynamics.mean_motion])
        # angular_vel_offsets = np.random.rand(3, self.scenario.number_of_satellites) * \
        #                       self.scenario.initial_state.angular_velocity_offset_magnitude
        # self.rot_states[0:1] = \
        #     self.orbital_mech.convert_orbital_elements_to_quaternion(true_anomalies=true_anomalies,
        #                                                              initial_angular_velocity=angular_vel,
        #                                                              inclination=self.scenario.orbital.inclination,
        #                                                              longitude=longitude,
        #                                                              initial_angle_offset=self.scenario.initial_state.attitude_offset,
        #                                                              initial_velocity_offset=angular_vel_offsets)

    def __find_osculation_offsets(self):

        self.__create_simulation()
        self.__initialise_simulation()
        start_epoch = self.scenario.simulation.start_epoch
        end_epoch = start_epoch + int(2 * np.pi / self.controller.dynamics.mean_motion)

        self.orbital_mech.set_initial_position(self.pos_states[0:1].T)
        self.orbital_mech.set_initial_orientation(self.rot_states[0:1].T)

        # Create propagation settings
        self.orbital_mech.create_propagation_settings(start_epoch=start_epoch,
                                                      end_epoch=end_epoch,
                                                      simulation_step_size=self.scenario.simulation.simulation_timestep)

        # Simulate system
        self.orbital_mech.simulate_system()
        indices = np.array(list(self.orbital_mech.dependent_variables_dict["keplerian state"].values())).reshape((-1,))
        kep_var = self.orbital_mech.dep_vars[:, indices]

        # Semi major axis
        semi_major_axes = kep_var[:, 0::6]
        self.orbital_element_offsets = np.zeros((6, semi_major_axes.shape[1]))
        self.orbital_element_offsets[0] = (self.scenario.physics.orbital_height + self.scenario.physics.radius_Earth) - \
                                          (np.max(semi_major_axes, axis=0) + np.min(semi_major_axes, axis=0)) / 2

        # Eccentricity
        # eccentricity = kep_var[:, 1::6]
        # self.orbital_element_offsets[1] = self.scenario.orbital.eccentricity - np.mean(eccentricity, axis=0)

        # Inclination
        inclinations = kep_var[:, 2::6]
        self.orbital_element_offsets[2] = np.deg2rad(self.scenario.orbital.inclination) - \
                                          (np.max(inclinations, axis=0) + np.min(inclinations, axis=0)) / 2

        # Periapsis
        periapsis = kep_var[:, 3::6]
        t = np.arange(0, periapsis.shape[0], 1).reshape((-1, 1)) * self.scenario.simulation.simulation_timestep
        # periapsis_osc = periapsis - self.orbital_mech.orbital_derivative[4] * t
        # self.orbital_element_offsets[3] = np.deg2rad(self.scenario.orbital.argument_of_periapsis) - \
        #                                   np.mean(periapsis_osc, axis=0)

        # RAAN
        RAAN = kep_var[:, 4::6]
        RAAN[:, RAAN[0] > np.pi] -= 2 * np.pi
        RAAN = np.unwrap(RAAN, axis=0)
        RAAN_osc = RAAN - self.orbital_mech.orbital_derivative[3] * t

        if self.reference_satellites_added:
            self.orbital_element_offsets[4] = np.deg2rad(
                np.hstack([self.satellite_longitudes, self.scenario.orbital.longitude])) - \
                                              (np.max(RAAN_osc, axis=0) + np.min(RAAN_osc, axis=0)) / 2
        else:
            self.orbital_element_offsets[4] = np.deg2rad(self.satellite_longitudes) - \
                                              (np.max(RAAN_osc, axis=0) + np.min(RAAN_osc, axis=0)) / 2
        # Anomaly
        # true_anomaly = kep_var[:, 5::6]
        # true_anomaly_osc = periapsis - self.orbital_mech.orbital_derivative[5] * t
        # self.orbital_element_offsets[5] = true_anomaly[0] - np.mean(true_anomaly_osc, axis=0)

        self.orbital_mech = None

        # plt.figure(1)
        # plt.plot(t, RAAN)
        # plt.plot(t, RAAN_osc, '--')
        # plt.show()

    def __run_simulation_timestep(self, time: int, initial_setup: bool = False, full_simulation: bool = False) -> None:
        """
        Run a single simulation timestep.

        :param time: The time for which the simulation is.
        :param initial_setup: Whether this simulation is to provide some initial values.
                              Simplified simulation in that case.
        :param full_simulation: Whether to run the complete simulation in one go. No control possible in that case.
                                Use time = 0 in that case.
        """
        if self.orbital_mech is None:
            self.__create_simulation()
        else:
            self.orbital_mech.update_engine_models(self.control_inputs_thrust, self.control_inputs_torque)

        if initial_setup:
            self.__initialise_simulation()
            start_epoch = self.scenario.simulation.start_epoch
            end_epoch = start_epoch + self.scenario.simulation.simulation_timestep
        elif full_simulation:
            start_epoch = self.scenario.simulation.start_epoch
            end_epoch = start_epoch + self.scenario.simulation.simulation_duration
        else:
            start_epoch = self.scenario.simulation.start_epoch + \
                          time * self.scenario.control.control_timestep
            end_epoch = start_epoch + self.scenario.control.control_timestep

        self.orbital_mech.set_initial_position(self.pos_states[time * self.control_simulation_ratio:
                                                               time * self.control_simulation_ratio + 1].T)

        # self.orbital_mech.set_initial_orientation(self.rot_states[time * self.control_simulation_ratio:
        #                                                           time * self.control_simulation_ratio + 1].T)

        # Create propagation settings
        self.orbital_mech.create_propagation_settings(start_epoch=start_epoch,
                                                      end_epoch=end_epoch,
                                                      simulation_step_size=self.scenario.simulation.simulation_timestep)

        # Simulate system
        self.orbital_mech.simulate_system()
        dep_vars_array = self.orbital_mech.dep_vars
        control_states = self.orbital_mech.get_states_for_dynamical_model(self.controller.dynamics)

        # Prevent issue where reference is negative and initial state close to 2 * pi from below
        problematic_states = (control_states[0, self.controller.angle_states] > np.pi) & (
                    self.controller.x_ref[self.controller.angle_states].reshape((-1,)) < 0)
        control_states[0, self.controller.angle_states[problematic_states]] -= 2 * np.pi

        # Store results
        if initial_setup:
            self.sls_states[0:1] = control_states[0:1]

            self.dep_vars = np.zeros((len(self.pos_states[:, 0]), dep_vars_array.shape[1]))
            self.dep_vars[0] = dep_vars_array[0]
            self.dep_vars_dict = self.orbital_mech.dependent_variables_dict
        else:
            if full_simulation:
                self.pos_states = self.orbital_mech.translation_states
                # self.rot_states = self.orbital_mech.rotation_states
                self.dep_vars = self.orbital_mech.dep_vars
                self.sls_states = control_states
                self.done_full_sim = True

                self.control_inputs = self.control_inputs_thrust.flatten()
            else:
                index_selection = np.arange(time * self.control_simulation_ratio + 1,
                                            (time + 1) * self.control_simulation_ratio + 1)

                self.pos_states[index_selection] = self.orbital_mech.translation_states[1:]
                # self.rot_states[index_selection] = self.orbital_mech.rotation_states[1:]
                self.dep_vars[index_selection] = dep_vars_array[1:]
                self.sls_states[index_selection] = control_states[1:self.control_simulation_ratio + 1]
                self.orbital_mech.update_mean_orbital_elements(dep_vars_array[-1, 1:])

                self.control_inputs[index_selection] = np.tile(self.control_inputs_thrust[:, :, 0].reshape((1, -1)), (self.control_simulation_ratio, 1))

        # Unwrap to prevent weird positions when positions are involved
        if isinstance(self.controller.dynamics, TranslationalDynamics) and not initial_setup and not full_simulation:
            unwrap_indices = np.arange(time * self.control_simulation_ratio,
                                       (time + 1) * self.control_simulation_ratio + 1)
            index_selection = np.ix_(unwrap_indices, self.controller.all_angle_states)
            self.sls_states[index_selection] = np.unwrap(self.sls_states[index_selection], axis=0)
            # print(f"Actual state: {self.sls_states[(time + 1) * self.control_simulation_ratio]}")

    def __synthesise_controller(self, time: int, simulation_length: int = 1) -> None:
        """
        Synthesise the controller and generate optimal control inputs.

        :param time: The time at which to synthesise the controller.
        :param simulation_length: How many inputs and states to use from the MPC results. Usually 1.
        """
        self.controller.set_initial_conditions(self.sls_states[time * self.control_simulation_ratio:
                                                               time * self.control_simulation_ratio + 1].T,
                                               true_anomalies=self.orbital_mech.true_anomalies)
        control_states, control_inputs = self.controller.simulate_system(t_horizon=1, noise=None,
                                                                         inputs_to_store=simulation_length,
                                                                         fast_computation=True,
                                                                         time_since_start=time * self.scenario.control.control_timestep,
                                                                         add_collision_avoidance=self.scenario.collision_avoidance,
                                                                         absolute_longitude_refs=np.kron(
                                                                             self.orbital_mech.initial_reference_state[
                                                                             :, 4],
                                                                             np.ones((1, self.number_of_planes))),
                                                                         current_true_anomalies=self.orbital_mech.true_anomalies)

        if isinstance(self.controller.dynamics, LinAttModel):
            self.control_inputs_torque = control_inputs.reshape((self.scenario.number_of_satellites,
                                                                 self.controller.dynamics.input_size, -1))
            self.control_inputs_thrust = 0 * self.control_inputs_torque
        elif isinstance(self.controller.dynamics, DifferentialDragDynamics):
            inputs_theta = control_inputs.reshape((self.scenario.number_of_satellites,
                                                   self.controller.dynamics.input_size, -1))

            inputs_sat = inputs_theta[0] - inputs_theta
            theta_ddot = inputs_sat / self.scenario.physics.mass / self.controller.dynamics.orbit_radius

            cyl_states = self.orbital_mech.convert_to_cylindrical_coordinates()

            r_avg = cyl_states[-1, 0::6] + 0.5 * self.controller.sampling_time * cyl_states[-1, 3::6]
            theta_dot_avg = cyl_states[-1, 4::6] + 0.5 * self.controller.sampling_time * theta_ddot.flatten()
            inputs_r = (-3 * self.controller.dynamics.mean_motion ** 2 * r_avg - 0.00001 * r_avg -
                        2 * self.controller.dynamics.orbit_radius * self.controller.dynamics.mean_motion
                        * theta_dot_avg).reshape((self.scenario.number_of_satellites,
                                                  self.controller.dynamics.input_size, -1))

            self.control_inputs_thrust = np.concatenate((inputs_r * self.scenario.physics.mass,
                                                         inputs_sat,
                                                         0 * inputs_theta), axis=1)
            self.control_inputs_torque = 0 * self.control_inputs_thrust

        else:
            # control_inputs[0::3, :] = 0
            # control_inputs[1::3, :] = 0
            self.control_inputs_thrust = control_inputs.reshape((self.scenario.number_of_satellites,
                                                                 self.controller.dynamics.input_size, -1))
            self.control_inputs_torque = 0 * self.control_inputs_thrust

    def simulate_system_closed_loop(self, print_progress: bool = True) -> None:
        """
        Run a simulation for the provided scenario with closed-loop control.

        :param print_progress: Whether to print the progress. Default is True.
        """
        self.__find_osculation_offsets()
        self.__run_simulation_timestep(0, initial_setup=True)

        progress = 0
        for t in range(self.t_horizon_control):
            if print_progress and t / self.t_horizon_control * 100 > progress:
                print(f"Progress: {int(t / self.t_horizon_control * 100 / 5) * 5}%")
                progress = int(t / self.t_horizon_control * 100 / 5) * 5 + 5

            # print(t)
            self.__synthesise_controller(t)
            self.__run_simulation_timestep(t)

    def simulate_system_single_shot(self) -> None:
        """
        Perform one optimisation loop and store all results.
        """
        self.__find_osculation_offsets()
        self.__run_simulation_timestep(0, initial_setup=True)

        self.__synthesise_controller(0, self.scenario.control.tFIR)
        self.__run_simulation_timestep(0, full_simulation=True)

    def simulate_system_no_control(self) -> None:
        """
        Run a simulation for the provided scenario without control.
        """
        self.__find_osculation_offsets()
        self.__run_simulation_timestep(0, initial_setup=True)

        # Set large set of zero inputs for simulation
        self.control_inputs_thrust = np.zeros((self.scenario.number_of_satellites, 3, self.t_horizon_simulation))
        self.control_inputs_torque = np.zeros((self.scenario.number_of_satellites, 3, self.t_horizon_simulation))
        self.__run_simulation_timestep(0, full_simulation=True)

    def simulate_system_controller_sim(self) -> None:
        """
        Run a simulation for the provided scenario with the controller model as a simulator.
        """
        # For initial value
        self.__find_osculation_offsets()
        self.__run_simulation_timestep(0, initial_setup=True)

        self.controller.set_initial_conditions(self.sls_states[0:1].T)
        self.controller.simulate_system(t_horizon=self.t_horizon_control, noise=None, progress=True,
                                        fast_computation=True)

    def simulate_system_controller_then_full_sim(self) -> None:
        """
        Simulate the controller with its own model, then use all inputs for simulation.
        """
        self.__find_osculation_offsets()
        self.__run_simulation_timestep(0, initial_setup=True)

        self.controller.set_initial_conditions(self.sls_states[0:1].T)
        states, inputs = self.controller.simulate_system(t_horizon=self.t_horizon_control, noise=None, progress=True,
                                                         fast_computation=True)

        # Set large set of zero inputs for simulation
        self.control_inputs_thrust = inputs.reshape((self.scenario.number_of_satellites,
                                                     self.controller.dynamics.input_size, -1))
        self.control_inputs_torque = 0 * self.control_inputs_thrust
        self.__run_simulation_timestep(0, full_simulation=True)

    def export_results(self) -> OrbitalMechSimulator:
        """
        Export the results in an OrbitalMechSimulator object.

        :return: OrbitalMechSimulator with all the obtained results.
        """
        orbital_sim = OrbitalMechSimulator(self.scenario, self.orbital_mech.reference_angle_offsets)
        orbital_sim.number_of_controlled_satellites = self.scenario.number_of_satellites
        orbital_sim.number_of_total_satellites = self.scenario.number_of_satellites + 1
        orbital_sim.simulation_timestep = self.scenario.simulation.simulation_timestep
        orbital_sim.translation_states = self.pos_states
        orbital_sim.rotation_states = self.rot_states
        orbital_sim.dep_vars = self.dep_vars
        orbital_sim.dependent_variables_dict = self.dep_vars_dict
        orbital_sim.controlled_satellite_names = self.orbital_mech.controlled_satellite_names
        orbital_sim.all_satellite_names = self.orbital_mech.all_satellite_names
        orbital_sim.satellite_mass = self.scenario.physics.mass
        orbital_sim.set_mean_motion_and_orbital_diff(self.controller.dynamics.mean_motion,
                                                     self.controller.dynamics.get_orbital_differentiation())
        orbital_sim.initial_reference_state = self.orbital_mech.initial_reference_state

        orbital_sim.filter.initialise_filter(self.orbital_mech.kalman_state)

        if self.scenario.model == Model.HCW:
            orbital_sim.cylindrical_states = self.sls_states
        elif self.scenario.model == Model.ROE:
            orbital_sim.quasi_roe = self.sls_states
        elif self.scenario.model == Model.ROE_V2:
            orbital_sim.roe = self.sls_states
        elif self.scenario.model == Model.BLEND:
            orbital_sim.blend_states = self.sls_states
        elif self.scenario.model == Model.BLEND_SMALL:
            orbital_sim.small_blend_states = self.sls_states
        # if self.done_full_sim:
        #     orbital_sim.cylindrical_states = self.orbital_mech.cylindrical_states
        #     orbital_sim.roe = self.orbital_mech.roe
        #     orbital_sim.quasi_roe = self.orbital_mech.quasi_roe
        #     orbital_sim.blend_states = self.orbital_mech.blend_states

        orbital_sim.thrust_forces = self.control_inputs
        orbital_sim.solver_time = self.controller.total_solver_time
        print(orbital_sim.solver_time)
        # print(self.orbital_mech.thrust_forces)

        return orbital_sim
