import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from tudatpy import util
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup, create_dynamics_simulator
from tudatpy.kernel.astro import element_conversion, frame_conversion
from tudatpy.util import result2array
from Space.EngineModel import ThrustModel, TorqueModel
from Visualisation import Plotting as Plot
from Visualisation import PlotResults as PlotRes
from Dynamics import HCWDynamics, ROEDynamics, SystemDynamics, AttitudeDynamics, DifferentialDragDynamics, BlendDynamics
from scipy.spatial.transform import Rotation
from Scenarios.MainScenarios import Scenario, Model
import Utils.Conversions as Conversion
from Filters.KalmanFilter import KalmanFilter
from Scenarios.OrbitalScenarios import OrbitGroup
from Simulator import Dynamics_simulator as mean_simulator


class OrbitalMechSimulator:
    """
    Class to simulate several satellites in the same orbit using a high-fidelity simulator. 
    """

    def __init__(self, scenario: Scenario, reference_angle_offsets: np.ndarray = None):
        self.kalman_state = None
        self.bodies = None
        self.central_bodies = None
        self.controlled_satellite_names = None
        self.all_satellite_names = None
        self.number_of_total_satellites = None
        self.number_of_controlled_satellites = None
        self.acceleration_model = None
        self.torque_model = None
        self.input_thrust_model = None
        self.input_torque_model = None
        self.torque_engine_models = None
        self.mass_rate_models = None
        self.satellite_mass = None
        self.simulation_timestep = None

        self.initial_position = None
        self.initial_orientation = None
        self.propagator_settings = None

        self.dependent_variables_trans = []
        self.dependent_variables_rot = []
        self.dependent_variables_mass = []
        self.dependent_variables_dict = {'Time': [0]}
        self.dependent_variables_idx = 1

        self.states = None
        self.dep_vars = None
        self.thrust_forces = None
        self.control_torques = None
        self.cylindrical_states = None
        self.quasi_roe = None
        self.roe = None
        self.euler_angles = None
        self.angular_velocities = None
        self.rsw_quaternions = None
        self.blend_states = None
        self.small_blend_states = None

        self.reference_satellite_added = False
        self.reference_satellite_names = []
        self.translation_states = None
        self.mass_states = None
        self.rotation_states = None

        self.mean_motion = None
        self.orbital_derivative = None
        self.gravitational_constant = scenario.physics.gravitational_parameter_Earth

        self.scenario = scenario
        self.filter = KalmanFilter(scenario.simulation.simulation_timestep, scenario.physics.radius_Earth,
                                   scenario.physics.gravitational_parameter_Earth,
                                   scenario.physics.J2_value * self.scenario.physics.J2_perturbation,
                                   scenario.control.control_timestep, scenario.physics.mass)
        self.filtered_oe = None
        self.filtered_oe_ref = None

        self.thrust_inputs = None
        self.initial_reference_state = None
        self.oe_mean_0 = None

        if isinstance(self.scenario.orbital, OrbitGroup):
            self.number_of_orbits = len(self.scenario.orbital.longitude)
        else:
            self.number_of_orbits = 1

        self.true_anomalies = None
        self.reference_angle_offsets = reference_angle_offsets
        self.number_of_simulation_steps = None
        self.initial_state_oe = None
        self.solver_time = 0

        self.update_translation_states = False
        self.sigma_A = None
        self.sigma_B = None

    def create_bodies(self, number_of_satellites: int, satellite_mass: float, satellite_inertia: np.ndarray[3, 3],
                      add_reference_satellite: bool = False, use_parameters_from_scenario: Scenario = None) -> None:
        """
        Create the different bodies used during a simulation.

        :param number_of_satellites: Number of satellites (excluding a possible virtual reference satellite) to include.
        :param satellite_mass: Mass of each satellite.
        :param satellite_inertia: Mass moment of inertia of each satellite.
        :param add_reference_satellite: Whether to add a (virtual) reference satellite.
        :param use_parameters_from_scenario: It is possible to use physical parameters set by the scenario
                                             if this is not None.
        """
        # Create default body settings for "Earth"
        bodies_to_create = ["Earth"]

        # Create default body settings for bodies_to_create,
        # with "Earth"/"J2000" as the global frame origin and orientation
        global_frame_origin = "Earth"
        global_frame_orientation = "J2000"
        body_settings = environment_setup.get_default_body_settings(
            bodies_to_create, global_frame_origin, global_frame_orientation)

        if use_parameters_from_scenario is not None:
            body_settings.get('Earth').shape_settings.radius = use_parameters_from_scenario.physics.radius_Earth
            body_settings.get("Earth").gravity_field_settings = environment_setup.gravity_field.spherical_harmonic(
                use_parameters_from_scenario.physics.gravitational_parameter_Earth,
                use_parameters_from_scenario.physics.radius_Earth * 1.00111886533,
                # Original value also slightly larger...
                body_settings.get("Earth").gravity_field_settings.normalized_cosine_coefficients,
                body_settings.get("Earth").gravity_field_settings.normalized_sine_coefficients,
                body_settings.get("Earth").gravity_field_settings.associated_reference_frame)

            self.gravitational_constant = use_parameters_from_scenario.physics.gravitational_parameter_Earth

            # body_settings.get('Earth').gravity_field_settings.gravitational_parameter =
            # body_settings.get('Earth').gravity_field_settings.reference_radius =
        # Create system of bodies (in this case only Earth)
        self.bodies = environment_setup.create_system_of_bodies(body_settings)

        # Create bodies for all the satellites
        self.satellite_mass = satellite_mass
        self.controlled_satellite_names = []
        self.number_of_controlled_satellites = number_of_satellites
        for i in range(self.number_of_controlled_satellites):
            satellite_name = f"Satellite_{i}"
            self.controlled_satellite_names.append(satellite_name)
            self.bodies.create_empty_body(satellite_name)
            self.bodies.get(satellite_name).mass = satellite_mass
            self.bodies.get(satellite_name).inertia_tensor = satellite_inertia

        if add_reference_satellite:
            self.reference_satellite_added = True
            self.number_of_total_satellites = self.number_of_controlled_satellites + self.number_of_orbits
            self.all_satellite_names = np.copy(self.controlled_satellite_names).tolist()
            for ref_sat in range(self.number_of_orbits):
                satellite_name = f"Satellite_ref_{ref_sat}"
                self.bodies.create_empty_body(satellite_name)
                self.bodies.get(satellite_name).mass = satellite_mass
                self.bodies.get(satellite_name).inertia_tensor = satellite_inertia

                self.all_satellite_names.append(satellite_name)
                self.reference_satellite_names.append(satellite_name)
        else:
            self.all_satellite_names = self.controlled_satellite_names
            self.number_of_total_satellites = self.number_of_controlled_satellites

        # Create a list with the central body for each satellite
        self.central_bodies = self.number_of_total_satellites * ["Earth"]

    def create_engine_models_thrust(self, control_timestep: int, thrust_inputs: np.ndarray,
                                    specific_impulse: float) -> None:
        """
        Add a thruster to provide given control inputs for thrust to each satellite.

        :param control_timestep: Timestep between each control input.
        :param thrust_inputs: Numpy array of control inputs of the shape (number_of_satellites, 3, time).
        :param specific_impulse: Specific impulse for each engine.
        """
        self.input_thrust_model = [None] * self.number_of_controlled_satellites
        self.thrust_inputs = thrust_inputs

        # Add thruster model for every satellite
        for idx, satellite_name in enumerate(self.controlled_satellite_names):
            # Set up the thrust model for the satellite
            self.input_thrust_model[idx] = ThrustModel(thrust_inputs[idx], control_timestep, specific_impulse,
                                                       self.bodies.get(satellite_name))

            rotation_model_settings = environment_setup.rotation_model.custom_inertial_direction_based(
                self.input_thrust_model[idx].get_thrust_direction, "J2000", "VehicleFixed")
            environment_setup.add_rotation_model(self.bodies, satellite_name, rotation_model_settings)

            # Define the thrust magnitude settings for the satellite from the custom functions
            thrust_magnitude_settings = propagation_setup.thrust.custom_thrust_magnitude(
                self.input_thrust_model[idx].get_thrust_magnitude, self.input_thrust_model[idx].get_specific_impulse)

            environment_setup.add_engine_model(satellite_name, satellite_name + "_thrust_engine",
                                               thrust_magnitude_settings, self.bodies)

    def create_engine_models_torque(self, control_timestep: int, torque_inputs: np.ndarray,
                                    specific_impulse: float) -> None:
        """
        Add a thruster to provide given control inputs for thrust to each satellite.

        :param control_timestep: Timestep between each control input.
        :param torque_inputs: Numpy array of control inputs of the shape (number_of_satellites, 3, time).
        :param specific_impulse: Specific impulse for each engine.
        """
        self.torque_engine_models = [None] * self.number_of_controlled_satellites
        self.input_torque_model = [None] * self.number_of_controlled_satellites

        # Add thruster model for every satellite
        for idx, satellite_name in enumerate(self.controlled_satellite_names):
            # Set up the thrust model for the satellite
            self.input_torque_model[idx] = TorqueModel(torque_inputs[idx], control_timestep,
                                                       self.bodies.get(satellite_name))

            # Define the thrust magnitude settings for the satellite from the custom functions
            self.torque_engine_models[idx] = propagation_setup.torque.custom_torque(
                self.input_torque_model[idx].get_torque)

    def update_engine_models(self, thrust_inputs: np.ndarray, torque_inputs: np.ndarray) -> None:
        """
        Update the engine models with these new inputs.

        :param thrust_inputs: Thrust inputs in shape (satellites, 3, time).
        :param torque_inputs: Torque inputs in shape (satellites, 3, time).
        """
        self.thrust_inputs = thrust_inputs
        for i in range(self.number_of_controlled_satellites):
            self.input_thrust_model[i].update_thrust(thrust_inputs[i])
            # self.input_torque_model[i].update_torque(torque_inputs[i])

    def create_acceleration_model(self, order_of_zonal_harmonics: int = 0) -> None:
        """
        Create an acceleration model of all satellites, including a desired order of zonal harmonics and possibly
        the thrust from the engines.

        :param order_of_zonal_harmonics: Order of zonal harmonics to include. If set to 0, point mass gravity is used.
        """
        if order_of_zonal_harmonics == 0:
            gravitational_function = propagation_setup.acceleration.point_mass_gravity()
        else:
            gravitational_function = propagation_setup.acceleration. \
                spherical_harmonic_gravity(order_of_zonal_harmonics, order_of_zonal_harmonics)

        acceleration_settings = dict()
        for satellite_name in self.controlled_satellite_names:
            # Define accelerations acting on a single satellite
            acceleration_settings_satellite = {
                "Earth": [gravitational_function],
                satellite_name: [propagation_setup.acceleration.thrust_from_all_engines()]
            }

            acceleration_settings[satellite_name] = acceleration_settings_satellite

        # Reference satellite has no thrusters
        if self.reference_satellite_added:
            for satellite_name in self.reference_satellite_names:
                acceleration_settings[satellite_name] = {"Earth": [gravitational_function]}

        # Create acceleration models
        self.acceleration_model = propagation_setup.create_acceleration_models(
            self.bodies, acceleration_settings, self.all_satellite_names, self.central_bodies
        )

    def create_torque_model(self) -> None:
        """
        Create a torque model for all satellites (including a possible reference satellite).
        """
        torque_settings = dict()

        if self.torque_engine_models is None:
            raise Exception("Add a torque provided by the engines first. ")

        for idx, satellite_name in enumerate(self.controlled_satellite_names):
            torque_settings_satellite = {"Earth": [propagation_setup.torque.second_degree_gravitational()],
                                         satellite_name: [self.torque_engine_models[idx]]
                                         }
            torque_settings[satellite_name] = torque_settings_satellite

        if self.reference_satellite_added:
            for satellite_name in self.reference_satellite_names:
                torque_settings[satellite_name] = \
                    {"Earth": [propagation_setup.torque.second_degree_gravitational()]}

        self.torque_model = propagation_setup.create_torque_models(self.bodies, torque_settings,
                                                                   self.controlled_satellite_names)

    def create_mass_rate_model(self) -> None:
        """
        Create a mass rate model for all satellites (except the possible reference satellite).
        """
        mass_rate_settings = dict()

        for satellite_name in self.controlled_satellite_names:
            mass_rate_settings[satellite_name] = [propagation_setup.mass_rate.from_thrust()]

        self.mass_rate_models = propagation_setup.create_mass_rate_models(
            self.bodies,
            mass_rate_settings,
            self.acceleration_model
        )

    def convert_orbital_elements_to_cartesian(self, true_anomalies: list[float], orbit_height: float,
                                              eccentricity: float = 0, inclination: float = 0,
                                              argument_of_periapsis: float = 0, longitude: list[float] = 0,
                                              orbital_element_offsets: list[int | float] = None) -> np.ndarray:
        """
        Create a desired state array in cartesian coordinates provided the orbital elements for all satellites,
        assuming the that all satellites are in the same orbit (but the true anomalies differ).

        :param true_anomalies: True anomalies of satellites in degrees.
        :param orbit_height: Height of the orbit above the surface of the Earth in m.
        :param eccentricity: Eccentricity of the orbit.
        :param inclination: Inclination of the orbit.
        :param argument_of_periapsis: Argument of periapsis of the orbit.
        :param longitude: List with the longitude of the ascending node of the orbit.
        :param orbital_element_offsets: List with offsets for the orbital elements.
        :return: Numpy array with the states of all satellites in cartesian coordinates.
        """
        if not len(true_anomalies) == self.number_of_controlled_satellites:
            raise Exception(f"Not enough true anomalies. Received {len(true_anomalies)} variables, "
                            f"but expected {self.number_of_controlled_satellites} variables. ")

        if len(longitude) == 1:
            longitude = longitude * self.number_of_controlled_satellites

        # Retrieve gravitational parameter
        earth_gravitational_parameter = self.bodies.get("Earth").gravitational_parameter

        # Retrieve Earth's radius
        earth_radius = self.bodies.get("Earth").shape_model.average_radius

        orbit_semi_major_axis = earth_radius + orbit_height
        orbit_eccentricity = eccentricity
        orbit_inclination = np.deg2rad(inclination)
        orbit_argument_of_periapsis = np.deg2rad(argument_of_periapsis)
        orbit_longitude = np.deg2rad(longitude)

        true_anomalies = [element_conversion.mean_to_true_anomaly(eccentricity, np.deg2rad(mean_anomaly)) for
                          mean_anomaly in true_anomalies]
        orbit_anomalies = true_anomalies

        if orbital_element_offsets is None:
            orbital_element_offsets = np.zeros((6, (len(true_anomalies) + self.number_of_orbits)))

        initial_states = np.array([])
        self.kalman_state = np.zeros((4 * self.number_of_controlled_satellites,))
        self.initial_state_oe = np.zeros((6 * self.number_of_controlled_satellites,))

        for idx, anomaly in enumerate(orbit_anomalies):
            self.kalman_state[idx * 4:(idx + 1) * 4] = np.array([orbit_semi_major_axis, orbit_eccentricity,
                                                                 orbit_inclination, orbit_longitude[idx]])
            self.initial_state_oe[idx * 6:(idx + 1) * 6] = np.array([orbit_semi_major_axis + orbital_element_offsets[0, idx],
                                                                     orbit_eccentricity + orbital_element_offsets[1, idx],
                                                                     orbit_inclination + orbital_element_offsets[2, idx],
                                                                     orbit_argument_of_periapsis+ orbital_element_offsets[3, idx],
                                                                     orbit_longitude[idx] + orbital_element_offsets[4, idx],
                                                                     anomaly + orbital_element_offsets[5, idx]])

            initial_states = np.append(initial_states,
                                       element_conversion.keplerian_to_cartesian_elementwise(
                                           gravitational_parameter=earth_gravitational_parameter,
                                           semi_major_axis=orbit_semi_major_axis + orbital_element_offsets[0, idx],
                                           eccentricity=orbit_eccentricity + orbital_element_offsets[1, idx],
                                           inclination=orbit_inclination + orbital_element_offsets[2, idx],
                                           argument_of_periapsis=orbit_argument_of_periapsis + orbital_element_offsets[
                                               3, idx],
                                           longitude_of_ascending_node=orbit_longitude[idx] + orbital_element_offsets[
                                               4, idx],
                                           true_anomaly=anomaly + orbital_element_offsets[5, idx]
                                       ))

        self.filter.initialise_filter(self.kalman_state)

        if self.reference_satellite_added:
            satellites_per_plane = self.number_of_controlled_satellites // self.number_of_orbits
            for ref_sat in range(self.number_of_orbits):
                initial_states = np.append(initial_states,
                                           element_conversion.keplerian_to_cartesian_elementwise(
                                               gravitational_parameter=earth_gravitational_parameter,
                                               semi_major_axis=orbit_semi_major_axis + orbital_element_offsets[
                                                   0, self.number_of_controlled_satellites + ref_sat],
                                               eccentricity=orbit_eccentricity + orbital_element_offsets[
                                                   1, self.number_of_controlled_satellites + ref_sat],
                                               inclination=orbit_inclination + orbital_element_offsets[
                                                   2, self.number_of_controlled_satellites + ref_sat],
                                               argument_of_periapsis=orbit_argument_of_periapsis +
                                                                     orbital_element_offsets[
                                                                         3, self.number_of_controlled_satellites + ref_sat],
                                               longitude_of_ascending_node=orbit_longitude[
                                                                               satellites_per_plane * ref_sat] +
                                                                           orbital_element_offsets[
                                                                               4, self.number_of_controlled_satellites + ref_sat],
                                               true_anomaly=orbital_element_offsets[
                                                   5, self.number_of_controlled_satellites + ref_sat]
                                               # Reference always at 0
                                           ))

        self.initial_reference_state = np.tile(np.array([orbit_semi_major_axis, orbit_eccentricity, orbit_inclination,
                                                         orbit_argument_of_periapsis, orbit_longitude[0], 0]),
                                               (self.number_of_orbits, 1))

        self.initial_reference_state[:, 4] = np.deg2rad(self.scenario.orbital.longitude)
        # print(self.initial_state_oe)
        return initial_states

    def convert_orbital_elements_to_quaternion(self, true_anomalies: list[float], initial_angular_velocity: np.ndarray,
                                               inclination: float = 0, longitude: float = 0,
                                               initial_angle_offset: Rotation = None,
                                               initial_velocity_offset: np.ndarray = None) -> np.array:
        """
        Convert a set of orbital elements to the initial orientation of the satellites when they are nadir pointing.

        :param true_anomalies: List of true anomalies for all controlled satellites.
        :param initial_angular_velocity: Initial rotational velocity in the inertial frame. This is the base value,
                                         specific offsets can be added through `initial_velocity_offset'.
        :param inclination: The inclination of the orbit for the satellites in degrees.
        :param longitude: Longitude of the ascending node in degrees.
        :param initial_angle_offset: An initial offset in the rotation. Should be a Rotation object.
        :param initial_velocity_offset: Offset in angular velocity around base for each satellite.
                                        Shape: (3, number_of_satellites)
        :return: Numpy array with initial state for the orientations.
        """
        initial_orientation = np.zeros((7 * self.number_of_total_satellites,))

        if initial_angle_offset is None:
            initial_angle_offset = Rotation.from_euler('x', 0, degrees=True)  # Zero rotation as offset

        if len(true_anomalies) != self.number_of_controlled_satellites:
            raise Exception("True anomalies should contain exactly 1 anomaly per controlled satellite!")
        else:
            true_anomalies.append(0)  # Add a zero for the reference satellite
            initial_velocity_offset = np.concatenate((initial_velocity_offset, np.zeros((3, 1))), axis=1)

        for idx, true_anomaly in enumerate(true_anomalies):
            # Create quaternion with minus signs because nadir
            initial_rotation = Rotation.from_euler('ZXZ', [-true_anomaly, -inclination, -longitude], degrees=True)
            initial_rotation *= initial_angle_offset
            initial_rotation_quat = initial_rotation.as_quat()[[3, 0, 1, 2]]  # Swap order for quaternions

            # Find angular velocities and concatenate
            angular_velocity = initial_rotation.apply(initial_angular_velocity + initial_velocity_offset[:, idx],
                                                      inverse=True)
            initial_state_idx = np.concatenate((initial_rotation_quat, angular_velocity))
            initial_orientation[idx * 7:(idx + 1) * 7] = initial_state_idx

        return initial_orientation

    def set_initial_position(self, initial_state_in_cartesian: np.ndarray) -> None:
        """
        Set the initial position for the translation.

        :param initial_state_in_cartesian: Initial state in cartesian coordinates. States for different satellites
                                           are stacked and each state itself is of shape (6, 1).
        """
        self.initial_position = initial_state_in_cartesian

    def set_initial_orientation(self, initial_rotation: np.ndarray[np.float64]) -> None:
        """
        Set the initial states for the rotation.

        :param initial_rotation: Initial state for the orientation of shape (7 * total_number_of_satellites, ).
                                 In the order of [q_0, omega_0, q_1, omega_1, etc.]
        """
        self.initial_orientation = initial_rotation

    def update_mean_orbital_elements(self, mean_orbital_elements) -> None:
        """
        Update the mean orbital elements vector.

        :param mean_orbital_elements: Initial state in mean orbital elements.
        """
        self.initial_state_oe = mean_orbital_elements

    def set_dependent_variables_translation(self, add_keplerian_state: bool = True, add_thrust_accel: bool = False,
                                            add_rsw_rotation_matrix: bool = False) -> None:
        """
        Add dependent variables for the translational propagation.

        :param add_keplerian_state: Whether to add keplerian states to the dependent variables.
        :param add_thrust_accel: Whether to add accelerations due to thrust to the dependent variables.
        :param add_rsw_rotation_matrix: Whether to add the RSW to inertial rotation matrix to the dependent variables.
        """
        if add_keplerian_state:
            keplerian_state_dict = {}  # Dict to keep track of indices for dependent variables
            for satellite in self.all_satellite_names:
                self.dependent_variables_trans.extend([
                    propagation_setup.dependent_variable.keplerian_state(satellite, "Earth")
                ])
                keplerian_state_dict[satellite] = np.arange(self.dependent_variables_idx,
                                                            self.dependent_variables_idx + 6)
                self.dependent_variables_idx += 6

            self.dependent_variables_dict["keplerian state"] = keplerian_state_dict

        if add_thrust_accel:
            thrust_acc_dict = {}
            for satellite in self.controlled_satellite_names:
                self.dependent_variables_trans.extend([
                    propagation_setup.dependent_variable.
                    single_acceleration(propagation_setup.acceleration.AvailableAcceleration.thrust_acceleration_type,
                                        satellite, satellite)
                ])
                thrust_acc_dict[satellite] = np.arange(self.dependent_variables_idx, self.dependent_variables_idx + 3)
                self.dependent_variables_idx += 3

            self.dependent_variables_dict["thrust acceleration"] = thrust_acc_dict

        # Can be used for the conversion to anything LVLH/RSW frame related, e.g. cylindrical coordinates
        # or euler angles
        if add_rsw_rotation_matrix:
            rsw_matrix_dict = {}
            for satellite in self.all_satellite_names:
                self.dependent_variables_trans.extend([
                    propagation_setup.dependent_variable.rsw_to_inertial_rotation_matrix(satellite, "Earth")
                ])
                rsw_matrix_dict[satellite] = np.arange(self.dependent_variables_idx, self.dependent_variables_idx + 9)
                self.dependent_variables_idx += 9

            self.dependent_variables_dict["rsw rotation matrix"] = rsw_matrix_dict

    def set_dependent_variables_rotation(self, add_control_torque: bool = True, add_torque_norm: bool = False,
                                         add_body_rotation_matrix: bool = True) -> None:
        """
        Add dependent variables for the attitude propagation.

        :param add_control_torque: Whether to add the control torques to the dependent variables.
        :param add_torque_norm: Whether to add the norm of the applied torques to the dependent variables.
        :param add_body_rotation_matrix: Whether to add the body to inertial rotation matrix to the dependent variables.
        """
        # Used for the conversion of moments between body and inertial frame.
        if add_body_rotation_matrix:
            body_matrix_dict = {}
            for satellite in self.controlled_satellite_names:
                self.dependent_variables_trans.extend([
                    propagation_setup.dependent_variable.inertial_to_body_fixed_rotation_frame(satellite)
                ])
                body_matrix_dict[satellite] = np.arange(self.dependent_variables_idx, self.dependent_variables_idx + 9)
                self.dependent_variables_idx += 9

            self.dependent_variables_dict["body rotation matrix"] = body_matrix_dict

        # Adding a control torque to the dependent variables does not seem possible to do directly.
        # Instead we save the total torque and the other torques, and subtract those later.
        if add_control_torque:
            torque_control_dict = {}
            for satellite in self.controlled_satellite_names:
                self.dependent_variables_rot.extend([
                    propagation_setup.dependent_variable.total_torque(satellite),
                    propagation_setup.dependent_variable.single_torque(
                        propagation_setup.torque.AvailableTorque.second_order_gravitational_type,
                        satellite, "Earth")
                ])
                torque_control_dict[satellite] = np.arange(self.dependent_variables_idx,
                                                           self.dependent_variables_idx + 6)
                self.dependent_variables_idx += 6

            self.dependent_variables_dict["control torque"] = torque_control_dict

        if add_torque_norm:
            torque_norm_dict = {}
            for satellite in self.controlled_satellite_names:
                self.dependent_variables_rot.extend([
                    propagation_setup.dependent_variable.total_torque_norm(satellite)
                ])
                torque_norm_dict[satellite] = self.dependent_variables_idx
                self.dependent_variables_idx += 1

            self.dependent_variables_dict["torque norm"] = torque_norm_dict

    def set_dependent_variables_mass(self, add_mass: bool = True) -> None:
        """
        Add dependent variables relating the mass propagation.

        :param add_mass: Whether to add the mass to the dependent variables.
        """
        if add_mass:
            mass_dict = {}
            for satellite in self.controlled_satellite_names:
                self.dependent_variables_mass.extend([
                    propagation_setup.dependent_variable.body_mass(satellite)
                ])
                mass_dict[satellite] = self.dependent_variables_idx
                self.dependent_variables_idx += 1

            self.dependent_variables_dict["mass"] = mass_dict

    def create_propagation_settings(self, start_epoch: float, end_epoch: float, simulation_step_size: float) -> None:
        """
        Create the propagation settings used for simulation. This method automatically selects the correct type of
        propagators.

        :param start_epoch: Start time of the simulation.
        :param end_epoch: End time of the simulation.
        :param simulation_step_size: Step size during the simulation.
        """
        self.simulation_timestep = simulation_step_size
        integrator_settings = propagation_setup.integrator.runge_kutta_4(self.simulation_timestep)

        # Small offset to prevent the simulation from doing another step at 4.99999 when going to 5
        termination_settings = propagation_setup.propagator.time_termination(end_epoch - 0.0001)

        self.number_of_simulation_steps = int((end_epoch + 0.00001 - start_epoch) / self.simulation_timestep)

        propagator_settings_list = []
        dependent_variables_list = []
        if self.acceleration_model is not None:
            translational_propagator_settings = propagation_setup.propagator.translational(
                self.central_bodies,
                self.acceleration_model,
                self.all_satellite_names,
                self.initial_position,
                start_epoch,
                integrator_settings,
                termination_settings,
                output_variables=self.dependent_variables_trans
            )
            propagator_settings_list.append(translational_propagator_settings)
            dependent_variables_list.extend(self.dependent_variables_trans)

        if self.torque_model is not None:
            rotational_propagator_settings = propagation_setup.propagator.rotational(
                self.torque_model,
                self.all_satellite_names,
                self.initial_orientation,
                start_epoch,
                integrator_settings,
                termination_settings,
                output_variables=self.dependent_variables_rot
            )
            propagator_settings_list.append(rotational_propagator_settings)
            dependent_variables_list.extend(self.dependent_variables_rot)

        if self.mass_rate_models is not None:
            initial_mass = []
            for satellite in self.controlled_satellite_names:
                initial_mass.append(self.bodies.get(satellite).mass)

            mass_propagator_settings = propagation_setup.propagator.mass(
                self.controlled_satellite_names,
                self.mass_rate_models,
                initial_mass,
                start_epoch,
                integrator_settings,
                termination_settings,
                output_variables=self.dependent_variables_mass
            )
            propagator_settings_list.append(mass_propagator_settings)
            dependent_variables_list.extend(self.dependent_variables_mass)

        if len(propagator_settings_list) == 0:
            raise Exception("No acceleration, torque or mass model defined."
                            "There is nothing to propagate.")
        elif len(propagator_settings_list) == 1:
            self.propagator_settings = propagator_settings_list[0]
        else:
            self.propagator_settings = propagation_setup.propagator.multitype(
                propagator_settings_list,
                integrator_settings,
                start_epoch,
                termination_settings,
                output_variables=dependent_variables_list
            )

    def simulate_system(self, disturbance_free: bool = False) -> (np.ndarray, np.ndarray):
        """
        Run a simulation of the system

        :return: Either a tuple of two Numpy arrays (if at least one dependent variable has been added) with the states
                 and the dependent variables, or only the states.
        """
        if disturbance_free:
            disturbance_to_use = None
        else:
            disturbance_to_use = self.scenario.disturbance
        # Create simulation object and propagate the dynamics
        if self.scenario.use_mean_simulator:
            # print(self.initial_state_oe)
            dynamics_simulator = mean_simulator.create_dynamics_simulator(self.initial_state_oe, self.thrust_inputs,
                                                                          self.scenario,
                                                                          self.number_of_simulation_steps,
                                                                          disturbance_to_use)
            self.thrust_forces = self.thrust_inputs.T
            self.update_translation_states = False
            # self.initial_state_oe = dynamics_simulator.dependent_variable_history[self.number_of_simulation_steps]
        else:
            with util.redirect_std():
                dynamics_simulator = create_dynamics_simulator(self.bodies, self.propagator_settings)
        # dynamics_simulator = create_dynamics_simulator(self.bodies, self.propagator_settings)

        states = dynamics_simulator.state_history
        dep_vars = dynamics_simulator.dependent_variable_history
        self.states = result2array(states)

        # Extract different types of states from self.states.
        # Assumption: translation always present, as a position is also needed for basic torque calculations.
        self.translation_states = self.states[:, 1:6 * self.number_of_total_satellites + 1]

        if self.mass_rate_models is not None:
            self.mass_states = self.states[:, 6 * self.number_of_total_satellites + 1:
                                              7 * self.number_of_total_satellites + 1]

        if self.mass_states is not None and self.torque_model is not None:
            self.rotation_states = self.states[:, 7 * self.number_of_total_satellites + 1:]
        elif self.torque_model is not None:
            self.rotation_states = self.states[:, 6 * self.number_of_total_satellites + 1:]

        # Reset variables possibly used before
        self.cylindrical_states = None
        self.quasi_roe = None
        self.euler_angles = None
        self.angular_velocities = None
        self.rsw_quaternions = None
        self.blend_states = None
        self.small_blend_states = None
        self.filtered_oe = None
        self.filtered_oe_ref = None

        if len(dep_vars) == 0:
            return self.states
        else:
            self.dep_vars = result2array(dep_vars)
            return self.states, self.dep_vars

    def filter_oe(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Filter the orbital elements of the reference and the controlled satellites.

        :return: Tuple with (oe_sat, oe_ref).
        """
        if self.filtered_oe_ref is None and self.filtered_oe is None:
            # Filter
            oe_sat = self.dep_vars[:,
                     self.dependent_variables_dict['keplerian state'][self.controlled_satellite_names[0]][0]:
                     self.dependent_variables_dict['keplerian state'][self.controlled_satellite_names[-1]][-1] + 1]

            time = np.arange(0, oe_sat.shape[0], self.simulation_timestep).reshape((-1, 1))

            oe_der = np.array([self.orbital_derivative[0], 0, self.orbital_derivative[2], self.orbital_derivative[4],
                               self.orbital_derivative[3], self.orbital_derivative[5]])

            if self.oe_mean_0 is None:
                self.oe_mean_0 = self.initial_reference_state.reshape((-1, 1, 6))
            self.filtered_oe_ref = self.oe_mean_0 + time * oe_der
            self.filtered_oe_ref[:, :, 5:6] += time * self.mean_motion
            self.filtered_oe_ref[:, :, [3, 5]] %= 2 * np.pi

            if self.filtered_oe_ref.shape[1] > 2:
                self.oe_mean_0 = self.filtered_oe_ref[:, -1:]

            if not self.scenario.use_mean_simulator:
                thrust_calculated = False
                if self.thrust_inputs is None:
                    self.get_thrust_forces_from_acceleration()
                    self.thrust_inputs = self.thrust_forces.T
                    thrust_calculated = True

                oe_sat[:, 4::6][:, oe_sat[0, 4::6] > np.pi] -= 2 * np.pi
                oe_sat[:, 4::6] = np.unwrap(oe_sat[:, 4::6], axis=0)

                self.filtered_oe = self.filter.filter_data(oe_sat,
                                                           self.thrust_inputs.reshape(
                                                               (3 * self.number_of_controlled_satellites, -1)).T,
                                                           update_state=oe_sat.shape[0] > 5,
                                                           inputs_with_same_sampling_time=thrust_calculated)
            else:
                self.filtered_oe = oe_sat

        return self.filtered_oe, self.filtered_oe_ref

    def convert_to_cylindrical_coordinates(self) -> np.ndarray:
        """
        Convert the states from cartesian coordinates to relative cylindrical coordinates.
        Store the result in self.cylindrical_states and return them.

        :return: Array with cylindrical coordinates in shape (t, 6 * number_of_controlled_satellites)
        """
        oe_filtered, oe_ref = self.filter_oe()
        self.cylindrical_states = Conversion.oe2cylindrical(oe_filtered, self.gravitational_constant, oe_ref,
                                                            self.reference_angle_offsets)

        return self.cylindrical_states

    def convert_to_quasi_roe(self) -> np.ndarray:
        """
        Convert the states from keplerian coordinates to quasi ROE.

        :return: Array with quasi ROE in shape (t, 6 * number_of_controlled_satellites)
        """
        oe_filtered, oe_ref = self.filter_oe()
        self.quasi_roe = Conversion.oe2quasi_roe(oe_filtered, oe_ref, self.reference_angle_offsets)

        return self.quasi_roe

    def convert_to_roe(self) -> np.ndarray:
        """
        Convert the states from keplerian coordinates to ROE.

        :return: Array with ROE in shape (t, 6 * number_of_controlled_satellites)
        """
        self.roe = np.zeros_like(self.translation_states[:, :-6])

        # kepler_ref_0 = self.dep_vars[0, self.dependent_variables_dict["keplerian state"][self.reference_satellite_name]]
        # time = np.arange(0, self.roe[:, 0].shape[0], self.simulation_timestep).reshape((-1, 1))
        # kepler_ref = kepler_ref_0 + time * self.orbital_derivative
        kepler_ref = self.dep_vars[:, self.dependent_variables_dict["keplerian state"][self.reference_satellite_name]]

        mean_anomaly_ref = np.zeros_like(kepler_ref[:, 0:1]) % (2 * np.pi)
        for t in range(kepler_ref.shape[0]):
            mean_anomaly_ref[t] = element_conversion.true_to_mean_anomaly(kepler_ref[t, 1], kepler_ref[t, 5]) % (
                    2 * np.pi)

        for idx, satellite_name in enumerate(self.controlled_satellite_names):
            kepler_sat = self.dep_vars[:, self.dependent_variables_dict["keplerian state"][satellite_name]]
            delta_a = (kepler_sat[:, 0:1] - kepler_ref[:, 0:1]) / kepler_ref[:, 0:1]
            delta_ex = kepler_sat[:, 1:2] - kepler_ref[:, 1:2]
            delta_ey = kepler_sat[:, 3:4] - kepler_ref[:, 3:4] + (kepler_sat[:, 4:5] - kepler_ref[:, 4:5]) * np.cos(
                kepler_ref[:, 2:3])
            delta_ix = kepler_sat[:, 2:3] - kepler_ref[:, 2:3]
            delta_iy = (kepler_sat[:, 4:5] - kepler_ref[:, 4:5]) * np.sin(kepler_ref[:, 2:3])

            mean_anomaly_sat = np.zeros_like(kepler_sat[:, 0:1])
            for t in range(kepler_sat.shape[0]):
                mean_anomaly_sat[t] = element_conversion.true_to_mean_anomaly(kepler_sat[t, 1], kepler_sat[t, 5]) % (
                        2 * np.pi)

            delta_lambda = mean_anomaly_sat - mean_anomaly_ref + np.sqrt(1 - kepler_ref[:, 1:2] ** 2) * delta_ey

            if delta_ey[0] > 0.999 * 2 * np.pi:
                delta_ey[0] = 0

            if delta_lambda[0] < 0:
                delta_lambda += 2 * np.pi

            self.roe[:, idx * 6:(idx + 1) * 6] = np.concatenate((delta_a, delta_lambda, delta_ex,
                                                                 delta_ey, delta_ix, delta_iy), axis=1)
        return self.roe

    def convert_to_blend_states(self) -> np.ndarray:
        """
        Convert the states from keplerian coordinates to blend states.

        :return: Array with blend states in shape (t, 6 * number_of_controlled_satellites)
        """
        # # Filter
        # oe_sat = self.dep_vars[:,
        #          self.dependent_variables_dict['keplerian state'][self.controlled_satellite_names[0]][0]:
        #          self.dependent_variables_dict['keplerian state'][self.controlled_satellite_names[-1]][-1] + 1]
        #
        # oe_ref = self.dep_vars[:, self.dependent_variables_dict["keplerian state"][self.reference_satellite_name]]
        #
        # oe_filtered = self.filter.filter_data(oe_sat,
        #                                       self.thrust_inputs.reshape(
        #                                           (3 * self.number_of_controlled_satellites, -1)).T,
        #                                       update_state=oe_sat.shape[0] > 5)
        #
        # if oe_sat.shape[0] > 5:
        #     fig, ax = plt.subplots(3, 2, figsize=(16, 9))
        #     ax = list(fig.get_axes())
        #
        #     ax[0].plot(oe_sat[:, 0::6])
        #     ax[0].plot(oe_filtered[:, 0::6], '--')
        #
        #     ax[1].plot(oe_sat[:, 1::6])
        #     ax[1].plot(oe_filtered[:, 1::6], '--')
        #
        #     ax[2].plot(oe_sat[:, 2::6])
        #     ax[2].plot(oe_filtered[:, 2::6], '--')
        #
        #     ax[3].plot(oe_sat[:, 3::6])
        #     ax[3].plot(oe_filtered[:, 3::6], '--')
        #
        #     ax[4].plot(oe_sat[:, 4::6])
        #     ax[4].plot(oe_filtered[:, 4::6], '--')
        #
        #     ax[5].plot(oe_sat[:, 5::6])
        #     ax[5].plot(oe_filtered[:, 5::6], '--')
        #
        #     plt.show()

        oe_filtered, oe_ref = self.filter_oe()
        self.blend_states = Conversion.oe2blend(oe_filtered, oe_ref, self.reference_angle_offsets)

        return self.blend_states

    def convert_to_small_blend_states(self) -> np.ndarray:
        """
        Convert the states from keplerian coordinates to small blend states.

        :return: Array with small blend states in shape (t, 4 * number_of_controlled_satellites)
        """
        oe_filtered, oe_ref = self.filter_oe()
        self.small_blend_states = Conversion.oe2small_blend(oe_filtered, oe_ref, self.reference_angle_offsets)
        self.true_anomalies = oe_filtered[-1, 5::6].tolist()

        return self.small_blend_states

    def convert_to_rsw_quaternions(self) -> np.ndarray:
        """
        The stored quaternions describe a rotation from body-fixed frame to inertial frame.
        This method calculates another set of quaternions: those from body-fixed frame to rsw frame.
        This can be useful for nadir pointing satellites.

        :return: Array with quaternions in shape (t, 4 * number_of_controlled_satellites).
        """
        self.rsw_quaternions = np.zeros((len(self.rotation_states[:, 0]), 4 * self.number_of_controlled_satellites))

        # Quaternions now describe rotation from body fixed frame to inertial frame.
        # Find the quaternions describing the rotation from body fixed frame to LVLH frame
        for i in range(self.number_of_controlled_satellites):
            for t in range(len(self.rotation_states[:, 0])):
                rsw_rotation_matrix = self.dep_vars[t, self.dependent_variables_dict["rsw rotation matrix"]
                [self.controlled_satellite_names[i]]].reshape((3, 3))
                quaternion_rsw2inertial = Rotation.from_matrix(rsw_rotation_matrix)

                q_scalar_first = self.rotation_states[t,
                                 i * 7:i * 7 + 4]  # We need to swap the order for scipy: scalar last
                quaternion_body2inertial = Rotation.from_quat(q_scalar_first[[1, 2, 3, 0]])
                # quaternion_body2rsw = quaternion_body2inertial * quaternion_rsw2inertial

                q_scalar_last = (quaternion_body2inertial * quaternion_rsw2inertial).as_quat()
                self.rsw_quaternions[t, i * 4:(i + 1) * 4] = q_scalar_last[[3, 0, 1, 2]]  # Put scalar first again

        return self.rsw_quaternions

    def convert_to_euler_angles(self) -> np.ndarray:
        """
        Convert the quaternions to euler angles.

        :return: Array with euler angles in shape (t, 3 * number_of_controlled_satellites)
        """
        self.euler_angles = np.zeros((len(self.rotation_states[:, 0]), 3 * self.number_of_controlled_satellites))

        # Transform that rsw quaternions to euler angles.
        if self.rsw_quaternions is None:
            self.convert_to_rsw_quaternions()

        # # Original paper used different convention of LVLH frame -> need rotation to match
        # rotation_convention = Rotation.from_euler('z', 90, degrees=True) * Rotation.from_euler('x', -90, degrees=True)

        for i in range(self.number_of_controlled_satellites):
            for t in range(len(self.rotation_states[:, 0])):
                q_scalar_first = self.rsw_quaternions[t, i * 4:(i + 1) * 4]
                # quaternion = Rotation.from_quat(q_scalar_first[[1, 2, 3, 0]]) * rotation_convention  # Put scalar last
                # pitch_yaw_roll = quaternion.as_euler('YZX')
                # self.euler_angles[t, i*3:(i+1)*3] = pitch_yaw_roll[[2, 0, 1]]
                pitch_yaw_roll = Rotation.from_quat(q_scalar_first[[1, 2, 3, 0]]).as_euler('ZXY')
                self.euler_angles[t, i * 3:(i + 1) * 3] = np.array([pitch_yaw_roll[2], -pitch_yaw_roll[0],
                                                                    -pitch_yaw_roll[1]])

        return self.euler_angles

    def convert_to_angular_velocities(self) -> np.ndarray:
        """
        Find the angular velocities of the body frame with respect to the LVLH frame.

        :return: Array with velocities in shape (t, 3 * number_of_satellites).
        """
        if self.mean_motion is None:
            raise Exception("Set the mean motion of the satellites first!")

        self.angular_velocities = np.zeros((len(self.rotation_states[:, 0]), 3 * self.number_of_controlled_satellites))

        for idx, satellite_name in enumerate(self.controlled_satellite_names):
            for t in range(len(self.angular_velocities[:, 0])):
                omega_inertial = self.rotation_states[t, idx * 7 + 4:(idx + 1) * 7]
                inertial_to_body_frame = self.dep_vars[
                    t, self.dependent_variables_dict["body rotation matrix"][satellite_name]].reshape((3, 3)).T
                omega_body = np.dot(inertial_to_body_frame, omega_inertial)
                omega_body = np.array([omega_body[1], -omega_body[2], -omega_body[0]])
                self.angular_velocities[t, idx * 3:(idx + 1) * 3] = omega_body - np.array([0, self.mean_motion, 0])
        return self.angular_velocities

    def convert_to_euler_state(self) -> np.ndarray:
        """
        Convert the states to a state with euler angles and angular velocities.

        :return: Array with [euler_angles_0, omega_0, euler_angles_1, omega_1, etc.]
        """

        if self.euler_angles is None:
            self.convert_to_euler_angles()

        if self.angular_velocities is None:
            self.convert_to_angular_velocities()

        states_euler = np.zeros((len(self.euler_angles[:, 0]), 2 * len(self.euler_angles[0])))

        for idx, satellite_name in enumerate(self.controlled_satellite_names):
            for t in range(len(states_euler[:, 0])):
                states_euler[t, idx * 6:idx * 6 + 3] = self.euler_angles[t, idx * 3:(idx + 1) * 3]
                states_euler[t, idx * 6 + 3:(idx + 1) * 6] = self.angular_velocities[t, idx * 3:(idx + 1) * 3]

        return states_euler

    def set_mean_motion_and_orbital_diff(self, mean_motion: float, orbital_diff: np.ndarray) -> None:
        """
        Set the mean motion of the satellites.

        :param mean_motion: The mean motion of the satellites in rad/s
        """
        self.mean_motion = mean_motion
        self.orbital_derivative = orbital_diff

    def get_states_for_dynamical_model(self, dynamical_model: SystemDynamics.TranslationalDynamics) -> np.ndarray:
        """
        Find the states corresponding to a dynamical model, e.g. in cylindrical or ROE coordinates.

        :param dynamical_model:The dynamical model that is used.
        :return: The states for the corresponding model.
        """
        if isinstance(dynamical_model, HCWDynamics.RelCylHCW):
            return self.convert_to_cylindrical_coordinates()
        elif isinstance(dynamical_model, ROEDynamics.QuasiROE):
            return self.convert_to_quasi_roe()
        elif isinstance(dynamical_model, ROEDynamics.ROE):
            return self.convert_to_roe()
        elif isinstance(dynamical_model, DifferentialDragDynamics.DifferentialDragDynamics):
            zero_satellite = np.tile(self.convert_to_cylindrical_coordinates()[:, 1:6:3],
                                     self.number_of_controlled_satellites - 1)
            return self.convert_to_cylindrical_coordinates()[:, 7::3] - zero_satellite
        elif isinstance(dynamical_model, AttitudeDynamics.LinAttModel):
            return self.convert_to_euler_state()
        elif isinstance(dynamical_model, BlendDynamics.BlendSmall):
            return self.convert_to_small_blend_states()
        elif isinstance(dynamical_model, BlendDynamics.Blend):
            return self.convert_to_blend_states()
        else:
            raise Exception("No conversion for this type of model has been implemented yet. ")

    def plot_states_of_dynamical_model(self, dynamical_model: SystemDynamics.TranslationalDynamics,
                                       figure: plt.figure = None) -> plt.figure:
        """
        Plot the states corresponding to a dynamical model, e.g. in cylindrical or ROE coordinates.

        :param dynamical_model:The dynamical model that is used.
        :param figure: Figure to plot to states onto. If None, a new one is created.
        :return: The figure with the states.
        """
        if isinstance(dynamical_model, HCWDynamics.RelCylHCW):
            return self.plot_cylindrical_states(figure=figure)
        elif isinstance(dynamical_model, ROEDynamics.QuasiROE):
            return self.plot_quasi_roe_states(figure=figure)
        elif isinstance(dynamical_model, ROEDynamics.ROE):
            return self.plot_roe_states(figure=figure)
        elif isinstance(dynamical_model, BlendDynamics.Blend):
            return self.plot_blend_states(figure=figure)
        elif isinstance(dynamical_model, BlendDynamics.BlendSmall):
            return self.plot_small_blend_states(figure=figure)
        else:
            raise Exception("No conversion for this type of model has been implemented yet. ")

    def get_thrust_forces_from_acceleration(self) -> np.ndarray:
        """
        Compute the thrust forces from the accelerations that were stored during the simulation.
        This provides an excellent overview of the applied forces, as opposed to the 'planned' forces
        you could get by storing the inputs provided to the simulator.

        Result is stored in self.thrust_forces and returned
        """
        # Accelerations are in inertial frame. Convert to RSW frame
        self.thrust_forces = np.zeros((len(self.translation_states[:, 0]), 3, self.number_of_controlled_satellites))
        for idx, satellite_name in enumerate(self.controlled_satellite_names):
            for t in range(len(self.translation_states[:, 0])):
                # Rotation matrix
                current_state = self.translation_states[t, idx * 6: (idx + 1) * 6]
                rsw_to_inertial_frame = frame_conversion.rsw_to_inertial_rotation_matrix(current_state)

                # Force in inertial frame
                acceleration_trust = self.dep_vars[
                    t, self.dependent_variables_dict["thrust acceleration"][satellite_name]]
                force_inertial = acceleration_trust.T * self.satellite_mass

                # Compute the thrust in the RSW frame
                thrust_rsw_frame = np.dot(rsw_to_inertial_frame.T, force_inertial)
                self.thrust_forces[t, :, idx] = thrust_rsw_frame

        return self.thrust_forces

    def find_control_torques_from_dependent_variables(self) -> None:
        """
        Extract the control torques from the dependent variables (if present).
        """
        self.control_torques = np.zeros((len(self.rotation_states[:, 0]), 3, self.number_of_controlled_satellites))
        for idx, satellite_name in enumerate(self.controlled_satellite_names):
            for t in range(len(self.rotation_states[:, 0])):
                all_torques = self.dep_vars[
                    t, self.dependent_variables_dict["control torque"][satellite_name]]
                control_torques_inertial = all_torques[:3] - all_torques[3:]
                inertial_to_body_frame = self.dep_vars[
                    t, self.dependent_variables_dict["body rotation matrix"][satellite_name]].reshape((3, 3)).T
                control_torques_body = np.dot(inertial_to_body_frame, control_torques_inertial)
                self.control_torques[t, :, idx] = np.array(
                    [control_torques_body[1], -control_torques_body[2], -control_torques_body[0]])

    def find_satellite_names_and_indices(self, satellite_names: list[str],
                                         state_length: int = 6) -> tuple[list[str], list[int]]:
        """
        Find the satellite names and corresponding indices of the array from the ones provided by the user.

        :param satellite_names: Names of the satellites to find. If none, all are returned.
        :param state_length: Length of the state. Defaults to six.
        :return: Tuple with a list of satellite names and a list of satellite indices.
        """
        if satellite_names is None:
            satellite_names = self.controlled_satellite_names

        # Find correct satellites indices
        satellite_indices = []
        for satellite_name in satellite_names:
            satellite_indices.append(self.all_satellite_names.index(satellite_name))

        satellite_indices = [x * state_length for x in satellite_indices]

        return satellite_names, satellite_indices

    def plot_3d_orbit(self, satellite_names: list[str] = None, figure: plt.figure = None) -> plt.figure:
        """
        Plot the orbit in 3D.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot into. If none, a new one is created.
        :return: Figure with the orbit in 3D.
        """
        if self.scenario.use_mean_simulator and not self.update_translation_states:
            for t in range(self.translation_states.shape[0]):
                for i in range(self.number_of_controlled_satellites):
                    self.translation_states[t, i * 6: i * 6 + 3] = element_conversion.keplerian_to_cartesian(
                        self.translation_states[t, i * 6: i * 6 + 6],
                        self.gravitational_constant)[:3]
            self.update_translation_states = True

        satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names)

        for idx, satellite_name in enumerate(satellite_names):
            figure = Plot.plot_3d_trajectory_complete(
                self.translation_states[:, satellite_indices[idx]:satellite_indices[idx] + 3],
                state_label_name=satellite_name,
                figure=figure)

        return figure

    def plot_3d_orbit_projection(self, satellite_names: list[str] = None, figure: plt.figure = None) -> plt.figure:
        """
        Plot the orbit in 3D projected on the reference orbit.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot into. If none, a new one is created.
        :return: Figure with the orbit in 3D.
        """
        if self.scenario.use_mean_simulator and not self.update_translation_states:
            for t in range(self.translation_states.shape[0]):
                for i in range(self.number_of_controlled_satellites):
                    self.translation_states[t, i * 6: i * 6 + 3] = element_conversion.keplerian_to_cartesian(
                        self.translation_states[t, i * 6: i * 6 + 6],
                        self.gravitational_constant)[:3]
            self.update_translation_states = True

        satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names)

        rot_mat = np.array([[1, 0, 0],
                            [0, np.cos(np.pi / 4), np.sin(np.pi / 4)],
                            [0, -np.sin(np.pi / 4), np.cos(np.pi / 4)]])

        trans_copy = np.zeros_like(self.translation_states)
        for t in range(self.translation_states.shape[0]):
            for i in range(self.number_of_controlled_satellites):
                trans_copy[t, i * 6: i * 6 + 3] = rot_mat @ self.translation_states[t, i * 6: i * 6 + 3]

        satellite_names = [r'$\Omega=0\;\mathrm{deg}$', r'$\Omega=15\;\mathrm{deg}$', r'$\Omega=30\;\mathrm{deg}$',
                           r'$\Omega=60\;\mathrm{deg}$']
        linestyles = ['-', '--', '-.', ':']

        for idx, satellite_name in enumerate(satellite_names):
            figure = Plot.plot_3d_trajectory_complete(
                trans_copy[:, satellite_indices[idx]:satellite_indices[idx] + 3],
                state_label_name=satellite_name,
                figure=figure,
                linestyle=linestyles[idx])

        return figure

    def create_animation(self, satellite_names: list[str] = None, figure: plt.figure = None) -> FuncAnimation:
        """
        Create an animation of the satellite movement.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot into. If none, a new one is created.
        :return: FuncAnimation object. Must be stored in a variable to keep animation alive.
        """
        satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names)

        for idx, satellite_name in enumerate(satellite_names):
            figure = Plot.plot_3d_position(
                self.translation_states[0, satellite_indices[idx]:satellite_indices[idx] + 3],
                state_label_name=satellite_name,
                figure=figure)
        satellite_points = figure.get_axes()[0].collections[1:]  # First one is the Earth

        animation = FuncAnimation(figure,
                                  func=lambda i: Plot.animation_function(i, satellite_points,
                                                                         self.translation_states, satellite_indices),
                                  frames=np.arange(1, len(self.translation_states[:, 0]), 25),
                                  interval=150,
                                  repeat=True,
                                  blit=False)

        return animation

    def plot_keplerian_states(self, satellite_names: list[str] = None, plot_argument_of_latitude=False,
                              figure: plt.figure = None, legend_name: str = None) -> plt.figure:
        """
        Plot the keplerian states.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param plot_argument_of_latitude: If true, plot argument of latitude instead of true anomaly.
        :param figure: Figure to plot into. If none, a new one is created.
        :return: Figure with the added states.
        """
        satellite_names, _ = self.find_satellite_names_and_indices(satellite_names)
        legend_names = [legend_name] + [None] * len(satellite_names)

        for idx, satellite_name in enumerate(satellite_names):
            figure = Plot.plot_keplerian_states(self.dep_vars[:, self.dependent_variables_dict["keplerian state"]
                                                                 [satellite_name]],
                                                self.simulation_timestep,
                                                plot_argument_of_latitude=plot_argument_of_latitude,
                                                legend_name=legend_names[idx],
                                                figure=figure)
        return figure

    def plot_kalman_states(self, satellite_names: list[str] = None, figure: plt.figure = None,
                           legend_name: str = None) -> plt.figure:
        """
        Plot the kalman states.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param plot_argument_of_latitude: If true, plot argument of latitude instead of true anomaly.
        :param figure: Figure to plot into. If none, a new one is created.
        :return: Figure with the added states.
        """
        satellite_names, _ = self.find_satellite_names_and_indices(satellite_names)
        legend_names = [legend_name] + [None] * len(satellite_names)

        for idx, satellite_name in enumerate(satellite_names):
            orbital_elements = self.dep_vars[:, self.dependent_variables_dict["keplerian state"][satellite_name]]
            # print(orbital_elements.shape)
            kalman_states = np.concatenate(
                (orbital_elements[:, 0:1], orbital_elements[:, 3:4] + orbital_elements[:, 5:6],
                 orbital_elements[:, 1:2] * np.cos(orbital_elements[:, 3:4]),
                 orbital_elements[:, 1:2] * np.sin(orbital_elements[:, 3:4]),
                 orbital_elements[:, 2:3], orbital_elements[:, 4:5]), axis=1)
            figure = Plot.plot_kalman_states(kalman_states,
                                             self.simulation_timestep,
                                             legend_name=legend_names[idx],
                                             figure=figure)
        return figure

    def plot_thrusts(self, satellite_names: list[str] = None, figure: plt.figure = None, legend_name: str = None,
                     **kwargs) -> plt.figure:
        """
        Plot the thrust forces.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot into. If none, a new one is created.
        :param legend_name: Name to put in the legend.
        :return: Figure with the thrust forces.
        """
        # Compute thrust forces if not yet done
        if self.thrust_forces is None:
            self.get_thrust_forces_from_acceleration()

        satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names, state_length=1)

        color_palette = list(mcolors.TABLEAU_COLORS.values())
        legend_names = [legend_name] + [None] * len(satellite_names)

        for idx, satellite_name in enumerate(satellite_names):
            figure = Plot.plot_thrust_forces(self.thrust_forces[:, :, satellite_indices[idx]],
                                             self.simulation_timestep,
                                             figure=figure,
                                             color=color_palette[idx % 10],
                                             legend_name=legend_names[idx], **kwargs)

        return figure

    def plot_torques(self, satellite_names: list[str] = None, figure: plt.figure = None) -> plt.figure:
        """
        Plot the control torques.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot into. If none, a new one is created.
        :return: Figure with the control torques.
        """
        # Compute control torques es if not yet done
        if self.control_torques is None:
            self.find_control_torques_from_dependent_variables()

        satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names, state_length=1)

        for idx, satellite_name in enumerate(satellite_names):
            figure = Plot.plot_control_torques(self.control_torques[:, :, satellite_indices[idx]],
                                               self.simulation_timestep,
                                               satellite_name=satellite_name,
                                               figure=figure)

        return figure

    def plot_cylindrical_states(self, satellite_names: list[str] = None, figure: plt.figure = None,
                                reference_angles: list[float] = None, legend_name: str = None,
                                states2plot: list = None, **kwargs) -> plt.figure:
        """
        Plot the relative cylindrical states.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot into. If none, a new one is created.
        :param reference_angles: Reference angles to plot.
        :param legend_name: Name for in the legend
        :param states2plot: Indices of which states to plot.
        :return: Figure with the added states.
        """
        # Find cylindrical states if not yet done
        if self.cylindrical_states is None:
            self.convert_to_cylindrical_coordinates()

        satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names)

        color_palette = list(mcolors.TABLEAU_COLORS.values())
        legend_names = [legend_name] + [None] * len(satellite_names)

        # Plot required states
        for idx, satellite_name in enumerate(satellite_names):
            rel_states = self.cylindrical_states[:, satellite_indices[idx]: satellite_indices[idx] + 6]

            # Plot relative error if possible
            if reference_angles is not None:
                rel_states[:, 1] -= reference_angles[self.all_satellite_names.index(satellite_name)]
                if rel_states[0, 1] > np.pi:
                    rel_states[:, 1] -= 2 * np.pi
                elif rel_states[0, 1] < -np.pi:
                    rel_states[:, 1] += 2 * np.pi

            figure = Plot.plot_cylindrical_states(rel_states,
                                                  self.simulation_timestep,
                                                  figure=figure, color=color_palette[idx % 10],
                                                  legend_name=legend_names[idx],
                                                  states2plot=states2plot,
                                                  **kwargs)
        return figure

    def plot_quasi_roe_states(self, satellite_names: list[str] = None, figure: plt.figure = None,
                              reference_angles: list[float] = None, legend_name: str = None) -> plt.figure:
        """
        Plot the quasi roe.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot into. If none, a new one is created.
        :param reference_angles: Reference angles to plot.
        :return: Figure with the added states.
        """
        # Find quasi roe states if not yet done
        if self.quasi_roe is None:
            self.convert_to_quasi_roe()

        satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names)
        legend_names = [legend_name] + [None] * len(satellite_names)

        # Plot required input forces
        for idx, satellite_name in enumerate(satellite_names):
            rel_states = self.quasi_roe[:, satellite_indices[idx]: satellite_indices[idx] + 6]

            # Plot relative error if possible
            if reference_angles is not None:
                rel_states[:, 1] -= reference_angles[idx]

            figure = Plot.plot_quasi_roe(rel_states,
                                         self.simulation_timestep,
                                         legend_name=legend_names[idx],
                                         figure=figure)

        return figure

    def plot_roe_states(self, satellite_names: list[str] = None, figure: plt.figure = None,
                        reference_angles: list[float] = None, legend_name: str = None) -> plt.figure:
        """
        Plot the roe.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot into. If none, a new one is created.
        :param reference_angles: Reference angles to plot.
        :return: Figure with the added states.
        """
        # Find roe states if not yet done
        if self.roe is None:
            self.convert_to_roe()

        satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names)
        legend_names = [legend_name] + [None] * len(satellite_names)

        # Plot required input forces
        for idx, satellite_name in enumerate(satellite_names):
            rel_states = self.roe[:, satellite_indices[idx]: satellite_indices[idx] + 6]

            # Plot relative error if possible
            if reference_angles is not None:
                rel_states[:, 1] -= reference_angles[idx]

                # if rel_states[0, 1] < -np.pi:
                #     rel_states[:, 1] += 2 * np.pi

            figure = Plot.plot_roe(rel_states,
                                   self.simulation_timestep,
                                   legend_name=legend_names[idx],
                                   figure=figure)

        return figure

    def plot_blend_states(self, satellite_names: list[str] = None, figure: plt.figure = None,
                          reference_angles: list[float] = None, legend_name: str = None) -> plt.figure:
        """
        Plot the blended states.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot into. If none, a new one is created.
        :param reference_angles: Reference angles to plot.
        :return: Figure with the added states.
        """
        # Find blended states if not yet done
        if self.blend_states is None:
            self.convert_to_blend_states()

        satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names, state_length=6)
        legend_names = [legend_name] + [None] * len(satellite_names)

        # Plot required input forces
        for idx, satellite_name in enumerate(satellite_names):
            rel_states = self.blend_states[:, satellite_indices[idx]: satellite_indices[idx] + 6]

            # Plot relative error if possible
            if reference_angles is not None:
                rel_states[:, 1] -= reference_angles[idx]
                if rel_states[0, 1] > np.pi:
                    rel_states[:, 1] -= 2 * np.pi

                if rel_states[0, 1] < -np.pi:
                    rel_states[:, 1] += 2 * np.pi

            figure = Plot.plot_blend(rel_states,
                                     self.simulation_timestep,
                                     legend_name=legend_names[idx],
                                     figure=figure)

        return figure

    def plot_small_blend_states(self, satellite_names: list[str] = None, figure: plt.figure = None,
                                reference_angles: list[float] = None, legend_name: str = None) -> plt.figure:
        """
        Plot the small blended states.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot into. If none, a new one is created.
        :param reference_angles: Reference angles to plot.
        :return: Figure with the added states.
        """
        # Find blended states if not yet done
        if self.small_blend_states is None:
            self.convert_to_small_blend_states()

        satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names, state_length=6)
        legend_names = [legend_name] + [None] * len(satellite_names)

        # Plot required input forces
        for idx, satellite_name in enumerate(satellite_names):
            rel_states = self.small_blend_states[:, satellite_indices[idx]: satellite_indices[idx] + 6]

            # Plot relative error if possible
            if reference_angles is not None:
                rel_states[:, 1] -= reference_angles[idx]
                if rel_states[0, 1] > np.pi:
                    rel_states[:, 1] -= 2 * np.pi

                if rel_states[0, 1] < -np.pi:
                    rel_states[:, 1] += 2 * np.pi

            figure = Plot.plot_blend_small(rel_states,
                                           self.simulation_timestep,
                                           legend_name=legend_names[idx],
                                           figure=figure)

        return figure

    def plot_controller_states(self, satellite_names: list[str] = None, figure: plt.figure = None,
                               reference_angles: list[float] = None, legend_name: str = None) -> plt.figure:
        """
        Plot the states of the controller.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot into. If none, a new one is created.
        :param reference_angles: Reference angles to plot.
        :return: Figure with the added states.
        """
        if self.scenario.model == Model.HCW:
            return self.plot_cylindrical_states(satellite_names, figure, reference_angles, legend_name)
        elif self.scenario.model == Model.ROE:
            return self.plot_quasi_roe_states(satellite_names, figure, reference_angles, legend_name)
        elif self.scenario.model == Model.BLEND:
            return self.plot_blend_states(satellite_names, figure, reference_angles, legend_name)

    def plot_quaternions(self, satellite_names: list[str] = None, figure: plt.figure = None) -> plt.figure:
        """
        Plot the quaternions from body-fixed to inertial frame over time.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot into. If none, a new one is created.
        :return: Figure with the added quaternions.
        """
        satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names, state_length=7)

        # Plot required input forces
        for idx, satellite_name in enumerate(satellite_names):
            figure = Plot.plot_quaternion(self.rotation_states[:, satellite_indices[idx]:
                                                                  satellite_indices[idx] + 4],
                                          self.simulation_timestep,
                                          satellite_name=satellite_name,
                                          figure=figure)

        return figure

    def plot_quaternions_rsw(self, satellite_names: list[str] = None, figure: plt.figure = None) -> plt.figure:
        """
        Plot the quaternions from body-fixed to rsw frame over time.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot into. If none, a new one is created.
        :return: Figure with the added quaternions.
        """
        # Find rsw quaternions if not yet done
        if self.rsw_quaternions is None:
            self.convert_to_rsw_quaternions()

        satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names, state_length=4)

        # Plot required input forces
        for idx, satellite_name in enumerate(satellite_names):
            figure = Plot.plot_quaternion(self.rsw_quaternions[:, satellite_indices[idx]:
                                                                  satellite_indices[idx] + 4],
                                          self.simulation_timestep,
                                          satellite_name=satellite_name,
                                          figure=figure)

        return figure

    def plot_euler_angles(self, satellite_names: list[str] = None, figure: plt.figure = None) -> plt.figure:
        """
        Plot the euler angles.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot into. If none, a new one is created.
        :return: Figure with the added states.
        """
        # Find euler angles if not yet done
        if self.euler_angles is None:
            self.convert_to_euler_angles()

        satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names, state_length=3)

        # Plot required euler angles
        for idx, satellite_name in enumerate(satellite_names):
            figure = Plot.plot_euler_angles(self.euler_angles[:, satellite_indices[idx]:
                                                                 satellite_indices[idx] + 3],
                                            self.simulation_timestep,
                                            satellite_name=satellite_name,
                                            figure=figure)

        return figure

    def plot_angular_velocities(self, satellite_names: list[str] = None, figure: plt.figure = None) -> plt.figure:
        """
        Plot the angular velocities in the body fixed frame.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot into. If none, a new one is created.
        :return: Figure with the added states.
        """
        # Find euler angles if not yet done
        if self.angular_velocities is None:
            self.convert_to_angular_velocities()

        satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names, state_length=3)

        # Plot required input forces
        for idx, satellite_name in enumerate(satellite_names):
            figure = Plot.plot_angular_velocities(self.angular_velocities[:, satellite_indices[idx]:
                                                                             satellite_indices[idx] + 3],
                                                  self.simulation_timestep,
                                                  satellite_name=satellite_name,
                                                  figure=figure)

        return figure

    def plot_theta_Omega(self, figure: plt.figure = None) -> plt.figure:
        """
        Create a plot for all satellites with the theta on the x-axis and the Omega on y-axis.

        :param figure: Figure to plot the results onto.

        :return: Figure with the results.
        """
        oe_filtered, oe_ref = self.filter_oe()
        theta = oe_filtered[:, 3::6] + oe_filtered[:, 5::6]
        Omega = oe_filtered[:, 4::6]

        if figure is None:
            figure, _ = plt.subplots(1, 1, figsize=(16, 9))

        ax = figure.get_axes()[0]
        theta_ref = self.mean_motion * np.linspace(0, theta.shape[0] - 1, theta.shape[0]).reshape(
            (-1, 1)) * self.simulation_timestep
        theta_norm = np.rad2deg(np.unwrap(theta - theta_ref, axis=0))
        Omega_norm = np.rad2deg(Omega)

        ax.plot(theta_norm, Omega_norm)
        ax.plot(theta_norm[0], Omega_norm[0], 'o', label='start')
        ax.plot(theta_norm[-1], Omega_norm[-1], 's', label='end')
        # ax.x_label('Theta [deg]')
        # ax.y_label('Omega [deg]')
        plt.legend()
        return figure

    def plot_main_states(self, satellite_indices: list[int] = None, figure: plt.figure = None, legend_name: str = '',
                         plot_duration: int = None, **kwargs) -> plt.figure:
        """
        Create a plot with the main results for the report.

        :param satellite_indices: Indices the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot the results onto.
        :param legend_name: Name for the legend
        :param plot_duration: Length of the plot in min.

        :return: Figure with the results.
        """
        oe_filtered, oe_ref = self.filter_oe()
        main_states = Conversion.oe2main(oe_filtered, oe_ref, self.reference_angle_offsets)

        if plot_duration is not None:
            main_states = main_states[:plot_duration * 60 + 1, :]

        radius = main_states[:, 0::3]
        theta = main_states[:, 1::3]
        Omega = main_states[:, 2::3]

        # satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names, state_length=1)

        if np.max(Omega) - np.min(Omega) < 1e-6:
            Omega *= 0

        if satellite_indices is None:
            satellite_indices = np.arange(self.number_of_controlled_satellites)

        for idx in satellite_indices:
            # satellite_idx = satellite_indices[idx]
            states = np.concatenate((radius[:, idx:idx + 1],
                                     theta[:, idx:idx + 1],
                                     Omega[:, idx:idx + 1]), axis=1)
            figure = PlotRes.plot_main_states_report(states=states, timestep=self.simulation_timestep, figure=figure,
                                                     legend_name=legend_name, **kwargs)

            if idx == satellite_indices[0]:
                legend_name = None

        return figure

    def plot_side_states(self, satellite_indices: list[int] = None, figure: plt.figure = None, legend_name: str = '',
                         plot_duration: int = None, **kwargs) -> plt.figure:
        """
        Create a plot with the side results for the report.

        :param satellite_indices: Indices the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot the results onto.
        :param legend_name: Name for the legend
        :param plot_duration: Length of the plot in min.

        :return: Figure with the results.
        """
        oe_filtered, oe_ref = self.filter_oe()
        # satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names, state_length=1)

        side_states = Conversion.oe2side(oe_filtered, oe_ref)

        if plot_duration is not None:
            side_states = side_states[:plot_duration * 60 + 1, :]

        e = side_states[:, 0::2]
        i = side_states[:, 1::2]

        if np.max(i) - np.min(i) < 1e-6:
            i *= 0

        if satellite_indices is None:
            satellite_indices = np.arange(self.number_of_controlled_satellites)

        for idx in satellite_indices:
            # satellite_idx = satellite_indices[idx]
            states = np.concatenate((e[:, idx:idx + 1],
                                     i[:, idx:idx + 1]), axis=1)
            figure = PlotRes.plot_side_states_report(states=states, timestep=self.simulation_timestep, figure=figure,
                                                     legend_name=legend_name, **kwargs)

            if idx == satellite_indices[0]:
                legend_name = None

        return figure

    def plot_inputs(self, satellite_indices: list[int] = None, figure: plt.figure = None, legend_name: str = None,
                    plot_duration: int = None, **kwargs) -> plt.figure:
        """
        Create a plot with the control inputs for the report.

        :param satellite_indices: Indices of the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot into. If none, a new one is created.
        :param legend_name: Name to put in the legend.
        :param plot_duration: Length of the plot in min.
        :return: Figure with the thrust forces.

        :return: Figure with the inputs.
        """
        # Compute thrust forces if not yet done
        if self.thrust_forces is None:
            self.get_thrust_forces_from_acceleration()

        forces = self.thrust_forces[1:]
        if plot_duration is not None:
            forces = forces[:plot_duration * 60 + 1, :]
        # satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names, state_length=3)

        # legend_names = [legend_name] + [None] * len(satellite_names)
        if satellite_indices is None:
            satellite_indices = np.arange(self.number_of_controlled_satellites)

        for idx in satellite_indices:
            inputs = np.concatenate((forces[:, idx * 3:idx * 3 + 1],
                                     forces[:, idx * 3 + 1:idx * 3 + 2],
                                     forces[:, idx * 3 + 2:idx * 3 + 3]), axis=1)
            figure = PlotRes.plot_inputs_report(inputs, self.simulation_timestep, figure=figure,
                                                legend_name=legend_name, **kwargs)
            if idx == satellite_indices[0]:
                legend_name = None

        return figure

    def plot_model_errors(self, model_values: np.ndarray, satellite_indices: list[int] = None, figure: plt.figure = None,
                          legend_name: str = '', plot_duration: int = None, **kwargs) -> plt.figure:
        """
        Create a plot with the model errors for the report.

        :param model_values: The values obtained from the model.
        :param satellite_indices: Indices the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot the results onto.
        :param legend_name: Name for the legend
        :param plot_duration: Length of the plot in min.

        :return: Figure with the results.
        """
        oe_filtered, oe_ref = self.filter_oe()
        main_states = Conversion.oe2main(oe_filtered, oe_ref, self.reference_angle_offsets)

        if plot_duration is not None:
            main_states = main_states[:plot_duration * 60 + 1, :]

        radius = main_states[:, 0::3]
        theta = main_states[:, 1::3]
        Omega = main_states[:, 2::3]

        # satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names, state_length=1)

        if np.max(Omega) - np.min(Omega) < 1e-6:
            Omega *= 0

        if self.scenario.model == Model.BLEND:
            main_states_model = Conversion.blend2main(model_values.T)
        elif self.scenario.model == Model.HCW:
            main_states_model = Conversion.hcw2main(model_values.T)
        elif self.scenario.model == Model.ROE:
            main_states_model = Conversion.roe2main(model_values.T)
        else:
            main_states_model = None

        radius_error = main_states_model[:-1, 0::3] - radius
        theta_error = main_states_model[:-1, 1::3] - theta
        Omega_error = main_states_model[:-1, 2::3] - Omega

        if satellite_indices is None:
            satellite_indices = np.arange(self.number_of_controlled_satellites)

        for idx in satellite_indices:
            # satellite_idx = satellite_indices[idx]
            states = np.concatenate((radius_error[:, idx:idx + 1],
                                     theta_error[:, idx:idx + 1],
                                     Omega_error[:, idx:idx + 1]), axis=1)
            figure = PlotRes.plot_main_states_report(states=states, timestep=self.scenario.control.control_timestep,
                                                     figure=figure, legend_name=legend_name, **kwargs)

            if idx == satellite_indices[0]:
                legend_name = None

        return figure

    def plot_radius_zoomed(self, satellite_indices: list[int] = None, figure: plt.figure = None, legend_name: str = '',
                           plot_duration: int = None, **kwargs) -> plt.figure:
        """
        Create a plot with the radius for the report.

        :param satellite_indices: Indices the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot the results onto.
        :param legend_name: Name for the legend
        :param plot_duration: Length of the plot in min.

        :return: Figure with the results.
        """
        oe_filtered, oe_ref = self.filter_oe()
        main_states = Conversion.oe2main(oe_filtered, oe_ref, self.reference_angle_offsets)

        if plot_duration is not None:
            main_states = main_states[:plot_duration * 60 + 1, :]

        radius = main_states[:, 0::3]

        # satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names, state_length=1)

        if satellite_indices is None:
            satellite_indices = np.arange(self.number_of_controlled_satellites)

        for idx in satellite_indices:
            # satellite_idx = satellite_indices[idx]
            states = radius[:, idx:idx + 1]
            figure = PlotRes.plot_radius_report(states=states, timestep=self.simulation_timestep, figure=figure,
                                                     legend_name=legend_name, **kwargs)

            if idx == satellite_indices[0]:
                legend_name = None

        return figure

    def plot_ex(self, satellite_indices: list[int] = None, figure: plt.figure = None, legend_name: str = '',
                           plot_duration: int = None, **kwargs) -> plt.figure:
        """
        Create a plot with the radius for the report.

        :param satellite_indices: Indices the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot the results onto.
        :param legend_name: Name for the legend
        :param plot_duration: Length of the plot in min.

        :return: Figure with the results.
        """
        if self.blend_states is None:
            self.convert_to_blend_states()

        if plot_duration is not None:
            self.blend_states = self.blend_states[:plot_duration * 60 + 1, :]

        ex = self.blend_states[:, 2::6]

        # satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names, state_length=1)

        if satellite_indices is None:
            satellite_indices = np.arange(self.number_of_controlled_satellites)

        for idx in satellite_indices:
            # satellite_idx = satellite_indices[idx]
            states = ex[:, idx:idx + 1]
            figure = PlotRes.plot_ex(states=states, timestep=self.simulation_timestep, figure=figure,
                                                     legend_name=legend_name, **kwargs)

            if idx == satellite_indices[0]:
                legend_name = None

        return figure

    def print_metrics(self) -> str:
        """
        Print metrics to score approaches
        """
        metric_values = self.find_metric_values()

        state_results = f"Mean radius: {metric_values[0]}, mean theta: {metric_values[1]} and mean omega: {metric_values[2]}\n"
        input_results = f"Mean u_r: {metric_values[3]}, mean u_t: {metric_values[4]}, mean u_n: {metric_values[5]} and mean norm: {metric_values[6]}"

        return state_results + input_results

    def find_metric_values(self) -> list[float]:
        """
        Find all the metric values.

        :return: List with all the metric values.
        """
        oe_filtered, oe_ref = self.filter_oe()
        main_states = Conversion.oe2main(oe_filtered, oe_ref, self.reference_angle_offsets)
        radius = main_states[:, 0::3]
        theta = main_states[:, 1::3]
        Omega = main_states[:, 2::3]

        if self.thrust_forces is None:
            self.get_thrust_forces_from_acceleration()

        mean_radius = np.mean(np.abs(radius))
        mean_theta = np.rad2deg(np.mean(np.abs(theta)))
        mean_Omega = np.rad2deg(np.mean(np.abs(Omega)))

        if len(self.thrust_forces.shape) == 1:
            self.thrust_forces = self.thrust_forces.reshape((-1, 3))
        inputs_r = self.thrust_forces[1:, 0::3]
        inputs_t = self.thrust_forces[1:, 1::3]
        inputs_n = self.thrust_forces[1:, 2::3]

        inputs_order = np.concatenate((inputs_r.reshape((-1, 1)), inputs_t.reshape((-1, 1)),
                                       inputs_n.reshape((-1, 1))), axis=1)

        mean_ur = np.mean(np.abs(inputs_r))
        mean_ut = np.mean(np.abs(inputs_t))
        mean_un = np.mean(np.abs(inputs_n))

        norms = np.mean(np.linalg.norm(inputs_order, axis=1))

        return [mean_radius, mean_theta, mean_Omega, mean_ur, mean_ut, mean_un, norms]

    def plot_relative_radius_and_height(self, satellite_names: list[str] = None, figure: plt.figure = None,
                                        reference_angles: list[float] = None, legend_name: str = None,
                                        states2plot: list = None, **kwargs) -> plt.figure:
        """
        Plot the relative cylindrical states (only radius and height).

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot into. If none, a new one is created.
        :param reference_angles: Reference angles to plot.
        :param legend_name: Name for in the legend
        :param states2plot: Indices of which states to plot.
        :return: Figure with the added states.
        """
        # Find cylindrical states if not yet done
        if self.cylindrical_states is None:
            self.convert_to_cylindrical_coordinates()

        satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names)

        satellite_names = [r'$\Omega=0\;\mathrm{deg}$', r'$\Omega=15\;\mathrm{deg}$', r'$\Omega=30\;\mathrm{deg}$',
                           r'$\Omega=60\;\mathrm{deg}$']
        linestyles = ['-', '--', '-.', ':']

        # Plot required states
        for idx, satellite_name in enumerate(satellite_names):
            rel_states = self.cylindrical_states[:, satellite_indices[idx]: satellite_indices[idx] + 6]

            # Plot relative error if possible
            if reference_angles is not None:
                rel_states[:, 1] -= reference_angles[self.all_satellite_names.index(satellite_name)]
                if rel_states[0, 1] > np.pi:
                    rel_states[:, 1] -= 2 * np.pi
                elif rel_states[0, 1] < -np.pi:
                    rel_states[:, 1] += 2 * np.pi

            figure = Plot.plot_cylindrical_radius_height(rel_states,
                                                         self.simulation_timestep,
                                                         figure=figure,
                                                         legend_name=satellite_name,
                                                         states2plot=states2plot,
                                                         linestyle=linestyles[idx],
                                                         **kwargs)
        return figure
