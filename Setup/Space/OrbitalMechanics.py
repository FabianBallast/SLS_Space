import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup, create_dynamics_simulator
from tudatpy.kernel.interface import spice
from tudatpy.kernel.astro import element_conversion, frame_conversion
from tudatpy.util import result2array
from Space.EngineModel import ThrustModel, TorqueModel
from Visualisation import Plotting as Plot
from Dynamics import HCWDynamics, ROEDynamics, SystemDynamics, AttitudeDynamics
from scipy.spatial.transform import Rotation

# Load spice kernels once, not for every object we create
spice.load_standard_kernels()


class OrbitalMechSimulator:
    """
    Class to simulate several satellites in the same orbit using a high-fidelity simulator. 
    """

    def __init__(self):
        self.bodies = None
        self.central_bodies = None
        self.controlled_satellite_names = None
        self.all_satellite_names = None
        self.number_of_total_satellites = None
        self.number_of_controlled_satellites = None
        self.acceleration_model = None
        self.torque_model = None
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
        self.euler_angles = None
        self.angular_velocities = None
        self.rsw_quaternions = None

        self.reference_satellite_added = False
        self.reference_satellite_name = "Satellite_ref"

        self.translation_states = None
        self.mass_states = None
        self.rotation_states = None

        self.mean_motion = None

    def create_bodies(self, number_of_satellites: int, satellite_mass: float, satellite_inertia: np.ndarray[3, 3],
                      add_reference_satellite: bool = False) -> None:
        """
        Create the different bodies used during a simulation.

        :param number_of_satellites: Number of satellites (excluding a possible virtual reference satellite) to include.
        :param satellite_mass: Mass of each satellite.
        :param satellite_inertia: Mass moment of inertia of each satellite.
        :param add_reference_satellite: Whether to add a (virtual) reference satellite.
        """
        # Create default body settings for "Earth"
        bodies_to_create = ["Earth"]

        # Create default body settings for bodies_to_create,
        # with "Earth"/"J2000" as the global frame origin and orientation
        global_frame_origin = "Earth"
        global_frame_orientation = "J2000"
        body_settings = environment_setup.get_default_body_settings(
            bodies_to_create, global_frame_origin, global_frame_orientation)

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
            self.bodies.create_empty_body(self.reference_satellite_name)
            self.bodies.get(self.reference_satellite_name).mass = satellite_mass
            self.bodies.get(self.reference_satellite_name).inertia_tensor = satellite_inertia

            self.number_of_total_satellites = self.number_of_controlled_satellites + 1
            self.reference_satellite_added = True
            self.all_satellite_names = self.controlled_satellite_names + [self.reference_satellite_name]
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
        # Add thruster model for every satellite
        for idx, satellite_name in enumerate(self.controlled_satellite_names):
            # Set up the thrust model for the satellite
            current_thrust_model = ThrustModel(thrust_inputs[idx], control_timestep, specific_impulse,
                                               self.bodies.get(satellite_name))

            rotation_model_settings = environment_setup.rotation_model.custom_inertial_direction_based(
                current_thrust_model.get_thrust_direction, "J2000", "VehicleFixed")
            environment_setup.add_rotation_model(self.bodies, satellite_name, rotation_model_settings)

            # Define the thrust magnitude settings for the satellite from the custom functions
            thrust_magnitude_settings = propagation_setup.thrust.custom_thrust_magnitude(
                current_thrust_model.get_thrust_magnitude, current_thrust_model.get_specific_impulse)

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

        # Add thruster model for every satellite
        for idx, satellite_name in enumerate(self.controlled_satellite_names):
            # Set up the thrust model for the satellite
            current_torque_model = TorqueModel(torque_inputs[idx], control_timestep, self.bodies.get(satellite_name))

            # Define the thrust magnitude settings for the satellite from the custom functions
            self.torque_engine_models[idx] = propagation_setup.torque.custom_torque(current_torque_model.get_torque)

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
            acceleration_settings[self.reference_satellite_name] = {"Earth": [gravitational_function]}

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
            torque_settings[self.reference_satellite_name] = \
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
                                              argument_of_periapsis: float = 0, longitude: float = 0) -> np.ndarray:
        """
        Create a desired state array in cartesian coordinates provided the orbital elements for all satellites,
        assuming the that all satellites are in the same orbit (but the true anomalies differ).

        :param true_anomalies: True anomalies of satellties in degrees.
        :param orbit_height: Height of the orbit above the surface of the Earth in m.
        :param eccentricity: Eccentricity of the orbit.
        :param inclination: Inclination of the orbit.
        :param argument_of_periapsis: Argument of periapsis of the orbit.
        :param longitude: Longitude of the ascending node of the orbit.
        :return: Numpy array with the states of all satellites in cartesian coordinates.
        """
        if not len(true_anomalies) == self.number_of_controlled_satellites:
            raise Exception(f"Not enough true anomalies. Received {len(true_anomalies)} variables, "
                            f"but expected {self.number_of_controlled_satellites} variables. ")

        # Retrieve gravitational parameter
        earth_gravitational_parameter = self.bodies.get("Earth").gravitational_parameter

        # Retrieve Earth's radius
        earth_radius = self.bodies.get("Earth").shape_model.average_radius

        orbit_semi_major_axis = earth_radius + orbit_height * 1e3
        orbit_eccentricity = eccentricity
        orbit_inclination = np.deg2rad(inclination)
        orbit_argument_of_periapsis = np.deg2rad(argument_of_periapsis)
        orbit_longitude = np.deg2rad(longitude)
        orbit_anomalies = np.deg2rad(true_anomalies)

        initial_states = np.array([])

        for anomaly in orbit_anomalies:
            initial_states = np.append(initial_states,
                                       element_conversion.keplerian_to_cartesian_elementwise(
                                           gravitational_parameter=earth_gravitational_parameter,
                                           semi_major_axis=orbit_semi_major_axis,
                                           eccentricity=orbit_eccentricity,
                                           inclination=orbit_inclination,
                                           argument_of_periapsis=orbit_argument_of_periapsis,
                                           longitude_of_ascending_node=orbit_longitude,
                                           true_anomaly=anomaly
                                       ))

        if self.reference_satellite_added:
            initial_states = np.append(initial_states,
                                       element_conversion.keplerian_to_cartesian_elementwise(
                                           gravitational_parameter=earth_gravitational_parameter,
                                           semi_major_axis=orbit_semi_major_axis,
                                           eccentricity=orbit_eccentricity,
                                           inclination=orbit_inclination,
                                           argument_of_periapsis=orbit_argument_of_periapsis,
                                           longitude_of_ascending_node=orbit_longitude,
                                           true_anomaly=0  # Reference always at 0
                                       ))

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
        termination_settings = propagation_setup.propagator.time_termination(end_epoch)

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

    def simulate_system(self) -> (np.ndarray, np.ndarray):
        """
        Run a simulation of the system

        :return: Either a tuple of two Numpy arrays (if at least one dependent variable has been added) with the states
                 and the dependent variables, or only the states.
        """
        # Create simulation object and propagate the dynamics
        dynamics_simulator = create_dynamics_simulator(self.bodies, self.propagator_settings)

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

        if len(dep_vars) == 0:
            return self.states
        else:
            self.dep_vars = result2array(dep_vars)
            return self.states, self.dep_vars

    def convert_to_cylindrical_coordinates(self) -> np.ndarray:
        """
        Convert the states from cartesian coordinates to relative cylindrical coordinates.
        Store the result in self.cylindrical_states and return them.

        :return: Array with cylindrical coordinates in shape (t, 6 * number_of_controlled_satellites)
        """
        # Find rotation matrices
        rsw_to_inertial_rot_mats = self.dep_vars[:, self.dependent_variables_dict["rsw rotation matrix"]
                                                    [self.reference_satellite_name]]
        inertial_to_rsw_rots = np.transpose(rsw_to_inertial_rot_mats.reshape(-1, 3, 3), axes=[0, 2, 1])

        number_of_states = self.translation_states.shape[0]
        states_rsw = np.zeros_like(self.translation_states)

        for i in range(number_of_states):
            inertial_to_rsw = inertial_to_rsw_rots[i]
            inertial_to_rsw_kron = np.kron(np.eye(2 * self.number_of_total_satellites), inertial_to_rsw)
            states_rsw[i, :] = inertial_to_rsw_kron @ self.translation_states[i]

        # Find reference position
        self.cylindrical_states = np.zeros_like(self.translation_states[:, :-6])
        ref_states_rsw = states_rsw[:, -6:]
        ref_rho = np.sqrt(ref_states_rsw[:, 0:1] ** 2 + ref_states_rsw[:, 1:2] ** 2)
        ref_theta = np.unwrap(np.arctan2(ref_states_rsw[:, 1:2], ref_states_rsw[:, 0:1]), axis=0)
        ref_z = ref_states_rsw[:, 2:3]
        ref_rho_dot = np.cos(ref_theta) * ref_states_rsw[:, 3:4] + np.sin(ref_theta) * ref_states_rsw[:, 4:5]
        ref_theta_dot = (-np.sin(ref_theta) * ref_states_rsw[:, 3:4] +
                         np.cos(ref_theta) * ref_states_rsw[:, 4:5]) / ref_rho
        ref_z_dot = ref_states_rsw[:, 5:6]

        # Find relative positions in cylindrical frame
        for satellite in range(self.number_of_controlled_satellites):
            states_satellite = states_rsw[:, satellite * 6:(satellite + 1) * 6]

            # Actual conversion
            rho = np.sqrt(states_satellite[:, 0:1] ** 2 + states_satellite[:, 1:2] ** 2) - ref_rho
            theta = np.unwrap(np.arctan2(states_satellite[:, 1:2], states_satellite[:, 0:1]), axis=0) - ref_theta
            z = states_satellite[:, 2:3] - ref_z
            rho_dot = np.cos(theta) * states_satellite[:, 3:4] + np.sin(theta) * states_satellite[:, 4:5] - ref_rho_dot
            theta_dot = (-np.sin(theta) * states_satellite[:, 3:4] +
                         np.cos(theta) * states_satellite[:, 4:5]) / (rho + ref_rho) - ref_theta_dot
            z_dot = states_satellite[:, 5:6] - ref_z_dot

            # Convert to range [0, 2 * pi]
            if theta[0] < 0:
                theta += 2 * np.pi

            self.cylindrical_states[:, satellite * 6:(satellite + 1) * 6] = np.concatenate((rho, theta, z,
                                                                                            rho_dot, theta_dot, z_dot),
                                                                                           axis=1)
        return self.cylindrical_states

    def convert_to_quasi_roe(self) -> np.ndarray:
        """
        Convert the states from keplerian coordinates to quasi ROE.

        :return: Array with quasi ROE in shape (t, 6 * number_of_controlled_satellites)
        """
        self.quasi_roe = np.zeros_like(self.translation_states[:, :-6])
        kepler_ref = self.dep_vars[:, self.dependent_variables_dict["keplerian state"][self.reference_satellite_name]]

        for idx, satellite_name in enumerate(self.controlled_satellite_names):
            kepler_sat = self.dep_vars[:, self.dependent_variables_dict["keplerian state"][satellite_name]]
            delta_a = (kepler_sat[:, 0:1] - kepler_ref[:, 0:1]) / kepler_ref[:, 0:1]
            delta_lambda = kepler_sat[:, 3:4] + kepler_sat[:, 5:6] + kepler_sat[:, 4:5] * np.cos(kepler_ref[:, 2:3]) - \
                           kepler_ref[:, 3:4] - kepler_ref[:, 5:6] - kepler_ref[:, 4:5] * np.cos(kepler_ref[:, 2:3])
            delta_ex = kepler_sat[:, 1:2] * np.cos(kepler_sat[:, 3:4]) - kepler_ref[:, 1:2] * np.cos(kepler_ref[:, 3:4])
            delta_ey = kepler_sat[:, 1:2] * np.sin(kepler_sat[:, 3:4]) - kepler_ref[:, 1:2] * np.sin(kepler_ref[:, 3:4])
            delta_ix = kepler_sat[:, 2:3] - kepler_ref[:, 2:3]
            delta_iy = (kepler_sat[:, 4:5] - kepler_ref[:, 4:5]) * kepler_ref[:, 2:3]
            self.quasi_roe[:, idx * 6:(idx + 1) * 6] = np.concatenate((delta_a, delta_lambda, delta_ex,
                                                                       delta_ey, delta_ix, delta_iy), axis=1)
        return self.quasi_roe

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

                q_scalar_first = self.rotation_states[t, i*7:i*7 + 4]  # We need to swap the order for scipy: scalar last
                quaternion_body2inertial = Rotation.from_quat(q_scalar_first[[1, 2, 3, 0]])
                # quaternion_body2rsw = quaternion_body2inertial * quaternion_rsw2inertial

                q_scalar_last = (quaternion_body2inertial * quaternion_rsw2inertial).as_quat()
                self.rsw_quaternions[t, i*4:(i+1)*4] = q_scalar_last[[3, 0, 1, 2]]  # Put scalar first again

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
                q_scalar_first = self.rsw_quaternions[t, i*4:(i+1)*4]
                # quaternion = Rotation.from_quat(q_scalar_first[[1, 2, 3, 0]]) * rotation_convention  # Put scalar last
                # pitch_yaw_roll = quaternion.as_euler('YZX')
                # self.euler_angles[t, i*3:(i+1)*3] = pitch_yaw_roll[[2, 0, 1]]
                pitch_yaw_roll = Rotation.from_quat(q_scalar_first[[1, 2, 3, 0]]).as_euler('ZXY')
                self.euler_angles[t, i * 3:(i + 1) * 3] = np.array([pitch_yaw_roll[2], -pitch_yaw_roll[0],
                                                                    -pitch_yaw_roll[1]])

        return self.euler_angles

    def convert_to_angular_velocities(self):
        if self.mean_motion is None:
            raise Exception("Set the mean motion of the satellites first!")

        self.angular_velocities = np.zeros((len(self.rotation_states[:, 0]), 3 * self.number_of_controlled_satellites))

        for idx, satellite_name in enumerate(self.controlled_satellite_names):
            for t in range(len(self.angular_velocities[:, 0])):
                omega_inertial = self.rotation_states[t, idx*7+4:(idx+1)*7]
                inertial_to_body_frame = self.dep_vars[
                    t, self.dependent_variables_dict["body rotation matrix"][satellite_name]].reshape((3, 3)).T
                omega_body = np.dot(inertial_to_body_frame, omega_inertial)
                omega_body = np.array([omega_body[1], -omega_body[2], -omega_body[0]])
                self.angular_velocities[t, idx*3:(idx+1)*3] = omega_body - np.array([0, self.mean_motion, 0])

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
                states_euler[t, idx*6:idx*6+3] = self.euler_angles[t, idx*3:(idx+1)*3]
                states_euler[t, idx*6+3:(idx+1)*6] = self.angular_velocities[t, idx*3:(idx+1)*3]

        return states_euler

    def set_mean_motion(self, mean_motion: float) -> None:
        """
        Set the mean motion of the satellites.

        :param mean_motion: The mean motion of the satellites in rad/s
        """
        self.mean_motion = mean_motion
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
        elif isinstance(dynamical_model, AttitudeDynamics.LinAttModel):
            return self.convert_to_euler_angles()
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
        else:
            raise Exception("No conversion for this type of model has been implemented yet. ")

    def get_thrust_forces_from_acceleration(self) -> None:
        """
        Compute the thrust forces from the accelerations that were stored during the simulation.
        This provides an excellent overview of the applied forces, as opposed to the 'planned' forces
        you could get by storing the inputs provided to the simulator.

        Result is stored in self.thrust_forces
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
                inertial_to_body_frame = self.dep_vars[t, self.dependent_variables_dict["body rotation matrix"][satellite_name]].reshape((3, 3)).T
                control_torques_body = np.dot(inertial_to_body_frame, control_torques_inertial)
                self.control_torques[t, :, idx] = np.array([control_torques_body[1], -control_torques_body[2], -control_torques_body[0]])

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
            satellite_indices.append(self.controlled_satellite_names.index(satellite_name))

        satellite_indices = [x * state_length for x in satellite_indices]

        return satellite_names, satellite_indices

    def plot_3d_orbit(self, satellite_names: list[str] = None, figure: plt.figure = None) -> plt.figure:
        """
        Plot the orbit in 3D.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot into. If none, a new one is created.
        :return: Figure with the orbit in 3D.
        """
        satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names)

        for idx, satellite_name in enumerate(satellite_names):
            figure = Plot.plot_3d_trajectory(self.translation_states[:, satellite_indices[idx]:satellite_indices[idx] + 3],
                                             state_label_name=satellite_name,
                                             figure=figure)
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
            figure = Plot.plot_3d_position(self.translation_states[0, satellite_indices[idx]:satellite_indices[idx] + 3],
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
                              figure: plt.figure = None) -> plt.figure:
        """
        Plot the keplerian states.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param plot_argument_of_latitude: If true, plot argument of latitude instead of true anomaly.
        :param figure: Figure to plot into. If none, a new one is created.
        :return: Figure with the added states.
        """
        satellite_names, _ = self.find_satellite_names_and_indices(satellite_names)

        for idx, satellite_name in enumerate(satellite_names):
            figure = Plot.plot_keplerian_states(self.dep_vars[:, self.dependent_variables_dict["keplerian state"]
                                                                 [satellite_name]],
                                                self.simulation_timestep,
                                                plot_argument_of_latitude=plot_argument_of_latitude,
                                                satellite_name=satellite_name,
                                                figure=figure)
        return figure

    def plot_thrusts(self, satellite_names: list[str] = None, figure: plt.figure = None) -> plt.figure:
        """
        Plot the thrust forces.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot into. If none, a new one is created.
        :return: Figure with the thrust forces.
        """
        # Compute thrust forces if not yet done
        if self.thrust_forces is None:
            self.get_thrust_forces_from_acceleration()

        satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names, state_length=1)

        for idx, satellite_name in enumerate(satellite_names):
            figure = Plot.plot_thrust_forces(self.thrust_forces[:, :, satellite_indices[idx]],
                                             self.simulation_timestep,
                                             satellite_name=satellite_name,
                                             figure=figure)

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
                                reference_angles: list[float] = None) -> plt.figure:
        """
        Plot the relative cylindrical states.

        :param satellite_names: Names of the satellites to plot. If none, all are plotted.
        :param figure: Figure to plot into. If none, a new one is created.
        :param reference_angles: Reference angles to plot.
        :return: Figure with the added states.
        """
        # Find cylindrical states if not yet done
        if self.cylindrical_states is None:
            self.convert_to_cylindrical_coordinates()

        satellite_names, satellite_indices = self.find_satellite_names_and_indices(satellite_names)

        # Plot required states
        for idx, satellite_name in enumerate(satellite_names):
            rel_states = self.cylindrical_states[:, satellite_indices[idx]: satellite_indices[idx] + 6]

            # Plot relative error if possible
            if reference_angles is not None:
                rel_states[:, 1] -= reference_angles[idx]

            figure = Plot.plot_cylindrical_states(rel_states,
                                                  self.simulation_timestep,
                                                  satellite_name=satellite_name,
                                                  figure=figure)
        return figure

    def plot_quasi_roe_states(self, satellite_names: list[str] = None, figure: plt.figure = None,
                              reference_angles: list[float] = None) -> plt.figure:
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

        # Plot required input forces
        for idx, satellite_name in enumerate(satellite_names):
            rel_states = self.quasi_roe[:, satellite_indices[idx]: satellite_indices[idx] + 6]

            # Plot relative error if possible
            if reference_angles is not None:
                rel_states[:, 1] -= reference_angles[idx]

            figure = Plot.plot_quasi_roe(rel_states,
                                         self.simulation_timestep,
                                         satellite_name=satellite_name,
                                         figure=figure)

        return figure

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
