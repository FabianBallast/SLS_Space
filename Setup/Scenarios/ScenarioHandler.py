from Dynamics.AttitudeDynamics import LinAttModel
from Dynamics.HCWDynamics import RelCylHCW
from Dynamics.ROEDynamics import QuasiROE
from Dynamics.SystemDynamics import TranslationalDynamics
from Space.OrbitalMechanics import OrbitalMechSimulator
from SLS.SLS_setup import SLSSetup
from Scenarios.ControlScenarios import Model
import numpy as np


class ScenarioHandler:
    """
    Create a class to deal with different scenarios and set up everything.
    """

    def __init__(self, scenario: dict) -> None:
        """
        Create the scenario handler object through a scenario.

        :param scenario: A dict with the scenario to run.
        """
        self.scenario = scenario
        self.sls = None
        self.orbital_mech = None

        self.control_inputs_thrust = np.zeros((self.scenario['number_of_sats'], 3, 2))
        self.control_inputs_torque = np.zeros((self.scenario['number_of_sats'], 3, 2))
        self.sls_states = None
        self.pos_states = None
        self.rot_states = None

        self.dep_vars = None
        self.dep_vars_dict = None

        self.t_horizon_control = None
        self.control_simulation_ratio = None
        self.t_horizon_simulation = None

    def create_sls_system(self) -> None:
        """
        Create the SLS optimiser.
        """
        if self.scenario['model'] == Model.ATTITUDE:
            control_model = LinAttModel(self.scenario)
        elif self.scenario['model'] == Model.ROE:
            control_model = QuasiROE(self.scenario)
        elif self.scenario['model'] == Model.HCW:
            control_model = RelCylHCW(self.scenario)
        else:
            raise Exception(f"Model type {self.scenario['model']} not recognized!")

        self.sls = SLSSetup(sampling_time=self.scenario['control']['control_timestep'], system_dynamics=control_model,
                            tFIR=self.scenario['control']['tFIR'])
        self.sls.create_system(number_of_systems=self.scenario['number_of_sats'])

        # Add cost matrices
        self.sls.create_cost_matrices()

        # Create x0 and x_ref
        self.sls.create_x0(number_of_dropouts=self.scenario['initial_state']['dropouts'])
        self.sls.create_reference()

    def create_storage_variables(self) -> None:
        """
        Create variables to store the results in.
        """
        self.t_horizon_control = int(np.ceil(self.scenario['simulation']['simulation_duration'] /
                                             self.scenario['control']['control_timestep'])) + 1
        self.control_simulation_ratio = int(self.scenario['control']['control_timestep'] /
                                            self.scenario['simulation']['simulation_timestep'])
        self.t_horizon_simulation = self.control_simulation_ratio * self.t_horizon_control

        # Also include reference for pos/rot states
        self.sls_states = np.zeros((self.t_horizon_simulation + 1, len(self.sls.x0)))
        self.pos_states = np.zeros((self.t_horizon_simulation + 1, (self.scenario['number_of_sats'] + 1) * 6))
        self.rot_states = np.zeros((self.t_horizon_simulation + 1, (self.scenario['number_of_sats'] + 1) * 7))

    def __create_simulation(self) -> None:
        """
        Create an OrbitalMechSim and set most of the parameters, except initial positions.
        """
        self.orbital_mech = OrbitalMechSimulator()
        self.orbital_mech.set_mean_motion(self.sls.dynamics.mean_motion)
        self.orbital_mech.create_bodies(number_of_satellites=self.scenario['number_of_sats'],
                                        satellite_mass=self.scenario['physics']['mass'],
                                        satellite_inertia=self.scenario['physics']['inertia_tensor'],
                                        add_reference_satellite=True,
                                        use_parameters_from_scenario=self.scenario)

        # Create thrust models
        self.orbital_mech.create_engine_models_thrust(control_timestep=self.scenario['control']['control_timestep'],
                                                      thrust_inputs=self.control_inputs_thrust,
                                                      specific_impulse=self.scenario['physics']['specific_impulse'])

        # Create torque models
        self.orbital_mech.create_engine_models_torque(control_timestep=self.scenario['control']['control_timestep'],
                                                      torque_inputs=self.control_inputs_torque,
                                                      specific_impulse=self.scenario['physics']['specific_impulse'])

        # Add acceleration to the model
        if self.scenario['physics']['J2_perturbations']:
            self.orbital_mech.create_acceleration_model(order_of_zonal_harmonics=2)  # Check if this works!
        else:
            self.orbital_mech.create_acceleration_model(order_of_zonal_harmonics=0)

        # Add torques to the satellites. Currently only adds 2nd degree grav torque.
        self.orbital_mech.create_torque_model()

        # Add dependent variables to track
        self.orbital_mech.set_dependent_variables_translation(add_keplerian_state=True,
                                                              add_rsw_rotation_matrix=True,
                                                              add_thrust_accel=True)
        self.orbital_mech.set_dependent_variables_rotation(add_control_torque=True, add_torque_norm=False)

    def __initialise_simulation(self) -> None:
        """
        Initialise the simulation by setting the correct initial positions.
        """
        # Set the initial position of the satellites
        if isinstance(self.sls.dynamics, LinAttModel):
            true_anomalies = np.linspace(0, 360, self.scenario['number_of_sats'], endpoint=False).tolist()
        else:
            true_anomalies = np.rad2deg(self.sls.x0[self.sls.angle_states].reshape((-1,))).tolist()

        self.pos_states[0:1] = \
            self.orbital_mech.convert_orbital_elements_to_cartesian(true_anomalies=true_anomalies,
                                                                    orbit_height=self.scenario['orbital']['orbital_height'],
                                                                    inclination=self.scenario['orbital']['inclination'],
                                                                    longitude=self.scenario['orbital']['longitude'])

        angular_vel = np.array([0, 0, -1 * self.sls.dynamics.mean_motion])
        angular_vel_offsets = np.random.rand(3, self.scenario['number_of_sats']) * \
                              self.scenario['initial_state']['angular_velocity_offset_magnitude']
        self.rot_states[0:1] = \
            self.orbital_mech.convert_orbital_elements_to_quaternion(true_anomalies=true_anomalies,
                                                                     initial_angular_velocity=angular_vel,
                                                                     inclination=self.scenario['orbital']['inclination'],
                                                                     longitude=self.scenario['orbital']['longitude'],
                                                                     initial_angle_offset=self.scenario['initial_state']['attitude_offset'],
                                                                     initial_velocity_offset=angular_vel_offsets)

    def __run_simulation_timestep(self, time: int, initial_setup: bool = False) -> None:
        """
        Run a single simulation timestep.

        :param time: The time for which the simulation is.
        :param initial_setup: Whether this simulation is to provide some initial values.
                              Simplified simulation in that case.
        """
        self.__create_simulation()

        if initial_setup:
            self.__initialise_simulation()
            start_epoch = self.scenario['simulation']['start_epoch']
            end_epoch = self.scenario['simulation']['simulation_timestep']
        else:
            start_epoch = self.scenario['simulation']['start_epoch'] + \
                          time * self.scenario['control']['control_timestep']
            end_epoch = self.scenario['simulation']['start_epoch'] + \
                        (time + 1) * self.scenario['control']['control_timestep']

        self.orbital_mech.set_initial_position(self.pos_states[time * self.control_simulation_ratio:
                                                               time * self.control_simulation_ratio + 1].T)
        self.orbital_mech.set_initial_orientation(self.rot_states[time * self.control_simulation_ratio:
                                                                  time * self.control_simulation_ratio + 1].T)

        # Create propagation settings
        self.orbital_mech.create_propagation_settings(start_epoch=start_epoch,
                                                      end_epoch=end_epoch,
                                                      simulation_step_size=self.scenario['simulation']['simulation_timestep'])

        # Simulate system
        self.orbital_mech.simulate_system()

        # Store results
        if not initial_setup:
            index_selection = np.arange(time * self.control_simulation_ratio + 1,
                                        (time + 1) * self.control_simulation_ratio + 1)

            dep_vars_array = self.orbital_mech.dep_vars
            self.pos_states[index_selection] = self.orbital_mech.translation_states[1:]
            self.rot_states[index_selection] = self.orbital_mech.rotation_states[1:]

            if self.dep_vars is None:
                self.dep_vars = np.zeros((self.t_horizon_simulation + 1, dep_vars_array.shape[1]))
                self.dep_vars[0] = dep_vars_array[0]
                self.dep_vars_dict = self.orbital_mech.dependent_variables_dict
            self.dep_vars[index_selection] = dep_vars_array[1:]
        else:
            index_selection = np.arange(0, 1)

        control_states = self.orbital_mech.get_states_for_dynamical_model(self.sls.dynamics)
        self.sls_states[index_selection] = control_states[1:self.control_simulation_ratio + 1]

        # Unwrap to prevent weird positions when positions are involved
        if isinstance(self.sls.dynamics, TranslationalDynamics) and not initial_setup:
            unwrap_indices = np.arange(time * self.control_simulation_ratio,
                                       (time + 1) * self.control_simulation_ratio + 1)
            index_selection = np.ix_(unwrap_indices, self.sls.angle_states)
            self.sls_states[index_selection] = np.unwrap(self.sls_states[index_selection], axis=0)

    def __synthesise_controller(self, time: int) -> None:
        """
        Synthesise the controller and generate optimal control inputs.

        :param time: The time at which to synthesise the controller.
        """
        self.sls.set_initial_conditions(self.sls_states[time * self.control_simulation_ratio:
                                                        time * self.control_simulation_ratio + 1].T)
        self.sls.simulate_system(t_horizon=1, noise=None, inputs_to_store=1)

        if isinstance(self.sls.dynamics, LinAttModel):
            self.control_inputs_torque = self.sls.u_inputs.reshape((self.scenario['number_of_sats'], 3, -1))
            self.control_inputs_thrust = 0 * self.control_inputs_torque
        else:
            self.control_inputs_thrust = self.sls.u_inputs.reshape((self.scenario['number_of_sats'], 3, -1))
            self.control_inputs_torque = 0 * self.control_inputs_thrust

    def simulate_system(self) -> None:
        """
        Run a simulation for the provided scenario.
        """
        self.__run_simulation_timestep(0, initial_setup=True)

        progress = 0
        for t in range(self.t_horizon_control):
            if t / self.t_horizon_control * 100 > progress:
                print(f"Progress: {int(t / self.t_horizon_control * 100)}%")
                progress = int(t / self.t_horizon_control * 100) + 1

            self.__synthesise_controller(t)
            self.__run_simulation_timestep(t)

    def export_results(self) -> OrbitalMechSimulator:
        """
        Export the results in an OrbitalMechSimulator object.

        :return: OrbitalMechSimulator with all the obtained results.
        """
        orbital_sim = OrbitalMechSimulator()
        orbital_sim.number_of_controlled_satellites = self.scenario['number_of_sats']
        orbital_sim.number_of_total_satellites = self.scenario['number_of_sats'] + 1
        orbital_sim.simulation_timestep = self.scenario['simulation']['simulation_timestep']
        orbital_sim.translation_states = self.pos_states
        orbital_sim.rotation_states = self.rot_states
        orbital_sim.dep_vars = self.dep_vars
        orbital_sim.dependent_variables_dict = self.dep_vars_dict
        orbital_sim.controlled_satellite_names = self.orbital_mech.controlled_satellite_names
        orbital_sim.satellite_mass = self.scenario['physics']['mass']
        orbital_sim.set_mean_motion(self.sls.dynamics.mean_motion)

        return orbital_sim

