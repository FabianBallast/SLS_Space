from enum import Enum
from Scenarios.OrbitalScenarios import OrbitalScenarios
from Scenarios.PhysicsScenarios import PhysicsScenarios
from Scenarios.InitialStateScenarios import InitialStateScenarios
from Scenarios.SimulationScenarios import SimulationScenarios
from Scenarios.ControlScenarios import ControlParameterScenarios, Model


class Scenario:
    """
    A class representing a scenario.
    """

    def __init__(self, orbital_scenario: OrbitalScenarios = OrbitalScenarios.equatorial_orbit,
                 physics_scenario: PhysicsScenarios = PhysicsScenarios.basic_physics,
                 simulation_scenario: SimulationScenarios = SimulationScenarios.sim_1_minute,
                 initial_state_scenario: InitialStateScenarios = InitialStateScenarios.no_state_error,
                 control_scenario: ControlParameterScenarios = ControlParameterScenarios.control_attitude_default,
                 number_of_satellites: int = 3, model: Model = Model.ATTITUDE, collision_avoidance: bool = False,
                 use_mean_simulator: bool = True):
        self.orbital = orbital_scenario.value
        self.physics = physics_scenario.value
        self.simulation = simulation_scenario.value
        self.initial_state = initial_state_scenario.value
        self.control = control_scenario.value
        self.number_of_satellites = number_of_satellites
        self.model = model
        self.collision_avoidance = collision_avoidance
        self.use_mean_simulator = use_mean_simulator


class AttitudeScenario(Scenario):
    """
    Class representing a scenario for attitude control.
    """
    def __init__(self, orbital_scenario: OrbitalScenarios = OrbitalScenarios.arbitrary_orbit,
                 physics_scenario: PhysicsScenarios = PhysicsScenarios.basic_physics,
                 simulation_scenario: SimulationScenarios = SimulationScenarios.sim_10_minute,
                 initial_state_scenario: InitialStateScenarios = InitialStateScenarios.small_state_error,
                 control_scenario: ControlParameterScenarios = ControlParameterScenarios.control_attitude_default,
                 number_of_satellites: int = 3):
        super().__init__(orbital_scenario, physics_scenario, simulation_scenario, initial_state_scenario,
                         control_scenario, number_of_satellites, model=Model.ATTITUDE)


class TranslationalScenario(Scenario):
    """
    Class representing a scenario for translational control.
    """
    def __init__(self, orbital_scenario: OrbitalScenarios = OrbitalScenarios.tilted_orbit_45deg,
                 physics_scenario: PhysicsScenarios = PhysicsScenarios.basic_physics,
                 simulation_scenario: SimulationScenarios = SimulationScenarios.sim_10_minute,
                 initial_state_scenario: InitialStateScenarios = InitialStateScenarios.no_state_error,
                 control_scenario: ControlParameterScenarios = ControlParameterScenarios.control_position_fine,
                 number_of_satellites: int = 5, model: Model = Model.HCW, collision_avoidance: bool = False):
        super().__init__(orbital_scenario, physics_scenario, simulation_scenario, initial_state_scenario,
                         control_scenario, number_of_satellites, model, collision_avoidance)


class ScenarioEnum(Enum):
    # Attitude scenarios
    simple_scenario_attitude = AttitudeScenario(initial_state_scenario=InitialStateScenarios.no_state_error)
    simple_scenario_attitude_scaled = AttitudeScenario(physics_scenario=PhysicsScenarios.basic_physics_scaled,
                                                       initial_state_scenario=InitialStateScenarios.no_state_error)
    advanced_scenario_attitude = AttitudeScenario(physics_scenario=PhysicsScenarios.advanced_grav_physics,
                                                  initial_state_scenario=InitialStateScenarios.large_state_error)
    advanced_scenario_attitude_scaled = AttitudeScenario(physics_scenario=PhysicsScenarios.advanced_grav_physics_scaled,
                                                         initial_state_scenario=InitialStateScenarios.large_state_error)

    # Translational scenarios
    position_keeping_scenario_translation_HCW = TranslationalScenario()
    position_keeping_scenario_translation_HCW_scaled = TranslationalScenario(physics_scenario=PhysicsScenarios.basic_physics_scaled)
    position_keeping_scenario_translation_ROE = TranslationalScenario(model=Model.ROE)
    position_keeping_scenario_translation_ROE_scaled = TranslationalScenario(physics_scenario=PhysicsScenarios.basic_physics_scaled,
                                                                             model=Model.ROE)
    simple_scenario_translation_HCW = TranslationalScenario(initial_state_scenario=InitialStateScenarios.small_state_error,
                                                            simulation_scenario=SimulationScenarios.sim_1_hour)
    simple_scenario_translation_HCW_scaled = TranslationalScenario(physics_scenario=PhysicsScenarios.basic_physics_scaled,
                                                                   initial_state_scenario=InitialStateScenarios.small_state_error,
                                                                   number_of_satellites=10,
                                                                   simulation_scenario=SimulationScenarios.sim_45_minute,
                                                                   control_scenario=ControlParameterScenarios.control_position_fine)
    simple_scenario_translation_ROE = TranslationalScenario(model=Model.ROE,
                                                            initial_state_scenario=InitialStateScenarios.small_state_error)
    simple_scenario_translation_ROE_scaled = TranslationalScenario(physics_scenario=PhysicsScenarios.basic_physics_scaled,
                                                                   model=Model.ROE,
                                                                   initial_state_scenario=InitialStateScenarios.small_state_error,
                                                                   control_scenario=ControlParameterScenarios.control_position_fine,
                                                                   number_of_satellites=10,
                                                                   simulation_scenario=SimulationScenarios.sim_45_minute)

    j2_scenario_pos_keep_HCW = TranslationalScenario(physics_scenario=PhysicsScenarios.advanced_grav_physics)
    j2_scenario_pos_keep_HCW_scaled = TranslationalScenario(physics_scenario=PhysicsScenarios.advanced_grav_physics_scaled,
                                                            control_scenario=ControlParameterScenarios.control_position_default,
                                                            simulation_scenario=SimulationScenarios.sim_45_minute,
                                                            number_of_satellites=10)
    j2_scenario_pos_keep_ROE = TranslationalScenario(physics_scenario=PhysicsScenarios.advanced_grav_physics, model=Model.ROE)
    j2_scenario_pos_keep_ROE_scaled = TranslationalScenario(physics_scenario=PhysicsScenarios.advanced_grav_physics_scaled, model=Model.ROE,
                                                            simulation_scenario=SimulationScenarios.sim_45_minute,
                                                            number_of_satellites=10)

    j2_scenario_moving_HCW = TranslationalScenario(physics_scenario=PhysicsScenarios.advanced_grav_physics,
                                                   initial_state_scenario=InitialStateScenarios.small_state_error)
    j2_scenario_moving_HCW_scaled = TranslationalScenario(physics_scenario=PhysicsScenarios.advanced_grav_physics_scaled,
                                                          initial_state_scenario=InitialStateScenarios.small_state_error,
                                                          control_scenario=ControlParameterScenarios.control_position_fine,
                                                          simulation_scenario=SimulationScenarios.sim_45_minute,
                                                          number_of_satellites=10)
    j2_scenario_moving_ROE = TranslationalScenario(physics_scenario=PhysicsScenarios.advanced_grav_physics,
                                                   model=Model.ROE,
                                                   initial_state_scenario=InitialStateScenarios.small_state_error)
    j2_scenario_moving_ROE_scaled = TranslationalScenario(physics_scenario=PhysicsScenarios.advanced_grav_physics_scaled,
                                                          initial_state_scenario=InitialStateScenarios.small_state_error,
                                                          model=Model.ROE,
                                                          simulation_scenario=SimulationScenarios.sim_45_minute,
                                                          number_of_satellites=10,
                                                          control_scenario=ControlParameterScenarios.control_position_fine)

    simple_scenario_translation_SimAn_scaled = TranslationalScenario(
        physics_scenario=PhysicsScenarios.basic_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        model=Model.DIFFERENTIAL_DRAG,
        number_of_satellites=10,
        simulation_scenario=SimulationScenarios.sim_45_minute,
        control_scenario=ControlParameterScenarios.control_differential_drag)

    j2_scenario_translation_SimAn_scaled = TranslationalScenario(
        physics_scenario=PhysicsScenarios.advanced_grav_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        model=Model.DIFFERENTIAL_DRAG,
        number_of_satellites=10,
        simulation_scenario=SimulationScenarios.sim_45_minute,
        control_scenario=ControlParameterScenarios.control_differential_drag)

    simple_scenario_translation_ROEV2_scaled = TranslationalScenario(
        physics_scenario=PhysicsScenarios.basic_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        model=Model.ROE_V2,
        number_of_satellites=10,
        simulation_scenario=SimulationScenarios.sim_30_minute,
        control_scenario=ControlParameterScenarios.control_position_default,
        orbital_scenario=OrbitalScenarios.eccentric_orbit
    )

    j2_scenario_translation_ROEV2_scaled = TranslationalScenario(
        physics_scenario=PhysicsScenarios.advanced_grav_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        model=Model.ROE_V2,
        number_of_satellites=10,
        simulation_scenario=SimulationScenarios.sim_30_minute,
        control_scenario=ControlParameterScenarios.control_position_default,
        orbital_scenario=OrbitalScenarios.eccentric_orbit
    )

    simple_scenario_translation_blend_scaled = TranslationalScenario(
        physics_scenario=PhysicsScenarios.basic_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        model=Model.BLEND,
        number_of_satellites=10,
        simulation_scenario=SimulationScenarios.sim_45_minute,
        control_scenario=ControlParameterScenarios.control_position_fine,
        orbital_scenario=OrbitalScenarios.tilted_orbit_45deg
    )

    j2_scenario_translation_blend_scaled = TranslationalScenario(
        physics_scenario=PhysicsScenarios.advanced_grav_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        model=Model.BLEND,
        number_of_satellites=10,
        simulation_scenario=SimulationScenarios.sim_45_minute,
        control_scenario=ControlParameterScenarios.control_position_fine,
        orbital_scenario=OrbitalScenarios.tilted_orbit_45deg
    )

    simple_scenario_translation_blend_small_scaled = TranslationalScenario(
        physics_scenario=PhysicsScenarios.basic_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        model=Model.BLEND_SMALL,
        number_of_satellites=10,
        simulation_scenario=SimulationScenarios.sim_10_minute,
        control_scenario=ControlParameterScenarios.control_position_default,
        orbital_scenario=OrbitalScenarios.tilted_orbit_45deg
    )

    j2_scenario_translation_blend_small_scaled = TranslationalScenario(
        physics_scenario=PhysicsScenarios.advanced_grav_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        model=Model.BLEND_SMALL,
        number_of_satellites=10,
        simulation_scenario=SimulationScenarios.sim_30_minute,
        control_scenario=ControlParameterScenarios.control_position_default,
        orbital_scenario=OrbitalScenarios.tilted_orbit_45deg
    )

    j2_scenario_moving_HCW_2_orbits = TranslationalScenario(physics_scenario=PhysicsScenarios.advanced_grav_physics_scaled,
                                                            initial_state_scenario=InitialStateScenarios.small_state_error,
                                                            control_scenario=ControlParameterScenarios.control_position_fine,
                                                            simulation_scenario=SimulationScenarios.sim_1_5_hour,
                                                            number_of_satellites=10,
                                                            orbital_scenario=OrbitalScenarios.tilted_orbit_45deg_group_2_close)

    j2_scenario_moving_blend_2_orbits = TranslationalScenario(
        physics_scenario=PhysicsScenarios.advanced_grav_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        control_scenario=ControlParameterScenarios.control_position_fine,
        simulation_scenario=SimulationScenarios.sim_1_5_hour,
        number_of_satellites=10,
        orbital_scenario=OrbitalScenarios.tilted_orbit_45deg_group_2_close,
        model=Model.BLEND)

    j2_scenario_moving_ROE_2_orbits = TranslationalScenario(
        physics_scenario=PhysicsScenarios.advanced_grav_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        control_scenario=ControlParameterScenarios.control_position_fine,
        simulation_scenario=SimulationScenarios.sim_1_5_hour,
        number_of_satellites=10,
        orbital_scenario=OrbitalScenarios.tilted_orbit_45deg_group_2_close,
        model=Model.ROE)

    simple_scenario_moving_blend_2_orbits = TranslationalScenario(
        physics_scenario=PhysicsScenarios.basic_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        control_scenario=ControlParameterScenarios.control_position_fine,
        simulation_scenario=SimulationScenarios.sim_1_5_hour,
        number_of_satellites=10,
        orbital_scenario=OrbitalScenarios.tilted_orbit_45deg_group_2_close,
        model=Model.BLEND)

    simple_scenario_HCW_2_orbits = TranslationalScenario(
        physics_scenario=PhysicsScenarios.basic_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        control_scenario=ControlParameterScenarios.control_position_fine,
        simulation_scenario=SimulationScenarios.sim_1_5_hour,
        number_of_satellites=10,
        orbital_scenario=OrbitalScenarios.tilted_orbit_45deg_group_2_close,
        model=Model.HCW)

    simple_scenario_ROE_2_orbits = TranslationalScenario(
        physics_scenario=PhysicsScenarios.basic_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        control_scenario=ControlParameterScenarios.control_position_fine,
        simulation_scenario=SimulationScenarios.sim_1_5_hour,
        number_of_satellites=10,
        orbital_scenario=OrbitalScenarios.tilted_orbit_45deg_group_2_close,
        model=Model.ROE)

    simple_scenario_ROE_6_orbits = TranslationalScenario(
        physics_scenario=PhysicsScenarios.basic_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        control_scenario=ControlParameterScenarios.control_position_fine,
        simulation_scenario=SimulationScenarios.sim_2_hour,
        number_of_satellites=36,
        orbital_scenario=OrbitalScenarios.tilted_orbit_45deg_group_6,
        model=Model.ROE)

    simple_scenario_moving_blend_6_orbits = TranslationalScenario(
        physics_scenario=PhysicsScenarios.basic_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        control_scenario=ControlParameterScenarios.control_position_fine,
        simulation_scenario=SimulationScenarios.sim_2_hour,
        number_of_satellites=36,
        orbital_scenario=OrbitalScenarios.tilted_orbit_45deg_group_6,
        model=Model.BLEND)

    simple_scenario_HCW_6_orbits = TranslationalScenario(
        physics_scenario=PhysicsScenarios.basic_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        control_scenario=ControlParameterScenarios.control_position_fine,
        simulation_scenario=SimulationScenarios.sim_2_hour,
        number_of_satellites=36,
        orbital_scenario=OrbitalScenarios.tilted_orbit_45deg_group_6,
        model=Model.HCW)

    j2_scenario_HCW_6_orbits = TranslationalScenario(
        physics_scenario=PhysicsScenarios.advanced_grav_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        control_scenario=ControlParameterScenarios.control_position_fine,
        simulation_scenario=SimulationScenarios.sim_2_hour,
        number_of_satellites=36,
        orbital_scenario=OrbitalScenarios.tilted_orbit_45deg_group_6,
        model=Model.HCW)

    j2_scenario_ROE_6_orbits = TranslationalScenario(
        physics_scenario=PhysicsScenarios.advanced_grav_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        control_scenario=ControlParameterScenarios.control_position_fine,
        simulation_scenario=SimulationScenarios.sim_2_hour,
        number_of_satellites=36,
        orbital_scenario=OrbitalScenarios.tilted_orbit_45deg_group_6,
        model=Model.ROE)

    j2_scenario_moving_blend_6_orbits = TranslationalScenario(
        physics_scenario=PhysicsScenarios.advanced_grav_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        control_scenario=ControlParameterScenarios.control_position_fine,
        simulation_scenario=SimulationScenarios.sim_2_hour,
        number_of_satellites=36,
        orbital_scenario=OrbitalScenarios.tilted_orbit_45deg_group_6,
        model=Model.BLEND)

    j2_scenario_moving_blend_6_orbits_ca = TranslationalScenario(
        physics_scenario=PhysicsScenarios.advanced_grav_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        control_scenario=ControlParameterScenarios.control_position_default,
        simulation_scenario=SimulationScenarios.sim_10_orbital_period,
        number_of_satellites=36,
        orbital_scenario=OrbitalScenarios.tilted_orbit_45deg_group_6,
        model=Model.BLEND,
        collision_avoidance=True)

    j2_scenario_moving_blend_small_2_orbits = TranslationalScenario(
        physics_scenario=PhysicsScenarios.advanced_grav_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        control_scenario=ControlParameterScenarios.control_position_default,
        simulation_scenario=SimulationScenarios.sim_5_orbital_period,
        number_of_satellites=10,
        orbital_scenario=OrbitalScenarios.tilted_orbit_45deg_group_2_close,
        model=Model.BLEND_SMALL)

    j2_scenario_moving_blend_small_6_orbits = TranslationalScenario(
        physics_scenario=PhysicsScenarios.advanced_grav_physics_scaled,
        initial_state_scenario=InitialStateScenarios.small_state_error,
        control_scenario=ControlParameterScenarios.control_position_default,
        simulation_scenario=SimulationScenarios.sim_10_minute,
        number_of_satellites=36,
        orbital_scenario=OrbitalScenarios.tilted_orbit_45deg_group_6,
        model=Model.BLEND_SMALL)

    projection_comparison = TranslationalScenario(
        physics_scenario=PhysicsScenarios.basic_physics_scaled,
        initial_state_scenario=InitialStateScenarios.no_state_error,
        control_scenario=ControlParameterScenarios.control_position_fine,
        simulation_scenario=SimulationScenarios.sim_orbital_period,
        number_of_satellites=4,
        orbital_scenario=OrbitalScenarios.projection_comparison,
        model=Model.HCW)

#
# simple_scenario_attitude = {'orbital': OrbitalScenarios.arbitrary_orbit.value,
#                             'physics': PhysicsScenarios.basic_physics.value,
#                             'simulation': simulation_scenarios['sim_1_minute'],
#                             'initial_state': initial_state_scenarios['no_state_error'],
#                             'control': control_scenarios['control_attitude_default'],
#                             'number_of_sats': 3,
#                             'model': Model.ATTITUDE}
#
# simple_scenario_attitude_scaled = {'orbital': orbital_scenarios['arbitrary_orbit'],
#                                    'physics': physics_scenarios['basic_physics_scaled'],
#                                    'simulation': simulation_scenarios['sim_1_minute'],
#                                    'initial_state': initial_state_scenarios['no_state_error'],
#                                    'control': control_scenarios['control_attitude_default'],
#                                    'number_of_sats': 3,
#                                    'model': Model.ATTITUDE}
#
# advanced_scenario_attitude = {'orbital': orbital_scenarios['arbitrary_orbit'],
#                               'physics': physics_scenarios['advanced_grav_physics'],
#                               'simulation': simulation_scenarios['sim_1_minute'],
#                               'initial_state': initial_state_scenarios['large_state_error'],
#                               'control': control_scenarios['control_attitude_default'],
#                               'number_of_sats': 3,
#                               'model': Model.ATTITUDE}
#
# advanced_scenario_attitude_scaled = {'orbital': orbital_scenarios['arbitrary_orbit'],
#                                      'physics': physics_scenarios['advanced_grav_physics_scaled'],
#                                      'simulation': simulation_scenarios['sim_1_minute'],
#                                      'initial_state': initial_state_scenarios['large_state_error'],
#                                      'control': control_scenarios['control_attitude_default'],
#                                      'number_of_sats': 3,
#                                      'model': Model.ATTITUDE}
#
# position_keeping_scenario_translation_HCW = {'orbital': orbital_scenarios['tilted_orbit_45deg'],
#                                              'physics': physics_scenarios['basic_physics'],
#                                              'simulation': simulation_scenarios['sim_10_minute'],
#                                              'initial_state': initial_state_scenarios['no_state_error'],
#                                              'control': control_scenarios['control_position_default'],
#                                              'number_of_sats': 5,
#                                              'model': Model.HCW}
#
# position_keeping_scenario_translation_HCW_scaled = {'orbital': orbital_scenarios['tilted_orbit_45deg'],
#                                                     'physics': physics_scenarios['basic_physics_scaled'],
#                                                     'simulation': simulation_scenarios['sim_10_minute'],
#                                                     'initial_state': initial_state_scenarios['no_state_error'],
#                                                     'control': control_scenarios['control_position_default'],
#                                                     'number_of_sats': 5,
#                                                     'model': Model.HCW}
#
# position_keeping_scenario_translation_ROE = {'orbital': orbital_scenarios['tilted_orbit_45deg'],
#                                              'physics': physics_scenarios['basic_physics'],
#                                              'simulation': simulation_scenarios['sim_10_minute'],
#                                              'initial_state': initial_state_scenarios['no_state_error'],
#                                              'control': control_scenarios['control_position_default'],
#                                              'number_of_sats': 5,
#                                              'model': Model.ROE}
#
# position_keeping_scenario_translation_ROE_scaled = {'orbital': orbital_scenarios['tilted_orbit_45deg'],
#                                                     'physics': physics_scenarios['basic_physics_scaled'],
#                                                     'simulation': simulation_scenarios['sim_10_minute'],
#                                                     'initial_state': initial_state_scenarios['no_state_error'],
#                                                     'control': control_scenarios['control_position_default'],
#                                                     'number_of_sats': 5,
#                                                     'model': Model.ROE}
#
# simple_scenario_translation_HCW = {'orbital': orbital_scenarios['tilted_orbit_45deg'],
#                                    'physics': physics_scenarios['basic_physics'],
#                                    'simulation': simulation_scenarios['sim_1_hour'],
#                                    'initial_state': initial_state_scenarios['small_state_error'],
#                                    'control': control_scenarios['control_position_default'],
#                                    'number_of_sats': 5,
#                                    'model': Model.HCW}
#
# simple_scenario_translation_HCW_scaled = {'orbital': orbital_scenarios['tilted_orbit_45deg'],
#                                           'physics': physics_scenarios['basic_physics_scaled'],
#                                           'simulation': simulation_scenarios['sim_10_minute'],
#                                           'initial_state': initial_state_scenarios['small_state_error'],
#                                           'control': control_scenarios['control_position_default'],
#                                           'number_of_sats': 5,
#                                           'model': Model.HCW}
#
# simple_scenario_translation_ROE = {'orbital': orbital_scenarios['tilted_orbit_45deg'],
#                                    'physics': physics_scenarios['basic_physics'],
#                                    'simulation': simulation_scenarios['sim_10_minute'],
#                                    'initial_state': initial_state_scenarios['small_state_error'],
#                                    'control': control_scenarios['control_position_default'],
#                                    'number_of_sats': 5,
#                                    'model': Model.ROE}
#
# simple_scenario_translation_ROE_scaled = {'orbital': orbital_scenarios['tilted_orbit_45deg'],
#                                           'physics': physics_scenarios['basic_physics_scaled'],
#                                           'simulation': simulation_scenarios['sim_10_minute'],
#                                           'initial_state': initial_state_scenarios['small_state_error'],
#                                           'control': control_scenarios['control_position_default'],
#                                           'number_of_sats': 5,
#                                           'model': Model.ROE}
#
# medium_scenario_translation_HCW = {'orbital': orbital_scenarios['tilted_orbit_45deg'],
#                                    'physics': physics_scenarios['advanced_grav_physics'],
#                                    'simulation': simulation_scenarios['sim_1_hour'],
#                                    'initial_state': initial_state_scenarios['small_state_error'],
#                                    'control': control_scenarios['control_position_default'],
#                                    'number_of_sats': 5,
#                                    'model': Model.HCW}
#
# medium_scenario_translation_ROE = {'orbital': orbital_scenarios['tilted_orbit_45deg'],
#                                    'physics': physics_scenarios['advanced_grav_physics'],
#                                    'simulation': simulation_scenarios['sim_1_hour'],
#                                    'initial_state': initial_state_scenarios['small_state_error'],
#                                    'control': control_scenarios['control_position_default'],
#                                    'number_of_sats': 5,
#                                    'model': Model.ROE}
#
# advanced_scenario_translation_HCW = {'orbital': orbital_scenarios['tilted_orbit_45deg'],
#                                      'physics': physics_scenarios['full_physics'],
#                                      'simulation': simulation_scenarios['sim_1_hour'],
#                                      'initial_state': initial_state_scenarios['large_state_error'],
#                                      'control': control_scenarios['control_position_default'],
#                                      'number_of_sats': 5,
#                                      'model': Model.HCW}
#
# advanced_scenario_translation_ROE = {'orbital': orbital_scenarios['tilted_orbit_45deg'],
#                                      'physics': physics_scenarios['full_physics'],
#                                      'simulation': simulation_scenarios['sim_1_hour'],
#                                      'initial_state': initial_state_scenarios['large_state_error'],
#                                      'control': control_scenarios['control_position_default'],
#                                      'number_of_sats': 5,
#                                      'model': Model.ROE}
