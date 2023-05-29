from matplotlib import pyplot as plt
from matplotlib.cbook import silent_list
import mpl_toolkits.mplot3d.art3d as p3
import numpy as np


def get_time_axis(states_over_time: np.ndarray, timestep: float, required_scale: float = 3600) -> np.ndarray:
    """
    Create the time axis used for plots.

    :param states_over_time: The states that have to be plotted of shape (t, any).
    :param timestep: Timestep between each state in s.
    :param required_scale: Timescale to which to convert. Default is hours (3600 s).
    :return: Array of shape (t, ) with the time in the required scale.
    """
    return np.arange(0, len(states_over_time[:, 0])) * timestep / required_scale


def get_figure_and_axes(figure: plt.figure, shape_of_plots: tuple) -> (plt.figure, tuple[plt.axes]):
    """
    Get the figure with its axes given a shape.

    :param figure: Figure that is possibly already present. Use None to create a new one.
    :param shape_of_plots: Shape of the required plot if no figure was provided.
    :return: Tuple with figure and a tuple of axes.
    """
    if figure is None:
        fig, _ = plt.subplots(shape_of_plots[0], shape_of_plots[1], figsize=(9, 12))
    else:
        fig = figure

    return fig, fig.get_axes()


def get_scaling_parameters(state: np.ndarray) -> tuple[int, str, int]:
    """
    Find different parameters used to deal with different scaling.

    :param state: Example of a state that has to be plotted in m.

    :return: Tuple with maximum size of plots, units to use in labels for plots and how much to divide each value by.
    """
    if np.linalg.norm(state) > 1000:
        return 7500, 'km', 1000
    else:
        return 60, 'm', 1


def create_3d_plot(max_distance: int, unit_label: str = 'km') -> tuple[plt.figure, plt.axes]:
    """
    Basis for a 3D plot with the Earth.

    :param max_distance: Maximum distance from satellite to Earth in km for scaling the plot.
    :param unit_label: The unit to use in the label. Defaults to km.
    :return: Tuple with figure and corresponding axes.
    """
    # Define a 3D figure using pyplot
    figure = plt.figure(figsize=(6, 6), dpi=150)
    ax = figure.add_subplot(111, projection='3d')
    ax.set_title(f'Satellite trajectories around Earth')
    ax.set_xlabel(f'x [{unit_label}]')
    ax.set_ylabel(f'y [{unit_label}]')
    ax.set_zlabel(f'z [{unit_label}]')
    ax.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue')
    ax.set_xlim(-max_distance, max_distance)
    ax.set_ylim(-max_distance, max_distance)
    ax.set_zlim(-max_distance, max_distance)

    return figure, ax


def plot_3d_trajectory(states: np.ndarray, state_label_name: str, figure: plt.figure = None) -> plt.figure:
    """
    Plot the trajectory of a satellite in 3D.

    :param states: States of the satellite in the shape (t, 3) in m.
    :param state_label_name: Label for the legend for these states.
    :param figure: Figure to plot the trajectories in. If not provided, a new one is created.
    :return: Figure with the added trajectory.
    """
    max_distance, units, scaling_factor = get_scaling_parameters(states[0])

    if figure is None:
        figure, ax = create_3d_plot(max_distance=max_distance, unit_label=units)
    else:
        ax = figure.get_axes()[0]

    # Plot the positional state history
    ax.plot(states[:, 0] / scaling_factor, states[:, 1] / scaling_factor,
            states[:, 2] / scaling_factor, label=state_label_name, linestyle='-')

    # Add the legend
    ax.legend()

    return figure


def plot_3d_position(position: np.ndarray, state_label_name: str, figure: plt.figure = None) -> plt.figure:
    """
    Plot the position of a satellite in 3D.

    :param position: Position of the satellite in the shape (3,) in m.
    :param state_label_name: Label for the legend for these states.
    :param figure: Figure to plot the trajectories in. If not provided, a new one is created.
    :return: Figure with the added trajectory.
    """
    max_distance, units, scaling_factor = get_scaling_parameters(position)

    if figure is None:
        figure, ax = create_3d_plot(max_distance=max_distance, unit_label=units)
    else:
        ax = figure.get_axes()[0]

    # Plot the positional state history
    ax.scatter(position[0] / scaling_factor, position[1] / scaling_factor, position[2] / scaling_factor,
               label=state_label_name, marker='o')

    # Add the legend
    ax.legend()

    return figure


def plot_onto_axes(states: np.ndarray, time: np.ndarray, axes_list: list[plt.axes], is_angle: list[bool],
                   y_label_names: list[str], state_label_name: str, unwrap_angles: bool = True, **kwargs) -> None:
    """
    Plot states on the provided axes with a given label and legend label name.
    :param states: The states (y-component) to be plotted.
    :param time: The time (x-component) to be plotted.
    :param axes_list: List with all the axes to plot onto.
    :param is_angle: List whether a certain value is an angle (and thus should be unwrapped/converted to deg)
    :param y_label_names: List with names of the y-axes.
    :param state_label_name: Name for the legend.
    :param unwrap_angles: Whether to unwrap angles. Default is True.
    :param kwargs: Kwargs for plotting purposes.
    :return: Nothing.
    """
    for idx, axes in enumerate(axes_list):
        state = states[:, idx]

        # If it is an angle, unwrap and convert to deg
        if is_angle[idx] and unwrap_angles:
            state = np.rad2deg(np.unwrap(state, axis=0))
        elif is_angle[idx]:
            state = np.rad2deg(state)

        # Plot data
        axes.plot(time, state, label=state_label_name, **kwargs)
        axes.set_xlabel('Time [hr]')
        axes.set_ylabel(y_label_names[idx])
        axes.set_xlim([min(time), max(time)])
        axes.grid()
        # axes.legend()

    plt.tight_layout()


def plot_keplerian_states(kepler_elements: np.ndarray, timestep: float, plot_argument_of_latitude: bool = False,
                          satellite_name: str = None, figure: plt.figure = None, **kwargs) -> plt.figure:
    """
    Method to plot the evolution of the orbital (kepler) elements over time.

    :param kepler_elements: 2D-array containing the orbital elements over time. Shape is (t, 6).
    :param timestep: Amount of time between each measurement in s.
    :param plot_argument_of_latitude: Plot the argument of latitude or the true anomaly.
    :param satellite_name: Name to place as a label for the legend.
    :param figure: Figure to plot the kepler elements into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added keplerian states.
    """
    # Plot Kepler elements as a function of time
    time_hours = get_time_axis(kepler_elements, timestep)

    fig, axes = get_figure_and_axes(figure, (3, 2))
    fig.suptitle('Evolution of Kepler elements over the course of the propagation')

    is_angle_list = [False, False, True, True, True, True]
    y_label_list = ['Semi-major axis [m]', 'Eccentricity [-]', 'Inclination [deg]',
                   'Argument of Periapsis [deg]', 'RAAN [deg]', "True anomaly [deg]"]

    if plot_argument_of_latitude:
        kepler_elements[:, 5] += kepler_elements[:, 3]
        y_label_list[5] = "Argument of latitude [deg]"

    plot_onto_axes(kepler_elements, time_hours, list(axes), is_angle_list, y_label_list, satellite_name, **kwargs)

    return fig


def plot_thrust_forces(inputs: np.ndarray, timestep: float, satellite_name: str = None,
                       figure: plt.figure = None, **kwargs) -> plt.figure:
    """
    Method to plot the control inputs over time.

    :param inputs: 2D-array with the inputs over time with shape (t, 3).
    :param timestep: Amount of time between each input in s.
    :param satellite_name: Name to place as a label for the legend.
    :param figure: Figure to plot the inputs into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added control inputs.
    """
    time_hours = get_time_axis(inputs, timestep)

    fig, axes = get_figure_and_axes(figure, (4, 1))
    fig.suptitle('Control inputs over the course of the propagation.')

    is_angle_list = [False] * 4
    y_label_list = ['u_rho [N]', 'u_theta [N]', 'u_z [N]', "norm(u) [N]"]
    inputs_and_norm = np.concatenate((inputs, np.linalg.norm(inputs, axis=1).reshape(-1, 1)), axis=1)
    plot_onto_axes(inputs_and_norm, time_hours, list(axes), is_angle_list, y_label_list, satellite_name, **kwargs)

    return fig


def plot_drag_forces(inputs: np.ndarray, timestep: float, satellite_name: str = None,
                     figure: plt.figure = None, **kwargs) -> plt.figure:
    """
    Method to plot the control inputs over time.

    :param inputs: 2D-array with the inputs over time with shape (t, 1).
    :param timestep: Amount of time between each input in s.
    :param satellite_name: Name to place as a label for the legend.
    :param figure: Figure to plot the inputs into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added control inputs.
    """
    time_hours = get_time_axis(inputs, timestep)

    fig, axes = get_figure_and_axes(figure, (1, 1))
    fig.suptitle('Control inputs over the course of the propagation.')

    is_angle_list = [False] * 1
    y_label_list = ['u_drag [N]']
    plot_onto_axes(inputs, time_hours, list(axes), is_angle_list, y_label_list, satellite_name, **kwargs)

    return fig


def plot_control_torques(inputs: np.ndarray, timestep: float, satellite_name: str = None,
                         figure: plt.figure = None, **kwargs) -> plt.figure:
    """
    Method to plot the control torques over time.

    :param inputs: 2D-array with the torques over time with shape (t, 3).
    :param timestep: Amount of time between each input in s.
    :param satellite_name: Name to place as a label for the legend.
    :param figure: Figure to plot the inputs into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added control inputs.
    """
    time_hours = get_time_axis(inputs, timestep)

    fig, axes = get_figure_and_axes(figure, (4, 1))
    fig.suptitle('Control torques over the course of the propagation.')

    is_angle_list = [False] * 4
    y_label_list = ['u_x [Nm]', 'u_y [Nm]', 'u_z [Nm]', "norm(u) [Nm]"]
    inputs_and_norm = np.concatenate((inputs, np.linalg.norm(inputs, axis=1).reshape(-1, 1)), axis=1)
    plot_onto_axes(inputs_and_norm, time_hours, list(axes), is_angle_list, y_label_list, satellite_name, **kwargs)

    return fig


def plot_cylindrical_states(cylindrical_states: np.ndarray, timestep: float, satellite_name: str = None,
                            figure: plt.figure = None, **kwargs) -> plt.figure:
    """
    Method to plot the relative cylindrical states over time.

    :param cylindrical_states: 2D-array with the cylindrical states over time with shape (t, 6).
    :param timestep: Amount of time between each state in s.
    :param satellite_name: Name to place as a label for the legend.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added cylindrical states.
    """
    time_hours = get_time_axis(cylindrical_states, timestep)

    fig, axes = get_figure_and_axes(figure, (3, 2))
    fig.suptitle('Evolution of relative cylindrical states compared to reference.')

    is_angle_list = [False, True, False, False, True, False]
    y_label_list = ['rho [m]', 'theta [deg]', 'z [m]', "rho_dot [m/s]", 'theta_dot [deg/s]', 'z_dot [m/s]']
    plot_onto_axes(cylindrical_states, time_hours, list(axes), is_angle_list, y_label_list, satellite_name, **kwargs)

    return fig


def plot_quasi_roe(quasi_roe_states: np.ndarray, timestep: float, satellite_name: str = None,
                   figure: plt.figure = None, **kwargs) -> plt.figure:
    """
    Method to plot the quasi ROE states over time.

    :param quasi_roe_states: 2D-array with the quasi ROE states over time with shape (t, 6).
    :param timestep: Amount of time between each state in s.
    :param satellite_name: Name to place as a label for the legend.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added quasi ROE states.
    """
    time_hours = get_time_axis(quasi_roe_states, timestep)

    fig, axes = get_figure_and_axes(figure, (3, 2))
    fig.suptitle('Evolution of quasi ROE.')

    is_angle_list = [False, True, False, False, True, True]
    y_label_list = ['delta a [-]', 'delta lambda [deg]',
                    'delta e_x [-]', "delta e_y [-]",
                    'delta i_x [deg]', 'delta i_y [deg]']
    plot_onto_axes(quasi_roe_states, time_hours, list(axes), is_angle_list, y_label_list, satellite_name, **kwargs)

    return fig


def plot_roe(roe_states: np.ndarray, timestep: float, satellite_name: str = None,
             figure: plt.figure = None, **kwargs) -> plt.figure:
    """
    Method to plot the ROE states over time.

    :param roe_states: 2D-array with the quasi ROE states over time with shape (t, 6).
    :param timestep: Amount of time between each state in s.
    :param satellite_name: Name to place as a label for the legend.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added ROE states.
    """
    time_hours = get_time_axis(roe_states, timestep)

    fig, axes = get_figure_and_axes(figure, (3, 2))
    fig.suptitle('Evolution of ROE.')

    is_angle_list = [False, True, False, True, True, True]
    y_label_list = ['delta a [-]', 'delta lambda [deg]',
                    'delta e_x [-]', "delta e_y [deg]",
                    'delta i_x [deg]', 'delta i_y [deg]']
    plot_onto_axes(roe_states, time_hours, list(axes), is_angle_list, y_label_list, satellite_name, **kwargs)

    return fig


def plot_quaternion(quaternion_states: np.ndarray, timestep: float, satellite_name: str = None,
                    figure: plt.figure = None, **kwargs) -> plt.figure:
    """
    Method to plot the quaternions over time.

    :param quaternion_states: 2D-array with the quaternions over time with shape (t, 4).
    :param timestep: Amount of time between each state in s.
    :param satellite_name: Name to place as a label for the legend.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added quaternions.
    """
    time_hours = get_time_axis(quaternion_states, timestep)

    fig, axes = get_figure_and_axes(figure, (4, 1))
    fig.suptitle('Evolution of quaternions.')

    is_angle_list = [False, False, False, False]
    y_label_list = ['q_1 [-]', 'q_2 [-]', 'q_3 [-]', 'q_4 [-]']
    plot_onto_axes(quaternion_states, time_hours, list(axes), is_angle_list, y_label_list,
                   satellite_name, **kwargs)

    return fig


def plot_euler_angles(euler_angles: np.ndarray, timestep: float, satellite_name: str = None,
                      figure: plt.figure = None, **kwargs) -> plt.figure:
    """
    Method to plot the euler angles over time.

    :param euler_angles: 2D-array with the euler angles over time in rad with shape (t, 3).
    :param timestep: Amount of time between each state in s.
    :param satellite_name: Name to place as a label for the legend.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added euler angles states.
    """
    time_hours = get_time_axis(euler_angles, timestep)

    fig, axes = get_figure_and_axes(figure, (3, 1))
    fig.suptitle('Evolution of euler angles.')

    is_angle_list = [True, True, True]
    y_label_list = ['Roll [deg]', 'Pitch [deg]', 'Yaw [deg]']
    plot_onto_axes(euler_angles, time_hours, list(axes), is_angle_list, y_label_list, satellite_name,
                   unwrap_angles=True, **kwargs)

    return fig


def plot_angular_velocities(angular_velocities: np.ndarray, timestep: float, satellite_name: str = None,
                            figure: plt.figure = None, **kwargs) -> plt.figure:
    """
    Method to plot the angular velocities over time.

    :param angular_velocities: 2D-array with the angular velocities over time in rad with shape (t, 3).
    :param timestep: Amount of time between each state in s.
    :param satellite_name: Name to place as a label for the legend.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added euler angles states.
    """
    time_hours = get_time_axis(angular_velocities, timestep)

    fig, axes = get_figure_and_axes(figure, (3, 1))
    fig.suptitle('Evolution of angular velocities.')

    is_angle_list = [True, True, True]
    y_label_list = ['omega_x [deg/s]', 'omega_y [deg/s]', 'omega_z [deg/s]']
    plot_onto_axes(angular_velocities, time_hours, list(axes), is_angle_list, y_label_list, satellite_name,
                   unwrap_angles=False, **kwargs)

    return fig


def animation_function(t: int, points: silent_list[p3.Path3DCollection], states: np.ndarray,
                       satellite_indices: list[int]) -> tuple[silent_list[p3.Path3DCollection], None]:
    """
    Helper function for an animation plot.

    :param t: Time index at which to plot.
    :param points: The points of the plot.
    :param states: The states to plot in shape (time_dimension, states_dimensions)
    :param satellite_indices: The indices of the satellites to plot.
    :return: Tuple with update points and None
    """
    _, _, scaling_factor = get_scaling_parameters(states[0, :3])

    for idx, point in enumerate(points):
        point.set_offsets(states[t, satellite_indices[idx]:satellite_indices[idx] + 2] / scaling_factor)
        point.set_3d_properties(zs=states[t, satellite_indices[idx] + 2] / scaling_factor, zdir='z')
    return points, None


def plot_differential_drag_states(diff_drag_states: np.ndarray, timestep: float, satellite_name: str = None,
                                  figure: plt.figure = None, **kwargs) -> plt.figure:
    """
    Method to plot the relative differential drag states over time.

    :param diff_drag_states: 2D-array with the differential drag states over time with shape (t, 2).
    :param timestep: Amount of time between each state in s.
    :param satellite_name: Name to place as a label for the legend.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added differential drag states.
    """
    time_hours = get_time_axis(diff_drag_states[:, 0:1], timestep)

    fig, axes = get_figure_and_axes(figure, (1, 1))
    fig.suptitle('Evolution of relative differential drag states compared to reference.')

    is_angle_list = [True]
    y_label_list = ['relative angle [deg]']
    plot_onto_axes(diff_drag_states[:, 0:1], time_hours, list(axes), is_angle_list, y_label_list, satellite_name, **kwargs)

    return fig

