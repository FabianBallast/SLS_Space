from __future__ import annotations
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.cbook import silent_list
import mpl_toolkits.mplot3d.art3d as p3
import numpy as np

mpl.rcParams["mathtext.fontset"] = 'cm'  # Better font for LaTex
# mpl.rcParams["axes.labelsize"] = 'xx-large'  # Better font for LaTex


def get_time_axis(states_over_time: np.ndarray, timestep: float, required_scale: float = 60) -> np.ndarray:
    """
    Create the time axis used for plots.

    :param states_over_time: The states that have to be plotted of shape (t, any).
    :param timestep: Timestep between each state in s.
    :param required_scale: Timescale to which to convert. Default is hours (3600 s).
    :return: Array of shape (t, ) with the time in the required scale.
    """
    return np.arange(0, states_over_time.shape[0]) * timestep / required_scale


def get_figure_and_axes(figure: plt.figure, shape_of_plots: tuple, sharex: bool = False) -> (plt.figure, tuple):
    """
    Get the figure with its axes given a shape.

    :param figure: Figure that is possibly already present. Use None to create a new one.
    :param shape_of_plots: Shape of the required plot if no figure was provided.
    :param sharex: Share the x-axis.
    :return: Tuple with figure and a tuple of axes.
    """
    if figure is None:
        fig, _ = plt.subplots(shape_of_plots[0], shape_of_plots[1], figsize=(16, shape_of_plots[0] * 3), sharex=sharex)
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


def create_full_3d_overview(max_distance: int, unit_label: str = 'km') -> tuple[plt.figure, plt.axes]:
    """
    Basis for a 3D plot with the Earth.

    :param max_distance: Maximum distance from satellite to Earth in km for scaling the plot.
    :param unit_label: The unit to use in the label. Defaults to km.
    :return: Tuple with figure and corresponding axes.
    """
    # Define a 3D figure using pyplot
    figure = plt.figure(figsize=(18, 5.5), constrained_layout=True)
    ax1 = figure.add_subplot(131, projection='3d')
    # ax1.set_title(f'Satellite trajectories around Earth')
    ax1.set_xlabel(f'x [{unit_label}]')
    ax1.set_ylabel(f'y [{unit_label}]')
    ax1.set_zlabel(f'z [{unit_label}]')
    # ax1.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue')
    ax1.set_xlim(-max_distance, max_distance)
    ax1.set_ylim(-max_distance, max_distance)
    ax1.set_zlim(-max_distance, max_distance)

    ax2 = figure.add_subplot(132, projection='3d')
    # ax2.set_title(f'Satellite trajectories around Earth')
    ax2.set_xlabel(f'x [{unit_label}]')
    ax2.set_ylabel(f'y [{unit_label}]')
    # ax2.set_zlabel(f'z [{unit_label}]')
    # ax2.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue')
    ax2.tick_params(axis='z', labelleft=False, bottom=False, top=False, left=False, right=False)
    ax2.set_xlim(-max_distance, max_distance)
    ax2.set_ylim(-max_distance, max_distance)
    ax2.set_zlim(-max_distance, max_distance)
    ax2.view_init(90, -90, 0)

    ax3 = figure.add_subplot(133, projection='3d')
    # ax3.set_title(f'Satellite trajectories around Earth')
    ax3.set_xlabel(f'x [{unit_label}]')
    # ax3.set_ylabel(f'y [{unit_label}]')
    ax3.tick_params(axis='y', labelleft=False, bottom=False, top=False, left=False, right=False)
    ax3.set_zlabel(f'z [{unit_label}]')
    # ax3.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue')
    ax3.set_xlim(-max_distance, max_distance)
    ax3.set_ylim(-max_distance, max_distance)
    ax3.set_zlim(-max_distance, max_distance)
    ax3.view_init(0, -90, 0)

    return figure, [ax1, ax2, ax3]


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

    plt.tight_layout()
    # Add the legend
    # ax.legend()

    return figure


def plot_3d_trajectory_complete(states: np.ndarray, state_label_name: str, figure: plt.figure = None, **kwargs) -> plt.figure:
    """
    Plot the trajectory of a satellite in 3D.

    :param states: States of the satellite in the shape (t, 3) in m.
    :param state_label_name: Label for the legend for these states.
    :param figure: Figure to plot the trajectories in. If not provided, a new one is created.
    :return: Figure with the added trajectory.
    """
    max_distance, units, scaling_factor = get_scaling_parameters(states[0])

    if figure is None:
        figure, ax_list = create_full_3d_overview(max_distance=max_distance, unit_label=units)
    else:
        ax_list = list(figure.get_axes())

    # Plot the positional state history
    for ax in ax_list:
        ax.plot(states[:, 0] / scaling_factor, states[:, 1] / scaling_factor,
                states[:, 2] / scaling_factor, label=state_label_name, **kwargs)

        # set_axes_equal(ax)

        if state_label_name is not None:
            ax.legend()
            state_label_name = None

    # Add the legend
    # ax_list[0].legend()

    return figure


def plot_3d_position(position: np.ndarray, state_label_name: str, figure: plt.figure = None, **kwargs) -> plt.figure:
    """
    Plot the position of a satellite in 3D.

    :param position: Position of the satellite in the shape (3,) in m.
    :param state_label_name: Label for the legend for these states.
    :param figure: Figure to plot the trajectories in. If not provided, a new one is created.
    :return: Figure with the added trajectory.
    """
    max_distance, units, scaling_factor = get_scaling_parameters(position[0])

    if figure is None:
        figure, ax = create_3d_plot(max_distance=max_distance, unit_label=units)
    else:
        ax = figure.get_axes()[0]

    # Plot the positional state history
    ax.scatter(position[:, 0] / scaling_factor, position[:, 1] / scaling_factor, position[:, 2] / scaling_factor,
               label=state_label_name, marker='o', **kwargs)

    # Add the legend
    ax.legend()

    return figure


def plot_onto_axes(states: np.ndarray, time: np.ndarray, axes_list: list[plt.axes], is_angle: list[bool],
                   y_label_names: list[str], legend_names: list[str | None], unwrap_angles: bool = True,
                   states2plot: list[int] = None, xlabel_plot: list[int] = None, **kwargs) -> None:
    """
    Plot states on the provided axes with a given label and legend label name.
    :param states: The states (y-component) to be plotted.
    :param time: The time (x-component) to be plotted.
    :param axes_list: List with all the axes to plot onto.
    :param is_angle: List whether a certain value is an angle (and thus should be unwrapped/converted to deg)
    :param y_label_names: List with names of the y-axes.
    :param legend_names: Name for the legend.
    :param unwrap_angles: Whether to unwrap angles. Default is True.
    :param states2plot: Indices of states to plot.
    :param kwargs: Kwargs for plotting purposes.
    :return: Nothing.
    """
    for idx, axes in enumerate(axes_list):
        state_idx = states2plot[idx]
        state = states[:, state_idx]

        # If it is an angle, unwrap and convert to deg
        if is_angle[state_idx] and unwrap_angles:
            state = np.rad2deg(np.unwrap(state, axis=0))
        elif is_angle[state_idx]:
            state = np.rad2deg(state)

        # Plot data
        axes.plot(time, state, label=legend_names[state_idx], **kwargs)
        if xlabel_plot is None or idx in xlabel_plot:
            axes.set_xlabel(r'$\mathrm{Time\;[min]}$', fontsize=14)
        axes.set_ylabel(y_label_names[state_idx], fontsize=14)
        axes.set_xlim([min(time), max(time)])
        axes.grid(True)

        if legend_names[state_idx] is not None:
            axes.legend(fontsize=12, loc='upper right')

    plt.tight_layout()


def plot_keplerian_states(kepler_elements: np.ndarray, timestep: float, plot_argument_of_latitude: bool = False,
                          legend_name: str = None, figure: plt.figure = None, states2plot: list = None,
                          **kwargs) -> plt.figure:
    """
    Method to plot the evolution of the orbital (kepler) elements over time.

    :param kepler_elements: 2D-array containing the orbital elements over time. Shape is (t, 6).
    :param timestep: Amount of time between each measurement in s.
    :param plot_argument_of_latitude: Plot the argument of latitude or the true anomaly.
    :param legend_name: Name to place as a label for the legend.
    :param figure: Figure to plot the kepler elements into. If not provided, a new one is created.
    :param states2plot: List with which states to plot.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added keplerian states.
    """
    # Plot Kepler elements as a function of time
    time_hours = get_time_axis(kepler_elements, timestep)

    if states2plot is None:
        fig, axes = get_figure_and_axes(figure, (3, 2))
        states2plot = [0, 1, 2, 3, 4, 5]
    else:
        fig, axes = get_figure_and_axes(figure, (1, len(states2plot)))
    fig.suptitle('Evolution of Kepler elements over the course of the propagation')

    is_angle_list = [False, False, True, True, True, True]
    y_label_list = ['Semi-major axis [m]', 'Eccentricity [-]', 'Inclination [deg]',
                   'Argument of Periapsis [deg]', 'RAAN [deg]', "True anomaly [deg]"]

    if plot_argument_of_latitude:
        kepler_elements[:, 5] += kepler_elements[:, 3]
        y_label_list[5] = "Argument of latitude [deg]"

    legend_names = [None] + [legend_name] + [None] * 4

    plot_onto_axes(kepler_elements, time_hours, list(axes), is_angle_list, y_label_list, legend_names=legend_names,
                   states2plot=states2plot, **kwargs)

    return fig


def plot_kalman_states(kalman_elements: np.ndarray, timestep: float,
                       legend_name: str = None, figure: plt.figure = None, states2plot: list = None,
                       **kwargs) -> plt.figure:
    """
    Method to plot the evolution of the orbital (kepler) elements over time.

    :param kalman_elements: 2D-array containing the orbital elements over time. Shape is (t, 6).
    :param timestep: Amount of time between each measurement in s.
    :param legend_name: Name to place as a label for the legend.
    :param figure: Figure to plot the kepler elements into. If not provided, a new one is created.
    :param states2plot: List with which states to plot.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added keplerian states.
    """
    # Plot Kepler elements as a function of time
    time_hours = get_time_axis(kalman_elements, timestep)

    if states2plot is None:
        fig, axes = get_figure_and_axes(figure, (3, 2))
        states2plot = [0, 1, 2, 3, 4, 5]
    else:
        fig, axes = get_figure_and_axes(figure, (1, len(states2plot)))
    fig.suptitle('Evolution of Kalman elements over the course of the propagation')

    is_angle_list = [False, True, False, False, True, True]
    y_label_list = ['Semi-major axis [m]', 'True latitude [deg]', 'e cos(omega) [-]',
                   'e sin(omega) [-]', 'Inclination [deg]', "RAAN [deg]"]

    legend_names = [None] + [legend_name] + [None] * 4

    plot_onto_axes(kalman_elements, time_hours, list(axes), is_angle_list, y_label_list, legend_names=legend_names,
                   states2plot=states2plot, **kwargs)

    return fig

def plot_thrust_forces(inputs: np.ndarray, timestep: float, legend_name: str = None,
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
    fig.suptitle(r'$\mathrm{Control \;inputs \;over \;the \;course \;of \;the \;propagation}$', fontsize=16)

    is_angle_list = [False] * 4
    y_label_list = [r'$\mathrm{u_\rho\;[N]}$', r'$\mathrm{u_\theta\;[N]}$', r'$\mathrm{u_z\;[N]}$', r"$\mathrm{||u||_2\;[N]}$"]
    inputs_and_norm = np.concatenate((inputs, np.linalg.norm(inputs, axis=1).reshape(-1, 1)), axis=1)
    legend_names = [legend_name] + 3 * [None]

    plot_onto_axes(inputs_and_norm, time_hours, list(axes), is_angle_list, y_label_list, legend_names=legend_names,
                   states2plot=[0, 1, 2, 3], **kwargs)

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


def plot_cylindrical_states(cylindrical_states: np.ndarray, timestep: float, legend_name: str = '',
                            figure: plt.figure = None, states2plot: list[int] = None, **kwargs) -> plt.figure:
    """
    Method to plot the relative cylindrical states over time.

    :param cylindrical_states: 2D-array with the cylindrical states over time with shape (t, 6).
    :param timestep: Amount of time between each state in s.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param legend_name: The name to pass to the legend
    :param states2plot: Which states to plot.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added cylindrical states.
    """
    time_hours = get_time_axis(cylindrical_states, timestep)

    if states2plot is None:
        fig, axes = get_figure_and_axes(figure, (3, 2))
        states2plot = [0, 1, 2, 3, 4, 5]
    else:
        fig, axes = get_figure_and_axes(figure, (1, len(states2plot)))

        if len(states2plot) != len(axes):
            axes = [axes[states2plot[0]]]

    # fig.suptitle(r'$\mathrm{Evolution\; of\; relative\; cylindrical\; states\; compared\; to\; reference}$', fontsize=16)

    is_angle_list = [False, True, False, False, True, False]
    y_label_list = [r'$\mathrm{\Delta r \; [m]}$', r'$\mathrm{\Delta \varphi \;[deg]}$', r'$\mathrm{\Delta z \;[m]}$',
                    r"$\mathrm{\Delta \dot{r}\; [m/s]}$", r'$\mathrm{\Delta \dot{\varphi}\;[deg/s]}$', r'$\mathrm{\Delta \dot{z}\;[m/s]}$']

    legend_names = [None] + [legend_name] + [None] * 4
    plot_onto_axes(cylindrical_states, time_hours, list(axes), is_angle_list,
                   y_label_list, legend_names=legend_names, states2plot=states2plot, xlabel_plot=[4, 5], **kwargs)

    return fig


def plot_cylindrical_radius_height(cylindrical_states: np.ndarray, timestep: float, legend_name: str = '',
                                  figure: plt.figure = None, states2plot: list[int] = None, **kwargs) -> plt.figure:
    """
    Method to plot the relative cylindrical states over time.

    :param cylindrical_states: 2D-array with the cylindrical states over time with shape (t, 6).
    :param timestep: Amount of time between each state in s.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param legend_name: The name to pass to the legend
    :param states2plot: Which states to plot.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added cylindrical states.
    """
    time_hours = get_time_axis(cylindrical_states, timestep)

    if states2plot is None:
        fig, axes = get_figure_and_axes(figure, (2, 1), sharex=True)
        states2plot = [0, 1]
    else:
        fig, axes = get_figure_and_axes(figure, (1, len(states2plot)))

        if len(states2plot) != len(axes):
            axes = [axes[states2plot[0]]]

    # fig.suptitle(r'$\mathrm{Evolution\; of\; relative\; cylindrical\; states\; compared\; to\; reference}$', fontsize=16)

    is_angle_list = [False, False]
    y_label_list = [r'$\mathrm{\Delta r \; [m]}$', r'$\mathrm{\Delta z \;[m]}$']

    legend_names = [None, legend_name]
    plot_onto_axes(cylindrical_states[:, [0,2]], time_hours, list(axes), is_angle_list,
                   y_label_list, legend_names=legend_names, states2plot=states2plot, xlabel_plot=[1], **kwargs)

    return fig


def plot_quasi_roe(quasi_roe_states: np.ndarray, timestep: float, legend_name: str = '',
                   figure: plt.figure = None, states2plot: list[int] = None, **kwargs) -> plt.figure:
    """
    Method to plot the quasi ROE states over time.

    :param quasi_roe_states: 2D-array with the quasi ROE states over time with shape (t, 6).
    :param timestep: Amount of time between each state in s.
    :param legend_name: Name to place as a label for the legend.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added quasi ROE states.
    """
    time_hours = get_time_axis(quasi_roe_states, timestep)

    if states2plot is None:
        fig, axes = get_figure_and_axes(figure, (3, 2))
        states2plot = [0, 1, 2, 3, 4, 5]
    else:
        fig, axes = get_figure_and_axes(figure, (1, len(states2plot)))

    # fig.suptitle(r'$\mathrm{Evolution\;of \;quasi \;ROE}$', fontsize=16)

    is_angle_list = [False, True, False, False, True, True]
    y_label_list = [r'$\mathrm{\delta a\;[-]}$', r'$\mathrm{\delta \lambda\;[deg]}$',
                    r'$\mathrm{\delta e_x\;[-]}$', r"$\mathrm{\delta e_y\;[-]}$",
                    r'$\mathrm{\delta i_x\;[deg]}$', r'$\mathrm{\delta i_y\;[deg]}$']

    legend_names = [None] + [legend_name] + [None] * 4
    plot_onto_axes(quasi_roe_states, time_hours, list(axes), is_angle_list,
                   y_label_list, legend_names=legend_names, states2plot=states2plot, xlabel_plot=[4, 5], **kwargs)

    return fig


def plot_roe(roe_states: np.ndarray, timestep: float, legend_name: str = '',
                   figure: plt.figure = None, states2plot: list[int] = None, **kwargs) -> plt.figure:
    """
    Method to plot the ROE states over time.

    :param roe_states: 2D-array with the quasi ROE states over time with shape (t, 6).
    :param timestep: Amount of time between each state in s.
    :param legend_name: Name to place as a label for the legend.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added ROE states.
    """
    time_hours = get_time_axis(roe_states, timestep)

    if states2plot is None:
        fig, axes = get_figure_and_axes(figure, (3, 2))
        states2plot = [0, 1, 2, 3, 4, 5]
    else:
        fig, axes = get_figure_and_axes(figure, (1, len(states2plot)))

    # fig.suptitle('Evolution of ROE.')

    is_angle_list = [False, True, False, True, True, True]
    y_label_list = ['delta a [-]', 'delta lambda [deg]',
                    'delta e_x [-]', "delta e_y [deg]",
                    'delta i_x [deg]', 'delta i_y [deg]']
    legend_names = [None] + [legend_name] + [None] * 4

    plot_onto_axes(roe_states, time_hours, list(axes), is_angle_list, y_label_list, legend_names,
                   states2plot=states2plot, xlabel_plot=[4, 5], **kwargs)

    return fig


def plot_blend(blend_states: np.ndarray, timestep: float, legend_name: str = '',
               figure: plt.figure = None, states2plot: list[int] = None, **kwargs) -> plt.figure:
    """
    Method to plot the blend states over time.

    :param blend_states: 2D-array with the quasi ROE states over time with shape (t, 6).
    :param timestep: Amount of time between each state in s.
    :param legend_name: Name to place as a label for the legend.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added blend states.
    """
    time_hours = get_time_axis(blend_states, timestep)

    if states2plot is None:
        fig, axes = get_figure_and_axes(figure, (3, 2))
        states2plot = [0, 1, 2, 3, 4, 5]
    else:
        fig, axes = get_figure_and_axes(figure, (1, len(states2plot)))

    # fig.suptitle('Evolution of states.')

    is_angle_list = [False, True, False, False, False, False]
    y_label_list = [r'$\delta r\mathrm{\;[m]}$', r'$\delta\lambda^f\mathrm{\;[deg]}$',
                    r'$\delta e_x^f \mathrm{\;[-]}$', r'$\delta e_y^f\mathrm{\;[-]}$',
                    r'$\delta \xi_x \mathrm{\;[-]}$', r'$\delta \xi_y \mathrm{\;[-]}$']
    legend_names = [None] + [legend_name] + [None] * 4

    plot_onto_axes(blend_states, time_hours, list(axes), is_angle_list, y_label_list, legend_names,
                   states2plot=states2plot, xlabel_plot=[4, 5], **kwargs)

    return fig


def plot_blend_small(blend_states: np.ndarray, timestep: float, legend_name: str = '',
               figure: plt.figure = None, states2plot: list[int] = None, **kwargs) -> plt.figure:
    """
    Method to plot the blend states over time.

    :param blend_states: 2D-array with the quasi ROE states over time with shape (t, 4).
    :param timestep: Amount of time between each state in s.
    :param legend_name: Name to place as a label for the legend.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added blend states.
    """
    time_hours = get_time_axis(blend_states, timestep)

    if states2plot is None:
        fig, axes = get_figure_and_axes(figure, (3, 2))
        states2plot = [0, 1, 2, 3, 4, 5]
    else:
        fig, axes = get_figure_and_axes(figure, (1, len(states2plot)))

    fig.suptitle('Evolution of states.')

    is_angle_list = [False, True, False, False, True, True]
    y_label_list = [r'$\delta\mathrm{r\;[-]}$', r'$\delta\theta\mathrm{\;[deg]}$', r'$\delta e cos f\mathrm{\;[-]}$',
                    r'$\delta e sin f\mathrm{\;[-]}$', r'$\delta\mathrm{i\;[deg]}$', r'$\delta\Omega\mathrm{\;[deg]}$']
    legend_names = [None] + [legend_name] + [None] * 4

    plot_onto_axes(blend_states, time_hours, list(axes), is_angle_list, y_label_list, legend_names,
                   states2plot=states2plot, **kwargs)

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
    y_label_list = [r'$q_1\;[-]$', r'$q_2\;[-]$', r'$q_3\;[-]$', r'$q_4\;[-]$']
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


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

