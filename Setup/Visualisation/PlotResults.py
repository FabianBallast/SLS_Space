from Visualisation.Plotting import get_time_axis, get_figure_and_axes
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.cbook import silent_list
import mpl_toolkits.mplot3d.art3d as p3
import numpy as np

mpl.rcParams["mathtext.fontset"] = 'cm'  # Better font for LaTex


def plot_onto_axes(states: np.ndarray, time: np.ndarray, axes_list: list[plt.axes], is_angle: list[bool],
                   y_label_names: list[str], legend_names: list[str | None], unwrap_angles: bool = True,
                   states2plot: list[int] = None, **kwargs) -> None:
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
        axes.set_xlabel(r'$\mathrm{Time\;[min]}$', fontsize=14)
        axes.set_ylabel(y_label_names[state_idx], fontsize=14)
        axes.set_xlim([min(time), max(time)])
        axes.grid(True)

        # if legend_names[state_idx] is not None:
        #     axes.legend(fontsize=12)

    plt.tight_layout()


def plot_main_states_report(states: np.ndarray, timestep: float, legend_name: str = '',
                            figure: plt.figure = None, states2plot: list[int] = None, **kwargs) -> plt.figure:
    """
    Method to plot the main states over time.

    :param states: 2D-array with the quasi ROE states over time with shape (t, 6).
    :param timestep: Amount of time between each state in s.
    :param legend_name: Name to place as a label for the legend.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added blend states.
    """
    time_hours = get_time_axis(states, timestep)

    if states2plot is None:
        fig, axes = get_figure_and_axes(figure, (3, 1))
        states2plot = [0, 1, 2]
    else:
        fig, axes = get_figure_and_axes(figure, (1, len(states2plot)))

    fig.suptitle('Evolution of states.')

    is_angle_list = [False, True, True]
    y_label_list = [r'$\delta r\mathrm{\;[m]}$', r'$\delta\theta\mathrm{\;[deg]}$', r'$\delta \Omega \mathrm{\;[deg]}$']
    legend_names = [legend_name] + [None] * 2

    plot_onto_axes(states, time_hours, list(axes), is_angle_list, y_label_list, legend_names,
                   states2plot=states2plot, **kwargs)

    return fig


def plot_side_states_report(states: np.ndarray, timestep: float, legend_name: str = '',
                            figure: plt.figure = None, states2plot: list[int] = None, **kwargs) -> plt.figure:
    """
    Method to plot the side states over time.

    :param states: 2D-array with the quasi ROE states over time with shape (t, 6).
    :param timestep: Amount of time between each state in s.
    :param legend_name: Name to place as a label for the legend.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added blend states.
    """
    time_hours = get_time_axis(states, timestep)

    if states2plot is None:
        fig, axes = get_figure_and_axes(figure, (1, 2))
        states2plot = [0, 1]
    else:
        fig, axes = get_figure_and_axes(figure, (1, len(states2plot)))

    fig.suptitle('Evolution of states.')

    is_angle_list = [False, True]
    y_label_list = [r'$\delta e\mathrm{\;[-]}$', r'$\delta i\mathrm{\;[deg]}$']
    legend_names = [None] + [legend_name]

    plot_onto_axes(states, time_hours, list(axes), is_angle_list, y_label_list, legend_names,
                   states2plot=states2plot, **kwargs)

    return fig


def plot_inputs_report(inputs: np.ndarray, timestep: float, legend_name: str = '',
                       figure: plt.figure = None, states2plot: list[int] = None, **kwargs) -> plt.figure:
    """
    Method to plot the inputs over time.

    :param inputs: 2D-array with the quasi ROE states over time with shape (t, 6).
    :param timestep: Amount of time between each state in s.
    :param legend_name: Name to place as a label for the legend.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added blend states.
    """
    time_hours = get_time_axis(inputs, timestep)

    if states2plot is None:
        fig, axes = get_figure_and_axes(figure, (3, 1))
        states2plot = [0, 1, 2]
    else:
        fig, axes = get_figure_and_axes(figure, (1, len(states2plot)))

    fig.suptitle('Control inputs')

    is_angle_list = [False, False, False]
    y_label_list = [r'$u_r \mathrm{\;[N]}$', r'$u_t\mathrm{\;[N]}$', r'$u_n \mathrm{\;[N]}$']
    legend_names = [legend_name] + [None] * 2

    plot_onto_axes(inputs, time_hours, list(axes), is_angle_list, y_label_list, legend_names,
                   states2plot=states2plot, **kwargs)

    return fig