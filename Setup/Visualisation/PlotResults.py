from Visualisation.Plotting import get_time_axis, get_figure_and_axes
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.cbook import silent_list
import mpl_toolkits.mplot3d.art3d as p3
import numpy as np

mpl.rcParams["mathtext.fontset"] = 'cm'  # Better font for LaTex


def plot_onto_axes(states: np.ndarray, time: np.ndarray, axes_list: list[plt.axes], is_angle: list[bool],
                   y_label_names: list[str], legend_names: list[str | None], unwrap_angles: bool = True,
                   states2plot: list[int] = None, xlabel_plot: list[int] = None, y_lim: list[float] = None,
                   constraint_value: list[int] | list[float] = None, **kwargs) -> None:
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
    :param xlabel_plot: Which axes to plot the x label on.
    :param y_lim: Enforce a limit for the y-axis. If none, automode of the plotter.
    :param kwargs: Kwargs for plotting purposes.
    :return: Nothing.
    """
    for idx, axes in enumerate(axes_list):
        state_idx = states2plot[idx]
        if constraint_value is None or constraint_value[idx] is None:
            state = states[:, state_idx]
        else:
            state = states
            axes.plot([min(time), max(time)], [constraint_value, constraint_value], 'r--', label='Constraint')
            axes.legend(fontsize=12, loc='upper right')

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

        if y_lim is not None:
            axes.set_ylim(y_lim)

        axes.grid(True)

        if legend_names[state_idx] is not None:
            axes.legend(fontsize=12, loc='upper right')



    plt.tight_layout()


def plot_main_states_report(states: np.ndarray, timestep: float, legend_name: str = None,
                            figure: plt.figure = None, states2plot: list[int] = None,
                            plot_radial_constraint: bool = False, **kwargs) -> plt.figure:
    """
    Method to plot the main states over time.

    :param states: 2D-array with the quasi ROE states over time with shape (t, 3).
    :param timestep: Amount of time between each state in s.
    :param legend_name: Name to place as a label for the legend.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added blend states.
    """
    time_hours = get_time_axis(states, timestep)

    if states2plot is None:
        fig, axes = get_figure_and_axes(figure, (3, 1), sharex=True)
        states2plot = [0, 1, 2]
    else:
        fig, axes = get_figure_and_axes(figure, (1, len(states2plot)))

    # fig.suptitle('Evolution of states.')

    is_angle_list = [False, True, True]
    y_label_list = [r'$\delta r\mathrm{\;[m]}$', r'$\delta\theta\mathrm{\;[deg]}$', r'$\delta \Omega \mathrm{\;[deg]}$']
    legend_names = [legend_name] + [None] * 2
    # print(legend_names)
    constraint_limits = [None] * 3
    if plot_radial_constraint:
        constraint_limits[0] = 0.1
    plot_onto_axes(states, time_hours, list(axes), is_angle_list, y_label_list, legend_names,
                   states2plot=states2plot, constraint_value=constraint_limits, xlabel_plot=[2], **kwargs)

    return fig


def plot_main_states_theta_Omega_report(states: np.ndarray, timestep: float, legend_name: str = None,
                            figure: plt.figure = None, states2plot: list[int] = None,
                            plot_radial_constraint: bool = False, **kwargs) -> plt.figure:
    """
    Method to plot the main states over time.

    :param states: 2D-array with the quasi ROE states over time with shape (t, 3).
    :param timestep: Amount of time between each state in s.
    :param legend_name: Name to place as a label for the legend.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added blend states.
    """
    time_hours = get_time_axis(states, timestep)

    if states2plot is None:
        fig, axes = get_figure_and_axes(figure, (2, 1), sharex=True)
        states2plot = [0, 1]
    else:
        fig, axes = get_figure_and_axes(figure, (1, len(states2plot)))

    # fig.suptitle('Evolution of states.')

    is_angle_list = [True, True]
    y_label_list = [r'$\delta\theta\mathrm{\;[deg]}$', r'$\delta \Omega \mathrm{\;[deg]}$']
    legend_names = [legend_name] + [None] * 1
    # print(legend_names)
    constraint_limits = [None] * 2
    if plot_radial_constraint:
        constraint_limits[0] = 0.1
    plot_onto_axes(states, time_hours, list(axes), is_angle_list, y_label_list, legend_names,
                   states2plot=states2plot, constraint_value=constraint_limits, xlabel_plot=[1], **kwargs)

    return fig

def plot_radius_theta_report(states: np.ndarray, timestep: float, legend_name: str = None,
                             figure: plt.figure = None, states2plot: list[int] = None, **kwargs) -> plt.figure:
    """
    Method to plot the main states over time.

    :param states: 2D-array with the quasi ROE states over time with shape (t, 3).
    :param timestep: Amount of time between each state in s.
    :param legend_name: Name to place as a label for the legend.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added blend states.
    """
    time_hours = get_time_axis(states, timestep)

    if states2plot is None:
        fig, axes = get_figure_and_axes(figure, (2, 1), sharex=True)
        states2plot = [0, 1]
    else:
        fig, axes = get_figure_and_axes(figure, (1, len(states2plot)))

    # fig.suptitle('Evolution of states.')

    is_angle_list = [False, True, True]
    y_label_list = [r'$\delta r\mathrm{\;[m]}$', r'$\delta\theta\mathrm{\;[deg]}$']
    legend_names = [legend_name] + [None] * 1
    # print(legend_names)
    plot_onto_axes(states, time_hours, list(axes), is_angle_list, y_label_list, legend_names,
                   states2plot=states2plot, xlabel_plot=[1], **kwargs)

    return fig


def plot_side_states_report(states: np.ndarray, timestep: float, legend_name: str = None,
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
        fig, axes = get_figure_and_axes(figure, (2, 1), sharex=True)
        states2plot = [0, 1]
    else:
        fig, axes = get_figure_and_axes(figure, (1, len(states2plot)))

    # fig.suptitle('Evolution of states.')

    is_angle_list = [False, True]
    y_label_list = [r'$\delta e\mathrm{\;[-]}$', r'$\delta i\mathrm{\;[deg]}$']
    legend_names = [legend_name] + [None]

    plot_onto_axes(states, time_hours, list(axes), is_angle_list, y_label_list, legend_names,
                   states2plot=states2plot, xlabel_plot=[1], **kwargs)

    return fig


def plot_inputs_report(inputs: np.ndarray, timestep: float, legend_name: str = None,
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
        fig, axes = get_figure_and_axes(figure, (3, 1), sharex=True)
        states2plot = [0, 1, 2]
    else:
        fig, axes = get_figure_and_axes(figure, (1, len(states2plot)))

    # fig.suptitle('Control inputs')

    is_angle_list = [False, False, False]
    y_label_list = [r'$u_r \mathrm{\;[N]}$', r'$u_t\mathrm{\;[N]}$', r'$u_n \mathrm{\;[N]}$']
    legend_names = [legend_name] + [None] * 2

    plot_onto_axes(inputs, time_hours, list(axes), is_angle_list, y_label_list, legend_names,
                   states2plot=states2plot, xlabel_plot=[2], **kwargs)

    return fig

def plot_planar_inputs_report(inputs: np.ndarray, timestep: float, legend_name: str = None,
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
        fig, axes = get_figure_and_axes(figure, (2, 1), sharex=True)
        states2plot = [0, 1]
    else:
        fig, axes = get_figure_and_axes(figure, (1, len(states2plot)))

    # fig.suptitle('Control inputs')

    is_angle_list = [False, False]
    y_label_list = [r'$u_r \mathrm{\;[N]}$', r'$u_t\mathrm{\;[N]}$']
    legend_names = [legend_name] + [None] * 1

    plot_onto_axes(inputs, time_hours, list(axes), is_angle_list, y_label_list, legend_names,
                   states2plot=states2plot, xlabel_plot=[1], **kwargs)

    return fig


def plot_radial_inputs_report(inputs: np.ndarray, timestep: float, legend_name: str = None,
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
        fig, axes = get_figure_and_axes(figure, (1, 1), sharex=True)
        states2plot = [0]
    else:
        fig, axes = get_figure_and_axes(figure, (1, len(states2plot)))

    # fig.suptitle('Control inputs')

    is_angle_list = [False]
    y_label_list = [r'$u_r \mathrm{\;[N]}$']
    legend_names = [legend_name]

    plot_onto_axes(inputs, time_hours, list(axes), is_angle_list, y_label_list, legend_names,
                   states2plot=states2plot, xlabel_plot=[0], **kwargs)

    return fig

def plot_radius_report(states: np.ndarray, timestep: float, legend_name: str = None,
                       figure: plt.figure = None, states2plot: list[int] = None, y_lim=None, **kwargs) -> plt.figure:
    """
    Method to plot the radius over time.

    :param states: 2D-array with the quasi ROE states over time with shape (t, 3).
    :param timestep: Amount of time between each state in s.
    :param legend_name: Name to place as a label for the legend.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added blend states.
    """
    time_hours = get_time_axis(states, timestep)

    if states2plot is None:
        fig, axes = get_figure_and_axes(figure, (1, 1), sharex=True)
        states2plot = [0]
    else:
        fig, axes = get_figure_and_axes(figure, (1, len(states2plot)))

    # fig.suptitle('Evolution of states.')

    is_angle_list = [False]
    y_label_list = [r'$\delta r\mathrm{\;[m]}$']
    legend_names = [legend_name] + [None] * 2
    # print(legend_names)
    plot_onto_axes(states, time_hours, list(axes), is_angle_list, y_label_list, legend_names,
                   states2plot=states2plot, xlabel_plot=[0], y_lim=y_lim, **kwargs)

    return fig


def plot_ex(states: np.ndarray, timestep: float, legend_name: str = None,
                       figure: plt.figure = None, states2plot: list[int] = None, **kwargs) -> plt.figure:
    """
    Method to plot the radius over time.

    :param states: 2D-array with the quasi ROE states over time with shape (t, 3).
    :param timestep: Amount of time between each state in s.
    :param legend_name: Name to place as a label for the legend.
    :param figure: Figure to plot the states into. If not provided, a new one is created.
    :param kwargs: Kwargs for plotting purposes.
    :return: Figure with the added blend states.
    """
    time_hours = get_time_axis(states, timestep)

    if states2plot is None:
        fig, axes = get_figure_and_axes(figure, (1, 1), sharex=True)
        states2plot = [0]
    else:
        fig, axes = get_figure_and_axes(figure, (1, len(states2plot)))

    # fig.suptitle('Evolution of states.')

    is_angle_list = [False]
    y_label_list = [r'$\delta e_x^f\mathrm{\;[m]}$']
    legend_names = [legend_name] + [None] * 2
    # print(legend_names)
    plot_onto_axes(states, time_hours, list(axes), is_angle_list, y_label_list, legend_names,
                   states2plot=states2plot, xlabel_plot=[0], **kwargs)

    return fig


def plot_theta_Omega(theta: np.ndarray, theta_ref: np.ndarray, Omega: np.ndarray, Omega_ref: np.ndarray,
                     figure: plt.figure = None) -> plt.figure:
    """
    Create the theta_Omega plot.

    :return: Figure with the plot.
    """
    theta_offset = -5
    Omega_offset = 5
    theta_unwrap = np.rad2deg(np.unwrap(theta - theta_ref, axis=0)) + theta_offset
    Omega_unwrap = np.rad2deg(np.unwrap(Omega - Omega_ref, axis=0)) + Omega_offset

    theta_wrap = theta_unwrap % 360
    Omega_wrap = Omega_unwrap % 360

    theta_wrap_true = np.rad2deg(np.unwrap(theta - theta_ref, axis=0)) % 360
    theta_wrap_true -= (theta_wrap_true > 180) * 360
    Omega_wrap_true = np.rad2deg(np.unwrap(Omega - Omega_ref, axis=0)) % 360

    theta_wrap -= (theta_wrap > 180) * 360
    # Omega_wrap -= (Omega_wrap > 180) * 360
    wrapped_thetas = np.any(np.abs(theta_wrap - theta_unwrap) > 1e-3, axis=0)
    wrapped_Omegas = np.any(np.abs(Omega_wrap - Omega_unwrap) > 1e-3, axis=0)

    non_wrapped_sats = ~wrapped_thetas & ~wrapped_Omegas
    wrapped_sats = ~non_wrapped_sats

    wrapped_idx = np.arange(wrapped_sats.shape[0])[wrapped_sats]
    first_breach_theta = np.argmax(
        np.abs(theta_wrap_true[:, wrapped_sats] - theta_unwrap[:, wrapped_sats] + theta_offset) > 1e-3, axis=0)
    first_breach_Omega = np.argmax(
        np.abs(Omega_wrap_true[:, wrapped_sats] - Omega_unwrap[:, wrapped_sats] + Omega_offset) > 1e-3, axis=0)

    before_breach_data = []
    after_breach_data = []

    for breach in range(len(first_breach_theta)):
        breach_idx = np.maximum(first_breach_theta[breach], first_breach_Omega[breach], dtype=int)

        if breach_idx > 0:
            before_breach_data.append((theta_wrap_true[0:breach_idx, wrapped_idx[breach]],
                                       Omega_wrap_true[0:breach_idx, wrapped_idx[breach]]))
        after_breach_data.append((theta_wrap_true[breach_idx:, wrapped_idx[breach]],
                                  Omega_wrap_true[breach_idx:, wrapped_idx[breach]]))

    if figure is None:
        figure, _ = plt.subplots(1, 1, figsize=(16, 9))

    ax = figure.get_axes()[0]

    ax.plot(theta_wrap[:, non_wrapped_sats] - theta_offset, Omega_wrap[:, non_wrapped_sats] - Omega_offset, 'b-')
    ax.plot(theta_wrap[0] - theta_offset, Omega_wrap[0] - Omega_offset, 'o', label='Start')
    ax.plot(theta_wrap[-1] - theta_offset, Omega_wrap[-1] - Omega_offset, 's', label='End')

    for i, breached_sat in enumerate(before_breach_data):
        ax.plot(breached_sat[0], breached_sat[1], 'b-')

    for i, breached_sat in enumerate(after_breach_data):
        ax.plot(breached_sat[0], breached_sat[1], 'b-')
    ax.set_xlabel(r'$\theta \; \mathrm{[deg]}$', fontsize=14)
    ax.set_ylabel(r'$\Omega \; \mathrm{[deg]}$', fontsize=14)
    plt.legend()
    ax.set_xlim([-190, 190])
    ax.set_ylim([-10, 390])
    ax.grid()
    return figure


def plot_theta_Omega_triple(theta: np.ndarray, theta_ref: np.ndarray, Omega: np.ndarray, Omega_ref: np.ndarray,
                     figure: plt.figure = None) -> plt.figure:
    """
    Create the theta_Omega plot.

    :return: Figure with the plot.
    """
    theta_offset = -5
    Omega_offset = 5
    theta_unwrap = np.rad2deg(np.unwrap(theta - theta_ref, axis=0)) + theta_offset
    Omega_unwrap = np.rad2deg(np.unwrap(Omega - Omega_ref, axis=0)) + Omega_offset

    theta_wrap = theta_unwrap % 360
    Omega_wrap = Omega_unwrap % 360

    theta_wrap_true = np.rad2deg(np.unwrap(theta - theta_ref, axis=0)) % 360
    theta_wrap_true -= (theta_wrap_true > 180) * 360
    Omega_wrap_true = np.rad2deg(np.unwrap(Omega - Omega_ref, axis=0)) % 360

    theta_wrap -= (theta_wrap > 180) * 360
    # Omega_wrap -= (Omega_wrap > 180) * 360
    wrapped_thetas = np.any(np.abs(theta_wrap - theta_unwrap) > 1e-3, axis=0)
    wrapped_Omegas = np.any(np.abs(Omega_wrap - Omega_unwrap) > 1e-3, axis=0)

    non_wrapped_sats = ~wrapped_thetas & ~wrapped_Omegas
    wrapped_sats = ~non_wrapped_sats

    wrapped_idx = np.arange(wrapped_sats.shape[0])[wrapped_sats]
    first_breach_theta = np.argmax(
        np.abs(theta_wrap_true[:, wrapped_sats] - theta_unwrap[:, wrapped_sats] + theta_offset) > 1e-3, axis=0)
    first_breach_Omega = np.argmax(
        np.abs(Omega_wrap_true[:, wrapped_sats] - Omega_unwrap[:, wrapped_sats] + Omega_offset) > 1e-3, axis=0)

    before_breach_data = []
    after_breach_data = []

    for breach in range(len(first_breach_theta)):
        breach_idx = np.maximum(first_breach_theta[breach], first_breach_Omega[breach], dtype=int)

        if breach_idx > 0:
            before_breach_data.append((theta_wrap_true[0:breach_idx, wrapped_idx[breach]],
                                       Omega_wrap_true[0:breach_idx, wrapped_idx[breach]]))
        after_breach_data.append((theta_wrap_true[breach_idx:, wrapped_idx[breach]],
                                  Omega_wrap_true[breach_idx:, wrapped_idx[breach]]))

    if figure is None:
        figure, _ = plt.subplots(1, 3, figsize=(21, 6), sharey=True)

    ax = figure.get_axes()[0]
    ax.plot(theta_wrap[0] - theta_offset, Omega_wrap[0] - Omega_offset, 'o')
    ax.set_xlabel(r'$\theta \; \mathrm{[deg]}$', fontsize=14)
    ax.set_ylabel(r'$\Omega \; \mathrm{[deg]}$', fontsize=14)
    ax.set_xlim([-190, 190])
    ax.set_ylim([-10, 390])
    ax.grid()
    ax.title.set_text('Starting Configuration')



    ax = figure.get_axes()[1]

    ax.plot(theta_wrap[:, non_wrapped_sats] - theta_offset, Omega_wrap[:, non_wrapped_sats] - Omega_offset, 'b-')


    for i, breached_sat in enumerate(before_breach_data):
        ax.plot(breached_sat[0], breached_sat[1], 'b-')

    for i, breached_sat in enumerate(after_breach_data):
        ax.plot(breached_sat[0], breached_sat[1], 'b-')
    ax.set_xlabel(r'$\theta \; \mathrm{[deg]}$', fontsize=14)
    # ax.set_ylabel(r'$\Omega \; \mathrm{[deg]}$', fontsize=14)
    ax.plot(theta_wrap[0] - theta_offset, Omega_wrap[0] - Omega_offset, 'o', label='Start')
    ax.plot(theta_wrap[-1] - theta_offset, Omega_wrap[-1] - Omega_offset, 's', label='End')
    ax.grid()
    ax.legend(loc='upper right')
    ax.title.set_text('Movement')
    ax.set_xlim([-190, 190])

    ax = figure.get_axes()[2]
    ax.plot(theta_wrap[-1] - theta_offset, Omega_wrap[-1] - Omega_offset, 's', color='#ff7f0e')
    ax.set_xlabel(r'$\theta \; \mathrm{[deg]}$', fontsize=14)
    # ax.set_ylabel(r'$\Omega \; \mathrm{[deg]}$', fontsize=14)
    ax.title.set_text('Ending Configuration')
    ax.set_xlim([-190, 190])
    ax.grid()

    plt.tight_layout()
    return figure


def plot_in_plane_constraints(values: np.ndarray, timestep, figure: plt.figure = None, y_lim: list = None, **kwargs) -> plt.figure:
    """
    Create a plot for the in-plane constraints.

    :param values: Value of the constraints.
    :param figure: Figure to plot into.
    :return: Figure with added data.
    """
    time_hours = get_time_axis(values, timestep)
    fig, axes = get_figure_and_axes(figure, (1, 1), sharex=True)
    states2plot = [0]
    # fig.suptitle('Evolution of states.')

    is_angle_list = [False]
    y_label_list = [r'$\lambda^f_2 - \lambda^f_1\mathrm{\;[deg]}$']
    legend_names = [None]
    # print(legend_names)
    # print(np.arange(269)[np.any(values > 100, axis=0)])
    plot_onto_axes(values, time_hours, list(axes), is_angle_list, y_label_list, legend_names,
                   states2plot=states2plot, xlabel_plot=[0], constraint_value=[5], y_lim=y_lim, **kwargs)

    return fig

def plot_out_of_plane_constraints(values: np.ndarray, timestep, figure: plt.figure = None, y_lim: list = None, **kwargs) -> plt.figure:
    """
    Create a plot for the in-plane constraints.

    :param values: Value of the constraints.
    :param figure: Figure to plot into.
    :return: Figure with added data.
    """
    time_hours = get_time_axis(values, timestep)
    fig, axes = get_figure_and_axes(figure, (1, 1), sharex=True)
    states2plot = [0]
    # fig.suptitle('Evolution of states.')

    is_angle_list = [False]
    y_label_list = [r'$|r_2 - r_1| + \alpha_\mathrm{w}|\theta^f_2 - \theta^f_1|\mathrm{\;[m]}$']
    legend_names = [None]
    # print(legend_names)
    # print(np.arange(269)[np.any(values > 100, axis=0)])
    plot_onto_axes(values, time_hours, list(axes), is_angle_list, y_label_list, legend_names,
                   states2plot=states2plot, xlabel_plot=[0], constraint_value=[0.01], y_lim=y_lim, **kwargs)

    return fig
