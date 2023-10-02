import matplotlib.pyplot as plt
import os
import pickle


def plot_data(plot_name: str) -> None:
    """
    Plot data for a given plot name.
    :param plot_name: The name of the plot (with corresponding data names)
    """
    solver_names = []
    solver_satellites = []
    solver_times = []

    for file in os.listdir("../Data"):
        if file.startswith(plot_name):
            with open(os.path.join("../Data", file), 'rb') as f:
                solver_satellites.append(pickle.load(f))
                solver_times.append(pickle.load(f))
                solver_names.append(pickle.load(f))

    fig = plt.figure()

    for idx, sat_arr in enumerate(solver_satellites):
        plt.loglog(sat_arr, solver_times[idx], label=solver_names[idx])

    plt.legend(fontsize=14)
    plt.xlabel(r'$\mathrm{Number \;of \;satellites \;[-]}$', fontsize=14)
    plt.ylabel(r'$\mathrm{Computation \;Time \;[s]}$', fontsize=14)
    plt.grid(True)
    fig.savefig('../Figures/' + plot_name + '.eps')


def plot_all() -> None:
    """
    Plot and save all plots.
    """
    for file in os.listdir("../Figures"):
        plot_data(file.removesuffix('.eps'))


if __name__ == '__main__':
    # plot_data('basic_MPC')
    plot_all()
    plt.show()
