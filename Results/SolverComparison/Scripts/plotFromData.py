import matplotlib.pyplot as plt
import os
import pickle

# legend_dict = {1: "Matrix",
#                2: "Vector",
#                3: "Sparse",
#                0: "Toolbox"}
# legend_dict = {0: "Gurobi",
#                1: "OSQP",
#                2: "cuOSQP",
#                3: "Toolbox"}
legend_dict = {1: "Reformulated",
               2: "Nonlinear",
               3: "cuOSQP",
               0: "Toolbox"}

style_dict = {0: {'color': 'red', 'linestyle': 'dashdot'},
              1: {'color': 'blue', 'linestyle': 'solid'},
              3: {'color': 'green', 'linestyle': 'dashed'},
              2: {'color': 'orange', 'linestyle': 'dotted'}}
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

    fig = plt.figure(figsize=(8, 4))

    for idx, sat_arr in enumerate(solver_satellites):
        plt.loglog(sat_arr, solver_times[idx], label=solver_names[idx])

    plt.legend(fontsize=12)
    plt.xlabel(r'$\mathrm{Number \;of \;satellites \;[-]}$', fontsize=12)
    plt.ylabel(r'$\mathrm{Computation \;Time \;[s]}$', fontsize=12)
    plt.grid(True)
    fig.savefig('../Figures/' + plot_name + '.eps')
    plt.tight_layout()


def plot_data_solver(data_names: list[str], plot_save_name: str) -> None:
    """
    Plot data for a given plot name.
    :param plot_name: The name of the plot (with corresponding data names)
    """
    solver_names = []
    solver_satellites = []
    solver_times = []

    for plot_name in data_names:
        for file in os.listdir("../Data"):
            if file.startswith(plot_name):
                with open(os.path.join("../Data", file), 'rb') as f:
                    solver_satellites.append(pickle.load(f))
                    solver_times.append(pickle.load(f))
                    solver_names.append(pickle.load(f))

    fig = plt.figure(figsize=(8, 4))

    for idx, sat_arr in enumerate(solver_satellites):
        plt.loglog(sat_arr, solver_times[idx], label=legend_dict[idx], **style_dict[idx])

    plt.legend(fontsize=12)
    plt.xlabel(r'$\mathrm{Number \;of \;satellites \;[-]}$', fontsize=12)
    plt.ylabel(r'$\mathrm{Computation \;Time \;[s]}$', fontsize=12)
    plt.grid(True)
    fig.savefig('../Figures/' + plot_save_name + '.eps')
    plt.tight_layout()


def plot_data_presentation(data_names: list[str]) -> None:
    """
    Plot data for a given plot name.
    """
    solver_names = []
    solver_satellites = []
    solver_times = []

    for plot_name in data_names:
        for file in os.listdir("../Data"):
            if file.startswith(plot_name) and not file.endswith("Exact"):
                with open(os.path.join("../Data", file), 'rb') as f:
                    solver_satellites.append(pickle.load(f))
                    solver_times.append(pickle.load(f))
                    solver_names.append(pickle.load(f))

    fig = plt.figure(figsize=(16, 4))

    for idx, sat_arr in enumerate(solver_satellites):
        plt.loglog(sat_arr, solver_times[idx], label=legend_dict[idx], **style_dict[idx])

    plt.legend(fontsize=12)
    plt.xlabel(r'$\mathrm{Number \;of \;satellites \;[-]}$', fontsize=12)
    plt.ylabel(r'$\mathrm{Computation \;Time \;[s]}$', fontsize=12)
    plt.grid(True)
    # fig.savefig('../Figures/' + plot_save_name + '.eps')
    plt.tight_layout()


def plot_all() -> None:
    """
    Plot and save all plots.
    """
    for file in os.listdir("../Figures"):
        plot_data(file.removesuffix('.eps'))


if __name__ == '__main__':
    # plot_data_solver(['basic_SLS_Gurobi', 'transformed_SLS_Gurobi', 'sparse_SLS_Gurobi'], 'Gurobi_comp')
    # plot_data_solver(['sparse_SLS_Gurobi', 'sparse_SLS_OSQP', 'sparse_SLS_cuOSQP', 'basic_SLS_SLS'], 'solver_comp')
    plot_data_presentation(['robust_SLS_Toolbox', 'robust_SLS_Gurobi'])
    # plot_data('robust_SLS')
    # plot_data_presentation(['basic_SLS_SLS', 'basic_SLS_Gurobi', 'transformed_SLS_Gurobi', 'sparse_SLS_Gurobi'])
    # plot_all()
    plt.show()
