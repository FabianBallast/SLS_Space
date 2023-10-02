import matplotlib.pyplot as plt
import os
import pickle


def plot_data(plot_name: str) -> None:
    """
    Plot data for a given plot name.
    :param plot_name: The name of the plot (with corresponding data names)
    """
    fig_name_list = ['main_states', 'side_states', 'inputs']
    fig_list = [None] * len(fig_name_list)

    for file in os.listdir("../Data"):
        if file.startswith(plot_name):
            with open(os.path.join("../Data", file), 'rb') as f:
                orbital_sim = pickle.load(f)
                fig_list[0] = orbital_sim.plot_main_states(figure=fig_list[0])
                fig_list[1] = orbital_sim.plot_side_states(figure=fig_list[1])
                fig_list[2] = orbital_sim.plot_inputs(figure=fig_list[2])

    for idx, fig in enumerate(fig_list):
        fig.savefig('../Figures/' + plot_name + '_' + fig_name_list[idx] + '.eps')


def plot_all() -> None:
    """
    Plot and save all plots.
    """
    for file in os.listdir("../Figures"):
        if file.endswith('_inputs.eps'):
            plot_data(file.removesuffix('_inputs.eps'))


if __name__ == '__main__':
    # plot_data('basic_MPC')
    plot_all()
    plt.show()
