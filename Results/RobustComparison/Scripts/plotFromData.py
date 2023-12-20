import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

linestyle_dict = {'NO': {'color': 'red', 'linestyle': 'solid', 'marker': 'None'},
                  "SIMPLE": {'color': 'green', 'linestyle': 'dotted', 'marker': 'None'},
                  'ADVANCED': {'color': 'blue', 'linestyle': 'dashed', 'marker': 'None'},
                  'FAST': {'color': 'blue', 'linestyle': 'dashed', 'marker': 'None'},
                  'EXACT': {'color': 'red', 'linestyle': 'solid', 'marker': 'None'}}

satellite_dict = {'robustness': [0],
                  'robustness_noise': [0]}

plot_duration = {'robustness': 20,
                 'robustness_noise': 20}

legend_dict = {'NO': 'Nominal',
               'SIMPLE': 'Robust Old',
               'ADVANCED': "Robust New",
               'FAST': 'QP formulation',
               'EXACT': 'Exact formulation'}


def plot_data_comparison(plot_name: str) -> None:
    """
    Plot data for a given plot name.
    :param plot_name: The name of the plot (with corresponding data names)
    """
    fig_search_list = ['NO', 'SIMPLE', 'ADVANCED']
    fig_name_list = ['main_states', 'side_states', 'inputs', 'radius', 'state_comparison']
    fig_list = [None] * len(fig_name_list)

    for fig_search in fig_search_list:
        for file in os.listdir("../Data"):
            if file.startswith(plot_name + "_" + fig_search):
                method = file.removeprefix(plot_name + "_")
                end = file.removeprefix(plot_name)
                type = file.removesuffix(end).removesuffix('_kepler').removesuffix('_j2')
                with open(os.path.join("../Data", file), 'rb') as f:
                    orbital_sim = pickle.load(f)

                    # print(method)
                    # print(orbital_sim.find_metric_values())
                    fig_list[0] = orbital_sim.plot_main_states(figure=fig_list[0], satellite_indices=satellite_dict[type],
                                                               **linestyle_dict[method], legend_name=legend_dict[method],
                                                               plot_duration=plot_duration[type])
                    fig_list[1] = orbital_sim.plot_side_states(figure=fig_list[1], satellite_indices=satellite_dict[type],
                                                               **linestyle_dict[method], legend_name=legend_dict[method],
                                                               plot_duration=plot_duration[type])
                    # if method == 'HCW':
                    #     continue
                    fig_list[2] = orbital_sim.plot_inputs(figure=fig_list[2], satellite_indices=satellite_dict[type],
                                                          **linestyle_dict[method], legend_name=legend_dict[method],
                                                          plot_duration=plot_duration[type])

                    if fig_search != "SIMPLE":
                        fig_list[3] = orbital_sim.plot_radius_zoomed(figure=fig_list[3], satellite_indices=satellite_dict[type],
                                                                     **linestyle_dict[method], legend_name=legend_dict[method],
                                                                     plot_duration=plot_duration[type])

                    fig_list[4] = orbital_sim.plot_ex(figure=fig_list[4], satellite_indices=satellite_dict[type],
                                                                 **linestyle_dict[method], legend_name=legend_dict[method],
                                                                 plot_duration=plot_duration[type])

    for idx, fig in enumerate(fig_list):
        fig.savefig('../Figures/' + plot_name + '_' + fig_name_list[idx] + '.eps')


def plot_individual_results(plot_name: str) -> None:
    """
    Plot the individual results of the simulation for the appendix.

    :param plot_name: The name of the plot (with corresponding data names).
    """
    fig = None

    for file in os.listdir("../Data"):
        if file.startswith(plot_name):
            method = file.removeprefix(plot_name + "_")
            with open(os.path.join("../Data", file), 'rb') as f:
                orbital_sim = pickle.load(f)
                fig = orbital_sim.plot_controller_states()
                fig.savefig(f'../Figures/{plot_name}_states_{method}.eps')

                fig = orbital_sim.plot_inputs()
                fig.savefig(f'../Figures/{plot_name}_inputs_{method}.eps')


def plot_sigma_comparison(state: int):
    """
    Plot a comparison of the sigma's over time.

    :param state: The state to plot.
    """
    for file in os.listdir("../Data"):
        if file.startswith('robustness_ADVANCED'):
            with open(os.path.join("../Data", file), 'rb') as f:
                orbital_sim = pickle.load(f)

                sigma_A = orbital_sim.sigma_A
                sigma_B = orbital_sim.sigma_B
                total_sigma = sigma_A + sigma_B

                fig = plt.figure(figsize=(16, 3))
                time = np.arange(sigma_A.shape[0]) / 60 * 10
                for t_pred in range(6):
                    plt.plot(time, sigma_A[:, t_pred, state] / total_sigma[:, t_pred, state], label=f"t={t_pred}")
                plt.xlabel(r'$\mathrm{Time\;[min]}$', fontsize=14)
                plt.ylabel(r'$\frac{\sigma_t^{3}(A)}{\sigma_t^{3}(A) + \sigma_t^{3}(B_2)}$', fontsize=17)
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.xlim([0, 20])
                fig.savefig(f'../Figures/sigma_comparison.eps')


def plot_exact_comparison():
    """
    Create a plot with the comparison between exact and fast model.
    """
    fig = None
    for file in os.listdir("../Data"):
        if file.startswith('exact_fast_comp'):
            method = file.removeprefix('exact_fast_comp_ADVANCED_')
            with open(os.path.join("../Data", file), 'rb') as f:
                orbital_sim = pickle.load(f)
                fig = orbital_sim.plot_main_states(figure=fig, satellite_indices=satellite_dict['robustness_noise'],
                                                    **linestyle_dict[method], legend_name=legend_dict[method],
                                                    plot_duration=plot_duration['robustness_noise'])

    fig.savefig('../Figures/exact_fast_comp.eps')

def plot_all() -> None:
    """
    Plot and save all plots.
    """
    for file in os.listdir("../Figures"):
        if file.endswith('_inputs.eps'):
            plot_data_comparison(file.removesuffix('_inputs.eps'))
            plot_individual_results(file.removesuffix('_inputs.eps'))


def plot_presentation(plot_name: str) -> None:
    """
    Plot for presentation.

    :param plot_name: The name of the plot (with corresponding data names).
    """
    fig_name_list = ['radius', 'radial_input']
    fig_list = [None] * len(fig_name_list)

    for file in os.listdir("../Data"):
        if file.startswith(plot_name):
            method = file.removeprefix(plot_name + "_")
            end = file.removeprefix(plot_name)
            # type = file.removesuffix(end).removesuffix('_kepler').removesuffix('_j2')
            with open(os.path.join("../Data", file), 'rb') as f:
                print(file)
                orbital_sim = pickle.load(f)
                fig_list[0] = orbital_sim.plot_radius(figure=fig_list[0], satellite_indices=[0],
                                                       **linestyle_dict[method], legend_name=legend_dict[method],
                                                       plot_duration=20)

                fig_list[1] = orbital_sim.plot_radial_input(figure=fig_list[1], satellite_indices=[0],
                                                          **linestyle_dict[method], legend_name=legend_dict[method],
                                                          plot_duration=20)


if __name__ == '__main__':
    # plot_data_comparison('robustness')
    # plot_individual_results('robustness')
    # plot_projection_comparison()
    # plot_exact_comparison()
    # plot_j2_comparison()
    # plot_all()
    # plot_sigma_comparison(2)
    plot_presentation('robustness_noise')
    plt.show()
