import os
import pickle
import numpy as np

legend_dict = {'NO': 'SLS',
               'SIMPLE': 'Old LSLS',
               'ADVANCED': "New LSLS"}
def generate_table(table_name: str) -> None:
    """
    Generate table for a given table name.
    :param table_name: The name of the table (with corresponding data names)
    """
    table_data = {}

    for file in os.listdir("../Data"):
        if file.startswith(table_name) and file.removeprefix(table_name+ "_") in legend_dict:
            method = legend_dict[file.removeprefix(table_name + "_")]
            with open(os.path.join("../Data", file), 'rb') as f:
                orbital_sim = pickle.load(f)
                table_data[method] = orbital_sim.find_metric_values()
                number_of_iterations = int(np.ceil(orbital_sim.scenario.simulation.simulation_duration / orbital_sim.scenario.control.control_timestep)) + 1
                table_data[method] += [orbital_sim.solver_time / number_of_iterations]

    with open(f"..\\Tables\\{table_name}.txt", "w") as text_file:
        print("\\begin{table}[!hbt]\n\\centering", file=text_file)
        print(f"\\caption{{Metric during}}\n\\label{{tab: met_{table_name}_robust}}", file=text_file)
        print(
            "\\begin{tabular}{c| c c c c c c c c} \n Model & $\\Bar{|r|}$ [m] & $\\Bar{|\\theta|}$ [deg] & $\\Bar{|\\Omega|}$ [deg] & "
            "$\\Bar{|u_r|}$ [N]& $\\Bar{|u_t|}$ [N]& $\\Bar{|u_n|}$ [N]& $\\Bar{\\norm{\\mathbf{u}}}_2$ [N] & $T_\\mathrm{sol}$ [s]\\\\ \\hline",
            file=text_file)

        for method in table_data:
            print(f"{method} & {'&'.join(f'{x:.2g}' for x in table_data[method])} \\\\", file=text_file)

        print("\\end{tabular}\n\\end{table}", file=text_file)


def generate_all() -> None:
    """
    Generate all tables
    """
    for file in os.listdir("../Figures"):
        if file.endswith('_inputs.eps'):
            generate_table(file.removesuffix('_inputs.eps'))


if __name__ == '__main__':
    # plot_data('double_plane_kepler')
    generate_all()
    # plt.show()
