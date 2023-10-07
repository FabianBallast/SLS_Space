import os
import pickle


def generate_table(table_name: str) -> None:
    """
    Generate table for a given table name.
    :param table_name: The name of the table (with corresponding data names)
    """
    table_data = {}

    for file in os.listdir("../Data"):
        if file.startswith(table_name):
            method = file.removeprefix(table_name + "_")
            with open(os.path.join("../Data", file), 'rb') as f:
                orbital_sim = pickle.load(f)
                table_data[method] = orbital_sim.find_metric_values()
                table_data[method] += [orbital_sim.solver_time]

    with open(f"..\\Tables\\{table_name}.txt", "w") as text_file:
        print("\\begin{table}[!hbt]\n\\centering", file=text_file)
        print(f"\\caption{{Metric during}}\n\\label{{tab: met_{table_name}_orbit_mech}}", file=text_file)
        print("\\begin{tabular}{c| c c c c c c c c} \n Model & $\\Bar{|r|}$ & $\\Bar{|\\theta|}$ & $\\Bar{|\\Omega|}$ & "
              "$\\Bar{|u_r|}$ & $\\Bar{|u_t|}$ & $\\Bar{|u_n|}$ & $\\Bar{\\norm{\\mathbf{u}}}_2$  & $T_\\mathrm{sol}$\\\\ \\hline", file=text_file)

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
