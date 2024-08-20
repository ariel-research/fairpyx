from experiments_csv import single_plot_results, multi_plot_results
from matplotlib import pyplot as plt
from pathlib import Path
import sys

def multi_multi_plot_results(results_csv_file:str, save_to_file_template:str, filter:dict, 
     x_field:str, y_fields:list[str], z_field:str, mean:bool, 
     subplot_field:str, subplot_rows:int, subplot_cols:int, sharey:bool, sharex:bool,
     legend_properties:dict):
     for y_field in y_fields:
          save_to_file=save_to_file_template.format(y_field)
          print(y_field, save_to_file)
          multi_plot_results(
               results_csv_file=results_csv_file,
               save_to_file=save_to_file,
               filter=filter, 
               x_field=x_field, y_field=y_field, z_field=z_field, mean=mean, 
               subplot_field=subplot_field, subplot_rows=subplot_rows, subplot_cols=subplot_cols, sharey=sharey, sharex=sharex,
               legend_properties=legend_properties,
               )


def plot_course_allocation_results_szws():
     filter={"num_of_agents": 100, "num_of_items": 25}
     y_fields=["utilitarian_value","egalitarian_value", "max_envy", "mean_envy",  "mean_deficit", "max_deficit", "num_with_top_1", "num_with_top_2", "num_with_top_3","runtime"]
     multi_multi_plot_results(
          results_csv_file="results/with_solver_algo_for_course_allocation_szws.csv",
          save_to_file_template="results/more_algo_for_course_allocation_szws_{}.png",
          filter=filter, 
          x_field="supply_ratio", y_fields=y_fields, z_field="algorithm", mean=True,
          subplot_field="num_of_popular_items", subplot_rows=2, subplot_cols=1, sharey=True, sharex=True,
          legend_properties={"size":6}, 
          )


def plot_course_allocation_results_ariel():
     filter = {"solver": ["no_solver", "SCIPY"]}
     y_fields=["utilitarian_value","egalitarian_value", "max_envy", "mean_envy",  "mean_deficit", "max_deficit", "num_with_top_1", "num_with_top_2", "num_with_top_3","runtime"]
     multi_multi_plot_results(
          results_csv_file="results/with_solver_algo_for_course_allocation_ariel.csv",
          save_to_file_template="results/with_solver_algo_for_course_allocation_ariel_{}.png",
          filter=filter, 
          x_field="random_seed", y_fields=y_fields, z_field="algorithm", mean=True,
          subplot_field="max_total_agent_capacity", subplot_rows=2, subplot_cols=2, sharey=True, sharex=True,
          legend_properties={"size":6}, 
          )

def plot_course_allocation_results_uniform():
     filter = {}
     y_fields=["utilitarian_value","egalitarian_value", "max_envy", "mean_envy",  "mean_deficit", "max_deficit", "num_with_top_1", "num_with_top_2", "num_with_top_3","runtime"]
     multi_multi_plot_results(
          results_csv_file="results/with_solver_algo_for_course_allocation_uniform.csv",
          save_to_file_template="results/more_algo_for_course_allocation_uniform_{}.png",
          filter=filter, 
          x_field="value_noise_ratio", y_fields=y_fields, z_field="algorithm", mean=True,
          subplot_field = "num_of_agents", subplot_rows=2, subplot_cols=2, sharey=True, sharex=True,
          legend_properties={"size":6}, 
          )


def plot_course_allocation_results_solvers_uniform(algo):
     filter = {"algorithm": algo}
     y_fields = ["runtime"]
     z_field = "solver"
     save_to_file_template = f"results/with_solver_algo_for_course_allocation_uniform_{algo}.png"
     multi_multi_plot_results(
          results_csv_file="results/with_solver_algo_for_course_allocation_uniform.csv",
          save_to_file_template=save_to_file_template,
          filter=filter,
          x_field="value_noise_ratio", y_fields=y_fields, z_field=z_field, mean=True,
          subplot_field="num_of_agents", subplot_rows=2, subplot_cols=2, sharey=True, sharex=True,
          legend_properties={"size": 6},
     )

def plot_course_allocation_results_solvers_szws(algo):
     filter = {"algorithm": algo}
     y_fields = ["runtime"]
     z_field = "solver"
     save_to_file_template = f"results/with_solver_algo_for_course_allocation_szws_{algo}.png"
     multi_multi_plot_results(
          results_csv_file="results/with_solver_algo_for_course_allocation_szws.csv",
          save_to_file_template=save_to_file_template,
          filter=filter,
          x_field="supply_ratio", y_fields=y_fields, z_field=z_field, mean=True,
          subplot_field="num_of_agents", subplot_rows=2, subplot_cols=2, sharey=True, sharex=True,
          legend_properties={"size": 6},
     )

def plot_course_allocation_results_solvers_ariel(algo):
     filter = {"algorithm": algo}
     y_fields = ["runtime"]
     z_field = "solver"
     save_to_file_template = f"results/with_solver_algo_for_course_allocation_ariel_{algo}.png"
     multi_multi_plot_results(
          results_csv_file="results/with_solver_algo_for_course_allocation_ariel.csv",
          save_to_file_template=save_to_file_template,
          filter=filter,
          x_field="random_seed", y_fields=y_fields, z_field=z_field, mean=True,
          subplot_field="max_total_agent_capacity", subplot_rows=2, subplot_cols=2, sharey=True, sharex=True,
          legend_properties={"size": 6},
     )




# plot_course_allocation_results_uniform()
# plot_course_allocation_results_szws()
plot_course_allocation_results_ariel()

algorithms_with_specific_solver = ["TTC_O_function", "SP_O_function", "OC_function"]
for algo in algorithms_with_specific_solver:
     # plot_course_allocation_results_solvers_uniform(algo)
     # plot_course_allocation_results_solvers_szws(algo)
     plot_course_allocation_results_solvers_ariel(algo)



######## OLD PLOTS




# multi_plot_results(
#      "results/fractional_course_allocation.csv", 
#      save_to_file=True,
#     #  filter={"num_of_items": [5,10,20,30]},  # ValueError: ('Lengths must match to compare', (760,), (4,))
#      filter={}, 
#      x_field="num_of_agents", y_field="runtime", z_field="algorithm", mean=True, 
#      subplot_field = "num_of_items", subplot_rows=2, subplot_cols=2, sharey=True, sharex=True,
#      legend_properties={"size":6}, 
#      )


# multi_plot_results(
#      "results/check_effect_of_name_size.csv", 
#      save_to_file="results/check_effect_of_name_size.png",
#      filter={}, 
#      x_field="agent_name_size", y_field="runtime", z_field="algorithm", mean=True, 
#      subplot_field = "item_name_size", subplot_rows=2, subplot_cols=2, sharey=True, sharex=True,
#      legend_properties={"size":6}, 
#      )

# multi_plot_results(
#      "results/many_to_many_matchings.csv", save_to_file=True,
#      filter={}, 
#      x_field="item_capacity", y_field="runtime", z_field="algorithm", subplot_field = "agent_capacity", 
#      mean=True, subplot_rows=2, subplot_cols=3, sharey=True, sharex=True,
#      legend_properties={"size":6}, ylim=(0,30), xlim=(0,40))

