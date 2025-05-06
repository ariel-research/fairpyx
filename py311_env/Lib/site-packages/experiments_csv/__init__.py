import pathlib
HERE = pathlib.Path(__file__).parent
__version__ = (HERE / "VERSION").read_text().strip()

from experiments_csv.Experiment import Experiment, logger

try:
    from experiments_csv.plot_results import plot_dataframe, single_plot_results, multi_plot_results, multi_multi_plot_results
except ImportError:
    pass               # plotting is not supported. 
