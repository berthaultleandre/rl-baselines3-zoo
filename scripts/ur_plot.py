"""
Plot training reward/success rate
"""
import argparse
import os
from sqlalchemy import false

import yaml
import warnings
from ur_plot_utils import *

import math
import numpy as np
import seaborn
from matplotlib import gridspec, pyplot as plt
from stable_baselines3.common.monitor import LoadMonitorResultsError, load_results
from stable_baselines3.common.results_plotter import ts2xy, window_func

def format_x_axis(xs, x_axis):
	# If the number of timesteps is greater than 10**(3*power+1),
	# divide timesteps by 10**(3*power) (e.g. for timesteps ranging
	# from 0 up to 100000, x_ticks will range from 0 up to 100 (x1000))
	x_label = x_label_dict[x_axis]
	powerOf1000 = 0
	if x_axis == X_STEP:
		powerOf1000 = max([math.floor(1/3*math.log(max(x)/10)) for x in xs])
	if powerOf1000 != 0:
		x_label=f"{x_label} (x{int(10**powerOf1000)})"
		xs = [x / 10 ** (powerOf1000) for x in xs]
	return xs, x_label

def get_figure_shape(n_subplots, args):
	n_row = args.row
	n_col = math.ceil(n_subplots / args.row)
	if args.col:
		n_col = min(n_col, args.col)

	while (n_row-1)*n_col >= n_subplots:
		n_row = n_row - 1
	return n_row, n_col

def plot_results(args) -> None:

	# Adapt axes
	args.x_axis, args.y_axis = adapt_axes(args.x_axis, args.y_axis)

	args.exp_folder = remove_duplicates(args.exp_folder)

	titles, all_labels = [], []
	x_labels, y_labels = [], []
	x_data_all, y_data_all = [], []

	# Gather data from files
	for subplot_index in range(len(args.x_axis)):
			
		x_axis = args.x_axis[subplot_index]
		y_axis = args.y_axis[subplot_index]

		y_label = y_label_dict[y_axis]

		title = title_dict[y_axis]

		x_data_agent, y_data_agent = [], []

		labels = []

		empty_axis = True
		
		for exp_folder in args.exp_folder:
			dirs = []
			algos = args.algo if args.algo else os.listdir(exp_folder)
			for algo in algos:
				log_path = os.path.join(exp_folder, algo.lower())
				if os.path.isdir(log_path):
					if not args.env:
						dirs.extend([os.path.join(log_path, f) for f in os.listdir(log_path)])
					else:
						for env in args.env:
							dirs.extend(get_env_dirs(log_path, env))

			# Gather data from folders
			for folder in dirs:
				data = load_data(folder, x_axis, y_axis, args)
				if data is not None:
					labels.append(get_experiment_info(folder))
					empty_axis = False
					x, y = data
					x_data_agent, y_data_agent = x_data_agent + [x], y_data_agent + [y]

		if not empty_axis:
			x_data_agent, x_label = format_x_axis(x_data_agent, x_axis)

			# Store data
			titles.append(title)
			x_labels.append(x_label) 
			y_labels.append(y_label)
			x_data_all.append(x_data_agent)
			y_data_all.append(y_data_agent)
			all_labels.append(labels)

	# Number of plots
	n_subplots = len(titles)
	if n_subplots == 0:
		print("No data to display")
		return

	# Create figure
	fig, gs = create_figure(n_subplots, args)

	legend_labels, legend_handles = [], []

	# Create subplots
	for subplot_index in range(n_subplots):

		ax = fig.add_subplot(gs[subplot_index])
		title = titles[subplot_index]
		x_label = x_labels[subplot_index]
		y_label = y_labels[subplot_index]
		ax.set(title=title, xlabel=x_label, ylabel=y_label)

		# Plot data on subplot
		for curve_index in range(len(x_data_all[subplot_index])):
			x_data = x_data_all[subplot_index][curve_index]
			y_data = y_data_all[subplot_index][curve_index]
			label = all_labels[subplot_index][curve_index]
			plot_data(ax, x_data, y_data, label)
			
		# Update legend
		handles_tmp, labels_tmp = ax.get_legend_handles_labels()
		for i in range(len(labels_tmp)):
			if labels_tmp[i] in legend_labels:
				handles_tmp[i].set_color(legend_handles[legend_labels.index(labels_tmp[i])].get_color())
			else:
				legend_labels.append(labels_tmp[i])
			legend_handles.append(handles_tmp[i])


	adjust_figure(fig, legend_labels, args)

	plt.show()

def plot_rates(args) -> None:

	args.x_axis = adapt_axes_rates(args.x_axis)

	titles = []
	x_labels = []
	x_data_all, y_data_all = [], []

	for x_axis_index in range(len(args.x_axis)):
		x_axis = args.x_axis[x_axis_index]
		for exp_folder in args.exp_folder:
			algos = args.algo if args.algo else os.listdir(exp_folder)
			for algo in algos:
				log_path = os.path.join(exp_folder, algo.lower())
				if os.path.isdir(log_path):
					if not args.env:
						dirs = [os.path.join(log_path, f) for f in os.listdir(log_path)]
					else:
						dirs = []
						for env in args.env:
							dirs.extend(get_env_dirs(log_path, env))

					# Gather data from folder
					for folder in dirs:

						x_data_agent, y_data_agent = [], []

						empty_axis = True
						for rate in rates:
							y_axis = rate
							data = load_data(folder, x_axis, y_axis, args)
							if data is not None:
								title = get_experiment_info(folder)
								empty_axis = False
								x, y = data
								x_data_agent, y_data_agent = x_data_agent + [x], y_data_agent + [y]
						
						if not empty_axis:
							x_data_agent, x_label = format_x_axis(x_data_agent, x_axis)
							# Store data
							titles.append(title)
							x_labels.append(x_label) 
							x_data_all.append(x_data_agent)
							y_data_all.append(y_data_agent)


	# Number of plots
	n_subplots = len(titles)
	if n_subplots == 0:
		print("No data to display")
		return

	# Create figure
	fig, gs = create_figure(n_subplots, args)

	# Create subplots
	rate_index = 0
	for subplot_index in range(n_subplots):

		ax = fig.add_subplot(gs[subplot_index])
		title = titles[subplot_index]
		x_label = x_labels[subplot_index]
		y_label = y_label_dict[Y_RATES]
		ax.set(title=title, xlabel=x_label, ylabel=y_label)

		# Plot data on subplot
		for curve_index in range(len(x_data_all[subplot_index])):
			x_data = x_data_all[subplot_index][curve_index]
			y_data = y_data_all[subplot_index][curve_index]
			label = rates[rate_index]
			plot_data(ax, x_data, y_data, label)
			rate_index = (rate_index + 1) % len(rates)

	legend_labels = [title_dict[rate] for rate in rates]
	adjust_figure(fig, legend_labels, args)

	plt.show()
	
def create_figure(n_subplots, args):
	# Number of rows and columns
	n_row, n_col = get_figure_shape(n_subplots, args)

	gs = gridspec.GridSpec(n_row, n_col)
	if not args.figsize:
		args.figsize = [6.4, 4.8]

	fig = plt.figure(figsize=args.figsize)

	return fig, gs

def adjust_figure(fig, legend_labels, args):
	# Create legend
	legend = fig.legend(legend_labels, loc='lower right')
	legend.set_draggable(True)

	# Set figure title
	if args.evaluation:
		figtitle = "Evaluation results"
	else:
		figtitle = "Training results"

	fig.suptitle(figtitle, fontsize=16)

	# Adjust subplots dimensions
	fig.subplots_adjust(wspace=0.3, hspace=0.6, bottom=9/80 + 2/80 * len(legend_labels))

def get_experiment_info(folder) -> str:
	experiment_index = get_experiment_index(folder)
	env = get_env_name(folder)
	algo = get_algo_name(folder)
	label = env + ": " + algo + " (exp " + experiment_index + ")"
	return label

def get_algo_name(folder) -> str:
	return folder.split('/')[-2].upper()

def get_experiment_index(folder) -> str:
	return folder.split('_')[1]

def get_env_name(folder) -> str:
	return folder.split('/')[-1].split('-')[0]

def load_data(folder, x_axis, y_axis, args):
	if not os.path.isdir(folder):
		return None
	
	if args.evaluation:
		subfolder = "eval"
	else:
		subfolder = "train"

	data_folder = os.path.join(folder, subfolder)
	
	if not os.path.isdir(data_folder):
		return None

	try:
		data_frame = load_results(data_folder)
	except LoadMonitorResultsError:
		return None

	if args.max_timesteps is not None:
		data_frame = data_frame[data_frame.l.cumsum() <= args.max_timesteps]

	try:
		y = np.array(data_frame[y_axis_dict[y_axis]])
	except KeyError:
		print(f"No data available for {data_folder}")
		return None
	
	if y_axis in rates:
		y = np.array([compare_status(status, y_axis) for status in y])

	if len(y) == 0:
		return None

	if args.evaluation:
		data = get_evaluation_data(y, folder)
	else:
		data = get_training_data(y, x_axis, data_frame, args)
	
	return data

def plot_data(ax, x, y, label) -> bool:

	if (len(x) == 0 or len(y) == 0):
		return False

	ax.plot(x, y, linewidth=2, label=label)
	return True

def get_training_data(y, x_axis, data_frame, args):
	x, _ = ts2xy(data_frame, x_axis_dict[x_axis])
	if "episode_window" in args and x.shape[0] < args.episode_window:
		#print("Warning: cannot plot because episode window is larger than data")
		warnings.warn("Cannot plot because episode window is larger than data")
		return None
	return window_func(x, y, args.episode_window, np.mean)

def get_evaluation_data(y, folder):

	env = get_env_name(folder)
	
	# dirs[0] should be the folder containing args.yml
	dirs = get_env_dirs(folder, env)
	
	if len(dirs) == 0:
		return None

	args_file = os.path.join(dirs[0], "args.yml")
	f = open(args_file, 'r')
	yaml_args = yaml.unsafe_load(f)

	freq = yaml_args["eval_freq"]
	ep = yaml_args["eval_episodes"]

	x = []
	y_tmp = []
	for i in range(int(len(y)/ep)):
		x.append((i+1)*freq)
		y_tmp.append(np.mean([y[int(i*ep+j)] for j in range(ep)]))

	return np.array(x), y_tmp

def parse_arguments():

	parser = argparse.ArgumentParser("Gather results, plot training reward/success")

	# Experience folders
	parser.add_argument(
		"-f", "--exp-folder", 
		help="Experience folder(s) to include", 
		nargs="+", 
		type=str, 
		default=["logs/"])

	# Algorithms
	parser.add_argument(
		"-a", "--algo", 
		help="Algorithm(s) to include", 
		nargs="+", 
		type=str)
	
	# Environments
	parser.add_argument(
		"-e", "--env", 
		help="Environment(s) to include (Gym id, full or partial)", 
		nargs="+", 
		type=str)

	# x-axis
	parser.add_argument(
		"-x", "--x-axis", 
		help="X-axis", 
		choices=x_axis_choices, 
		nargs="+", 
		type=str, 
		default=[X_STEP])
		
	# y-axis
	parser.add_argument(
		"-y", "--y-axis", 
		help="Y-axis", 
		choices=y_axis_choices, 
		nargs="+", 
		type=str, 
		default=[Y_ALL])

	# Max timesteps
	parser.add_argument(
		"-max", "--max-timesteps", 
		help="Max number of timesteps to display", 
		type=int)

	# Rolling window size
	parser.add_argument(
		"-w", "--episode-window", 
		help="Rolling window size (in episodes)", 
		type=int, 
		default=200)

	# Plot rates
	parser.add_argument(
		"-rates", "--rates", 
		help="Plot rates", 
		action='store_true')

	# Use evaluation data
	parser.add_argument(
		"-eval", "--evaluation", 
		help="Use evaluation data", 
		action='store_true')

	# Max rows
	parser.add_argument(
		"-r", "--row",
		help="Figure preferred number of rows",
		type=int,
		default=3)

	# Max columns
	parser.add_argument(
		"-c", "--col",
		help="Figure preferred max number of columns",
		type=int)

	# Figure size
	parser.add_argument(
		"-size", "--figsize", 
		help="Figure size as [width, height] in inches", 
		nargs=2, 
		type=int)

	return parser.parse_args()

def set_font_size():
	plt.rc('font', size=SIZE_DEFAULT)          # controls default text sizes
	plt.rc('axes', titlesize=SIZE_AXES_TITLE)    # fontsize of the axes title
	plt.rc('axes', labelsize=SIZE_AXES_LABEL)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=SIZE_AXES_TICK)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=SIZE_AXES_TICK)    # fontsize of the tick labels
	plt.rc('legend', fontsize=SIZE_LEGEND)    # legend fontsize
	plt.rc('figure', titlesize=SIZE_FIGURE_TITLE)  # fontsize of the figure title

def adapt_axes(x_axis, y_axis):
	x_axis = remove_duplicates(x_axis)
	y_axis = remove_duplicates(y_axis)

	if Y_ALL in y_axis:
		y_axis = y_axis_list
	elif Y_RATES in y_axis:
		i = y_axis.index(Y_RATES)
		y_axis = y_axis[:i] + rates + y_axis[i+1:]

	if X_ALL in x_axis:
		x_axis = x_axis_list

	x_axis = remove_duplicates(x_axis)
	y_axis = remove_duplicates(y_axis)
	
	x_axis, y_axis = len(y_axis) * x_axis, [ax for y_ax in y_axis for ax in len(x_axis) * [y_ax]]

	return x_axis, y_axis

def adapt_axes_rates(x_axis):

	x_axis = remove_duplicates(x_axis)

	if X_ALL in x_axis:
		x_axis = x_axis_list

	x_axis = remove_duplicates(x_axis)

	return x_axis

def check_args(args):
	assert args.row > 0
	if args.col:
		assert args.col > 0
	assert args.episode_window > 0
	assert not args.max_timesteps or args.max_timesteps > 0

def main():
	seaborn.set()

	set_font_size()

	args = parse_arguments()
	check_args(args)

	plot_rates(args) if args.rates else plot_results(args)
		
	

if __name__ == '__main__':
	main()
