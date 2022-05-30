"""
Plot training reward/success rate
"""
import argparse
from cmath import inf
import os
import warnings
from ur_plot_utils import *

import math
import numpy as np
import seaborn
from matplotlib import gridspec, pyplot as plt
from stable_baselines3.common.monitor import LoadMonitorResultsError, load_results
from stable_baselines3.common.results_plotter import window_func

import pandas as pd
from typing import Callable, List, Optional, Tuple

def format_axis(data:List[float], axis:str) -> Tuple[List[float], str]:
	label = axis_label_dict[axis]
	if axis == KEY_STEP or axis == KEY_EPISODE:
		powerOf10 = max([math.floor(1/3*(math.log(max(x),10)-1)) for x in data])
		if powerOf10 != 0:
			label=f"{label} (x{10**(3*powerOf10)})"
			data = [x / 10 ** (3*powerOf10) for x in data]
	return data, label

def get_figure_shape(n_subplots, args) -> Tuple[int, int]:
	n_row = args.row
	n_col = math.ceil(n_subplots / args.row)
	if args.col:
		n_col = min(n_col, args.col)
	while n_row*(n_col-1) >= n_subplots:
		n_col = n_col - 1
	while n_row*n_col < n_subplots:
		n_col = n_col + 1
	while (n_row-1)*n_col >= n_subplots:
		n_row = n_row - 1
	return n_row, n_col

def gather_rates_data(x1_axis, x2_axis, args):
	x1_data, x2_data, x1_labels, x2_labels = [], [], [], []
	y_data, titles = [], []
	two_x_axes = x2_axis is not None
	# Gather all experience folders
	exp_folders = []
	for log_folder in args.log_folder:
		exp_folders.extend(get_experiment_folders(log_folder, args.algo, args.env))

	# Gather data from folder
	for exp_folder in exp_folders:

		x1_data_tmp, y_data_tmp = [], []
		if (two_x_axes):
			x2_data_tmp = []

		empty_axis = True
		for rate in rates:
			y_axis = rate

			data1 = load_data(exp_folder, x1_axis, y_axis, args)
			if (two_x_axes):
				data2 = load_data(exp_folder, x2_axis, y_axis, args)

			if data1 is not None and (not two_x_axes or data2 is not None):
				title = get_experiment_info(exp_folder)
				empty_axis = False
				x1_data_tmp.append(data1[0])
				y_data_tmp.append(data1[1])
				if (two_x_axes):
					x2_data_tmp.append(data2[0])
		
		if not empty_axis:
			x1_data_tmp, x1_label = format_axis(x1_data_tmp, x1_axis)

			# Store data
			titles.append(title)
			x1_labels.append(x1_label) 
			x1_data.append(x1_data_tmp)
			y_data.append(y_data_tmp)
			if (two_x_axes):
				x2_data_tmp, x2_label = format_axis(x2_data_tmp, x2_axis)
				x2_labels.append(x2_label) 
				x2_data.append(x2_data_tmp)
	return x1_data, x2_data, x1_labels, x2_labels, y_data, titles
				
def gather_data(x1_axis, x2_axis, args):

	x1_data, x2_data, x1_labels, x2_labels = [], [], [], []
	y_data, y_labels, all_labels, titles = [], [], [], []
	
	two_x_axes = x2_axis is not None

	for subplot_index in range(len(args.y_axis)):

		y_axis = args.y_axis[subplot_index]

		x1_data_tmp, y_data_tmp = [], []
		if (two_x_axes):
			x2_data_tmp = []

		labels = []

		# Gather data from folders
		empty_axis = True
		for log_folder in args.log_folder:
			exp_folders = get_experiment_folders(log_folder, args.algo, args.env)
			for exp_folder in exp_folders:
				if (two_x_axes):
					data2 = load_data(exp_folder, x2_axis, y_axis, args)
				title = str(title_dict[y_axis])
				data1 = load_data(exp_folder, x1_axis, y_axis, args)
				if data1 is not None and (not two_x_axes or data2 is not None):
					labels.append(get_experiment_info(exp_folder))
					empty_axis = False
					x1_data_tmp.append(data1[0])
					y_data_tmp.append(data1[1])
					if (two_x_axes):
						x2_data_tmp.append(data2[0])

		if not empty_axis:
			x1_data_tmp, x1_label = format_axis(x1_data_tmp, x1_axis)
			y_data_tmp, y_label = format_axis(y_data_tmp, y_axis)
			# Store data
			titles.append(title)
			x1_labels.append(x1_label) 
			y_labels.append(y_label)
			x1_data.append(x1_data_tmp)
			y_data.append(y_data_tmp)
			if (two_x_axes):
				x2_data_tmp, x2_label = format_axis(x2_data_tmp, x2_axis)
				x2_labels.append(x2_label) 
				x2_data.append(x2_data_tmp)
			all_labels.append(labels)

	return x1_data, x2_data, x1_labels, x2_labels, y_data, y_labels, all_labels, titles

def plot_results(args) -> None:

	### Init ###
	args.x_axis = adapt_x_axis(args.x_axis)
	if (not args.rates):
		args.y_axis = adapt_y_axis(args.x_axis, args.y_axis)

	args.log_folder = remove_duplicates(args.log_folder)

	two_x_axes = (len(args.x_axis) == 2)
	x1_axis = args.x_axis[0]
	x2_axis = args.x_axis[1] if two_x_axes else None

	if args.rates:
		x1_data, x2_data, x1_labels, x2_labels, y_data, titles = gather_rates_data(x1_axis, x2_axis, args)
	else:
		x1_data, x2_data, x1_labels, x2_labels, y_data, y_labels, all_labels, titles = gather_data(x1_axis, x2_axis, args) 
		

	# Number of subplots
	n_subplots = len(titles)
	if n_subplots == 0:
		print("No data to display")
		return

	# Create figure
	fig, gs = create_figure(n_subplots, args)

	if args.rates:
		rate_index = 0
		legend_labels = [title_dict[rate] for rate in rates]
	else:
		legend_labels, legend_handles = [], []

	# Create subplots
	for plt_idx in range(n_subplots):

		ax = fig.add_subplot(gs[plt_idx])

		title = titles[plt_idx]
		x_label = x1_labels[plt_idx]
		if args.rates:
			y_label = axis_label_dict[KEY_RATES]
		else:
			y_label = y_labels[plt_idx]

		ax.set(title=title, xlabel=x_label, ylabel=y_label)

		if (two_x_axes):
			x2_label = x2_labels[plt_idx]
			ax2 = ax.twiny()
			ax2.set(xlabel=x2_label)
			# Hide twin axis grid lines
			ax2.grid(False)

		# Plot data on subplot
		for cur_idx in range(len(x1_data[plt_idx])):
			x1_plt = x1_data[plt_idx][cur_idx]
			y_plt = y_data[plt_idx][cur_idx]
			if args.rates:
				label = rates[rate_index]
			else:
				label = all_labels[plt_idx][cur_idx]
			plot_data(ax, x1_plt, y_plt, label, args.line_width, False)
			if (two_x_axes):
				x2_plt = x2_data[plt_idx][cur_idx]
				plot_data(ax2, x2_plt, np.zeros(len(x2_plt)), "", args.line_width, True)
			if args.rates:
				rate_index = (rate_index + 1) % len(rates)

		if not args.rates:
			# Update legend
			handles_tmp, labels_tmp = ax.get_legend_handles_labels()
			for i in range(len(labels_tmp)):
				if labels_tmp[i] in legend_labels:
					handles_tmp[i].set_color(legend_handles[legend_labels.index(labels_tmp[i])].get_color())
				else:
					legend_labels.append(labels_tmp[i])
				legend_handles.append(handles_tmp[i])

	adjust_figure(fig, args.figure_title, legend_labels)

	plt.show()

def get_experiment_folders(log_folder, algos, envs) -> List[str]:
	dirs = []
	algos = algos if algos else os.listdir(log_folder)
	for algo in algos:
		log_path = os.path.join(log_folder, algo.lower())
		if os.path.isdir(log_path):
			if not envs:
				dirs.extend([os.path.join(log_path, f) for f in os.listdir(log_path)])
			else:
				for env in envs:
					dirs.extend(get_env_dirs(log_path, env))
	return dirs
	
def create_figure(n_subplots, args) -> Tuple[plt.Figure, gridspec.GridSpec]:
	n_row, n_col = get_figure_shape(n_subplots, args)
	gs = gridspec.GridSpec(n_row, n_col)
	fig = plt.figure(figsize=args.figsize)
	return fig, gs

def adjust_figure(fig, title, legend_labels) -> None:
	legend = fig.legend(legend_labels, loc='lower right')
	legend.set_draggable(True)
	fig.suptitle(title)
	fig.tight_layout()

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

def get_dataset_limits(dataset, min_val, max_val):
	low_limit = 0
	high_limit = len(dataset)
	i = 0
	found_min = False
	for e in dataset:
		if (not found_min and e >= min_val):
			low_limit = i
			found_min = True
		if (e >= max_val):
			high_limit = i - 1
			break
		i = i + 1
	if not found_min:
		return None
	return low_limit, high_limit

def get_episode_limits(data_frame, args) -> Tuple[int, int]:
	first_episode, last_episode = 0, len(data_frame)
	# Episodes
	if args.min_episodes is not None or args.max_episodes is not None:
		min_val = min(max(0, args.min_episodes), len(data_frame)) if args.min_episodes else 0
		max_val = min(max(0, args.max_episodes), len(data_frame)) if args.max_episodes else len(data_frame)
		if min_val > max_val:
			print(f"Min cannot be greater than max (episodes)")
			return None
		first_episode = max(first_episode, min_val)
		last_episode = min(last_episode, max_val)
	# Steps
	if args.min_steps is not None or args.max_steps is not None:
		min_val = args.min_steps if args.min_steps else 0
		max_val = args.max_steps if args.max_steps else inf
		if min_val > max_val:
			print(f"Min cannot be greater than max (steps)")
			return None
		limits = get_dataset_limits(np.cumsum(data_frame.loc[:,'l'].values), min_val, max_val)
		if limits is not None:
			first_episode = max(first_episode, limits[0])
			last_episode = min(last_episode, limits[1])
			
	# Time
	if args.min_time is not None or args.max_time is not None:
		min_val = args.min_time if args.min_time else 0
		max_val = args.max_time if args.max_time else inf
		if min_val >= max_val:
			print(f"Min cannot be greater than max (time)")
			return None
		limits = get_dataset_limits(data_frame.loc[:,'t'] / 3600.0, min_val, max_val)
		if limits is not None:
			first_episode = max(first_episode, limits[0])
			last_episode = min(last_episode, limits[1])

	return first_episode, last_episode

def load_data(folder, x_axis, y_axis, args) -> Tuple[List[float], List[float]]:
	if not os.path.isdir(folder):
		return None

	try:
		data_frame = load_results(folder)
	except LoadMonitorResultsError:
		return None

	# Get episode limit according to min/max args
	episode_limits = get_episode_limits(data_frame, args)

	if episode_limits is None:
		return None

	if y_axis in x_axis_list:
		y = dataframe_to_axis(data_frame, y_axis, episode_limits)
	else:
		try:
			y = np.array(data_frame.loc[episode_limits[0]:episode_limits[1]-1, axis_data_dict[y_axis]])
		except KeyError:
			print(f"No data available for {folder}")
			return None
		
		# If y_axis is a rate, then y is a list of string (e.g. ["collision", "success", ...])
		# We transform y into a list of float using compare_status
		if y_axis in rates:
			y = 100 * np.array([compare_status(status, y_axis) for status in y])

	if len(y) == 0:
		return None

	x = dataframe_to_axis(data_frame, x_axis, episode_limits)
	if not y_axis in x_axis_list:
		if args.episode_window and x.shape[0] < args.episode_window:
			warnings.warn("Cannot plot because episode window is larger than data")
			return None
		x, y = window_func(x, y, args.episode_window, np.mean)

	return x, y

def dataframe_to_axis(data_frame: pd.DataFrame, axis_key: str, episode_limits) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Decompose a data frame variable to x ans ys

	:param data_frame: the input data
	:param x_axis: the axis for the x and y output
		(can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
	:return: the x and y output
	"""
	[first_episode, last_episode] = episode_limits
	x_data_name = axis_data_dict[axis_key]

	if x_data_name == X_TIMESTEPS:
		x_var = np.sum(data_frame.loc[:first_episode-1,'l']) + np.cumsum(data_frame.loc[first_episode:last_episode-1,'l'].values)

	elif x_data_name == X_EPISODES:
		x_var = np.arange(first_episode, last_episode)

	elif x_data_name == X_WALLTIME:
		x_var = data_frame.loc[first_episode:last_episode-1,'t'].values / 3600.0

	else:
		raise NotImplementedError

	return x_var

def plot_data(ax, x, y, label, line_width, hide) -> bool:
	if (len(x) == 0 or len(y) == 0):
		return False
	ax.plot(x, y, linewidth=line_width, label=label, linestyle=('None' if hide else '-'))
	return True

def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser("Gather results, plot training reward/success")

	# Experience folders
	parser.add_argument(
		"-f", "--log-folder", 
		help="Experience folder(s) to include", 
		nargs="+", 
		type=str, 
		default=LOG_FOLDER_DEFAULT)

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

	# X-axis
	parser.add_argument(
		"-x", "--x-axis", 
		help="X-axis", 
		choices=x_axis_choices, 
		nargs="+", 
		type=str, 
		default=X_AXIS_DEFAULT)
		
	# Y-axis
	parser.add_argument(
		"-y", "--y-axis", 
		help="Y-axis", 
		choices=y_axis_choices, 
		nargs="+", 
		type=str, 
		default=Y_AXIS_DEFAULT)

	# Min timesteps
	parser.add_argument(
		"-min_steps", "--min-steps", 
		help="Min number of timesteps to display", 
		type=int)

	# Max timesteps
	parser.add_argument(
		"-max_steps", "--max-steps", 
		help="Max number of timesteps to display", 
		type=int)

	# Min episodes
	parser.add_argument(
		"-min_episodes", "--min-episodes", 
		help="Min episodes to display", 
		type=int)

	# Max episodes
	parser.add_argument(
		"-max_episodes", "--max-episodes", 
		help="Max episodes to display", 
		type=int)

	# Min time
	parser.add_argument(
		"-min_time", "--min-time", 
		help="Min time in hours", 
		type=float)

	# Max timesteps
	parser.add_argument(
		"-max_time", "--max-time", 
		help="Max time in hours", 
		type=float)

	# Rolling window size
	parser.add_argument(
		"-w", "--episode-window", 
		help="Rolling window size (in episodes)", 
		type=int, 
		default=ROLLING_WINDOW_DEFAULT)

	# Plot rates
	parser.add_argument(
		"-rates", "--rates", 
		help="Plot rates", 
		action='store_true')

	# Max rows
	parser.add_argument(
		"-r", "--row",
		help="Figure preferred count of rows",
		type=int,
		default=ROW_COUNT_DEFAULT)

	# Max columns
	parser.add_argument(
		"-c", "--col",
		help="Figure preferred max number of columns",
		type=int)

	# Figure title
	parser.add_argument(
		"-title", "--figure_title", 
		help="Figure title", 
		type=str,
		default=FIGURE_TITLE_DEFAULT)

	# Line width
	parser.add_argument(
		"-line_width", 
		help="Line size", 
		type=float,
		default=LINE_WIDTH_DEFAULT)

	# Font size
	parser.add_argument(
		"-font_size", 
		help="Font size", 
		type=int,
		default=FONT_SIZE_DEFAULT)

	# Figure size
	parser.add_argument(
		"-size", "--figsize", 
		help="Figure size as [width, height] in inches", 
		nargs=2, 
		type=int,
		default=FIGURE_SIZE_DEFAULT)

	return parser.parse_args()

def set_font_parameters(fontsize) -> None:
	ds = fontsize - FONT_SIZE_DEFAULT
	#print(plt.rcParams.keys())
	plt.rcParams["axes.titleweight"] = BOLD
	plt.rcParams["figure.titleweight"] = BOLD
	plt.rc('font', size=FONT_SIZE_DEFAULT + ds)              # controls default text sizes
	plt.rc('axes', titlesize=FONT_SIZE_AXES_TITLE + ds)      # fontsize of the axes title
	plt.rc('axes', labelsize=FONT_SIZE_AXES_LABEL + ds)      # fontsize of the x and y labels
	plt.rc('xtick', labelsize=FONT_SIZE_AXES_TICK + ds)      # fontsize of the tick labels
	plt.rc('ytick', labelsize=FONT_SIZE_AXES_TICK + ds)      # fontsize of the tick labels
	plt.rc('legend', fontsize=FONT_SIZE_LEGEND + ds)    	   # legend fontsize
	plt.rc('figure', titlesize=FONT_SIZE_FIGURE_TITLE + ds)  # fontsize of the figure title

def adapt_y_axis(x_axis, y_axis) -> List[str]:
	y_axis = remove_duplicates(y_axis)
	if KEY_ALL in y_axis:
		y_axis = y_axis_list
	elif KEY_RATES in y_axis:
		i = y_axis.index(KEY_RATES)
		y_axis = y_axis[:i] + rates + y_axis[i+1:]
	y_axis = remove_duplicates(y_axis)
	y_axis = [ax for ax in y_axis if ax not in x_axis]
	return y_axis

def adapt_x_axis(x_axis) -> List[str]:
	x_axis = remove_duplicates(x_axis)
	if len(x_axis) > 2:
		raise ValueError("Argument x_axis does not accept more than two values")
	return x_axis

def check_args(args) -> None:
	if args.row <= 0:
		raise ValueError("Argument row must be > 0")
	if args.col and args.col <= 0:
		raise ValueError("Argument col must be > 0")
	if args.episode_window and args.episode_window <= 0:
		raise ValueError("Argument w must be > 0")

def main() -> None:
	seaborn.set()
	args = parse_arguments()
	check_args(args)
	set_font_parameters(args.font_size)
	plot_results(args)

if __name__ == '__main__':
	main()
