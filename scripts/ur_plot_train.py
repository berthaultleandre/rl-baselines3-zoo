"""
Plot training reward/success rate
"""
import argparse
import os

import yaml 

import math
import numpy as np
import seaborn
from matplotlib import pyplot as plt
from stable_baselines3.common.monitor import LoadMonitorResultsError, load_results
from stable_baselines3.common.results_plotter import X_EPISODES, X_TIMESTEPS, X_WALLTIME, ts2xy, window_func



y_label_dict = {
	"success": "Success Rate",
	"max_steps_exceeded": "Max Steps Exceeded Rate",
	"collision": "Collision Rate",
	"all": "All",
	"reward": "Episodic Reward",
	"length": "Episode Length",
}

y_axis_dict = {
	"success": "final_status",
	"max_steps_exceeded": "final_status",
	"collision": "final_status",
	"all": "final_status",
	"reward": "r",
	"length": "l",
}

x_label_dict = {
	"steps": "Timesteps",
	"episodes": "Episodes",
	"time": "Walltime (in hours)",
}

x_axis_dict = {
		"steps": X_TIMESTEPS,
		"episodes": X_EPISODES,
		"time": X_WALLTIME,
}

def plot_data(ax, x_axis, y_axis, args):

	if args.evaluation:
		subfolder = "eval"
		label_prefix = "Evaluation"
	else:
		subfolder = "train"
		label_prefix = "Training"

	#x_label = x_label_dict[args.x_axis]
	#x_axis = x_axis_dict[args.x_axis]

	y_label = label_prefix + " " + y_label_dict[y_axis]
	#y_axis = [y_axis_dict[y] for y in args.y_axis]

	for algo in args.algos: ###
			
		log_path = os.path.join(args.exp_folder, algo.lower()) ###
		dirs = []

		for env in envs:
			
			dirs.extend(
			[
				os.path.join(log_path, folder)
				for folder in os.listdir(log_path)
				if (env in folder and os.path.isdir(os.path.join(log_path, folder)))
			]
			)

		ax.set_title(y_label)
		ax.set(xlabel=f"{x_label_dict[x_axis]}", ylabel=y_label)

		for folder in dirs:
			data_folder = os.path.join(folder, subfolder)
			
			if not os.path.isdir(data_folder):
				continue
			
			try:
				data_frame = load_results(data_folder)
			except LoadMonitorResultsError:
				continue
			if args.max_timesteps is not None:
				data_frame = data_frame[data_frame.l.cumsum() <= args.max_timesteps]
			try:
				y = np.array(data_frame[y_axis_dict[y_axis]])
			except KeyError:
				print(f"No data available for {folder}")
				continue
				
			if y_axis in ["collision", "success", "max_steps_exceeded"]:
				y = compare_status(y, status_to_int(y_axis))

			if args.evaluation:
				x, y_mean = get_eval_data(y, folder, env, data_frame)
			else:
				x, _ = ts2xy(data_frame, x_axis_dict[x_axis])
			
			# Do not plot the smoothed curve at all if the timeseries is shorter than window size.
			if args.evaluation or x.shape[0] >= args.episode_window:
				# Compute and plot rolling mean with window of size args.episode_window
				
				if not args.evaluation:
					x, y_mean = window_func(x, y, args.episode_window, np.mean)
				
				experiment_idx = folder.split('_')[1]
				label = env + ": " + algo + " (exp " + experiment_idx + ")"###
				
				ax.plot(x, y_mean, linewidth=2, label=label)
		
def compare_status(y, status):
	for i in range(len(y)):
		y[i] = 1 if y[i] == status else 0
	return y

def status_to_int(status):
	if status == "collision":
		return -1
	if status == "success":
		return +1
	if status == "max_steps_exceeded":
		return 0

def get_eval_data(y, folder, env, data_frame):
	dirs = []
	dirs.extend(
	[
		os.path.join(folder, f)
		for f in os.listdir(folder)
		if (env in f and os.path.isdir(os.path.join(folder, f)))
	]
	)
	args_file = os.path.join(dirs[0], "args.yml")
	f = open(args_file, 'r')
	yaml_args = yaml.unsafe_load(f)
	freq = yaml_args["eval_freq"]
	ep = yaml_args["eval_episodes"]
	x = []
	y_tmp = []
	for i in range(int(len(data_frame)/ep)):
		x.append((i+1)*freq)
		y_tmp.append(np.mean([y[int(i*ep+j)] for j in range(ep)]))
	y = y_tmp
	x = np.array(x)
	return x, y
    

y_axis_list = ["reward", "success", "length", "max_steps_exceeded", "collision"]

if __name__ == '__main__':
	# Activate seaborn
	seaborn.set()

	parser = argparse.ArgumentParser("Gather results, plot training reward/success")
	parser.add_argument("-a", "--algos", help="Algorithms to include", nargs="+", type=str, required=True) ###
	parser.add_argument("-e", "--env", help="Environment(s) to include", nargs="+", type=str, required=True)
	parser.add_argument("-f", "--exp-folder", help="Folders to include", type=str, required=True)
	parser.add_argument("--figsize", help="Figure size, width, height in inches.", nargs=2, type=int, default=[6.4, 4.8])
	parser.add_argument("--fontsize", help="Font size", type=int, default=14)
	parser.add_argument("-max", "--max-timesteps", help="Max number of timesteps to display", type=int)
	parser.add_argument("-x", "--x-axis", help="X-axis", choices=["steps", "episodes", "time"], nargs="+", type=str, default="steps")
	parser.add_argument("-y", "--y-axis", help="Y-axis", choices=["all"] + y_axis_list, nargs="+", type=str, default="reward")
	parser.add_argument("-w", "--episode-window", help="Rolling window size", type=int, default=100)
	parser.add_argument("-evaluation", help="Use evaluation data", action='store_true')

	args = parser.parse_args()

	args.algos = [algo.upper() for algo in args.algos] ###

	envs = args.env

	if "all" in args.y_axis:
		args.y_axis = y_axis_list

	if len(args.x_axis) > 1 and len(args.x_axis) != len(args.y_axis):
		raise ValueError("Size of x_axis and does not match size of y_axis")

	if len(args.x_axis) == 1:
		args.x_axis = len(args.y_axis) * args.x_axis


	n_plot = len(args.y_axis)

	n_row = min(2, n_plot)
	n_col = math.ceil(n_plot / 2)

	fig, axs = plt.subplots(n_row, n_col, figsize=args.figsize)

	handles = []
	labels = []

	subplot_index = 0
	for col in range(n_col):
		for row in range(n_row):
			
			if (subplot_index == n_plot):
				fig.delaxes(axs[-1,-1])
				break

			if n_col == 1:
				ax = axs[row]
			else:
				ax = axs[row][col]

			plot_data(ax, args.x_axis[subplot_index], args.y_axis[subplot_index], args)
			handles_tmp, labels_tmp = ax.get_legend_handles_labels()
			for i in range(len(labels_tmp)):
				if not labels_tmp[i] in labels:
					handles.append(handles_tmp[i])
					labels.append(labels_tmp[i])
			subplot_index = subplot_index + 1

	fig.legend(handles.reverse(), labels.reverse(), loc='lower center')
	fig.tight_layout()
	plt.show()

