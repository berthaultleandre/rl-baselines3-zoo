"""
Plot training reward/success rate
"""
import argparse
import os

import math
import numpy as np
import seaborn
from matplotlib import pyplot as plt
from stable_baselines3.common.monitor import LoadMonitorResultsError, load_results
from stable_baselines3.common.results_plotter import X_EPISODES, X_TIMESTEPS, X_WALLTIME, ts2xy, window_func

# Activate seaborn
seaborn.set()

parser = argparse.ArgumentParser("Gather results, plot training reward/success")
parser.add_argument("-a", "--algos", help="Algorithms to include", nargs="+", type=str, required=True) ###
parser.add_argument("-e", "--env", help="Environment(s) to include", nargs="+", type=str, required=True)
parser.add_argument("-f", "--exp-folder", help="Folders to include", type=str, required=True)
parser.add_argument("--figsize", help="Figure size, width, height in inches.", nargs=2, type=int, default=[6.4, 4.8])
parser.add_argument("--fontsize", help="Font size", type=int, default=14)
parser.add_argument("-max", "--max-timesteps", help="Max number of timesteps to display", type=int)
parser.add_argument("-w", "--episode-window", help="Rolling window size", type=int, default=100)

args = parser.parse_args()

X = ["steps", "episodes", "time"]
Y = ["reward", "length"]

figure, axis = plt.subplots(len(X), len(Y))

#algo = args.algo ###
args.algos = [algo.upper() for algo in args.algos] ###

envs = args.env

x_axes = [{
		"steps": X_TIMESTEPS,
		"episodes": X_EPISODES,
		"time": X_WALLTIME,
		}[x] for x in X]
x_labels = [{
	"steps": "Timesteps",
	"episodes": "Episodes",
	"time": "Walltime (in hours)",
	}[x] for x in X]
y_axes = [{
	"success": "is_success",
	"reward": "r",
	"length": "l",
}[y] for y in Y]
y_labels = [{
	"success": "Success Rate",
	"reward": "Episodic Reward",
	"length": "Episode Length",
}[y] for y in Y]

for row in range(len(X)):
	x_axis = x_axes[row]
	x_label = x_labels[row]
	for col in range(len(Y)):
		y_axis = y_axes[col]
		y_label = y_labels[col]
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

				#axis[row,col].set_title(y_label)
				axis[row,col].set(xlabel=f'{x_label}', ylabel=y_label)
				for folder in dirs:
					try:
						print(folder)
						data_frame = load_results(folder)
					except LoadMonitorResultsError:
						continue
					if args.max_timesteps is not None:
						data_frame = data_frame[data_frame.l.cumsum() <= args.max_timesteps]
					try:
						y = np.array(data_frame[y_axis])
					except KeyError:
						print(f"No data available for {folder}")
						continue
					x, _ = ts2xy(data_frame, x_axis)

					# Do not plot the smoothed curve at all if the timeseries is shorter than window size.
					if x.shape[0] >= args.episode_window:
						# Compute and plot rolling mean with window of size args.episode_window
						x, y_mean = window_func(x, y, args.episode_window, np.mean)

						axis[row,col].plot(x, y_mean, linewidth=2)

labels = [env + ": " + algo for algo in args.algos for env in envs]
figure.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.5)
figure.subplots_adjust(bottom=0.15)
figure.legend(labels = labels, loc = 'lower center') ###
plt.show() ###
