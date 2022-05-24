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

import yaml ###########

# Activate seaborn
seaborn.set()

parser = argparse.ArgumentParser("Gather results, plot training reward/success")
parser.add_argument("-a", "--algos", help="Algorithms to include", nargs="+", type=str, required=True) ###
parser.add_argument("-e", "--env", help="Environment(s) to include", nargs="+", type=str, required=True)
parser.add_argument("-f", "--exp-folder", help="Folders to include", type=str, required=True)
parser.add_argument("--figsize", help="Figure size, width, height in inches.", nargs=2, type=int, default=[6.4, 4.8])
parser.add_argument("--fontsize", help="Font size", type=int, default=14)
parser.add_argument("-max", "--max-timesteps", help="Max number of timesteps to display", type=int)
parser.add_argument("-y", "--y-axis", help="Y-axis", choices=["reward", "length", "success", "max_steps_exceeded", "collision"], type=str, default="reward")

args = parser.parse_args()

args.algos = [algo.upper() for algo in args.algos] ###

envs = args.env

x_label = "Timesteps"

y_axis = {
	"success": "final_status",
	"max_steps_exceeded": "final_status",
	"collision": "final_status",
	"reward": "r",
	"length": "l",
}[args.y_axis]
y_label = {
	"success": "Success Rate",
	"max_steps_exceeded": "Max Steps Exceeded Rate",
	"collision": "Collision Rate",
	"reward": "Episodic Reward",
	"length": "Episode Length",
}[args.y_axis]

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
    args_file = os.path.join(folder, env, "args.yml")
    f = open(args_file, 'r')
    yaml_args = yaml.unsafe_load(f)
    freq = yaml_args["eval_freq"]
    ep = yaml_args["eval_episodes"]
    print("freq:" + str(freq))
    x = []
    y_tmp = []
    for i in range(int(len(data_frame)/ep)):
        x.append((i+1)*freq)
        y_tmp.append(np.mean([y[int(i*ep+j)] for j in range(ep)]))
    y = y_tmp
    x = np.array(x)
    return x, y
    
subfolder = "eval"
label_prefix = "Evaluation"
    
y_label = label_prefix + " " + y_label

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

plt.figure(y_label, figsize=args.figsize)
plt.title(y_label, fontsize=args.fontsize)
plt.xlabel(f"{x_label}", fontsize=args.fontsize)
plt.ylabel(y_label, fontsize=args.fontsize)

for folder in dirs:
	print(folder)
	data_folder = os.path.join(folder, subfolder)
	
	if not os.path.isdir(data_folder):
		continue
	
	try:
		print(data_folder)
		data_frame = load_results(data_folder)
	except LoadMonitorResultsError:
		continue
	if args.max_timesteps is not None:
		data_frame = data_frame[data_frame.l.cumsum() <= args.max_timesteps]
	try:
		y = np.array(data_frame[y_axis])
	except KeyError:
		print(f"No data available for {folder}")
		continue
	    
	if args.y_axis in ["collision", "success", "max_steps_exceeded"]:
	    y = compare_status(y, status_to_int(args.y_axis))
	        
	x, y_mean = get_eval_data(y, folder, env, data_frame)
	
	
	experiment_idx = folder.split('_')[1]
	label = env + ": " + algo + " (exp " + experiment_idx + ")"###

	plt.plot(x, y_mean, linewidth=2, label=label)

plt.legend() ###
plt.tight_layout()
plt.show() ###
