"""
Utils
"""

import os
from stable_baselines3.common.results_plotter import X_EPISODES, X_TIMESTEPS, X_WALLTIME

# FONT SIZE
SIZE_FIGURE_TITLE = 16
SIZE_AXES_TITLE = 12
SIZE_AXES_TICK = 8
SIZE_LEGEND = 10
SIZE_AXES_LABEL = 10
SIZE_DEFAULT = 10

# Y
Y_ALL = 'all'

Y_RATES = 'rates'
Y_SUCCESS = 'success'
Y_COLLISION = 'collision'
Y_MAX_STEPS_EXCEEDED = 'max_steps_exceeded'
Y_FAILURE = 'failure'

Y_LENGTH = 'length'
Y_REWARD = 'reward'

failure_rates = [Y_MAX_STEPS_EXCEEDED, Y_COLLISION, Y_FAILURE]
rates = [Y_SUCCESS] + failure_rates
y_axis_list = [Y_REWARD, Y_LENGTH] + rates
y_axis_choices = [Y_ALL, Y_RATES] + y_axis_list

# X
X_ALL = 'all'
X_STEP = 'steps'
X_EPISODE = 'episodes'
X_TIME = 'time'

x_axis_list = [X_STEP, X_EPISODE, X_TIME]
x_axis_choices = [X_ALL] + x_axis_list

title_dict = {
	Y_SUCCESS: 'Success Rate',
	Y_MAX_STEPS_EXCEEDED: 'Max Steps Exceeded Rate',
	Y_COLLISION: 'Collision Rate',
	Y_FAILURE : 'Failure Rate',
	Y_RATES : 'All Rates',
	Y_ALL: 'All',
	Y_REWARD: 'Episodic Reward',
	Y_LENGTH: 'Episode Length',
}

y_label_dict = {
	Y_SUCCESS: 'Rate (%)',
	Y_MAX_STEPS_EXCEEDED: 'Rate (%)',
	Y_FAILURE : 'Rate (%)',
	Y_COLLISION: 'Rate (%)',
	Y_RATES : 'Rate (%)',
	Y_REWARD: 'Reward',
	Y_LENGTH: 'Steps',
}

y_axis_dict = {
	Y_SUCCESS: 'final_status',
	Y_MAX_STEPS_EXCEEDED: 'final_status',
	Y_COLLISION: 'final_status',
	Y_ALL: 'final_status',
	Y_RATES : 'final_status',
	Y_FAILURE: 'final_status',
	Y_REWARD: 'r',
	Y_LENGTH: 'l',
}

x_label_dict = {
	X_STEP: 'Timesteps',
	X_EPISODE: 'Episodes',
	X_TIME: 'Walltime (in hours)',
}

x_axis_dict = {
		X_STEP: X_TIMESTEPS,
		X_EPISODE: X_EPISODES,
		X_TIME: X_WALLTIME,
}

def compare_status(status1, status2) -> bool:
	if (status1 == Y_ALL or status2 == Y_ALL):
		return True
	if (status1 == Y_FAILURE):
		return status2 in failure_rates
	if (status2 == Y_FAILURE):
		return status1 in failure_rates
	return status1 == status2

def remove_duplicates(l):
	return list(dict.fromkeys(l))

def get_env_dirs(path, env_name):
	dirs = []
	dirs.extend([
		os.path.join(path, folder)
		for folder in os.listdir(path)
		if (env_name in folder and os.path.isdir(os.path.join(path, folder)))
	])
	return dirs