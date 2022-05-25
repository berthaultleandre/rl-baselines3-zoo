"""
UR Plot Utils
"""

import os
from stable_baselines3.common.results_plotter import X_EPISODES, X_TIMESTEPS, X_WALLTIME


BOLD = "bold"

# FONT SIZE
FONT_SIZE_FIGURE_TITLE = 20
FONT_SIZE_AXES_TITLE = 12
FONT_SIZE_AXES_TICK = 8
FONT_SIZE_LEGEND = 10
FONT_SIZE_AXES_LABEL = 10

# AXIS KEYS

KEY_STEP = 'steps'
KEY_EPISODE = 'episodes'
KEY_TIME = 'time'

KEY_ALL = 'all'

KEY_REWARD = 'reward'
KEY_LENGTH = 'length'

KEY_RATES = 'rates'
KEY_SUCCESS = 'success'
KEY_FAILURE = 'failure'
KEY_COLLISION = 'collision'
KEY_MAX_STEPS_EXCEEDED = 'max_steps_exceeded'

status = [KEY_SUCCESS, KEY_FAILURE, KEY_MAX_STEPS_EXCEEDED, KEY_COLLISION]
rates = [KEY_SUCCESS, KEY_FAILURE, KEY_MAX_STEPS_EXCEEDED, KEY_COLLISION]
failure_rates = [KEY_FAILURE, KEY_MAX_STEPS_EXCEEDED, KEY_COLLISION]

x_axis_list = [KEY_STEP, KEY_EPISODE, KEY_TIME]
x_axis_choices = x_axis_list

y_axis_list = [KEY_REWARD, KEY_LENGTH] + rates + x_axis_list
y_axis_choices = [KEY_ALL, KEY_RATES] + y_axis_list

X_AXIS_DEFAULT = [KEY_STEP, KEY_EPISODE]
Y_AXIS_DEFAULT = [KEY_ALL]
FIGURE_SIZE_DEFAULT = [12.8, 9.6]
LINE_WIDTH_DEFAULT = 2
ROW_COUNT_DEFAULT = 3
ROLLING_WINDOW_DEFAULT = 100
FONT_SIZE_DEFAULT = 10
FIGURE_TITLE_DEFAULT = "Training results"
LOG_FOLDER_DEFAULT = ["logs/"]


title_dict = {
	KEY_SUCCESS: 'Success Rate',
	KEY_MAX_STEPS_EXCEEDED: 'Max Steps Exceeded Rate',
	KEY_COLLISION: 'Collision Rate',
	KEY_FAILURE : 'Failure Rate',
	KEY_RATES : 'All Rates',
	KEY_ALL: 'All',
	KEY_REWARD: 'Episodic Reward',
	KEY_LENGTH: 'Episode Length',
	KEY_STEP: 'Timesteps',
	KEY_EPISODE: 'Episodes',
	KEY_TIME: 'Walltime',
}

axis_label_dict = {
	KEY_SUCCESS: 'Rate (%)',
	KEY_MAX_STEPS_EXCEEDED: 'Rate (%)',
	KEY_FAILURE : 'Rate (%)',
	KEY_COLLISION: 'Rate (%)',
	KEY_RATES : 'Rate (%)',
	KEY_REWARD: 'Reward',
	KEY_LENGTH: 'Length',
	KEY_STEP: 'Timesteps',
	KEY_EPISODE: 'Episodes',
	KEY_TIME: 'Walltime (in hours)',
}

axis_data_dict = {
	KEY_SUCCESS: 'final_status',
	KEY_MAX_STEPS_EXCEEDED: 'final_status',
	KEY_COLLISION: 'final_status',
	KEY_ALL: 'final_status',
	KEY_RATES : 'final_status',
	KEY_FAILURE: 'final_status',
	KEY_REWARD: 'r',
	KEY_LENGTH: 'l',
	KEY_STEP: X_TIMESTEPS,
	KEY_EPISODE: X_EPISODES,
	KEY_TIME: X_WALLTIME,
}

def compare_status(status1, status2) -> bool:
	if (not status1 in status or not status2 in status):
		raise ValueError()
	if (status1 == KEY_ALL or status2 == KEY_ALL):
		return True
	if (status1 == KEY_FAILURE):
		return status2 in failure_rates
	if (status2 == KEY_FAILURE):
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