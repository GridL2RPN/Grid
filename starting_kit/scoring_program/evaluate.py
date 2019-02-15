import os
import sys
import re
import numpy as np
import yaml

import collections

input_dir = sys.argv[1]
output_dir = sys.argv[2]

print(input_dir)
print(output_dir)

submit_dir = os.path.join(input_dir, 'res')

if not os.path.isdir(submit_dir):
    print("%s doesn't exist" % submit_dir)

if os.path.isdir(submit_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    phase_log_filepath = os.path.abspath(os.path.join(submit_dir, 'runner.log'))

    # Reads the log file in order to retrieve the rewards it contains
    with open(phase_log_filepath, 'r') as f:
        logs = f.read()
    # Rewards lines should be similar to '- reward: -1.05; cumulative reward:'
    reward_pattern = r'- INFO - step (\d+)/\d+ - reward:(.*?); cumulative reward: (.*?)\n'
    rewards_astring = re.findall(reward_pattern, logs)
    rewards_asmatrix = np.asarray([list(map(float, reward_line)) for reward_line in rewards_astring])
    
    cumulative_reward = rewards_asmatrix[-1, 2]
    step = len(rewards_asmatrix)

    try:
        metadata = yaml.load(open(os.path.join(submit_dir, 'metadata'), 'r'))
        duration = metadata['elapsedTime']
    except:
        duration = 0

    output_filename = os.path.join(output_dir, 'scores.txt')
    with open(output_filename, 'w') as f:
        f.write("score: {}\n".format(cumulative_reward))
        f.write("Duration: %0.6f\n" % duration)
        f.close()
    print("step : {}, cumulative rewards : {}".format(step,cumulative_reward ))
