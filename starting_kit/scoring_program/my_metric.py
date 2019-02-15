'''Examples of organizer-provided metrics.
You can just replace this code by your own.
Make sure to indicate the name of the function that you chose as metric function
in the file metric.txt. E.g. mse_metric, because this file may contain more 
than one function, hence you must specify the name of the function that is your metric.'''

import os
import sys
import re
import numpy as np
import csv
import collections

def reward(log_filedir):
    # Reads the log file in order to retrieve the rewards it contains
    phase_log_filepath = os.path.abspath(os.path.join(log_filedir, 'runner.log'))

    with open(phase_log_filepath, 'r') as f:
        logs = f.read()

    # Rewards lines should be similar to '- reward: -1.05; cumulative reward:'
    reward_pattern = r'- INFO - step (\d+)/\d+ - reward:(.*?); cumulative reward: (.*?)\n'
    rewards_astring = re.findall(reward_pattern, logs)
    rewards_asmatrix = np.asarray([list(map(float, reward_line)) for reward_line in rewards_astring])
    
    cumulative_reward = rewards_asmatrix[-1, 2]
    
    step = rewards_asmatrix[-1,0]

    print("step : {}, cumulative rewards : {}".format(int(step),cumulative_reward ))


def rescale_metric():
    print("aa")

    metric = Rescaler("scoring_program/hard") 
    return (lambda x, iter: metric.rescale(x, iter))



def rescale(score, donothing, bruteforce):
    return (score - donothing ) / ( bruteforce - donothing)

def rescale_list(scores, donothing, bruteforce):
    return [rescale(scores[i], donothing[i], bruteforce[i]) for i in range(max(len(scores), len(donothing), len(bruteforce)))]


class Rescaler(object):
    """docstring for rescaler"""
    def __init__(self, name="hard"):
        self.donothing = self.read_rewards(name+"0")
        self.bruteforce=  self.read_rewards(name+"1")

    def rescale_list(self, score):
        return rescale_list(score, self.donothing, self.bruteforce)

    def rescale(self, score,iter=-1):
        if isinstance(score, collections.Iterable):
            return self.rescale(score[iter],iter)
        else :
            return rescale(score, self.donothing[iter], self.bruteforce[iter])

    def read_rewards(self, name):
        ret = list()
        with open(name+'.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                ret.append(float(row[2]))
        return ret


