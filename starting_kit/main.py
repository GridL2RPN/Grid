import os
import time
import logging
import sys
from importlib import reload

root_dir = 'starting_kit/'
model_dir = root_dir + 'example_submission/'
problem_dir = root_dir + 'ingestion_program/'  
score_dir =  root_dir + 'scoring_program/'
input_dir = root_dir + 'public_data/'
output_dir = root_dir + 'output/'

from sys import path; path.append(root_dir); path.append(model_dir); path.append(problem_dir); path.append(score_dir);
path.append(input_dir); path.append(output_dir);


import pypownet.environment
import pypownet.runner
#import libscores
from scoring_program import libscores
from libscores import get_metric

from example_submission import my_agents

data_dir = 'starting_kit/public_data'

environment = pypownet.environment.RunEnv(parameters_folder=os.path.abspath(data_dir),
                                              game_level="hard",
                                              chronic_looping_mode='natural', start_id=0,
                                              game_over_mode="soft")


metric_name, scoring_function = get_metric()
print('Using scoring metric:', metric_name)

start = time.time()
NUMBER_ITERATIONS = 50 # The number of iterations can be changed

submission_dir = 'example_submission'
sys.path.append(submission_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_path = os.path.abspath(os.path.join(output_dir, 'runner.log'))


open(log_path, 'w').close()

submitted_controler = my_agents.CustomAgent(environment)
# Instanciate a runner, that will save the run statistics within the log_path file, to be parsed and processed
# by the scoring program
phase_runner = pypownet.runner.Runner(environment, submitted_controler, verbose=True, vverbose=False,
                                      log_filepath=log_path)
phase_runner.ch.setLevel(logging.ERROR)
# Run the planned experiment of this phase with the submitted model
score = phase_runner.loop(iterations=NUMBER_ITERATIONS)
print("cumulative rewards : {}".format(score))
end = time.time()
print(end-start)
