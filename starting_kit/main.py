import os
import time
import logging
import sys
from importlib import reload


import argparse
from pypownet.environment import RunEnv
from pypownet.runner import Runner
import example_submission.my_agents
import example_submission.baseline_agents



parser = argparse.ArgumentParser(description='CLI tool to run experiments using PyPowNet.')
parser.add_argument('-a', '--agent', metavar='AGENT_CLASS', default='DoNothingAgent', type=str,
                    help='class to use for the agent (must be within the \'pypownet/agent.py\' file); '
                         'default class Agent')
parser.add_argument('-n', '--niter', type=int, metavar='NUMBER_ITERATIONS', default='1000',
                    help='number of iterations to simulate (default 1000)')
parser.add_argument('-p', '--parameters', metavar='PARAMETERS_FOLDER', default='./public_data/', type=str,
                    help='parent folder containing the parameters of the simulator to be used (folder should contain '
                         'configuration.json and reference_grid.m)')
parser.add_argument('-lv', '--level', metavar='GAME_LEVEL', type=str, default='hard',
                    help='game level of the timestep entries to be played (default \'easy\')')
parser.add_argument('-s', '--start-id', metavar='CHRONIC_START_ID', type=int, default=0,
                    help='id of the first chronic to be played (default 0)')
parser.add_argument('-lm', '--loop-mode', metavar='CHRONIC_LOOP_MODE', type=str, default='fixed',
                    help='the way the game will loop through chronics of the specified game level: "natural" will'
                         ' play chronic in alphabetical order, "random" will load random chronics ids and "fixed"'
                         ' will always play the same chronic folder (default "fixed")')
parser.add_argument('-m', '--game-over-mode', metavar='GAME_OVER_MODE', type=str, default='hard',
                    help='game over mode to be played: either "soft", and after each game over the simulator will load '
                         'the next timestep of the same chronic; or "hard", and after each game over the simulator '
                         'will load the first timestep of the next grid, depending on --loop-mode parameter (default '
                         '"soft")')
parser.add_argument('-r', '--render', action='store_true',
                    help='render the power network observation at each timestep (not available if --batch is not 1)')
parser.add_argument('-la', '--latency', type=float, default=None,
                    help='time to sleep after each frame plot of the renderer (in seconds); note: there are multiple'
                         ' frame plots per timestep (at least 2, varies)')
parser.add_argument('-v', '--verbose', action='store_true', default=True,
                    help='display live info of the current experiment including reward, cumulative reward')
parser.add_argument('-vv', '--vverbose', action='store_true',
                    help='display live info + observations and actions played')




def main():
    args = parser.parse_args()
    env_class = RunEnv
    agent_class = eval('example_submission.my_agents.{}'.format(args.agent))

    # Instantiate environment and agent
    env = env_class(parameters_folder=args.parameters, game_level=args.level,
                    chronic_looping_mode=args.loop_mode, start_id=args.start_id,
                    game_over_mode=args.game_over_mode, renderer_latency=args.latency)
    agent = agent_class(env)
    # Instantiate game runner and loop
    runner = Runner(env, agent, args.render, args.verbose, args.vverbose)
    final_reward = runner.loop(iterations=args.niter)
    print("Obtained a final reward of {}".format(final_reward))


if __name__ == "__main__":
    main()












"""
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
"""
