import os
import sys
import pypownet.environment
import pypownet.runner
import logging
import pydoc
import time
overall_start = time.time()

PHASE_NUMBER = 1  # Numbering starts at 1

NUMBER_ITERATIONS = 1000  # Number of iterations of the simulator to be played
VERBOSE = True
VVERBOSE = False

GAME_LEVEL = 'hard'
GAME_OVER_MODE = 'soft'  # 'hard'
START_ID = 8

def main():
    # read arguments
    input_dir = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])
    program_dir = os.path.abspath(sys.argv[3])
    submission_dir = os.path.abspath(sys.argv[4])

    # create output dir if not existing
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if VERBOSE:
        print("input dir: {}".format(input_dir))
        print("output dir: {}".format(output_dir))
        print("program dir: {}".format(program_dir))
        print("submission dir: {}".format(submission_dir))

        print("input content", os.listdir(input_dir))
        print("output content", os.listdir(output_dir))
        print("program content", os.listdir(program_dir))
        print("submission content", os.listdir(submission_dir))

    # add proper directories to path
    sys.path.append(program_dir)
    sys.path.append(submission_dir)

    try:
        import submission
    except ImportError:
        raise ImportError('The submission folder should contain a file submission.py containing your controler named '
                          'as the class Submission.')

    # Instantiate and run the agents on both validation and test sets (simultaneously)
    environment = pypownet.environment.RunEnv(parameters_folder=input_dir,
                                              game_level=GAME_LEVEL,
                                              chronic_looping_mode='natural', start_id=START_ID,
                                              game_over_mode=GAME_OVER_MODE)
    try:
        submitted_controler = submission.Submission(environment)
    except:
        raise Exception('Did not find a class named Submission within submission.py; your submission controler should'
                        ' be a class named Submission in submission.py file directly within the ZIP submission file.')
    log_path = os.path.abspath(os.path.join(output_dir, 'runner.log'))
    print('log file path', log_path)

    open(log_path, 'w').close()

    # Instanciate a runner, that will save the run statistics within the log_path file, to be parsed and processed
    # by the scoring program
    phase_runner = pypownet.runner.Runner(environment, submitted_controler, verbose=VERBOSE, vverbose=VVERBOSE,
                                          log_filepath=log_path)  # vverbose should be False otherwise any one
                                                                  # can see the testing phase injections and co
    phase_runner.ch.setLevel(logging.ERROR)
    # Run the planned experiment of this phase with the submitted model
    phase_runner.loop(iterations=NUMBER_ITERATIONS)

    overall_time_spent = time.time() - overall_start
    
    if VERBOSE:
        print("Overall time spent %5.2f sec " % overall_time_spent)

if __name__ == "__main__":
    main()
