import pypownet.agent
import pypownet.environment
import example_submission.preprocessing
import numpy as np
import os
import itertools
import functools
import csv
import random
import math
from time import gmtime, strftime
import scipy
import _pickle as cPickle
from sklearn import svm
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


class DoNothingAgent(pypownet.agent.Agent):
    def __init__(self, environment):
        super().__init__(environment)

    def act(self, observation):
        """ Produces an action given an observation of the environment. Takes as argument an observation of the current
        power grid, and returns the chosen action."""
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)
        #print(" DO NOTHING AGENT !!! ")
        action_space = self.environment.action_space

        # Implement your policy here
        # Example of the do-nothing policy that produces no action (i.e. an action that does nothing) each time
        do_nothing_action = action_space.get_do_nothing_action()

        # Sanity check: verify the good overall structure of the returned action; raises exceptions if not valid
        assert action_space.verify_action_shape(do_nothing_action)
        return do_nothing_action

class ActIOnManager(object):
    def __init__(self, destination_path='saved_actions.csv', delete=True):
        self.actions = []
        self.destination_path = destination_path
        print('Storing actions at', destination_path)

        # Delete last path with same name by default!!!
        if delete and os.path.exists(destination_path):
            os.remove(destination_path)

    def dump(self, action):
        with open(self.destination_path, 'a') as f:
            f.write(','.join([str(int(switch)) for switch in action.as_array()]) + '\n')

    def dumpState(self, state):
        with open(self.destination_path, 'a') as f:
            f.write(','.join([str(float(switch)) for switch in state]) + '\n')

    def dumpReward(self, reward):
        with open(self.destination_path, 'a') as f:
            f.write(str(reward) + '\n')

    @staticmethod
    def load(filepath):
        with open(filepath, 'r') as f:
            lines = f.read().splitlines()
        actions = [[int(l) for l in line.split(',')] for line in lines]
        assert 0 in np.unique(actions) and 1 in np.unique(actions) and len(np.unique(actions)) == 2
        return actions

class GreedySearch(pypownet.agent.Agent):
    """ This agent is a tree-search model of depth 1, that is constrained to modifiying at most 1 substation
    configuration or at most 1 line status. This controler used the simulate method of the environment, by testing
    every 1-line status switch action, every new configuration for substations with at least 4 elements, as well as
    the do-nothing action. Then, it will seek for the best reward and return the associated action, expecting
    the maximum reward for the action pool it can reach.
    Note that the simulate method is only an approximation of the step method of the environment, and in three ways:
    * simulate uses the DC mode, while step is in AC
    * simulate uses only the predictions given to the player to simulate the next timestep injections
    * simulate can not compute the hazards that are supposed to come at the next timestep
    """

    def __init__(self, environment):
        super().__init__(environment)
        self.verbose = False
        self.epsilon = 0.1
        self.ioman = ActIOnManager(destination_path='saved_actions.csv')
        self.ioman2 = ActIOnManager(destination_path='saved_states.csv')
        self.ioman3  = ActIOnManager(destination_path='saved_rewards.csv')
        random.seed()

    def actGS(self, observation):
        import itertools

         # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)
        action_space = self.environment.action_space

        number_lines = action_space.lines_status_subaction_length
        # Will store reward, actions, and action name, then eventually pick the maximum reward and retrieve the
        # associated values
        rewards, actions, names = [], [], []

        # Test doing nothing
        if self.verbose:
            print(' Simulation with no action', end='')
        action = action_space.get_do_nothing_action()
        reward_aslist = self.environment.simulate(action, do_sum=False)
        reward = sum(reward_aslist)
        if self.verbose:
            print('; reward: [', ', '.join(['%.2f' % c for c in reward_aslist]), '] =', reward)
        rewards.append(reward)
        actions.append(action)
        names.append('no action')

        # Test every line opening
        for l in range(number_lines):
            if self.verbose:
                print(' Simulation with switching status of line %d' % l, end='')
            action = action_space.get_do_nothing_action()
            action_space.set_lines_status_switch_from_id(action=action, line_id=l, new_switch_value=1)
            reward_aslist = self.environment.simulate(action, do_sum=False)
            reward = sum(reward_aslist)
            if self.verbose:
                print('; reward: [', ', '.join(['%.2f' % c for c in reward_aslist]), '] =', reward)
            rewards.append(reward)
            actions.append(action)
            names.append('switching status of line %d' % l)

        # For every substation with at least 4 elements, try every possible configuration for the switches
        for substation_id in action_space.substations_ids:
            substation_n_elements = action_space.get_number_elements_of_substation(substation_id)
            if 6 > substation_n_elements > 3:
                # Look through all configurations of n_elements binary vector with first value fixed to 0
                for configuration in list(itertools.product([0, 1], repeat=substation_n_elements - 1)):
                    new_configuration = [0] + list(configuration)
                    if self.verbose:
                        print(' Simulation with change in topo of sub. %d with switches %s' % (
                            substation_id, repr(new_configuration)), end='')
                    # Construct action
                    action = action_space.get_do_nothing_action()
                    action_space.set_switches_configuration_of_substation(action=action,
                                                                          substation_id=substation_id,
                                                                          new_configuration=new_configuration)
                    reward_aslist = self.environment.simulate(action, do_sum=False)
                    reward = sum(reward_aslist)
                    if self.verbose:
                        print('; reward: [', ', '.join(['%.2f' % c for c in reward_aslist]), '] =', reward)
                    rewards.append(reward)
                    actions.append(action)
                    names.append('change in topo of sub. %d with switches %s' % (substation_id,
                                                                                 repr(new_configuration)))

        # Take the best reward, and retrieve the corresponding action
        best_reward = max(rewards)
        best_index = rewards.index(best_reward)
        best_action = actions[best_index]
        best_action_name = names[best_index]

        # Dump best action into stored actions file
        self.ioman.dump(best_action)
        self.ioman3.dumpReward(best_reward)
        self.ioman2.dumpState(observation.as_array())

        if self.verbose:
            print('Action chosen: ', best_action_name, '; expected reward %.4f' % best_reward)

        return best_action

    def actRNS(self, observation):
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)
        action_space = self.environment.action_space

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action()

        # Select a random substation ID on which to perform node-splitting
        target_substation_id = np.random.choice(action_space.substations_ids)
        expected_target_configuration_size = action_space.get_number_elements_of_substation(target_substation_id)
        # Choses a new switch configuration (binary array)
        target_configuration = np.random.choice([0, 1], size=(expected_target_configuration_size,))

        action_space.set_switches_configuration_of_substation(action=action,
                                                              substation_id=target_substation_id,
                                                              new_configuration=target_configuration)

        # Ensure changes have been done on action
        current_configuration, _ = action_space.get_switches_configuration_of_substation(action, target_substation_id)
        assert np.all(current_configuration == target_configuration)

        # Dump best action into stored actions file
         #self.ioman.dump(action)

        return action

    def act(self, observation):
        x = random.random()
        if x<= self.epsilon:
            return self.actRNS(observation)
        else:
            return self.actGS(observation)



class QLearningAgent(pypownet.agent.Agent):
    """
    This agent uses a Q-Learning strategy to interact with the environment. He first creates an action set from the actions taken with the GreedySearchAgent and the sorts all the states that he encountered with the GreedySearch in a discrete state set defined with the couples (s,a) where s is the state and a the action took by the GreedySearch acting as a label.
    """


    def __init__(self, environment):
        random.seed()
        super().__init__(environment)
        self.verbose = True
        #Loads data from the files
        prepro = example_submission.preprocessing.Preprocessing("saved_actions.csv","saved_states.csv","saved_rewards.csv")
        #Returns the preprocessed data
        self.data = prepro.main()

        #If there is no previous version of the classifier, computes a new one
        #if not(os.path.isfile('my_classifier.pkl')):
        if False:
            print("step1")
            del(prepro)
            #Creates a set to train the classifier
            X = self.data[0]
            y = self.data[1]
            #Computes the keys for each action
            #Each key represents the label associated to the state
            y_label = []
            for i in range(len(y)):
                y_label.append(self.compute_action_key(y[i]))
            print("step2")
            #Trains the classifier
            self.agent = SVC(kernel = 'linear', C=1).fit(X, y_label)
            #stores the classifier in a file
            with open('my_classifier.pkl', 'wb') as fid:
                cPickle.dump(self.agent, fid)
        else :
            with open('my_classifier.pkl', 'rb') as fid:
                self.agent = cPickle.load(fid)
        print("step3")
        #Initiates all the attributes of the agent
        self.Q = self.init_Q()
        self.epsilon = 0.1
        self.gamma = 0.5
        self.alpha = 0.5

    """
    Initiates the Q_pi(s,a) function with for every state an array of zeros (one zero for each action in the action set)
    """

    def init_Q(self):
        Q = dict()
        for i in self.data[2]:
            Q[i] = np.zeros(len(self.data[2]))
        return Q

    """
    Updates the Q_pi(s,a) function after the action a was taken with the state s: consequent_observation is the new state, rewards_aslist is the reward for the action a on state s
    """

    def feed_reward(self, action, consequent_observation, rewards_aslist):
        #Stores the new state s' as an array and classify it in the state_set
        new_state = consequent_observation.as_array()
        new_state = self.agent.predict([new_state])
        index = -1
        actionkey = self.compute_action_key(action)
        #gets the index of the action that was taken in the action_set
        for i in range(len(self.data[2])):
            if actionkey == self.data[2][i]:
                index = i
                break
        assert index != -1
        #Updates the Q_pi(s,a) function using the Q-Learning equation : Q_pi(s,a) = (1-alpha)*Q_pi(s,a) + alpha*(r(s,a,s')+gamma*max_{a in action_set}Q(s',a)
        temp = self.alpha * ( sum(rewards_aslist) + self.gamma * np.amax(self.Q[new_state[0]]))
        temp2 = self.Q[self.current_state[0]][index]
        self.Q[self.current_state[0]][index] = (1-self.alpha)*temp2 + temp

    """
    Chooses a random action in the action_set which is not the best action for state s
    """

    def choose_random_action(self, tab):
        #Stores the index of the best action
        best_index = np.argmax(tab)
        i = best_index
        #chooses a random index for the action and returns it if it is not the best action
        while (i == best_index):
            i = random.randint(0, len(tab)-1)
        return self.data[2][i]

    """
    Chooses the best action in the action_set for state s
    """

    def choose_best_action(self, tab):
        return self.data[2][np.argmax(tab)]

    """
    Given an key, returns the corresponding action
    """

    def decode_from_key(self, key):
        action = np.zeros(len(key))
        #creates an array of the size of an action and compares each cell with the corresponding char. If the char is "1" then the cell becomes a 1
        #Example : "101011" as key and [0,0,0,0,0,0] as array:
        # [1,0,0,0,0,0] -> [1,0,1,0,0,0] -> [1,0,1,0,1,0] -> [1,0,1,0,1,1]
        for i in range(len(key)):
            if key[i] == "1":
                action[i] = 1
        return action

    """
    Given an action as array, computes a key
    """

    def compute_action_key(self, array):
        key =""
        #creates a string key of the size of an action as array (each cell value of the action is put as a string in the key
        #Example : [1,0,1,0,1,1] as array:
        # "" -> "1" -> "01" -> "101" -> "1010" -> "10101" -> "101011"
        for i in range(len(array)):
            key = key + str(array[i])
        return key

    """
    Choose an action following an epsilon-greedy strategy
    """


    def act(self, observation):
        #Stores the current state in an attribute to use it later for the update of the Q_pi(s,a) function
        self.current_state = observation.as_array()
        action_space = self.environment.action_space
        #Classifies the current state in the state_set
        self.current_state = self.agent.predict([self.current_state])
        #Epsilon-greedy strategy for the action : if the random < epsilon then a non optimal random action is choosen else the optimal action
        if (random.random()<self.epsilon):
            return action_space.array_to_action(self.decode_from_key(self.choose_random_action(self.Q[self.current_state[0]])))
        else:
            return action_space.array_to_action(self.decode_from_key(self.choose_best_action(self.Q[self.current_state[0]])))











class ImitationAgent(pypownet.agent.Agent):

    def __init__(self, environment):
        random.seed()
        super().__init__(environment)
        prepro = example_submission.preprocessing.Preprocessing("saved_actions.csv","saved_states.csv","saved_rewards.csv")
        self.data = prepro.main()
        X = self.data[0]
        y = self.data[1]
        y_label = []
        for i in range(len(y)):
            y_label.append(self.compute_action_key(y[i]))
        self.agent = SVC(kernel = 'linear', C=1).fit(X, y_label)

    def compute_action_key(self, array):
        key =""
        for i in range(len(array)):
            key = key + str(array[i])
        return key

    def decode_from_key(self, key):
        action = np.zeros(len(key[0]))
        for i in range(len(key[0])):
            if key[0][i] == "1":
                action[i] = 1
        return action

    def act(self, observation):
        state = observation.as_array()
        id_action = self.agent.predict([state])
        action_space = self.environment.action_space
        return action_space.array_to_action(self.decode_from_key(id_action))
