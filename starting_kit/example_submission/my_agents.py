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

    def __init__(self, environment):
        super().__init__(environment)
        self.verbose = True
        prepro = example_submission.preprocessing.Preprocessing("saved_actions.csv","saved_states.csv","saved_rewards.csv")
        self.policy = prepro.main()
        self.epsilon = 0.1
        self.delta = 0.1

    """
    Détermine la politique à partir des fichiers préprocessés
    """

    def compute_policy(self):
        policy = dict()
        for state in self.state_set:
            policy[state] = []
        return policy

    """
    Charge l'ensemble des id des actions depuis un fichier préprocessé
    """

    def load_actions(self,f):
        pass

    """
    Renvoie l'action associée à l'id donnée
    """

    def decode_action(self, id_action):
        pass

    """
    Renvoie l'état associé à l'id donnée
    """

    def decode_state(self, id_state):
        pass

    """
    Charge l'ensemble des id des états depuis un fichier préprocessé
    """

    def load_states(self, f):
        pass



    def compare_states(self, state_ref, state_check):
        assert len(state_ref) == len(state_check)
        newState = np.zeros(len(state_ref))
        for i in range(len(state_ref)):
            newState[i] = state_ref[i] - state_check[i]
        return (np.linalg.norm(newState) <= self.delta)






    """
    Détermine une action à partir de l'ensemble de données fournies et d'une observation de l'état courant
    """

    def act(self, observation):
        state = observation.as_array()
        action_space = self.environment.action_space
        for i in range(len(self.policy[0])):
            if (self.compare_states(self.policy[0][i], state)):
                draw = random.random()
                if (draw < self.epsilon):
                    index = random.randint(0,len(self.policy[1][i]))
                    return action_space.array_to_action(self.policy[1][i][index])
                else:
                    best_rew = -1000
                    index = -1
                    for j in range(len(self.policy[2][i])):
                       if (self.policy[2][i][j] > best_rew):
                          best_rew = self.policy[2][i][j]
                          best_index = j
                    return action_space.array_to_action(self.policy[1][i][best_index])
        self.policy[0].append(state)
        self.policy[1].append(self.policy[1][0])
        self.policy[2].append(np.zeros(len(self.policy[2][0])))
        index = random.randint(0,len(self.policy[1][-1])-1)
        return action_space.array_to_action(self.policy[1][-1][index])
        print("x")
        return action_space.get_do_nothing_action()

    """
    Préprocesse les fichiers .csv contenant les etats, les actions et les rewards correspondantes
    """

    def preprocessing(self, actions_file, states_file, reward_file):
        hashmap = dict()
        actions = []
        rewards = []
        action_set = dict()
        with open(actions_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                actions.append(row)
        with open(reward_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = ',')
            for row in csv_reader:
                rewards.append(row[0])
        for i in range(len(actions)):
            rewards[i] = float(rewards[i])
            for j in range(len(actions[i])):
                actions[i][j] = int(actions[i][j])
            idAction = self.hash_function_action(actions[i])
            #print(idAction)
            action_set[idAction] = actions[i]
        """
        Pour chaque état, hache l'état puis vérifie si la clé existe déjà. Si elle n'existe pas on la rajoute à la table avec l'action et la reward correspondante, sinon on ajoute l'action et la reward à la liste
        """
        with open(states_file) as csv_file:
            csv_reader= csv.reader(csv_file, delimiter=',')
            line_indice = 0
            for row in csv_reader:
                for i in range(len(row)):
                    row[i] = float(row[i])
                idState = self.hash_function_state(row)
                if not(idState in hashmap):
                    hashmap[idState] = [ [self.hash_function_action(actions[line_indice])],[rewards[line_indice]] ]
                    #print(hashmap[idState])
                else:
                    #print("xxx")
                    temp0 = hashmap[idState][0]
                    temp0.append(self.hash_function_action(actions[line_indice]))
                    temp1 = hashmap[idState][1]
                    temp1.append(rewards[line_indice])
                    hashmap[idState] = [ temp0,temp1 ]
                line_indice += 1
        for key, vals in hashmap.items():
            print(key, " : ", vals)
        print("##################################################################")
        for key, vals in action_set.items():
            print(key, " : ", vals)
        #print(action_set)
        return hashmap, action_set

    def hash_function_state(self, array):
        sumArray = 0
        for i in range(len(array)):
            sumArray = sumArray + int(array[i]*100)
        return sumArray

    def hash_function_action(self, array):
        idArray = ""
        for i in range(len(array)):
            if (array[i] == 1):
                idArray += str(i)
        return idArray


from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


class ImitationAgent(pypownet.agent.Agent):

    def __init__(self, environment):
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
