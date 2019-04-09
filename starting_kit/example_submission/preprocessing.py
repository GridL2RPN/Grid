import numpy as np
import csv





class Preprocessing(object):

    def __init__(self, actions_file, states_file, rewards_file):

        #Initialisation d'un delta arbitraire pour la comparaison d'états

        self.delta = 0.1

        actions = []
        rewards = []
        states = []

        #On charge dans trois tableaux différents, les fichiers d'actions, états et rewards qui sont sous la forme de chaines de string
        # Importing actions stored
        with open(actions_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                actions.append(row)
        with open(rewards_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = ',')
            for row in csv_reader:
                rewards.append(row[0])
        with open(states_file) as csv_file: 
            csv_reader= csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                states.append(row)

        #On initialise les attributs d'actions, rewards et states du preprocessing à la bonne taille en numpy array vides

        self.actions = np.empty((len(actions), len(actions[0])), dtype = int)
        self.rewards = np.empty(len(rewards), dtype = float)
        self.states = np.empty((len(states), len(states[0])), dtype = float)

        #On convertit les tableaux temporaires et on stocke les valeurs dans les attributs du preprocessing

        for i in range(len(actions)):
            self.rewards[i] = float(rewards[i])
            for j in range(len(actions[i])):
                self.actions[i][j] = int(actions[i][j])
            for j in range(len(states[i])):
                self.states[i][j] = float(states[i][j])

    """
    Suppression de tous les doublons d'actions pour n'en garder qu'un exemplaire de chaque représenté sous la forme d'un dictionnaire qui à chaque clé d'action associe l'action
    """

    def reduce_actions(self):
        action_set = dict()
        for i in range(len(self.actions)):
            action_set[self.compute_action_key(self.actions[i])] = self.actions[i]
        return action_set

    """
    Calcule la clé d'une action
    """

    def compute_action_key(self, array):
        key =""
        for i in range(len(array)):
            key = key + str(array[i])
        return key

    """
    Compare la norme de la différence entre deux vecteurs et renvoie True si elle est inférieure à un seuil delta arbitraire
    """

    def compare_states(self, state_ref, state_check):
        assert len(state_ref) == len(state_check)
        newState = np.zeros(len(state_ref))
        for i in range(len(state_ref)):
            newState[i] = state_ref[i] - state_check[i]
        return (np.linalg.norm(newState) <= self.delta)

    """
    Détermine une première version de politique à partir des données
    """

    def compute_policy(self):
        policy = [dict()]
        new_states = [self.states[0]]
        new_actions = [[self.actions[0]]]
        new_rewards = [[self.rewards[0]]]
        for i in range(1,len(self.states)):
            exists = False
            for j in range(len(new_states)):
                if not(exists):
                    if self.compare_states(new_states[j],self.states[i]):
                       new_actions[j].append(self.actions[i])
                       new_rewards[j].append(self.rewards[i])
                       exists = True
            if not(exists):
                new_states.append(self.states[i])
                new_actions.append([self.actions[i]])
                new_rewards.append([self.rewards[i]])
        for i in range(len(new_states)):
            for j in self.action_set:
                if not(self.array_in_arraySet(self.action_set[j],new_actions[i])):
                    new_actions[i].append(self.action_set[j])
                    new_rewards[i].append(-1.0)
        return [ new_states, new_actions, new_rewards]

    def array_in_arraySet(self, array, arraySet):
        for i in range(len(arraySet)):
            assert len(array) == len(arraySet[i])
            equals = True
            for j in range(len(array)):
                if array[j] != arraySet[i][j]:
                    equals = False
            if equals:
                return True
        return False


    """
    Méthode principale renvoyant les données préprocessées
    """

    def main(self):
        self.action_set = self.reduce_actions()
        actions_label = []
        for i in self.action_set:
            actions_label.append(i)
        return [self.states, self.actions, actions_label]
        #return self.compute_policy()

    def mainQLearning(self):
        return self.compute_policy()
