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
    Méthode principale renvoyant les données préprocessées
    """

    def main(self):
        self.action_set = self.reduce_actions()
        actions_label = []
        for i in self.action_set:
            actions_label.append(i)
        return [self.states, self.actions, actions_label]

