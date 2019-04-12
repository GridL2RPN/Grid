import pypownet.agent
import pypownet.environment
import numpy as np
import os
import preprocessing
import random

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


class Submission(pypownet.agent.Agent):

    def __init__(self, environment):
        random.seed()
        super().__init__(environment)
        prepro = preprocessing.Preprocessing("program/saved_actions.csv","program/saved_states.csv","program/saved_rewards.csv")
        self.data = prepro.main()
        X = self.data[0]
        y = self.data[1]
        y_label = []
        for i in range(len(y)):
            y_label.append(self.compute_action_key(y[i]))
        self.agent = MLPClassifier(learning_rate = 'adaptive', activation = 'logistic').fit(X, y_label)

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
