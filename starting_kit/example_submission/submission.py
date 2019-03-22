import pypownet.agent
import pypownet.environment
import numpy as np
import os

class Submission(pypownet.agent.Agent):
    """
    An example of a baseline controler that randomly switches the status of one random power line per timestep (if the
    random line is previously online, switch it off, otherwise switch it on).
    """

    def __init__(self, environment):
        super().__init__(environment)
        self.verbose = True

    def chooseAction(self, template, rewardRef):
        if 0 in template:
            actions = []
            for i in range(len(template)):
                test = template.copy()
                if test[i] != 1:
                    test[i] = 1
                    actions.append(test)
            rewards = []
            for act in actions:
                act = self.environment.action_space.array_to_action(act)
                rewards.append(sum(self.environment.simulate(act, do_sum = False)))
            rewards = np.asarray(rewards)
            best_index = np.argmax(rewards)
            if rewards[best_index] > rewardRef:
                return actions[best_index]
        return template

    def act(self, observation):
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)
        action_space = self.environment.action_space

        # Create template of action with no switch activated (do-nothing action)
        bestAction = np.zeros(action_space.action_length)
        stop = 1
        cpt = 0
        while(True):
            rew = sum(self.environment.simulate(action_space.array_to_action(bestAction), do_sum = False))
            newBestAction = self.chooseAction(bestAction,rew)
            if (np.array_equal(newBestAction,bestAction)):
                break
            bestAction = newBestAction
            cpt = cpt+1
            if cpt == stop:
                break
        reward_aslist = self.environment.simulate(action_space.array_to_action(bestAction), do_sum=False)
        reward = sum(reward_aslist)
        if self.verbose:
            print('reward: [', ', '.join(['%.2f' % c for c in reward_aslist]), '] =', reward)
        return action_space.array_to_action(bestAction)
