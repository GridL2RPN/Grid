import pypownet.agent
import pypownet.environment
import numpy as np
import os

class CustomAgent(pypownet.agent.Agent):
    """
    An example of a baseline controler that randomly switches the status of one random power line per timestep (if the
    random line is previously online, switch it off, otherwise switch it on).
    """

    def __init__(self, environment):
        super().__init__(environment)
        self.verbose = True

    def act(self, observation):
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)
        action_space = self.environment.action_space

        # Create template of action with no switch activated (do-nothing action)
        action = action_space.get_do_nothing_action()

        # Select lines to switch
        if True :
            lines_load = observation.get_lines_capacity_usage()
            nb_lines = len(lines_load)
            assert nb_lines == action_space.lines_status_subaction_length
            for i in range(nb_lines):
                lines_status = action_space.get_lines_status_switch_from_id(action,i)
                if lines_status == 0:
                    action_space.set_lines_status_switch_from_id(action=action,line_id=i,new_switch_value=0)
                if lines_load[i] > 1:
                    action_space.set_lines_status_switch_from_id(action=action,line_id=i,new_switch_value=1)
                    action_name = 'switching status of line %d' % i
                    if self.verbose:
                        print('Action chosen: ', action_name, '; expected reward %.4f' % reward)


        # Test the reward on the environment
        reward_aslist = self.environment.simulate(action, do_sum=False)
        reward = sum(reward_aslist)
        if self.verbose:
            print('reward: [', ', '.join(['%.2f' % c for c in reward_aslist]), '] =', reward)


        return action

        # No learning (i.e. self.feed_reward does pass)
