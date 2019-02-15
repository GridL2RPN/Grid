import pypownet.agent
import pypownet.environment
import numpy as np

class DoNothingAgent(pypownet.agent.Agent):
    def __init__(self, environment):
        super().__init__(environment)

    def act(self, observation):
        """ Produces an action given an observation of the environment. Takes as argument an observation of the current
        power grid, and returns the chosen action."""
        # Sanity check: an observation is a structured object defined in the environment file.
        assert isinstance(observation, pypownet.environment.Observation)

        action_space = self.environment.action_space

        # Implement your policy here
        # Example of the do-nothing policy that produces no action (i.e. an action that does nothing) each time
        do_nothing_action = action_space.get_do_nothing_action()

        # Sanity check: verify the good overall structure of the returned action; raises exceptions if not valid
        assert action_space.verify_action_shape(do_nothing_action)
        return do_nothing_action

class RandomLineSwitch(pypownet.agent.Agent):
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

        # Randomly switch one line
        l = np.random.randint(action_space.lines_status_subaction_length)
        action_space.set_lines_status_switch_from_id(action=action,
                                                     line_id=l,
                                                     new_switch_value=1)

        # Test the reward on the environment
        reward_aslist = self.environment.simulate(action, do_sum=False)
        reward = sum(reward_aslist)
        if self.verbose:
            print('reward: [', ', '.join(['%.2f' % c for c in reward_aslist]), '] =', reward)

        action_name = 'switching status of line %d' % l
        if self.verbose:
            print('Action chosen: ', action_name, '; expected reward %.4f' % reward)

        return action

        # No learning (i.e. self.feed_reward does pass)


class RandomNodeSplitting(pypownet.agent.Agent):
    """ Implements a "random node-splitting" agent: at each timestep, this controler will select a random substation
    (id), then select a random switch configuration such that switched elements of the selected substations change the
    node within the substation on which they are directly wired.
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

        # Select a random substation ID on which to perform node-splitting
        target_substation_id = np.random.choice(action_space.substations_ids)
        expected_target_configuration_size = action_space.get_number_elements_of_substation(target_substation_id)
        # Choses a new switch configuration (binary array)
        target_configuration = np.random.choice([0, 1], size=(expected_target_configuration_size,))

        action_space.set_switches_configuration_of_substation(action=action,
                                                              substation_id=target_substation_id,
                                                              new_configuration=target_configuration)

        # Test the reward on the environment
        reward_aslist = self.environment.simulate(action, do_sum=False)
        reward = sum(reward_aslist)
        if self.verbose:
            print('reward: [', ', '.join(['%.2f' % c for c in reward_aslist]), '] =', reward)

        action_name = 'change in topo of sub. %d with switches %s' % (target_substation_id,
                                                                                 repr(target_configuration))
        if self.verbose:
            print('Action chosen: ', action_name, '; expected reward %.4f' % reward)


        # Ensure changes have been done on action
        current_configuration, _ = action_space.get_switches_configuration_of_substation(action, target_substation_id)
        assert np.all(current_configuration == target_configuration)

        return action