import random

from abc import ABC, abstractmethod

class MultiArmedBandit(object):
    """
    A class containing multiple different bandit arms that can be selected for
    some reward.
    
    Note that each bandit should be instance of Bandit so they each have their
    own random and thus do not depend on the order selected in.
    Instead, the ith pull of a specific bandit should result in the same reward
    across all of the different algorithms being run on it.
    
    Parameters
    --------
    bandits : array-like containing Bandit class, shape [num_bandits,]
        All of the possible bandits that can be selec4ted.
    """
    def __init__(self, bandits):
        self._bandits = bandits
        
    def regenerate(self):
        """
        Will regenerate the random seed for each of the bandits. Can be used to
        allow running the algorithms again with slightly different rewards.
        """
        for bandit in self._bandits:
            bandit.regenerate()
    
    def reset(self):
        """
        Will reset the random state for all of the bandits. Should be used between
        running different algorithms on the bandits to ensure there are no differences
        between rewards provided.
        """
        for bandit in self._bandits:
            bandit.reset()
    
    def pull_bandit(self, bandit_chosen):
        """
        Pull the selected bandit, returning the result of pulling that bandit once.
        """
        return self._bandits[bandit_chosen].pull()
    
    def actions(self):
        """
        Returns all of the valid actions.
        """
        return [i for i in range(len(self._bandits))]
    
    def expected_value_per_bandit(self):
        """
        Returns the expected value for each bandit.
        Will be array-like, shape [num_bandits,].
        """
        return [bandit.expected_value() for bandit in self._bandits]
    
    def regret_per_action(self):
        """
        Returns the regret for selecting each bandit, which is the difference between
        the optimal expected value and the bandits expected value.
        Will be array-like, shape [num_bandits,].
        """
        expected_values = self.expected_value_per_bandit()
        best_value = max(expected_values)
        
        return [best_value - bandit.expected_value() for bandit in self._bandits]

class Bandit(ABC):
    """
    A bandit arm that can be selected for some reward drawn from a distribution.
    
    Contains its own random, so the ith pull of an instance will result in the
    same reward, no matter the order + how many times other bandits have been pulled.\
    
    Parameters
    --------
    seed
        Seed to use for random.seed.
    """
    def __init__(self, seed = None):
        self.random = random.Random()
        self.regenerate(seed)
        
    def regenerate(self, seed = None):
        """
        Will re-seed random and then update the random state.
        
        Parameters
        --------
        seed
            Seed to use for random.seed.
        """
        self.random.seed(seed)
        
        self._random_state = self.random.getstate()

    def reset(self):
        """
        Reset random to its intial state before this bandit was pulled, so the
        ith pull will return the same reward.
        """
        self.random.setstate(self._random_state)
    
    @abstractmethod
    def pull(self):
        """
        Return a reward drawn from this bandits distribution.
        """
        pass
    
    @abstractmethod
    def expected_value(self):
        """
        Return the expected value for this bandit.
        """
        pass
    
    @abstractmethod
    def describe(self):
        """
        Returns a formatted string to describe the implementation + parameters
        for this bandit.
        """
        pass

class PercentageBandit(Bandit):
    """
    A bandit that returns success_reward with chance success_chance and
    returns otherwise returns fail_reward (chance 1 - success_chance).
    
    Parameters
    --------
    success_chance : float [0, 1]
        Chance for the bandit to return the success_reward.
    
    success_reward : numeric
        Reward returned on success.
        
    fail_reward : numeric
        Reward returned on fail.
    """
    def __init__(self, success_chance, success_reward=1, fail_reward=0):
        super().__init__()
        self._success_chance = success_chance
        self._success_reward = success_reward
        self._fail_reward = fail_reward
    
    def pull(self):
        """
        Return a reward drawn from this bandits distribution.
        """
        if self.random.random() < self._success_chance:
            return self._success_reward
        return self._fail_reward
    
    def expected_value(self):
        """
        Return the expected value for this bandit.
        """
        return (self._success_chance * self._success_reward +
                (1 - self._success_chance) * self._fail_reward)
    
    def describe(self):
        """
        Returns a formatted string to describe the implementation + parameters
        for this bandit.
        """
        return "PercentageBandit ({}%), reward {} fail {}".format(self._success_chance,
                                 self._success_reward, self._fail_reward)

