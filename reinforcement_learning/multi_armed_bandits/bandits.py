import random
import numbers

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
        All of the possible bandits that can be selected.
    """
    def __init__(self, bandits):
        self._bandits = bandits
        
        self._regrets = self.regret_per_action()
        
    def regenerate(self):
        """
        Will regenerate the random seed for each of the bandits. Can be used to
        allow running the algorithms again with slightly different rewards.
        """
        for bandit in self._bandits:
            bandit.regenerate()
        self._regrets = self.regret_per_action()
    
    def reset(self):
        """
        Will reset the random state for all of the bandits. Should be used between
        running different algorithms on the bandits to ensure there are no differences
        between rewards provided.
        """
        for bandit in self._bandits:
            bandit.reset()
        self._regrets = self.regret_per_action()
    
    def pull_bandit(self, bandit_chosen):
        """
        Pull the selected bandit, returning the result of pulling that bandit once.
        """
        regret = self._regrets[bandit_chosen]
        reward = self._bandits[bandit_chosen].pull()
        
        # If the expected_value for all bandits has changed, will NEED to
        # regenerate everything - best may have changed.
        if self._bandits[bandit_chosen].expected_value_changed():
            self._regrets = self.regret_per_action()
        
        return reward, regret
    
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
        
        It is recommended to use the regret return from pull_bandit to calculate regret,
        because the true regret per action can change, but the list returned by
        this function will not be updated.
        """
        expected_values = self.expected_value_per_bandit()
        best_value = max(expected_values)
        
        return [best_value - bandit.expected_value() for bandit in self._bandits]


class Bandit(ABC):
    """
    A bandit arm that can be selected for some reward drawn from a distribution.
    
    Contains its own random, so the ith pull of an instance will result in the
    same reward, no matter the order + how many times other bandits have been pulled.
    
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
        
        self._on_regenerate()
        
    def _on_regenerate(self):
        """
        Hook for when regenerating the bandit.
        """
        pass

    def reset(self):
        """
        Reset random to its intial state before this bandit was pulled, so the
        ith pull will return the same reward.
        """
        self.random.setstate(self._random_state)
        
        self._on_reset()
        
    def _on_reset(self):
       """
       Hook for when resetting the bandit.
       """
       pass
    
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
    
    def expected_value_changed(self):
        """
        Has the expected value changed. Should only be called when the bandit
        was just pulled.
        """
        # By default, most don't change their expected value.
        return False
    
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


class NormalDistributionBandit(Bandit):
    """
    A bandit that returns a reward drawn from normal (Gaussian) distribution with mean u
    and standard deviation sigma.
    
    Parameters
    --------
    mean
        Mean value for normal distribution rewards are drawn from.
    
    sigma
        Standard deviation for normal distribution rewards are drawn from.
    """
    def __init__(self, mean, sigma):
        super().__init__()
        self._mean = mean
        self._sigma = sigma
    
    def pull(self):
        """
        Return a reward drawn from this bandits distribution.
        """
        return self.random.gauss(self._mean, self._sigma)
    
    def expected_value(self):
        """
        Return the expected value for this bandit.
        """
        return self._mean
    
    def describe(self):
        """
        Returns a formatted string to describe the implementation + parameters
        for this bandit.
        """
        return "NormalDistributionBandit Mean {} Std Dev {}".format(self._mean, self._sigma)


class KNormalDistributionBandits(MultiArmedBandit):
    """
    Specialization of MultiArmedBandit that will create K normal distribution
    bandits by drawing a mean from a normal distribution for each.
    """
    def __init__(self, K, mean_selection_sigma = 1, bandit_sigma = 1):
        # Setup bandits with fake means. Will change their means in regenerate
        bandits =  [NormalDistributionBandit(0, bandit_sigma)
                   for _ in range(K)]
        super().__init__(bandits)
        self._mean_selection_sigma = mean_selection_sigma
        
        self.regenerate()
    
    def regenerate(self):
        """
        Will regenerate the random seed + mean for each of the bandits. Can be used to
        allow running the algorithms again with slightly different rewards.
        """
        for bandit in self._bandits:
            bandit._mean = random.gauss(0, self._mean_selection_sigma)


class SequenceBandit(Bandit):
    """
    Will return values from the different sequences it is given. They can either be a
    number or another Bandit in which case pull, reset, regenerate, etc. will be
    called as necessary.
    
    Note that if repeat and expected_value_is_current are False, the expected value
    will essentially be just be the expeted value of the last item.
    
    Parameters
    --------
    sequence : array-like, shape [n_items,]
        Array of values to return from. Should be a tuple of (item, count).
        If item is a number, will return it without any alteration, otherwise,
        will treat it like a Bandit.
        Cannot be a tuple!
        
    repeat : boolean
        After reaching end of sequence, should it repeat the sequence. If False,
        will continue to return the last element in sequence.
        
    expected_value_is_current : boolean
        If true, expected value will be the expected value of current item in sequence.
        Otherwise, it will be the weighted mean of expected value for all elements.
    
    seed
        Seed to use for random.seed.
    """
    # Weight to use for last item in sequence if not repeating.
    _final_weight = 100000000
    
    def __init__(self, sequence, repeat, expected_value_is_current=True, seed=None):
        # Ensure the count of last element is huge
        if not repeat:
            # It can't be a tuple, since will be changed.
            if isinstance(sequence, tuple):
                sequence = [c for c in sequence]
            # Change the count of last item in sequence to be essentially infinite
            sequence[-1] = (sequence[-1][0], SequenceBandit._final_weight)
        
        self._sequence = sequence
        
        super().__init__(seed=seed)
        
        self._current_return = 0
        self._current_return_count = 0
        
        self._expected_value_is_current = expected_value_is_current
    
    def pull(self):
        current = self._sequence[self._current_return]
        item = current[0]
        
        # Increase and check count
        self._current_return_count += 1
        if self._current_return_count >= current[1]:
            self._current_return += 1
            self._current_return_count = 0
            
            if self._current_return >= len(self._sequence):
                # Can assume wants repeat, otherwise would be infinite.
                self._current_return = 0
        
        # Return the item.
        if self._is_numeric(item):
            return item
        else:
            return item.pull()
        
    def expected_value_changed(self):
        # Only changed if we are reporting the current and it changed.
        return self._expected_value_is_current and self._current_return_count == 0
                
    def expected_value(self):
        # If only relying on current expected, then should just return the current
        # items expected value.
        if self._expected_value_is_current:
            current = self._sequence[self._current_return]
            item = current[0]
            if not self._is_numeric(item):
                item = item.expected_value()
            return item
        
        total = 0
        count = 0
        
        for current in self._sequence:
            item = current[0]
            if not self._is_numeric(item):
                item = item.expected_value()
                
            c = current[1]
            total += item * c
            count += c
        
        return total / count
    
    def describe(self):
        described_sequence = []
        for current in self._sequence:
            item = current[0]
            if not self._is_numeric(item):
                item = item.describe()
            
            described_sequence.append([item, current[1]])
        
        return "SequenceBandit {}".format(described_sequence)
    
    def _is_numeric(self, item_in_sequence):
        """
        Returns true if the item is a bandit, and should be pulled.
        Otherwise, assumes it is a number and returns normally.
        """
        return isinstance(item_in_sequence, numbers.Number)
    
    def _on_regenerate(self):
        """
        Need to make sure that any bandits this contains are reset properly.
        """
        self._current_return = 0
        self._current_return_count = 0
        
        for current in self._sequence:
            item = current[0]
            if not self._is_numeric(item):
                item.regenerate()
    
    def _on_reset(self):
        """
        Need to make sure that any bandits this contains are reset properly.
        """
        self._current_return = 0
        self._current_return_count = 0
        
        for current in self._sequence:
            item = current[0]
            if not self._is_numeric(item):
                item.reset()
