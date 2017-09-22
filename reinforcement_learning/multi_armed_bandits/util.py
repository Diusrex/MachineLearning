import numpy as np

def run_multiple_times(num_times, num_steps, bandit, algorithms):
    """
    What do we return? Total rewards, % equal to best at each step, regret by step
    """
    algorithm_total_rewards = [[] for algorithm in algorithms]
    algorithm_regret_by_step = [[[] for step in range(num_steps)] for algorithm in algorithms]
    
    # TODO: This should be multi-threaded - order doesn't matter, and would make it
    # far faster!
    for _ in range(num_times):
        bandit.regenerate()
        results_for_run = run_multiple_algorithms(num_steps, bandit, algorithms)
        
        regrets = bandit.regret_per_action()
        for algo in range(len(algorithms)):
            rewards = results_for_run[algo].rewards
            action_selected = results_for_run[algo].actions
            
            algorithm_total_rewards[algo].append(sum(rewards))
            
            for step in range(num_steps):
                action = action_selected[step]
                algorithm_regret_by_step[algo][step].append(regrets[action])
    
    return (algorithm_total_rewards, algorithm_regret_by_step)

def run_multiple_algorithms(num_steps, bandit, algorithms):
    results = []
    for algorithm in algorithms:
        bandit.reset()
        algorithm.reset()
        results.append(run(num_steps, bandit, algorithm))
    
    return results

def run(num_steps, bandit, algorithm):
    """
    TODO - Comments
    bandit - has try 
    """
    rewards = []
    actions = []
    for _ in range(num_steps):
        action = algorithm.select_action()
        reward = bandit.pull_bandit(action)
        algorithm.update_from_action(action, reward)
        
        rewards.append(reward)
        actions.append(action)
    
    class Wrapper(object):
        def __init__(self, rewards, actions):
            self.rewards = rewards
            self.actions = actions
    
    return Wrapper(rewards, actions)