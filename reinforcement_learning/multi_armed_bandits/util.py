import numpy as np
import copy
from multiprocessing import Pool as ThreadPool

_TOTAL_REWARD = 0
_ALL_REGRETS = 1

def run_multiple_times(num_times, num_steps, bandit, algorithms):
    """
    What do we return? Total rewards, % equal to best at each step, regret by step
    """
    pool = None
    try:
        pool = ThreadPool()
    except Exception:
        print("Unable to create ThreadPool, will be single-threaded.")
        pass # Won't do multi-threading then.
    
    if pool is not None:
        results = pool.map(run_and_combine_results,
                           [(num_steps, copy.deepcopy(bandit), copy.deepcopy(algorithms))
                            for _ in range(num_times)])
        pool.close()
    else:
        results = [run_multiple_algorithms(num_steps, bandit, algorithms) for _ in range(num_times)]
        
    algorithm_total_rewards = [[] for algorithm in algorithms]
    algorithm_regret_by_step = [[[] for step in range(num_steps)] for algorithm in algorithms]
    for results_for_run in results:
        for algo in range(len(algorithms)):
            regrets_of_actions = results_for_run[algo][_ALL_REGRETS]
            
            algorithm_total_rewards[algo].append(results_for_run[algo][_TOTAL_REWARD])
            
            for step in range(num_steps):
                algorithm_regret_by_step[algo][step].append(regrets_of_actions[step])
    
    return (algorithm_total_rewards, algorithm_regret_by_step)


def run_and_combine_results(args):
    num_steps, bandit, algorithms = args
    return run_multiple_algorithms(num_steps, bandit, algorithms)

def run_multiple_algorithms(num_steps, bandit, algorithms):
    bandit.regenerate()
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
    total_reward = 0
    
    regret_by_action = bandit.regret_per_action()
    regrets = []
    for _ in range(num_steps):
        action = algorithm.select_action()
        reward = bandit.pull_bandit(action)
        algorithm.update_from_action(action, reward)
        
        total_reward += reward
        regrets.append(regret_by_action[action])
    
    return [total_reward, regrets]