# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")

from reinforcement_learning.explore_exploit.estimator import AverageEstimator
from reinforcement_learning.explore_exploit.greedy_variants import GreedyExploreExploit, EpsilonGreedyExploreExploit, GreedyOptimisticInitExploreExploit
from reinforcement_learning.explore_exploit.ucb import UCB, UCB_V
from reinforcement_learning.multi_armed_bandits.util import run_multiple_times
from reinforcement_learning.multi_armed_bandits.bandits import MultiArmedBandit, PercentageBandit
from reinforcement_learning.multi_armed_bandits.graphing import GraphCumulativeRegret, GraphMeanRegret, GraphMeanPercentChoseBestAtStep

def main(num_runs=20, num_steps=500):
    bandits = MultiArmedBandit([PercentageBandit(0.5), PercentageBandit(0.70), PercentageBandit(0.65),
                                PercentageBandit(0.70), PercentageBandit(0.67), PercentageBandit(0.63),
                                PercentageBandit(0.98), PercentageBandit(0.71), PercentageBandit(0.75)])
    
    actions = bandits.actions()
    approaches = [GreedyExploreExploit(actions, AverageEstimator()),
                 EpsilonGreedyExploreExploit(0.1, actions, AverageEstimator()),
                 GreedyOptimisticInitExploreExploit(actions, AverageEstimator(), 1, 2),
                 GreedyOptimisticInitExploreExploit(actions, AverageEstimator(), 1, 10),
                 UCB(actions, AverageEstimator()),
                 UCB_V(actions, AverageEstimator()),]
    
    algorithm_total_rewards, algorithm_regret_by_step =\
        run_multiple_times(num_runs, num_steps, bandits, approaches)
    
    GraphCumulativeRegret(num_steps, algorithm_regret_by_step, approaches)
    GraphMeanRegret(num_steps, algorithm_regret_by_step, approaches)
    GraphMeanPercentChoseBestAtStep(num_steps, algorithm_regret_by_step, approaches)

if __name__ == "__main__":
    main(num_runs=1000, num_steps=2000)