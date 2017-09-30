# Add base directory of project to path.
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../..")

from reinforcement_learning.explore_exploit.estimator import AverageEstimator
from reinforcement_learning.explore_exploit.greedy_variants import GreedyExploreExploit, EpsilonGreedyExploreExploit, GreedyOptimisticInitExploreExploit
from reinforcement_learning.explore_exploit.ucb import UCB
from reinforcement_learning.multi_armed_bandits.util import run_multiple_times
from reinforcement_learning.multi_armed_bandits.bandits import MultiArmedBandit, PercentageBandit
from reinforcement_learning.multi_armed_bandits.graphing import GraphCumulativeRegret, GraphMeanRegret, GraphMeanPercentChoseBestAtStep

def main(num_runs=20, num_steps=500):
    bandits = MultiArmedBandit([PercentageBandit(0.5), PercentageBandit(0.75), PercentageBandit(0.72)])
    
    actions = bandits.actions()
    
    approaches = [GreedyExploreExploit(actions, AverageEstimator()),
                 EpsilonGreedyExploreExploit(0.1, actions, AverageEstimator()),
                 GreedyOptimisticInitExploreExploit(actions, AverageEstimator(), 1, 2),
                 GreedyOptimisticInitExploreExploit(actions, AverageEstimator(), 1, 10),
                 GreedyOptimisticInitExploreExploit(actions, AverageEstimator(), 1, 50),
                 UCB(actions, AverageEstimator()),]
    
    algorithm_total_rewards, algorithm_regret_by_step =\
        run_multiple_times(num_runs, num_steps, bandits, approaches)
    
    GraphCumulativeRegret(num_steps, algorithm_regret_by_step, approaches)
    GraphMeanRegret(num_steps, algorithm_regret_by_step, approaches)
    GraphMeanPercentChoseBestAtStep(num_steps, algorithm_regret_by_step, approaches)

if __name__ == "__main__":
    main(num_runs=1000, num_steps=2000)