import numpy as np
import matplotlib.pyplot as plt

def GraphCumulativeRegret(num_steps, algorithm_regret_by_step, approaches):
    display_every = int(num_steps / 100)
    plt.figure()
    for algo in range(len(approaches)):
        mean_regret_by_step = np.mean(
                np.cumsum(algorithm_regret_by_step[algo], axis=0), axis=1)
        plt.plot(range(0, num_steps, display_every),
                 mean_regret_by_step[::display_every], label=approaches[algo].describe())
    
    plt.legend(fontsize=8)
    plt.title("Mean Bandits Cumulative Regret")
    plt.show()

def GraphMeanRegret(num_steps, algorithm_regret_by_step, approaches):
    display_every = int(num_steps / 100)
    plt.figure()
    for algo in range(len(approaches)):
        mean_regret_by_step = np.mean(algorithm_regret_by_step[algo], axis=1)
        plt.plot(range(0, num_steps, display_every),
                 mean_regret_by_step[::display_every], label=approaches[algo].describe())
    
    plt.legend(fontsize=8)
    plt.title("Bandits Mean Regret")
    plt.show()

def GraphMeanPercentChoseBestAtStep(num_steps, algorithm_regret_by_step, approaches):
    display_every = int(num_steps / 100)
    plt.figure()
    for algo in range(len(approaches)):
        precent_correct_by_step = np.mean(np.array(algorithm_regret_by_step[algo]) == 0.0, axis=1)
        plt.plot(range(0, num_steps, display_every),
                  precent_correct_by_step[::display_every], label=approaches[algo].describe())
    plt.axis([1, num_steps, 0.0, 1.0])
    plt.legend(fontsize=8)
    plt.title("Bandits Percent Chose Best at Step")
    plt.show()