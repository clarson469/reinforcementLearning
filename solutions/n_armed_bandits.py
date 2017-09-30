# simplest N-armed bandits problem / solution (see: Sutton, Barto Chapter 2.2)
# AIM given a "bandit" with n arms, where each arm yields some reward,
# learn to optimise the reward yielded by that bandit

# PARAMS (via command-line)
# @ epsilons -> array of epsilon hyperparameters
#       entered as comma-deliniated values
#       default value -> [0, 0.01, 0.1, 1]
#       N.B. values must be in range 0 <= n <= 1

import numpy as np
import matplotlib.pyplot as plt
from util.iter_count import IterCount

def learn(epsilon):
    numBandits, numArms, numPlays = (2000, 10, 1000)
    bandits = np.random.normal(0, 1, (numBandits, numArms))
    best = np.argmax(bandits, axis=1)
    estimates, activated, cumRewards = [np.zeros(bandits.shape) for i in range(3)]

    rewards = np.zeros((numBandits, numPlays))
    isOptimal = np.zeros(rewards.shape)

    ic = IterCount('Play number {0} of {1}'.format("{}", numPlays))

    for i in range(numPlays):

        ic.update()

        explore = np.zeros(numBandits)
        explore[np.random.random(numBandits) <= epsilon] = 1
        arm = np.argmax(estimates, axis=1)
        arm[explore == 1] = np.random.randint(0, numArms, np.count_nonzero(explore))

        isOptimal[:, i][arm == best] = 1
        reward = np.random.normal(0, 1, numBandits) + bandits[range(numBandits), arm]
        rewards[:, i] = reward

        activated[range(numBandits), arm] += 1
        cumRewards[range(numBandits), arm] += reward
        estimates[range(numBandits), arm] = cumRewards[range(numBandits), arm] / activated[range(numBandits), arm]

    ic.exit()

    return rewards, isOptimal


def run(epsilons):
    cmap = plt.cm.get_cmap('jet', len(epsilons))
    for i, epsilon in enumerate(epsilons):
        print('Learning with epsilon = {}'.format(epsilon))
        rewards, isOptimal = learn(epsilon)
        plt.plot(range(1000), np.mean(rewards, axis=0), c=cmap(i))

    plt.legend(['Epsilon: {}'.format(e) for e in epsilons])
    plt.show()
