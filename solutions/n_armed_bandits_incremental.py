# same as "n_armed_bandits_softmax", but using incremental action values
# instead of averaging cumulative rewards to determine reward estimates

# PARAMS (via command-line)
# @ temperatures -> array of temperature hyperparameters
#       entered as comma-deliniated values
#       default value -> [0.1, 0.5, 1.0]
#       N.B. this must be a positive value

import numpy as np
import matplotlib.pyplot as plt
from util.iter_count import IterCount
from .solution_util import softmax

def learn(temperature):

    numBandits, numArms, numPlays = (2000, 10, 1000)

    bandits = np.random.normal(0, 1, (numBandits, numArms))
    best = np.argmax(bandits, axis=1)

    activated, estimates = [np.zeros(bandits.shape) for i in range(2)]

    rewards, isOptimal = [np.zeros((numBandits, numPlays)) for i in range(2)]

    ic = IterCount('Play number {0} of {1}'.format("{}", numPlays))

    for i in range(numPlays):

        ic.update()

        arm = softmax(estimates, temperature)

        isOptimal[:, i][arm == best] = 1
        reward = np.random.normal(0, 1, numBandits) + bandits[range(numBandits), arm]
        rewards[:, i] = reward

        activated[range(numBandits), arm] += 1
        estimates[range(numBandits), arm] += (reward - estimates[range(numBandits), arm]) / activated[range(numBandits), arm]

    ic.exit()

    return rewards, isOptimal


def run(temperatures):
    cmap = plt.cm.get_cmap('jet', len(temperatures))
    for i, temperature in enumerate(temperatures):
        print('Learning with temperature = {}'.format(temperature))
        rewards, isOptimal = learn(temperature)
        plt.plot(range(1000), np.mean(rewards, axis=0), c=cmap(i))

    plt.legend(['Temperature: {}'.format(t) for t in temperatures])
    plt.show()
