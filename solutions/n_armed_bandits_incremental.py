# same as "n_armed_bandits_softmax", but using incremental action values
# instead of averaging cumulative rewards to determine reward estimates

# PARAMS (via command-line)
# @ temperatures -> array of temperature hyperparameters
#       entered as comma-deliniated values
#       default value -> [0.01, 0.03, 0.1]
#       N.B. this must be a positive value

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from util.iter_count import IterCount
from util.cmap import colormap
from .solution_util import softmax

import settings

config = settings.n_armed_bandits

def learn(temperature):

    numBandits, numArms, numPlays = (config['numBandits'], config['numArms'], config['numPlays'])

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

    print('Running with settings:')
    print('\tnumBandits: {0}\tnumArms: {1}\tnumPlays: {2}\n'.format(config['numBandits'], config['numArms'], config['numPlays']))

    fig = plt.figure(1, (8,8))
    reward_plot = fig.add_subplot(211)
    optimal_plot = fig.add_subplot(212)

    cmap = colormap(len(temperatures))
    for i, temperature in enumerate(temperatures):
        print('Learning with temperature = {}'.format(temperature))
        rewards, isOptimal = learn(temperature)
        reward_plot.plot(range(config['numPlays']), np.mean(rewards, axis=0), c=cmap(i))
        optimal_plot.plot(range(config['numPlays']), np.mean(isOptimal, axis=0) * 100, c=cmap(i))

    reward_plot.legend(['Temperature: {}'.format(t) for t in temperatures])
    optimal_plot.legend(['Temperature: {}'.format(t) for t in temperatures])

    yticks = mtick.FormatStrFormatter('%.0f%%')
    optimal_plot.yaxis.set_major_formatter(yticks)
    plt.show()
