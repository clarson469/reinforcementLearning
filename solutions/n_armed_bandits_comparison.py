# approaches the n-armed bandit problem from a different angle,
# doing away with estimated action-values in favour of preferences based on
# a single reference reward (the average of _all_ received rewards)

# takes no direct parameters (though this may change)
# instead, modify the "n_armed_bandits" dict in "settings.py" to change the
# hyperparamters for this solution

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from util.iter_count import IterCount
from util.cmap import colormap
from .solution_util import pref_softmax

import settings

config = settings.n_armed_bandits

def learn(alpha, beta, init_ref):

    numBandits, numArms, numPlays = (config['numBandits'], config['numArms'], config['numPlays'])

    bandits = np.random.normal(0, 1, (numBandits, numArms))

    best = np.argmax(bandits, axis=1)

    reward_ref = np.zeros(numBandits) + init_ref
    preferences = np.zeros(bandits.shape)

    rewards, isOptimal = [np.zeros((numBandits, numPlays)) for i in range(2)]

    ic = IterCount('Play number {0} of {1}'.format("{}", numPlays))

    for i in range(numPlays):

        ic.update()

        arm = pref_softmax(preferences)

        isOptimal[:, i][arm == best] = 1
        reward = np.random.normal(0, 1, numBandits) + bandits[range(numBandits), arm]
        rewards[:, i] = reward

        reward_diff = reward - reward_ref

        preferences[range(numBandits), arm] += beta * reward_diff
        reward_ref += alpha * reward_diff

    ic.exit()

    return rewards, isOptimal

def run():

    print('Running with settings:')
    print('\tnumBandits: {0}\n\tnumArms: {1}\n\tnumPlays: {2}\n\talpha: {3}\n\tbeta: {4}\n\tinit_ref: {5}\n'.format(config['numBandits'], config['numArms'], config['numPlays'], config['alpha'], config['beta'], config['init_ref']))


    fig = plt.figure(1, (8,8))
    reward_plot = fig.add_subplot(211)
    optimal_plot = fig.add_subplot(212)

    print('Learning...')
    rewards, isOptimal = learn(config['alpha'], config['beta'], config['init_ref'])
    reward_plot.plot(range(config['numPlays']), np.mean(rewards, axis=0), c=(0,1,1))
    optimal_plot.plot(range(config['numPlays']), np.mean(isOptimal, axis=0) * 100, c=(0,1,1))

    yticks = mtick.FormatStrFormatter('%.0f%%')
    optimal_plot.yaxis.set_major_formatter(yticks)

    plt.show()
