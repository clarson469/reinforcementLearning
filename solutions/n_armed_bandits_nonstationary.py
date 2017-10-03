# modifies the N-armed Bandits problem into a non-stationary one
# that is, where the reward value of the bandit arms shifts randomly
#
# compares the use of a sample-average estimate method (used in previous soutions)
# with a non-stationary one, that weights more recent information / rewards
# more heavily than less recent

# PARAMS (via command-line)
# @ alphas -> array of alpha hyperparameters
#       entered as comma-deliniated values
#       default value -> [0.01, 0.1, 1.0]
#       N.B. values must satisfy 0 < a <= 1

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from util.iter_count import IterCount
from .solution_util import softmax

from settings import n_armed_bandits as config

def learn(action_value, alpha, temperature):

    numBandits, numArms, numPlays = (config['numBandits'], config['numArms'], config['numPlays'])

    bandits = np.random.normal(0, 1, (numBandits, numArms))

    if action_value == 'sample-average':
        activated, estimates = [np.zeros(bandits.shape) for i in range(2)]
    else:
        estimates = np.zeros(bandits.shape)

    rewards, isOptimal = [np.zeros((numBandits, numPlays)) for i in range(2)]

    ic = IterCount('Play numer {0} of {1}'.format("{}", numPlays))

    for i in range(numPlays):

        ic.update()

        bandits += np.random.normal(0, 0.2, (bandits.shape))

        best = np.argmax(bandits, axis=1)

        arm = softmax(estimates, temperature)

        isOptimal[:, i][arm == best] = 1
        reward = np.random.normal(0, 1, numBandits) + bandits[range(numBandits), arm]
        rewards[:, i] = reward

        if action_value == 'sample-average':
            activated[range(numBandits), arm] += 1
            estimates[range(numBandits), arm] += (reward - estimates[range(numBandits), arm]) / activated[range(numBandits), arm]

        else:
            estimates[range(numBandits), arm] *= 1 - alpha
            estimates[range(numBandits), arm] += alpha * reward

    ic.exit()

    return rewards, isOptimal


def run(alphas):

    print('Running with settings:')
    print('\tnumBandits: {0}\tnumArms: {1}\tnumPlays: {2}\n'.format(config['numBandits'], config['numArms'], config['numPlays']))

    fig = plt.figure(1, (8,8))
    reward_plot = fig.add_subplot(211)
    optimal_plot = fig.add_subplot(212)

    temperature = 0.1
    cmap = plt.cm.get_cmap('jet', len(alphas) + 1)

    print('Learning with sample-average action value estimate')
    rewards, isOptimal = learn('sample-average', None, temperature)
    reward_plot.plot(range(config['numPlays']), np.mean(rewards, axis=0), c=cmap(0))
    optimal_plot.plot(range(config['numPlays']), np.mean(isOptimal, axis=0) * 100, c=cmap(0))

    print('Learning with nonstationary action value estimate')
    for i, alpha in enumerate(alphas):
        print('Learning with alpha = {}'.format(alpha))
        rewards, isOptimal = learn('nonstationary', alpha, temperature)
        reward_plot.plot(range(config['numPlays']), np.mean(rewards, axis=0), c=cmap(i+1))
        optimal_plot.plot(range(config['numPlays']), np.mean(isOptimal, axis=0) * 100, c=cmap(i+1))

    reward_plot.legend(['Sample-Average'] + ['Alpha: {}'.format(a) for a in alphas])
    optimal_plot.legend(['Sample-Average'] + ['Alpha: {}'.format(a) for a in alphas])

    yticks = mtick.FormatStrFormatter('%.0f%%')
    optimal_plot.yaxis.set_major_formatter(yticks)
    plt.show()
