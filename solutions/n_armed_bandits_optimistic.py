# modifies the N-armed Bandit problem, by introducing a variable initial estimate

# PARAMS (via command-line)
# @ init_values -> array of initial value estimate hyperparameters
#       entered as comma-deliniated values
#       default -> [0.0, 1.0, 5.0]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from util.iter_count import IterCount
from util.cmap import colormap
from .solution_util import softmax

import settings

config = settings.n_armed_bandits

def learn(temperature, alpha, init_value):

    numBandits, numArms, numPlays = (config['numBandits'], config['numArms'], config['numPlays'])

    bandits = np.random.normal(0, 1, (numBandits, numArms))

    best = np.argmax(bandits, axis=1)

    estimates = np.zeros(bandits.shape) + init_value

    rewards, isOptimal = [np.zeros((numBandits, numPlays)) for i in range(2)]

    ic = IterCount('Play number {0} of {1}'.format("{}", numPlays))

    for i in range(numPlays):

        ic.update()

        arm = softmax(estimates, temperature)

        isOptimal[:, i][arm == best] = 1
        reward = np.random.normal(0, 1, numBandits) + bandits[range(numBandits), arm]
        rewards[:, i] = reward

        estimates[range(numBandits), arm] *= 1 - alpha
        estimates[range(numBandits), arm] += alpha * reward

    ic.exit()

    return rewards, isOptimal

def run(init_values):
    temperature, alpha = 0.1, 0.1

    fig = plt.figure(1, (8,8))
    reward_plot = fig.add_subplot(211)
    optimal_plot = fig.add_subplot(212)

    cmap = colormap(len(init_values))

    for i, init_value in enumerate(init_values):
        print('Learning with init_value = {}'.format(init_value))
        rewards, isOptimal = learn(temperature, alpha, init_value)
        reward_plot.plot(range(config['numPlays']), np.mean(rewards, axis=0), c=cmap(i))
        optimal_plot.plot(range(config['numPlays']), np.mean(isOptimal, axis=0) * 100, c=cmap(i))

    reward_plot.legend(['Initial Value: {}'.format(i) for i in init_values])
    optimal_plot.legend(['Initial Value: {}'.format(i) for i in init_values])

    yticks = mtick.FormatStrFormatter('%.0f%%')
    optimal_plot.yaxis.set_major_formatter(yticks)

    plt.show()
