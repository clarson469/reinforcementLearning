import numpy as np
import matplotlib.pyplot as plt
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

        bandits += np.random.random(bandits.shape) * 2 - 1

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


def run(action_value, alphas):

    print('Running with settings:')
    print('\tnumBandits: {0}\tnumArms: {1}\tnumPlays: {2}\n'.format(config['numBandits'], config['numArms'], config['numPlays']))

    temperature = 0.5
    if action_value == 'sample-average':
        print('Learning with sample-average action value estimate')
        rewards, isOptimal = learn(action_value, None, temperature)
        plt.plot(range(config['numPlays']), np.mean(rewards, axis=0), '-b')
        plt.title('Sample Average Action Value Estimates')
        plt.show()
    else:
        cmap = plt.cm.get_cmap('jet', len(alphas))
        for i, alpha in enumerate(alphas):
            print('Learning with alpha = {}'.format(alpha))
            rewards, isOptimal = learn(action_value, alpha, temperature)
            plt.plot(range(config['numPlays']), np.mean(rewards, axis=0), c=cmap(i))

        plt.title('NonStationary Action Value Estimate')
        plt.legend(['Alpha: {}'.format(a) for a in alphas])
        plt.show()
