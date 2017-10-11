# solution variables
n_armed_bandits = {
    'numBandits': 2000,
    'numArms': 10,
    'numPlays': 1000,
    # alpha must satisfy 0 < alpha <= 1
    'alpha': 0.1,
    # beta must satisfy 0 < beta <= 1
    'beta': 0.1,
    # init_ref can be any value, but because it gets exponentiated, don't make it too high
    'init_ref': 0
}

# general variables
colors = ['00ffff', 'ff99ff', '9999ff']
