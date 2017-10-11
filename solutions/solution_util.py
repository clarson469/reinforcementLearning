import numpy as np

def e_greedy(estimates, epsilon):
    numBandits, numArms = estimates.shape

    explore = np.zeros(numBandits)
    explore[np.random.random(numBandits) <= epsilon] = 1
    arm = np.argmax(estimates, axis=1)
    arm[explore == 1] = np.random.randint(0, numArms, np.count_nonzero(explore))

    return arm

def softmax(estimates, temperature):
    probs = (np.exp(estimates / temperature).T / np.sum(np.exp(estimates / temperature), axis=1)).T
    cumProb = probs.cumsum(axis=1)
    return (np.random.random(estimates.shape) < cumProb).argmax(axis=1)

def pref_softmax(preferences):
    probs = (np.exp(preferences).T / np.sum(np.exp(preferences), axis=1)).T
    cumProb = probs.cumsum(axis=1)
    return (np.random.random(preferences.shape) < cumProb).argmax(axis=1)
