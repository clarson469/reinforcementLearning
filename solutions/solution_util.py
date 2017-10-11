import numpy as np

def e_greedy(estimates, epsilon):
    numBandits, numArms = estimates.shape

    explore = np.zeros(numBandits)
    explore[np.random.random(numBandits) <= epsilon] = 1
    arm = np.argmax(estimates, axis=1)
    arm[explore == 1] = np.random.randint(0, numArms, np.count_nonzero(explore))

    return arm

def softmax(estimates, temperature):
    temp_est = estimates.T / temperature
    exponents = np.exp(temp_est - np.max(temp_est))
    dist = exponents / np.sum(exponents, axis=0)

    return (np.random.random(temp_est.shape) < dist.cumsum(axis=0)).argmax(axis=0)

def pref_softmax(preferences):
    pref = preferences.T
    exponents = np.exp(pref - np.max(pref))
    dist = exponents / np.sum(exponents, axis=0)

    return (np.random.random(pref.shape) < dist.cumsum(axis=0)).argmax(axis=0)
