import numpy as np

def softmax(estimates, temperature):
    probs = np.exp(estimates / temperature) / np.sum(np.exp(estimates / temperature), axis=0)
    return np.argmax(probs, axis=1)
