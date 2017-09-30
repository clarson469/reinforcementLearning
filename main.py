import sys
from solutions import *

def main():
    arg_len = len(sys.argv)
    if arg_len == 1:
        print('Enter solution to run...')
        return

    if sys.argv[1] == 'n_armed_bandits':
        if arg_len == 2:
            epsilons = [0, 0.01, 0.1, 1.0]
        else:
            epsilons = [float(e) for e in sys.argv[2].split(',')]
        n_armed_bandits.run(epsilons)

    elif sys.argv[1] == 'n_armed_bandits_softmax':
        if arg_len == 2:
            temperatures = [0.1, 0.5, 1.0]
        else:
            temperatures = [float(t) for t in sys.argv[2].split(',')]
        n_armed_bandits_softmax.run(temperatures)

if __name__ == '__main__':
    main()
