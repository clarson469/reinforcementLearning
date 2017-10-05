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

    elif sys.argv[1] == 'n_armed_bandits_incremental':
        if arg_len == 2:
            temperatures = [0.1, 0.5, 1.0]
        else:
            temperatures=  [float(t) for t in sys.argv[2].split(',')]
        n_armed_bandits_incremental.run(temperatures)

    elif sys.argv[1] == 'n_armed_bandits_nonstationary':
        if arg_len == 2:
            alphas = [0.01, 0.1, 1.0]
        else:
            alphas = [float(a) for a in sys.arv[2].split(',')]
        n_armed_bandits_nonstationary.run(alphas)

    elif sys.argv[1] == 'n_armed_bandits_optimistic':
        if arg_len == 2:
            init_values = [0.0, 1.0, 5.0]
        else:
            init_values = [float(i) for i in sys.argv[2].split(',')]
        n_armed_bandits_optimistic.run(init_values)

    elif sys.argv[1] == 'n_armed_bandits_comparison':
        n_armed_bandits_comparison.run()


if __name__ == '__main__':
    main()
