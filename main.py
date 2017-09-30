import sys
from solutions import n_armed_bandits

def main():
    if len(sys.argv) == 1:
        print('Enter solution to run...')
        return

    if sys.argv[1] == 'n_armed_bandits':
        n_armed_bandits.run()

if __name__ == '__main__':
    main()
