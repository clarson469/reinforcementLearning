# reinforcementLearning
Solutions (such as they are) for Sutton and Barto's "Reinforcement Learning: An Introduction"

## Contains
 * Simplest N-armed Bandit example, as per Chapter 2.1 / 2.2

## Running
Run main.py, with the name of whichever solution you want to run as a command-line argument, plus any additional arguments the solution may require (see the solution files individually for those)

#### A Note on requirements.txt
This repo uses the NumPy and Scipy libraries, but specifically it uses numpy+mkl, which contains numpy and required DLLs for the Intel Math Kernel Library - basically, numpy and (probably) scipy should be installed from a pre-downloaded wheel rather than by just directing pip (or equivalent) to the requirements file. Those can be found here -- [numpy+mkl](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy) & [scipy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy).
