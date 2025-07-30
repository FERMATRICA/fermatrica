"""
Optimize outer layer of the model.

Both global and local algorithms are supported. As for now two derivative-free algos are included:
1. COBYLA constrained optimization by linear approximations (local)
2. PyGad genetic algorithm (global)

However, FERMATRICA architecture allows fast and simple adding new algorithms, and some
algos could be added later.

Derivative-free approach allows optimising without calculating (analytical) gradient what could be
very complex and time-consuming. However, some derivative algo (e.g. GS) could be added later
at least for some most popular transformations.
"""


import fermatrica.optim.objective
import fermatrica.optim.globals
import fermatrica.optim.locals
import fermatrica.optim.pygad
