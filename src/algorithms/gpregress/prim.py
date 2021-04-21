import numpy as np
from algorithms.gpregress.classes import Primitives
from algorithms.gpregress.math import protected_div, protected_sqrt

def sq(x):
    return np.power(x, 2)

def primitives1():
    primitives = Primitives()
    primitives.add(np.multiply, 2)
    primitives.add(protected_div, 2, 'div')
    primitives.add(protected_sqrt, 1, 'sqrt')
    primitives.add(np.cos, 1)
    primitives.add(np.sin, 1)
    primitives.add(sq, 1, 'sq')
    return primitives