from algorithms.gpregress.classes import Node, SumNode
from algorithms.gpregress.math import G, derive_a, Xmat_from_inputX
import numpy as np


def print_tree(t, level=0):
    if isinstance(t.value, SumNode):
        print("  " * level + 'SumNode')
        print_tree(t.left, level + 1)
        print_tree(t.right, level + 1)
    elif isinstance(t.value, Node):
        print("  " * level + t.value.name)
        print_tree(t.left, level + 1)
        if t.value.atr == 2:
            print_tree(t.right, level + 1)
    else:
        print("  " * level + str(t.value))


def eval_tree(t, X):
    if isinstance(t.value, Node):
        if t.value.a is not None:
            return G(eval_tree(t.left, X), eval_tree(t.right, X), t.value.a)
        else:
            print('ERROR: node has null parameters')
            return
    else:
        if len(X.shape) == 1:
            return np.array([X[t.value]])
        else:
            return X[:, t.value]


def mult_reg(x1, x2, y):
    return derive_a(Xmat_from_inputX(x1, x2), y)


def gmdh(t, X, y):
    nd = t.value
    if isinstance(nd, Node):
        if nd.a is not None:
            return
        else:
            nl = gmdh(t.left, X, y)
            nr = gmdh(t.right, X, y)
            nd.a = mult_reg(nl, nr, y)
            return nd.update(nl, nr, y, t.n_nodes)
    else:
        return X[:, nd]