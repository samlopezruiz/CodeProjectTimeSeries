from copy import deepcopy
from algorithms.gpregress.math import G, score_error, Xmat_from_inputX, derive_a
import math
import numpy as np


class Primitives:
    def __init__(self):
        self.primitives = []
        self.attrs = []
        self.names = []
        self.num = 0

    def add(self, primitive, atr, name=None):
        if name is None:
            if hasattr(primitive, '__name__'):
                name = primitive.__name__
            else:
                name = 'op' + str(len(self.primitives))

        self.primitives.append(primitive)
        self.attrs.append(atr)
        self.names.append(name)
        self.num += 1

    def get(self, i):
        return self.primitives[i], self.attrs[i], self.names[i]

    def get_rand(self):
        i = np.random.randint(self.num)
        return self.primitives[i], self.attrs[i], self.names[i]


def generate(t, depth, n_terminals, primitives, n=3, id=0):
    if depth < 1:
        return
    elif depth == 1:
        t.value = np.random.randint(0, n_terminals)
        t.right, t.left = None, None
        t.depth = 1
        t.n_nodes = 0
    else:
        i = np.random.randint(n)
        if i == 0:
            t.value = SumNode(id=id)
            t.left = Tree(parent=t)
            t.right = Tree(parent=t)
            id = generate(t.left, depth - 1, n_terminals, primitives, id=id + 1)
            id = generate(t.right, depth - 1, n_terminals, primitives, id=id + 1)
        elif i == 1:
            op, atr, name = primitives.get_rand()
            t.value = Node(op, atr, name, id=id)
            t.left = Tree(parent=t)
            id = generate(t.left, depth - 1, n_terminals, primitives, id=id + 1)
            if atr == 2:
                t.right = Tree(parent=t)
                id = generate(t.right, depth - 1, n_terminals, primitives, id=id+1)
            else:
                t.right = None
            t.n_nodes = (t.right.n_nodes if atr == 2 else 0) + t.left.n_nodes + 1
        else:
            t.value = np.random.randint(0, n_terminals)
            t.right, t.left = None, None
            t.n_nodes = 0
    return id


class SumNode:

    def __init__(self, a=None, id=1):
        if a is None:
            # vals = np.random.rand(6)
            # signs = (np.random.rand(6) > 0.5).astype(int) * 2 - 1
            # self.a = vals * signs
            self.a = None
            self.mse = np.nan
            self.mdl = np.nan
            self.id = id
        else:
            self.a = a

    def eval(self, x1, x2):
       return G(x1, x2, self.a)

    def update(self, x1, x2, y, n_nodes):
        self.y_pred = self.eval(x1, x2)
        self.mse = score_error(y, self.y_pred)
        tree_coding_length = 0.5 * n_nodes * 6 * math.log(len(x1))
        exception_coding_length = 0.5 * len(x1) * math.log(self.mse)
        self.mdl = tree_coding_length + exception_coding_length

    def get_eval(self):
        return self.y_pred


class Node:

    def __init__(self, op, atr, name, id=1):
        self.mse = np.nan
        self.mdl = np.nan
        self.id = id
        self.atr = atr
        self.name = name
        self.op = op
        self.y_pred = None

    def eval(self, x1, x2):
        if self.atr == 1:
            return self.op(x1)
        else:
            return self.op(x1, x2)

    def update(self, x1, x2, y, n_nodes):
        self.y_pred = self.eval(x1, x2)
        self.mse = score_error(y, self.y_pred)
        tree_coding_length = 0.5 * n_nodes * 6 * math.log(len(x1))
        exception_coding_length = 0.5 * len(x1) * math.log(self.mse)
        self.mdl = tree_coding_length + exception_coding_length

    def get_eval(self):
        return self.y_pred


class Tree:

    def __init__(self, parent=None):
        self.depth = None
        self.value = None
        self.right = None
        self.left = None
        self.n_nodes = 0
        self.parent = parent
        self.level = 0


class Individual:

    def __init__(self, n_terminals, primitives, depth=5, n=1, id=0):
        tree = Tree()
        generate(tree, depth=depth, n_terminals=n_terminals, primitives=primitives, n=n, id=id)
        self.tree = deepcopy(tree)
        self.nodes = []
        self.mses = []
        self.mdls = []
        self.ids = []
        self.levels = []
        self.depth = 0
        self.get_lists(self.tree)

    def update_lists(self):
        self.nodes = []
        self.mses = []
        self.mdls = []
        self.ids = []
        self.levels = []
        self.get_lists(self.tree)
        self.update_n_nodes_(self.tree)
        self.depth = max(self.levels)

    def get_mdl(self):
        # first element is root
        return self.mdls[0]

    def get_n_nodes(self):
        # number of all nodes
        return len(self.nodes)

    def update_n_nodes(self):
        self.update_n_nodes_(self.tree)

    def update_n_nodes_(self, t):
        n = t.value
        if isinstance(n, SumNode):
            t.n_nodes = self.update_n_nodes_(t.left) + self.update_n_nodes_(t.right)
            return t.n_nodes
        elif isinstance(n, Node):
            if n.atr == 2:
                t.n_nodes = self.update_n_nodes_(t.left) + self.update_n_nodes_(t.right)
            else:
                t.n_nodes = self.update_n_nodes_(t.left)
            return t.n_nodes
        elif n is None:
            print("ERROR: node is None")
        else:
            return 1

    def get_lists(self, t, level=1):
        n = t.value
        t.level = level
        self.levels.append(level)
        self.nodes.append(n)
        if isinstance(n, SumNode):
            self.mses.append(n.mse)
            self.mdls.append(n.mdl)
            self.ids.append(n.id)
            self.get_lists(t.left, level+1)
            self.get_lists(t.right, level+1)
        elif isinstance(n, Node):
            self.mses.append(np.nan) #(n.mse)
            self.mdls.append(np.nan) #(n.mdl)
            self.ids.append(np.nan) #(n.id)
            self.get_lists(t.left, level + 1)
            if n.atr == 2:
                self.get_lists(t.right, level + 1)
        elif n is None:
            print("ERROR: node is None")
        else:
            self.ids.append(np.nan)
            self.mses.append(np.nan)
            self.mdls.append(np.nan)

    def print_tree(self):
        self.print_tree_(self.tree)

    def print_tree_(self, t, level=0):
        if isinstance(t.value, SumNode):
            print("  " * level + 'SumNode'+str(t.value.id))
            self.print_tree_(t.left, level + 1)
            self.print_tree_(t.right, level + 1)
        elif isinstance(t.value, Node):
            print("  " * level + t.value.name+str(t.value.id))
            self.print_tree_(t.left, level + 1)
            if t.value.atr == 2:
                self.print_tree_(t.right, level + 1)
        else:
            print("  " * level + str(t.value))

    def gmdh(self, X, y):
        self.gmdh_process(self.tree, X, y)
        self.update_lists()

    def gmdh_process(self, t, X, y):
        nd = t.value
        if isinstance(nd, SumNode):
            if nd.a is None:
                nl = self.gmdh_process(t.left, X, y)
                nr = self.gmdh_process(t.right, X, y)
                nd.a = derive_a(Xmat_from_inputX(nl, nr), y)
                nd.update(nl, nr, y, t.n_nodes)
                return nd.get_eval()
            else:
                return nd.get_eval()
        if isinstance(nd, Node):
            if nd.y_pred is None:
                nl = self.gmdh_process(t.left, X, y)
                nr = None
                if nd.atr == 2:
                    nr = self.gmdh_process(t.right, X, y)
                nd.update(nl, nr, y, t.n_nodes)
                return nd.get_eval()
            else:
                return nd.get_eval()
        else:
            # x1 = X[:, nd]
            # print('terminal: ', nd,  x1.shape)
            return X[:, nd]

    def predict(self, X):
        return self.eval_tree(self.tree, np.array(X))

    def is_valid(self):
        return self.valid(self.tree)

    def valid(self, t):
        if isinstance(t.value, SumNode):
            if t.value.a is not None:
                if not self.valid(t.left):
                    return False
                if not self.valid(t.right):
                    return False
                return True
            else:
                return False
        elif isinstance(t.value, Node):
            if t.value.y_pred is not None:
                if not self.valid(t.left):
                    return False
                if t.value.atr == 2:
                    if not self.valid(t.right):
                        return False
                return True
            else:
                return False
        else:
            return True

    def predict(self, X):
        return self.eval_tree(self.tree, X)

    def eval_tree(self, t, X):
        if isinstance(t.value, SumNode):
            if t.value.a is not None:
                return t.value.eval(self.eval_tree(t.left, X), self.eval_tree(t.right, X))
                    # G(self.eval_tree(t.left, X), self.eval_tree(t.right, X), t.value.a)
            else:
                print('ERROR: node has null parameters')
                return
        elif isinstance(t.value, Node):
            if t.value.y_pred is not None:
                if t.value.atr == 1:
                    return t.value.eval(self.eval_tree(t.left, X), None)
                else:
                    return t.value.eval(self.eval_tree(t.left, X), self.eval_tree(t.right, X))
            else:
                print('ERROR: y_pred is null')
                return
        else:
            try:
                if len(X.shape) == 1:
                    return np.array([X[t.value]])
                else:
                    return X[:, t.value]
            except:
                print(X.shape)
                return None




