from copy import deepcopy
from algorithms.stroganoff.math import G, score_error, Xmat_from_inputX, derive_a
import math
import numpy as np


def generate(t, depth, n_terminals, p=0.5, id=0):
    if depth < 1:
        return
    elif depth == 1:
        t.value = np.random.randint(0, n_terminals)
        t.right, t.left = None, None
        t.depth = 1
        t.n_nodes = 0
    else:
        if np.random.rand() > p:
            t.value = Node(id=id)
            t.right = Tree(parent=t)
            t.left = Tree(parent=t)
            id = generate(t.left, depth - 1, n_terminals, p=0.5, id=id+1)
            id = generate(t.right, depth - 1, n_terminals, p=0.5, id=id+1)
            t.depth = max(t.right.depth, t.left.depth) + 1
            t.n_nodes = t.right.n_nodes + t.left.n_nodes + 1
        else:
            t.value = np.random.randint(0, n_terminals)
            t.right, t.left = None, None
            t.n_nodes = 0
            t.depth = 1
    return id


class Node:

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


class Tree:

    def __init__(self, parent=None):
        self.value = None
        self.right = None
        self.left = None
        # self.depth = 0
        self.n_nodes = 0
        self.parent = parent
        self.level = 0


class Individual:

    def __init__(self, n_terminals, depth=5, p=0.0, id=0):
        tree = Tree()
        generate(tree, depth=depth, n_terminals=n_terminals, p=p, id=id)
        self.tree = deepcopy(tree)
        self.nodes = []
        self.mses = []
        self.mdls = []
        self.ids = []
        self.levels = []
        self.depth = self.tree.depth
        self.get_lists(self.tree)

    def update_lists(self):
        self.nodes = []
        self.mses = []
        self.mdls = []
        self.ids = []
        self.levels = []
        self.get_lists(self.tree)
        self.update_n_nodes_(self.tree) #added
        self.depth = max(self.levels)

    def get_mdl(self):
        # first element is root
        return self.mdls[0]

    def get_n_nodes(self):
        # number of all nodes
        return len(self.nodes)

    def get_lists(self, t, level=1):
        n = t.value
        t.level = level
        self.levels.append(level)
        self.nodes.append(n)
        if isinstance(n, Node):
            self.mses.append(n.mse)
            self.mdls.append(n.mdl)
            self.ids.append(n.id)
            self.get_lists(t.left, level+1)
            self.get_lists(t.right, level+1)
        elif n is None:
            print("ERROR: node is None")
        else:
            self.ids.append(np.nan)
            self.mses.append(np.nan)
            self.mdls.append(np.nan)

    def print_tree(self):
        self.print_tree_(self.tree)

    def print_tree_(self, t, level=0):
        if isinstance(t.value, Node):
            print("  " * level + 'Node'+str(t.value.id))
            self.print_tree_(t.right, level + 1)
            self.print_tree_(t.left, level + 1)
        else:
            print("  " * level + str(t.value))

    def gmdh(self, X, y):
        self.gmdh_process(self.tree, X, y)
        self.update_lists()

    def gmdh_process(self, t, X, y):
        nd = t.value
        if isinstance(nd, Node):
            if nd.a is None:
                nl = self.gmdh_process(t.left, X, y)
                nr = self.gmdh_process(t.right, X, y)
                nd.a = derive_a(Xmat_from_inputX(nl, nr), y)
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
        if isinstance(t.value, Node):
            if t.value.a is not None:
                if not self.valid(t.left):
                    return False
                if not self.valid(t.right):
                    return False
                return True
            else:
                return False
        else:
            return True

    def update_n_nodes(self):
        self.update_n_nodes_(self.tree)

    def update_n_nodes_(self, t):
        n = t.value
        if isinstance(n, Node):
            t.n_nodes = self.update_n_nodes_(t.left) + self.update_n_nodes_(t.right)
            return t.n_nodes
        elif n is None:
            print("ERROR: node is None")
        else:
            return 1

    def eval_tree(self, t, X):
        if isinstance(t.value, Node):
            if t.value.a is not None:
                return G(self.eval_tree(t.left, X), self.eval_tree(t.right, X), t.value.a)
            else:
                print('ERROR: node has null parameters')
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


