from copy import deepcopy
import numpy as np
from algorithms.gpregress.classes import SumNode, Node


def get_subtrees(child1, child2):
    # w1 = (child1.nodes[np.nanargmax(child1.mdls[1:]) + 1])
    # w2 = (child2.nodes[np.nanargmax(child2.mdls[1:]) + 1])
    # b1 = (child1.nodes[np.nanargmin(child1.mdls[1:]) + 1])
    # b2 = (child2.nodes[np.nanargmin(child2.mdls[1:]) + 1])
    # old_subt_1, _ = find_subtree(child1.tree, w1)
    # old_subt_2, _ = find_subtree(child2.tree, w2)
    # new_subt_1, _ = find_subtree(child1.tree, b1)
    # new_subt_2, _ = find_subtree(child2.tree, b2)
    w1 = (child1.ids[np.nanargmax(child1.mdls[1:]) + 1])
    w2 = (child2.ids[np.nanargmax(child2.mdls[1:]) + 1])
    b1 = (child1.ids[np.nanargmin(child1.mdls[1:]) + 1])
    b2 = (child2.ids[np.nanargmin(child2.mdls[1:]) + 1])
    old_subt_1, _ = find_subtree_by_id(child1.tree, w1)
    old_subt_2, _ = find_subtree_by_id(child2.tree, w2)
    new_subt_1, _ = find_subtree_by_id(child1.tree, b1)
    new_subt_2, _ = find_subtree_by_id(child2.tree, b2)
    new_subt_1 = deepcopy(new_subt_1)
    new_subt_2 = deepcopy(new_subt_2)
    return new_subt_1, new_subt_2, old_subt_1, old_subt_2


def find_subtree(tree, node, depth=0):
    if isinstance(tree.value, SumNode):
        # print(depth, node, tree.value)
        if tree.value == node:
            return tree, depth
        if tree.value == node:
            return tree, depth
        res = find_subtree(tree.left, node, depth=depth + 1)
        if res is not None:
            return res
        res = find_subtree(tree.right, node, depth=depth + 1)
        if res is not None:
            return res
    else:
        return None


def find_subtree_by_id(tree, node_id, depth=0):
    if isinstance(tree.value, SumNode):
        # print(depth, node, tree.value)
        if tree.value.id == node_id:
            return tree, depth
        if tree.value.id == node_id:
            return tree, depth
        res = find_subtree_by_id(tree.left, node_id, depth=depth + 1)
        if res is not None:
            return res
        res = find_subtree_by_id(tree.right, node_id, depth=depth + 1)
        if res is not None:
            return res
    if isinstance(tree.value, Node):
        if tree.value.id == node_id:
            return tree, depth
        if tree.value.id == node_id:
            return tree, depth
        res = find_subtree_by_id(tree.left, node_id, depth=depth + 1)
        if res is not None:
            return res
        if tree.value.atr == 2:
            res = find_subtree_by_id(tree.right, node_id, depth=depth + 1)
            if res is not None:
                return res
    else:
        return None


def substitute_subtrees(old_subt, new_subt):
    parent_subt = old_subt.parent
    new_subt.parent = parent_subt

    if parent_subt.left == old_subt:
        # replace left subtree with new subtree
        parent_subt.left = deepcopy(new_subt)
    elif parent_subt.right == old_subt:
        # replace right subtree with new subtree
        parent_subt.right = deepcopy(new_subt)

    backtrack_derived(parent_subt)


def backtrack_derived(t):
    t.value.a = None
    t.value.mdl = np.nan
    t.value.mse = np.nan
    # t.depth = max(t.left.depth, t.right.depth) + 1
    if t.parent is None:
        return
    else:
        backtrack_derived(t.parent)


def rand_node(ind):
    if np.isfinite(ind.mdls[1:]).any() == True: #ind.depth > 2:
        mdl = np.nan
        while np.isnan(mdl):
            node_ix = np.random.randint(len(ind.mdls) - 1) + 1
            mdl = ind.mdls[node_ix]
        return ind.ids[node_ix], node_ix
    else:
        return None, None