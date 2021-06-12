import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random
from time import time, sleep
from queue import PriorityQueue
from tqdm.auto import tqdm



def h_norm(l):
    m = np.min(l)
    M = np.max(l)
    if m == M:
        return np.ones(l.shape) * (1 / l.shape[0])
    temp_l = l - m
    return temp_l / np.sum(temp_l)


class Node:
    def __init__(self,move):
        self.current_move = move
        self.visited = False
        self.children = []
        self.parent = None
        self.number_of_visits = 0  # np.zeros(len(self.children)).astype(int)
        self.number_of_wins = 0  # np.zeros(len(self.children)).astype(int)
        # self.is_terminal = False
        self.changed = True
        # self.heuristic_value = 0

    def get_level(self):
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent
        return level

    def print_tree(self):
        spaces = ' ' * self.get_level() * 3
        prefix = spaces + "|__" if self.parent else ""
        if self.parent:
            print(prefix + str(self.current_move))

        if self.children:
            for child in self.children:
                child.print_tree()


def UCT(number_of_visits, number_of_wins, c=10, q=0.5):
    zeros = np.where(number_of_visits == 0)
    if zeros[0].size == 0:
        total = np.sum(number_of_visits)
        final_vector = h_norm((number_of_wins + np.abs(np.min(number_of_wins))) / number_of_visits + c * np.sqrt(
            np.log(total) / number_of_visits))
        # final_vector = v * q + (1 - q) * heuristics
        return np.argmax(final_vector)
    else:
        return np.random.choice(zeros[0])


def select(tree, path):
    # if tree.is_terminal:
    #     return tree
    if not tree.children:  # in other words - is a leaf
        if tree.visited:
            for move in range(9):
                temp_node = Node(move)
                temp_node.parent = tree
                tree.children.append(temp_node)
            tree.number_of_visits = np.zeros(len(tree.children)).astype(int)
            tree.number_of_wins = np.zeros(len(tree.children)).astype(int)
            path.append(np.random.randint(0,9))
            return tree.children[0]
        else:
            tree.visited = True
            return tree
    else:
        index = UCT(tree.number_of_visits, tree.number_of_wins, tree.heuristic_value)
        path.append(index)
        child = tree.children[index]
        return select(child, path)


def simulation(path,nodes_to_rollout):
    actions = path + [np.random.randint(0, 9) for _ in range(nodes_to_rollout)]
    # predicted_energy_loss = neural_net(actions)
    # cost, mean = multicriteria_objective_function(predicted_energy_loss)
    # return cost


def backpropagation(node, payout, path):
    while node.parent:
        index = path.pop(-1)
        node.parent.number_of_wins[index] += payout * node.parent.color
        node.parent.number_of_visits[index] += 1
        node = node.parent


def MCTS():
    tree = Node(0)
    for i in range(9):
        temp_node = Node(i)
        temp_node.parent = tree
        tree.children.append(temp_node)
    tree.number_of_visits = np.zeros(len(tree.children)).astype(int)
    tree.number_of_wins = np.zeros(len(tree.children)).astype(int)
    tree.visited = True

    count = 0
    t = 0.5
    t_0 = time()

    while True:
        count += 1
        # SELECTION
        if time() - t_0 > t:
            break
        path = []
        #     print("select")
        node = select(tree, path)

        # SIMULATION
        #     print("simulation")
        payout = simulation(node)

        # BACKPROPAGATION
        #     print("backpropagation")
        backpropagation(node, payout, path)

    move_id = np.argmax((tree.number_of_wins + 0.01) / (tree.number_of_visits + 0.01))
    return tree.children[move_id].current_move


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
