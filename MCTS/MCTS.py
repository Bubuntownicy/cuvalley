import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random
from time import time, sleep
from queue import PriorityQueue
from tqdm.auto import tqdm
from objective_function.multicriteria_objective_function import multicriteria_objective_function
from furnace_simulator.predict import get_model
from data.data_getter import data_getter4 as data_getter


def h_norm(l):
    m = np.min(l)
    M = np.max(l)
    if m == M:
        return np.ones(l.shape) * (1 / l.shape[0])
    temp_l = l - m
    return temp_l / np.sum(temp_l)


def move_to_value(move):
    if move < 9:
        return [2000, 2200, 2375, 2565, 2750, 2950, 3125, 3315, 3500][move]
    return move


class Node:
    def __init__(self, move):
        self.current_move = move
        self.visited = False
        self.children = []
        self.parent = None
        self.number_of_visits = 0
        self.number_of_wins = 0
        self.changed = True

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
        return np.argmax(final_vector)
    else:
        return np.random.choice(zeros[0])


def select(tree, path):
    if not tree.children:
        if tree.visited:
            for move in range(9):
                temp_node = Node(move)
                temp_node.parent = tree
                tree.children.append(temp_node)
            tree.number_of_visits = np.zeros(len(tree.children)).astype(int)
            tree.number_of_wins = np.zeros(len(tree.children)).astype(float)
            path.append(np.random.randint(0, 9))
            return tree.children[path[-1]]
        else:
            tree.visited = True
            return tree
    else:
        index = UCT(tree.number_of_visits, tree.number_of_wins)
        path.append(index)
        child = tree.children[index]
        return select(child, path)


def combine(data, actions, losses):
    combined = np.vstack([np.array(losses), np.array(actions)]).T
    return np.vstack([data, combined])


def simulation(path, neural_net, data, predicted_actions=200):
    actions = []
    loss = []
    for i in range(predicted_actions):
        temp_data = combine(data, actions, loss)[-predicted_actions:]
        # print(temp_data)
        loss.append(neural_net.predict(np.array([temp_data]))[0][0])
        if i < len(path):
            actions.append(move_to_value(path[i]))
        else:
            actions.append(move_to_value(np.random.randint(0, 9)))
    cost, mean = multicriteria_objective_function(data[-200:,0])
    return cost, actions[0], loss[0]


def backpropagation(node, payout, path):
    while node.parent:
        index = path.pop(-1)
        node.parent.number_of_wins[index] += payout
        node.parent.number_of_visits[index] += 1
        node = node.parent


def MCTS(data, turns=5, t=180):
    neural_net = get_model(r"..\furnace_simulator\furnace_model")
    predicted_data = []
    actions = []
    for turn in tqdm(range(turns)):
        tree = Node(-1)
        for i in range(9):
            temp_node = Node(i)
            temp_node.parent = tree
            tree.children.append(temp_node)
        tree.number_of_visits = np.zeros(len(tree.children)).astype(int)
        tree.number_of_wins = np.zeros(len(tree.children)).astype(float)
        tree.visited = True

        count = 0
        t_0 = time()

        while True:
            count += 1
            # SELECTION
            if time() - t_0 > t:
                break
            path = []
            node = select(tree, path)

            # SIMULATION
            payout, action, loss = simulation(path, neural_net, data)
            data = combine(data, action, loss)
            predicted_data.append(loss)

            # BACKPROPAGATION
            backpropagation(node, payout, path)

        move_id = np.argmax((tree.number_of_wins + 0.01) / (tree.number_of_visits + 0.01))
        actions.append(tree.children[move_id].current_move)
    print("Optymalnym ustawieniem przepływu powietrze w danym momencie jest:",actions[-1])
    plt.plot(data[-200:, 0])
    plt.plot(data[-200:, 1] / 150)
    plt.show()
