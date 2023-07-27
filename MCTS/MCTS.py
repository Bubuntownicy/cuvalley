import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random
from time import time, sleep
from queue import PriorityQueue
from tqdm.auto import tqdm
from objective_function.multicriteria_objective_function import multicriteria_objective_function
from furnace_simulator.predict import get_model



def h_norm(l):
    m = np.min(l)
    M = np.max(l)
    if m == M:
        return np.ones(l.shape) * (1 / l.shape[0])
    temp_l = l - m
    return temp_l / np.sum(temp_l)

def move_to_value(move):
    return [2000,2200,2375,2565,2750,2950,3125,3315,3500][move]

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
        index = UCT(tree.number_of_visits, tree.number_of_wins)
        path.append(index)
        child = tree.children[index]
        return select(child, path)


def simulation(path,l,neural_net,predicted_actions=400):
    actions_encrypted = l[-100:]+path + [np.random.randint(0, 9) for _ in range(predicted_actions-len(path)-len(l[-100:]))]
    actions = [[move_to_value(action)] for action in actions_encrypted]
    predicted_energy_loss = neural_net.predict([actions])
    cost, mean = multicriteria_objective_function(predicted_energy_loss)
    return cost


def backpropagation(node, payout, path):
    while node.parent:
        index = path.pop(-1)
        node.parent.number_of_wins[index] += payout
        node.parent.number_of_visits[index] += 1
        node = node.parent


def MCTS():
    neural_net = get_model(r"..\furnace_simulator\furnace_model")
    l = []
    loss = [2750]*400
    for turn in tqdm(range(800)):
        tree = Node(0)
        for i in range(9):
            temp_node = Node(i)
            temp_node.parent = tree
            tree.children.append(temp_node)
        tree.number_of_visits = np.zeros(len(tree.children)).astype(int)
        tree.number_of_wins = np.zeros(len(tree.children)).astype(int)
        tree.visited = True

        count = 0
        t = 1
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
            payout = simulation(path,l,neural_net)

            # BACKPROPAGATION
            #     print("backpropagation")
            backpropagation(node, payout, path)
            print(count)

        move_id = np.argmax((tree.number_of_wins + 0.01) / (tree.number_of_visits + 0.01))
        l.append(tree.children[move_id].current_move)
        if turn>401:
            l2 = [move_to_value(move) for move in l]
            loss.append(neural_net.predict([[[action] for action in l2]])[0][0])
    print(repr(l2))
    print(repr(loss))
    plt.plot(loss)
    plt.show()
    plt.plot(l2)
    plt.show()

if __name__ == '__main__':
    # np.random.seed(0)
    # random.seed(0)
    print(MCTS())