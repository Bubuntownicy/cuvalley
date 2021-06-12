import numpy as np
import matplotlib.pyplot as plt

def model(previous_data,current_move):
    pass

def plot_data(energy_loss, colling_level):
    plt.plot(energy_loss)
    # plt.plot(colling_level)
    plt.show()

if __name__ == '__main__':
    data = [line.split(',') for line in open(r"C:\Users\user\Downloads\lol.txt")]
    energy_loss = []
    colling_level = []
    for col,loss in data:
        energy_loss.append(float(loss[:-2]))
        colling_level.append(float(col))
    plot_data(energy_loss,colling_level)
